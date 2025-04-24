# train_qwen.py
import os
import json
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import wandb
import deepspeed
from deepspeed.accelerator import get_accelerator
from rewardmodel import RewardModel
import shutil
from dataset import PairwiseDataset, load_pairwise_samples
import torch.nn.functional as F


def evaluate(model, dataloader, device, margin_loss_fn):
    """
    在验证集上计算平均 loss 和 ranking accuracy，
    正候选得分 > 负候选得分视为预测正确。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids_pos = batch["input_ids_pos"].to(device)
            attention_mask_pos = batch["attention_mask_pos"].to(device)
            input_ids_neg = batch["input_ids_neg"].to(device)
            attention_mask_neg = batch["attention_mask_neg"].to(device)
            
            # 直接调用模型获得得分
            pos_score = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos)  # [batch]
            neg_score = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg)  # [batch]
            # 这里构造一个全部为1的 target 张量，其实 MarginRankingLoss 会将目标与 (pos - neg) 比较
            target = torch.ones(pos_score.size(), device=device)
            loss = margin_loss_fn(pos_score, neg_score, target)
            total_loss += loss.item() * pos_score.size(0)
            total_correct += (pos_score > neg_score).sum().item()
            total_samples += pos_score.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def score_sentence(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # 截取 logits（去除最后一时刻，因为没有对应的 target token）
    logits = outputs.logits[:, :-1, :]          # shape: [batch, seq_len-1, vocab_size]
    target_ids = input_ids[:, 1:]                 # shape: [batch, seq_len-1]
    target_mask = attention_mask[:, 1:].float()   # shape: [batch, seq_len-1]
    
    # 限制 logits 值，避免过大或过小导致 log_softmax 极值
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    
    # 计算对数概率
    log_probs = torch.log_softmax(logits, dim=-1)  # shape: [batch, seq_len-1, vocab_size]
    # 按照 target token 的索引，收集对应的 log 概率
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len-1]
    
    # 将 pad 区域的 log_probs 置 0（因为 target_mask 处于 0 的位置应不参与计算）
    token_log_probs = token_log_probs * target_mask
    
    # 计算每个样本的有效 token 数量，避免除零
    lengths = target_mask.sum(dim=1).clamp(min=1.0)
    sentence_score = token_log_probs.sum(dim=1) / lengths  # 每个样本的平均对数概率
    
    # 将非数（nan）或无限值替换为 -100（或者你认为合适的极端负值）
    sentence_score = torch.nan_to_num(sentence_score, nan=-100.0, posinf=-100.0, neginf=-100.0)
    return sentence_score

def main(args):
    # 设置环境变量（也可在命令行中设置）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 初始化 DeepSpeed 分布式环境
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device = torch.device("cuda", local_rank)

    # 主进程初始化 wandb
    if torch.distributed.get_rank() == 0:
        wandb.init(project=args.wandb_project, config=vars(args))

    # 加载分词器，确保 pad_token 被设置
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载训练和验证数据（需要你自定义 dataset.py 提供 load_pairwise_samples 和 PairwiseDataset）
    train_dirs = args.train_dirs.split(',')
    val_dirs = args.val_dirs.split(',')
    train_pairs = load_pairwise_samples(train_dirs)
    val_pairs = load_pairwise_samples(val_dirs)
    print(f"Loaded {len(train_pairs)} training pairs; {len(val_pairs)} validation pairs.")

    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=args.max_length)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=2)

    model = RewardModel(args.model_name)

    # 初始化优化器、调度器及 loss 函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    margin_loss_fn = nn.MarginRankingLoss(margin=args.margin)

    # 使用 DeepSpeed.initialize() 封装模型、优化器和其他配置
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=args.deepspeed_config
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        ds_engine.train()
        # 在训练循环中
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids_pos = batch["input_ids_pos"].to(device)
            attention_mask_pos = batch["attention_mask_pos"].to(device)
            input_ids_neg = batch["input_ids_neg"].to(device)
            attention_mask_neg = batch["attention_mask_neg"].to(device)
            
            # 建议先关闭 autocast 以调试数值稳定性，或者在全精度下运行
            with torch.amp.autocast('cuda'):
                # 直接调用模型 forward 获得得分
                pos_score = ds_engine(input_ids=input_ids_pos, attention_mask=attention_mask_pos)  # shape: [batch]
                neg_score = ds_engine(input_ids=input_ids_neg, attention_mask=attention_mask_neg)  # shape: [batch]
                # 使用 pairwise ranking loss： -log(sigmoid(pos_score - neg_score))
                loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
                    
            ds_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            ds_engine.step()
            scheduler.step()
            get_accelerator().empty_cache()

            if step % args.log_steps == 0 and torch.distributed.get_rank() == 0:
                print(f"Epoch {epoch} step {step}/{len(train_dataloader)} loss: {loss.item()}")
                wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

        # 每个 epoch 后在验证集上进行评估
        val_loss, val_acc = evaluate(ds_engine.module, val_dataloader, device, margin_loss_fn)
        if torch.distributed.get_rank() == 0:
            print(f"Epoch {epoch} validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
            wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "epoch": epoch})
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0 # 重置计数器
                print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
                # --- 开始修改 ---
                # 获取未包装的模型 (去除 DeepSpeed wrapper)
                model_to_save = ds_engine.module
                # 定义保存目录 (例如使用 best_model 或 checkpoint 目录)
                save_directory = os.path.join(args.output_dir, "best_model") # 可以自定义目录名
                if os.path.exists(save_directory):
                        shutil.rmtree(save_directory) # 清理旧的最佳模型
                os.makedirs(save_directory, exist_ok=True)

                # 使用 save_pretrained 保存完整模型权重和配置
                model_to_save.save_pretrained(save_directory)
                # 同时保存 tokenizer，方便后续加载
                tokenizer.save_pretrained(save_directory)
                print(f"Best model saved to {save_directory}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    if torch.distributed.get_rank() == 0:
        print("Training finished. Saving final model...")
        final_save_directory = os.path.join(args.output_dir, "final_model") # 可以自定义目录名
        if os.path.exists(final_save_directory):
            shutil.rmtree(final_save_directory) # 清理旧的最终模型 (如果需要)
        os.makedirs(final_save_directory, exist_ok=True)

        # 获取未包装的模型
        model_to_save = ds_engine.module
        # 保存最终模型
        model_to_save.save_pretrained(final_save_directory)
        # 保存 tokenizer
        tokenizer.save_pretrained(final_save_directory)
        print(f"Final model saved to {final_save_directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeed 多卡训练 Qwen2.5-0.5B 模型的微调示例")
    parser.add_argument("--train_dirs", type=str,
                        default="../VisualWebArena/reddit_gpt4v_som,../VisualWebArena/shopping_gpt4v_som",
                        help="训练集 JSON 文件所在的目录，多个目录用逗号分隔")
    parser.add_argument("--val_dirs", type=str,
                        default="../VisualWebArena/classifieds_gpt4v_som",
                        help="验证集 JSON 文件所在的目录，多个目录用逗号分隔")
    # 修改默认预训练模型为 Qwen2.5-0.5B
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="预训练模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="reward_model_out",
                        help="保存微调后模型的目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="每个 GPU 上的 batch size，建议设置较低以节省显存")
    parser.add_argument("--epochs", type=int, default=20, help="训练的总 epoch 数")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--margin", type=float, default=1.0, help="ranking loss 中的 margin")
    parser.add_argument("--log_steps", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--wandb_project", type=str, default="reward_model_project", help="Weights & Biases 项目名称")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="连续多少个 epoch 验证准确率无提升后提前停止")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="由分布式启动器传入的 Local rank")
    args = parser.parse_args()
    main(args)