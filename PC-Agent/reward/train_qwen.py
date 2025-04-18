import os
import json
import random
import argparse
import torch
import torch.nn as nn
# 导入 RandomSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import wandb
# 移除 deepspeed 导入
# import deepspeed
# from deepspeed.accelerator import get_accelerator # 移除
from rewardmodel import RewardModel # 确保这个类能被正确导入
import shutil
from dataset import PairwiseDataset, load_pairwise_samples # 确保这个也能被正确导入
import torch.nn.functional as F
# 导入 GradScaler 用于混合精度
from torch.cuda.amp import GradScaler, autocast

def evaluate(model, dataloader, device):
    """
    在验证集上计算平均 loss 和 ranking accuracy，
    正候选得分 > 负候选得分视为预测正确。
    (移除了 margin_loss_fn 参数，因为训练和评估用同一个 loss 定义更清晰)
    """
    model.eval() # 设置为评估模式
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids_pos = batch["input_ids_pos"].to(device)
            attention_mask_pos = batch["attention_mask_pos"].to(device)
            input_ids_neg = batch["input_ids_neg"].to(device)
            attention_mask_neg = batch["attention_mask_neg"].to(device)

            # 使用 autocast 进行推理，以防模型内部也使用混合精度
            with autocast(enabled=True): # 显式启用 autocast
                pos_score = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos)  # [batch]
                neg_score = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg)  # [batch]
                # 计算 pairwise ranking loss
                loss = -torch.mean(F.logsigmoid(pos_score - neg_score))

            total_loss += loss.item() * pos_score.size(0)
            total_correct += (pos_score > neg_score).sum().item()
            total_samples += pos_score.size(0)

    # 避免除以零
    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# score_sentence 函数（如果需要可以保留，此处未修改）
# ...

def main(args):
    # 设置环境变量（可选）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # 可能有助于显存管理

    # 移除 DeepSpeed 分布式初始化
    # deepspeed.init_distributed()

    # 设置单 GPU 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not found, using CPU!")

    # 初始化 wandb (不再需要检查 rank)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        print("Weights & Biases initialized.")
    else:
        print("Weights & Biases disabled.")


    # 加载分词器
    print(f"Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    print("Loading datasets...")
    train_dirs = args.train_dirs.split(',')
    val_dirs = args.val_dirs.split(',')
    train_pairs = load_pairwise_samples(train_dirs)
    val_pairs = load_pairwise_samples(val_dirs)
    print(f"Loaded {len(train_pairs)} training pairs from {args.train_dirs}")
    print(f"Loaded {len(val_pairs)} validation pairs from {args.val_dirs}")

    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=args.max_length)

    # 使用 RandomSampler 替换 DistributedSampler
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers) # 使用参数 num_workers

    # 验证集采样器保持不变
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers) # 使用参数 num_workers
    print(f"Dataloaders created with batch size: {args.batch_size}")

    # 初始化模型并移动到设备
    print(f"Initializing RewardModel with base: {args.model_name}")
    model = RewardModel(args.model_name)
    model.to(device) # 将模型移动到 GPU
    print("Model moved to device:", device)

    # 初始化优化器、调度器 (标准方式)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    print(f"Total training steps: {total_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps), # 使用比例计算 warmup steps
        num_training_steps=total_steps
    )

    # 初始化 GradScaler 用于混合精度训练
    scaler = GradScaler(enabled=True) # 显式启用
    print("Initialized AdamW optimizer, linear scheduler, and GradScaler for AMP.")

    best_val_acc = 0.0
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train() # 设置为训练模式
        # train_sampler.set_epoch(epoch) # RandomSampler 不需要这个
        total_train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # 将数据移动到设备
            input_ids_pos = batch["input_ids_pos"].to(device)
            attention_mask_pos = batch["attention_mask_pos"].to(device)
            input_ids_neg = batch["input_ids_neg"].to(device)
            attention_mask_neg = batch["attention_mask_neg"].to(device)

            optimizer.zero_grad() # 清空梯度

            # 使用 autocast 进行前向传播
            with autocast(enabled=True): # 显式启用 autocast
                pos_score = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos)  # shape: [batch]
                neg_score = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg)  # shape: [batch]
                # 计算 pairwise ranking loss
                loss = -torch.mean(F.logsigmoid(pos_score - neg_score))

            # 使用 scaler 进行反向传播
            scaler.scale(loss).backward()

            # 可选：梯度裁剪 (在 unscale_ 之前或之后都可以，通常在之后)
            # scaler.unscale_(optimizer) # 先 unscale 梯度
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm) # 使用参数 max_grad_norm

            # 使用 scaler 更新权重
            scaler.step(optimizer)
            scaler.update() # 更新 scaler 状态

            scheduler.step() # 更新学习率

            total_train_loss += loss.item()

            # 日志打印 (不再需要检查 rank)
            if step % args.log_steps == 0:
                avg_step_loss = total_train_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e}")
                if args.use_wandb:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/avg_loss": avg_step_loss,
                        "train/learning_rate": current_lr,
                        "epoch": epoch + 1, # 从 1 开始记录 epoch
                        "step": step
                    })

        # --- Epoch 结束 ---
        avg_epoch_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")

        # 每个 epoch 后在验证集上进行评估
        print(f"Starting evaluation for epoch {epoch+1}...")
        val_loss, val_acc = evaluate(model, val_dataloader, device) # 直接传递 model

        print(f"Epoch {epoch+1} Validation Results -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        if args.use_wandb:
            wandb.log({
                "validation/loss": val_loss,
                "validation/accuracy": val_acc,
                "epoch": epoch + 1
            })

        # 保存最佳模型逻辑 (不再需要检查 rank)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0 # 重置计数器
            print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")

            # model_to_save = ds_engine.module # 不再需要
            model_to_save = model # 直接使用 model

            save_directory = os.path.join(args.output_dir, "best_model")
            if os.path.exists(save_directory):
                shutil.rmtree(save_directory)
            os.makedirs(save_directory, exist_ok=True)

            model_to_save.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print(f"Best model saved to {save_directory}")
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epochs.")

        # 早停逻辑
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
            break

    # 训练结束后保存最终模型 (不再需要检查 rank)
    print("Training finished. Saving final model...")
    final_save_directory = os.path.join(args.output_dir, "final_model")
    if os.path.exists(final_save_directory):
        shutil.rmtree(final_save_directory)
    os.makedirs(final_save_directory, exist_ok=True)

    # model_to_save = ds_engine.module # 不再需要
    model_to_save = model # 直接使用 model

    model_to_save.save_pretrained(final_save_directory)
    tokenizer.save_pretrained(final_save_directory)
    print(f"Final model saved to {final_save_directory}")

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single GPU Training Example for Reward Model")
    # 数据和模型路径参数
    parser.add_argument("--train_dirs", type=str,
                        default="../VisualWebArena/reddit_gpt4v_som,../VisualWebArena/shopping_gpt4v_som",
                        help="Training data directories (comma-separated)")
    parser.add_argument("--val_dirs", type=str,
                        default="../VisualWebArena/classifieds_gpt4v_som",
                        help="Validation data directories (comma-separated)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="reward_model_out_single_gpu", # 修改默认输出目录
                        help="Directory to save the finetuned model")
    # 训练超参数
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU (adjust based on VRAM)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs") # 减少默认 epochs 方便测试
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup steps ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (if used)")
    # Loss 参数 (如果 evaluate 中用 MarginLoss)
    # parser.add_argument("--margin", type=float, default=1.0, help="Margin for MarginRankingLoss")
    # 其他参数
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
    parser.add_argument("--log_steps", type=int, default=10, help="Logging interval (steps)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--use_wandb", action='store_true', help="Enable Weights & Biases logging") # 添加 W&B 开关
    parser.add_argument("--wandb_project", type=str, default="reward_model_single_gpu", help="Weights & Biases project name")

    # 移除 DeepSpeed 和 LoRA 相关参数
    # parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed config path")
    # parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    # parser.add_argument("--lora_r", type=int, default=8, help="LoRA r")
    # ... 其他 LoRA 参数 ...

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)