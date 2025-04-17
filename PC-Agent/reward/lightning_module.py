import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from rewardmodel import RewardModel  # 请确保 rewardmodel.py 支持接收模型名称

class RewardModelLightning(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters(hparams)
        # 加载预训练 RewardModel（例如 Mistral-7B）
        self.model = RewardModel(self.hparams.model_name)
        # 应用 LoRA 微调配置
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=self.hparams.lora_target_modules.split(","),
            lora_dropout=self.hparams.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.margin_loss_fn = nn.MarginRankingLoss(margin=self.hparams.margin)
        # 使用 torchmetrics 记录验证指标
        self.val_loss_metric = torchmetrics.MeanMetric()
        self.val_accuracy_metric = torchmetrics.Accuracy()
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def training_step(self, batch, batch_idx):
        # 正样本与负样本分别计算 score
        device = batch["input_ids_pos"].device
        pos_scores = self(input_ids=batch["input_ids_pos"],
                          attention_mask=batch["attention_mask_pos"])
        neg_scores = self(input_ids=batch["input_ids_neg"],
                          attention_mask=batch["attention_mask_neg"])
        target = torch.ones(pos_scores.size(), device=device)
        loss = self.margin_loss_fn(pos_scores, neg_scores, target)
        # 记录训练 loss（步与 epoch 内均记录）
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        device = batch["input_ids_pos"].device
        pos_scores = self(input_ids=batch["input_ids_pos"],
                          attention_mask=batch["attention_mask_pos"])
        neg_scores = self(input_ids=batch["input_ids_neg"],
                          attention_mask=batch["attention_mask_neg"])
        target = torch.ones(pos_scores.size(), device=device)
        loss = self.margin_loss_fn(pos_scores, neg_scores, target)
        # 简单方式：以比较结果计算 accuracy（注意：这里直接平均各样本准确率）
        accuracy = (pos_scores > neg_scores).float().mean()
        # 更新 torchmetrics 记录
        self.val_loss_metric.update(loss)
        self.val_accuracy_metric.update(accuracy)
        # 同步记录每个 step 的值
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return {"val_loss": loss, "val_accuracy": accuracy}
    
    def on_validation_epoch_end(self):
        # 在验证 epoch 结束后计算 aggregated 指标并记录
        avg_loss = self.val_loss_metric.compute()
        avg_acc = self.val_accuracy_metric.compute()
        self.log("avg_val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("avg_val_accuracy", avg_acc, prog_bar=True, sync_dist=True)
        # 重置指标
        self.val_loss_metric.reset()
        self.val_accuracy_metric.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        # 通过 Trainer 提供的 estimated_stepping_batches 自动计算总步数
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}