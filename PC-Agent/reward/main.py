# main.py
import pytorch_lightning as pl
from lightning_module import RewardModelLightning
from data_module import RewardDataModule
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", type=str, default="../VisualWebArena/reddit_gpt4v_som,../VisualWebArena/shopping_gpt4v_som")
    parser.add_argument("--val_dirs", type=str, default="../VisualWebArena/classifieds_gpt4v_som")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--wandb_project", type=str, default="reward_model_project")
    args = parser.parse_args()

    hparams = {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    }

    dm = RewardDataModule(
        train_dirs=args.train_dirs,
        val_dirs=args.val_dirs,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    model = RewardModelLightning(hparams)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=7,
        strategy="deepspeed_stage_3",  # 使用 DeepSpeed ZeRO Stage 3 (你也可以指定其它 DeepSpeed 策略)
        precision="16-mixed",  # 使用混合精度
        log_every_n_steps=10
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()