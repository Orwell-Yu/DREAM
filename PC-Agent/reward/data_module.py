# data_module.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from dataset import PairwiseDataset, load_pairwise_samples
from transformers import AutoTokenizer

class RewardDataModule(pl.LightningDataModule):
    def __init__(self, train_dirs, val_dirs, model_name, max_length, batch_size, num_workers=2):
        super().__init__()
        self.train_dirs = train_dirs.split(',')
        self.val_dirs = val_dirs.split(',')
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_pairs = load_pairwise_samples(self.train_dirs)
        self.val_pairs = load_pairwise_samples(self.val_dirs)
        self.train_dataset = PairwiseDataset(self.train_pairs, self.tokenizer, max_length=self.max_length)
        self.val_dataset = PairwiseDataset(self.val_pairs, self.tokenizer, max_length=self.max_length)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset) if self.trainer.use_ddp else None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        sampler = SequentialSampler(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)