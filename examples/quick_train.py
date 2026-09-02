import os
import torch
from openahi.tokenizer.tokenizer import SimpleTokenizer
from openahi.data.dataset import load_dataset_from_file
from openahi.config import DEFAULT_CONFIG
from openahi.models.composter import ComposterModel
from openahi.training.trainer import Trainer


def run_quick_train():
    tokenizer = SimpleTokenizer()
    ds = load_dataset_from_file(os.path.join('data', 'tiny.txt'), tokenizer, DEFAULT_CONFIG.block_size)
    model = ComposterModel(DEFAULT_CONFIG)
    trainer = Trainer(model, tokenizer, DEFAULT_CONFIG)
    trainer.train(ds, batch_size=2, epochs=2, lr=1e-3, save_path='checkpoints/composter.pt')


if __name__ == '__main__':
    run_quick_train()
