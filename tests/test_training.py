import os
import torch
from openahi.tokenizer.tokenizer import SimpleTokenizer
from openahi.data.dataset import load_dataset_from_file
from openahi.config import DEFAULT_CONFIG
from openahi.models.composter import ComposterModel
from openahi.training.trainer import Trainer


def test_train_and_checkpoint(tmp_path):
    tokenizer = SimpleTokenizer()
    ds = load_dataset_from_file(os.path.join('data', 'tiny.txt'), tokenizer, DEFAULT_CONFIG.block_size)
    model = ComposterModel(DEFAULT_CONFIG)
    trainer = Trainer(model, tokenizer, DEFAULT_CONFIG)
    ckpt = tmp_path / 'composter.pt'
    trainer.train(ds, batch_size=2, epochs=1, lr=1e-3, save_path=str(ckpt))
    assert ckpt.exists()
    data = torch.load(str(ckpt), map_location='cpu')
    assert 'model_state_dict' in data

