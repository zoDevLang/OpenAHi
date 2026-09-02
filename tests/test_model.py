import torch
from openahi.config import DEFAULT_CONFIG
from openahi.models.composter import ComposterModel
from openahi.tokenizer.tokenizer import SimpleTokenizer


def test_model_forward_shapes():
    config = DEFAULT_CONFIG
    model = ComposterModel(config)
    tok = SimpleTokenizer()
    ids = tok.encode("Hi", add_bos=True, add_eos=True)
    import torch
    x = torch.tensor([ids[:config.block_size]], dtype=torch.long)
    logits, loss = model(x, x)
    assert logits.shape[0] == 1
    assert logits.shape[1] == x.shape[1]
    assert logits.shape[2] == config.vocab_size

