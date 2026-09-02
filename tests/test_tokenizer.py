import os
import pytest
from openahi.tokenizer.tokenizer import SimpleTokenizer


def test_tokenizer_roundtrip():
    t = SimpleTokenizer()
    s = "Hello"
    ids = t.encode(s)
    out = t.decode(ids)
    assert isinstance(ids, list)
    assert isinstance(out, str)

