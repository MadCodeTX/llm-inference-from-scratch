import pytest
from inference_from_scratch.tokenizer import ByteTokenizer, CharTokenizer, BPETokenizer


def test_byte_roundtrip_unicode_emoji():
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
    for s in ["hello", "héllo wörld", "emoji 🎉🚀", "combining é", "", "日本語テスト"]:
        ids = tok.encode(s)
        assert all(0 <= i < 256 for i in ids)
        assert tok.decode(ids) == s


def test_byte_multibyte_len():
    tok = ByteTokenizer()
    # rocket emoji is 4 utf-8 bytes
    assert len(tok.encode("🚀")) == 4


def test_char_roundtrip():
    text = "hello world, hello inference!"
    tok = CharTokenizer.build(text)
    assert tok.vocab_size == len(set(text))
    for s in ["hello", "world", text, ""]:
        assert tok.decode(tok.encode(s)) == s
    with pytest.raises(ValueError):
        tok.encode("xyz unseen")


def test_bpe_merges_reduce_and_roundtrip():
    text = "the quick brown fox jumps over the lazy dog. " * 20 + "the the the the " * 20
    tok = BPETokenizer.train(text, vocab_size=300)
    assert tok.vocab_size == 300
    assert len(tok.merges) == 44  # 300-256
    byte_len = len(text.encode("utf-8"))
    ids = tok.encode(text)
    assert len(ids) < byte_len  # merges compress repetitive text
    assert tok.decode(ids) == text
    # roundtrip on training alphabet subsets
    for s in ["the quick", "brown fox", "the the the"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_unicode_roundtrip():
    text = "héllo héllo héllo wörld wörld 🎉🎉🎉 " * 10
    tok = BPETokenizer.train(text, vocab_size=280)
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_bpe_invalid_vocab():
    with pytest.raises(ValueError):
        BPETokenizer.train("hello", vocab_size=100)
