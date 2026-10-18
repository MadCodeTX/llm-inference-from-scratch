"""Tests for Ch 09: safetensors by hand, HF config, weight mapping.

The safetensors fixtures are built with ``struct`` + ``json`` only --
the same bytes the format spec describes -- so the hand parser is
tested against the spec, not against the library it dissects. The
mapping tests enumerate a live ``DecoderLM``'s real ``state_dict()``
keys (never a guessed list): a fake HF checkpoint is built by
renaming the tiny model's own weights *inversely*, then mapped back,
validated, loaded into a fresh model, and required to reproduce the
original logits exactly. Tiny CPU dims throughout (d_model<=64,
n_layer<=2).
"""

import json
import struct

import pytest
import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.model import DecoderLM
from inference_from_scratch.weights import (
    LazySafetensorsFile,
    format_safetensors_table,
    load_hf_config,
    load_safetensors,
    map_hf_state_dict,
    read_safetensors_header,
    safetensors_table,
    validate_state_dict,
)


# --- hand-built safetensors bytes (struct + json only) ---------------------

_TORCH_TO_ST = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.int64: "I64",
    torch.int32: "I32",
}


def _write_safetensors(path, tensors, metadata=None, pad_after_first=0):
    """Write a real safetensors file by hand: <Q length + JSON + bytes.

    ``tensors`` is an ordered {name: Tensor} (float/int dtypes numpy
    can serialise). Offsets honour ``pad_after_first`` zero bytes of
    gap after the first tensor, so readers must use the header's
    offsets rather than assuming tensors are concatenated.
    Returns the header dict that was written.
    """
    header = {}
    chunks = []
    offset = 0
    for idx, (name, tensor) in enumerate(tensors.items()):
        raw = tensor.numpy().tobytes()
        header[name] = {
            "dtype": _TORCH_TO_ST[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
        if idx == 0 and pad_after_first:
            chunks.append(b"\x00" * pad_after_first)
            offset += pad_after_first
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)
    path.write_bytes(blob)
    return header


def test_header_roundtrip_hand_built_bytes(tmp_path):
    path = tmp_path / "tiny.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "model.layers.0.mlp.down_proj.weight": torch.arange(6, dtype=torch.int64).reshape(2, 3),
    }
    written = _write_safetensors(path, tensors, metadata={"format": "pt"})

    header = read_safetensors_header(path)
    assert header == written  # exact roundtrip, metadata included
    assert header["model.embed_tokens.weight"] == {
        "dtype": "F32",
        "shape": [3, 4],
        "data_offsets": [0, 48],
    }
    assert header["__metadata__"] == {"format": "pt"}


def test_header_table_reports_dtype_shape_offsets(tmp_path):
    path = tmp_path / "tiny.safetensors"
    _write_safetensors(
        path,
        {
            "a": torch.zeros(2, 3, dtype=torch.float32),  # 24 bytes
            "b": torch.zeros(4, dtype=torch.int64),  # 32 bytes
        },
    )
    rows = safetensors_table(read_safetensors_header(path))
    assert rows == [
        {"name": "a", "dtype": "F32", "shape": [2, 3], "numel": 6,
         "begin": 0, "end": 24, "nbytes": 24},
        {"name": "b", "dtype": "I64", "shape": [4], "numel": 4,
         "begin": 24, "end": 56, "nbytes": 32},
    ]
    text = format_safetensors_table(read_safetensors_header(path))
    assert "F32" in text and "[2, 3]" in text and "[24, 56)" in text


def test_load_safetensors_eager_and_lazy_agree(tmp_path):
    path = tmp_path / "tiny.safetensors"
    tensors = {
        "w": torch.randn(4, 5),
        "ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "scalar": torch.tensor(2.5, dtype=torch.float64),
    }
    _write_safetensors(path, tensors, pad_after_first=8)  # gap: offsets rule

    eager = load_safetensors(path)
    assert list(eager) == list(tensors)
    for name, want in tensors.items():
        assert eager[name].dtype == want.dtype
        assert eager[name].shape == want.shape
        assert torch.equal(eager[name], want)

    with LazySafetensorsFile(path) as lazy:
        assert lazy.keys() == list(tensors)
        assert len(lazy) == 3
        assert "w" in lazy and "nope" not in lazy
        # Lazy: one tensor at a time, same values as eager.
        assert torch.equal(lazy.load("ids"), tensors["ids"])
        with pytest.raises(KeyError):
            lazy.load("nope")


def test_load_safetensors_bfloat16_bits(tmp_path):
    # BF16 has no numpy dtype: hand the writer raw uint16 bit patterns.
    want = torch.tensor([1.5, -2.25, 0.0], dtype=torch.bfloat16)
    raw = want.view(torch.uint16).numpy().tobytes()
    header = {"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, len(raw)]}}
    header_bytes = json.dumps(header).encode()
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + raw)

    out = load_safetensors(path)["x"]
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, want)


def test_header_rejects_truncated_and_malformed(tmp_path):
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x05\x00")  # not even the 8-byte prefix
    with pytest.raises(ValueError, match="8-byte"):
        read_safetensors_header(short)

    bad_json = tmp_path / "bad.safetensors"
    payload = b"not json"
    bad_json.write_bytes(struct.pack("<Q", len(payload)) + payload)
    with pytest.raises(ValueError, match="not valid JSON"):
        read_safetensors_header(bad_json)

    bad_span = tmp_path / "span.safetensors"
    header = {"x": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 4]}}
    hb = json.dumps(header).encode()
    bad_span.write_bytes(struct.pack("<Q", len(hb)) + hb + b"\x00" * 16)
    with pytest.raises(ValueError, match="needs 16"):
        read_safetensors_header(bad_span)


# --- load_hf_config ---------------------------------------------------------

HF_FIXTURE = {
    "architectures": ["Qwen2ForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 256,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


def test_load_hf_config_from_fixture_dict():
    cfg = load_hf_config(HF_FIXTURE)
    assert cfg == ModelConfig(
        vocab_size=128, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=256, rope_theta=1000000.0, rms_eps=1e-6,
        tie_embeddings=True, norm_type="rmsnorm", mlp_type="swiglu",
        pos_type="rope", attention_bias=True,
    )
    assert cfg.head_dim == 8


def test_load_hf_config_attention_bias_explicit_and_family():
    explicit_false = dict(HF_FIXTURE, attention_bias=False)
    assert load_hf_config(explicit_false).attention_bias is False
    explicit_true = {
        k: v for k, v in HF_FIXTURE.items() if k != "architectures"
    }
    explicit_true["attention_bias"] = True
    assert load_hf_config(explicit_true).attention_bias is True
    llama_like = {k: v for k, v in HF_FIXTURE.items() if k != "architectures"}
    llama_like["model_type"] = "llama"
    assert load_hf_config(llama_like).attention_bias is False
    qwen_like = dict(llama_like, model_type="qwen2")
    assert load_hf_config(qwen_like).attention_bias is True


def test_load_hf_config_gqa_defaults_when_kv_heads_missing():
    fixture = {k: v for k, v in HF_FIXTURE.items() if k != "num_key_value_heads"}
    cfg = load_hf_config(fixture)
    assert cfg.n_kv_head == cfg.n_head == 4  # MHA default, not a guess at GQA

    fixture_none = dict(fixture, num_key_value_heads=None)
    assert load_hf_config(fixture_none).n_kv_head == 4


def test_load_hf_config_from_file_and_directory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(HF_FIXTURE))
    from_file = load_hf_config(tmp_path / "config.json")
    from_dir = load_hf_config(tmp_path)
    assert from_file == from_dir == load_hf_config(HF_FIXTURE)

    with pytest.raises(FileNotFoundError):
        load_hf_config(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="hidden_size"):
        load_hf_config({"vocab_size": 10})


# --- map_hf_state_dict + validate_state_dict --------------------------------

def _tiny_config(**overrides):
    # HF-like flavour: RMSNorm + SwiGLU + RoPE, GQA (2 KV heads < 4 Q).
    kwargs = dict(
        vocab_size=48, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=32,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    return DecoderLM(_tiny_config(**overrides)).eval()


def _to_hf_name(our_key: str) -> str:
    """Inverse of the documented mapping (test-side, independent)."""
    if our_key == "tok_embedding.weight":
        return "model.embed_tokens.weight"
    if our_key == "final_norm.weight":
        return "model.norm.weight"
    if our_key == "lm_head.weight":
        return "lm_head.weight"
    parts = our_key.split(".")  # layers.{i}.<rest...>
    layer = parts[1]
    if parts[2] == "attn_norm":
        return f"model.layers.{layer}.input_layernorm.weight"
    if parts[2] == "mlp_norm":
        return f"model.layers.{layer}.post_attention_layernorm.weight"
    if parts[2] == "attn":
        return f"model.layers.{layer}.self_attn.{parts[3]}.weight"
    if parts[2] == "mlp":
        return f"model.layers.{layer}.mlp.{parts[3]}.weight"
    raise AssertionError(f"unhandled DecoderLM key {our_key!r}")


def test_map_targets_are_real_decoderlm_state_dict_keys():
    # Enumerated from a live model -- the mapping may not invent names.
    model = _tiny_model()
    real_keys = set(model.state_dict())
    assert {
        "tok_embedding.weight", "final_norm.weight", "lm_head.weight",
        "layers.0.attn_norm.weight", "layers.0.attn.q_proj.weight",
        "layers.0.attn.k_proj.weight", "layers.0.attn.v_proj.weight",
        "layers.0.attn.o_proj.weight", "layers.0.mlp_norm.weight",
        "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.1.attn.o_proj.weight", "layers.1.mlp.down_proj.weight",
    } <= real_keys

    hf = {_to_hf_name(k): v for k, v in model.state_dict().items()}
    assert set(map_hf_state_dict(hf)) == real_keys


def test_inverse_map_load_roundtrip_logits_identical():
    model = _tiny_model(seed=7, tie_embeddings=False)
    own = model.state_dict()
    hf_sd = {_to_hf_name(k): v.clone() for k, v in own.items()}
    # Sanity: the fake checkpoint really uses HF spellings.
    assert "model.embed_tokens.weight" in hf_sd
    assert "model.layers.1.self_attn.q_proj.weight" in hf_sd
    assert "model.layers.0.input_layernorm.weight" in hf_sd
    assert "model.layers.0.mlp.gate_proj.weight" in hf_sd
    assert "model.norm.weight" in hf_sd

    mapped = map_hf_state_dict(hf_sd)
    assert set(mapped) == set(own)
    for our_key, hf_key in ((k, _to_hf_name(k)) for k in own):
        assert mapped[our_key] is hf_sd[hf_key]  # rename only, no copy

    report = validate_state_dict(mapped, model)
    assert report == {"missing": [], "unexpected": [],
                      "shape_mismatch": [], "ok": True}

    fresh = _tiny_model(seed=99, tie_embeddings=False)  # different init
    assert fresh.load_state_dict(mapped, strict=True)
    idx = torch.randint(0, 48, (2, 6))
    with torch.no_grad():
        want, _ = model(idx)
        got, _ = fresh(idx)
    assert torch.equal(got, want)  # logits IDENTICAL after the roundtrip


def test_validate_reports_missing_unexpected_and_shape_mismatch():
    model = _tiny_model(tie_embeddings=False)
    mapped = map_hf_state_dict(
        {_to_hf_name(k): v for k, v in model.state_dict().items()}
    )
    del mapped["layers.0.mlp.up_proj.weight"]  # missing
    mapped["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(4)
    mapped["layers.1.attn.q_proj.weight"] = torch.zeros(3, 3)  # wrong shape

    report = validate_state_dict(mapped, model)
    assert report["missing"] == ["layers.0.mlp.up_proj.weight"]
    assert report["unexpected"] == [
        "model.layers.0.self_attn.rotary_emb.inv_freq"
    ]
    assert report["shape_mismatch"] == [
        ("layers.1.attn.q_proj.weight", (32, 32), (3, 3))
    ]
    assert report["ok"] is False


def test_map_passes_unknown_keys_through_for_validation():
    sentinel = torch.ones(2)
    mapped = map_hf_state_dict(
        {"model.embed_tokens.weight": torch.zeros(48, 32),
         "some.future.buffer": sentinel}
    )
    assert mapped["some.future.buffer"] is sentinel
    assert mapped["tok_embedding.weight"].shape == (48, 32)


def test_map_renames_every_documented_layer_key():
    i = 3
    hf = {
        f"model.layers.{i}.input_layernorm.weight": torch.zeros(32),
        f"model.layers.{i}.post_attention_layernorm.weight": torch.zeros(32),
        f"model.layers.{i}.self_attn.q_proj.bias": torch.zeros(32),
        f"model.layers.{i}.self_attn.o_proj.weight": torch.zeros(32, 32),
        f"model.layers.{i}.mlp.down_proj.weight": torch.zeros(32, 64),
        "model.norm.weight": torch.zeros(32),
    }
    mapped = map_hf_state_dict(hf)
    assert set(mapped) == {
        f"layers.{i}.attn_norm.weight",
        f"layers.{i}.mlp_norm.weight",
        f"layers.{i}.attn.q_proj.bias",  # kept visible; validate flags it
        f"layers.{i}.attn.o_proj.weight",
        f"layers.{i}.mlp.down_proj.weight",
        "final_norm.weight",
    }
    # The renamed bias has no home in our bias-free projections:
    # validation must surface it as unexpected, not drop it silently.
    report = validate_state_dict(mapped, _tiny_model())
    assert f"layers.{i}.attn.q_proj.bias" in report["unexpected"]


def test_biased_checkpoint_validates_and_roundtrips():
    model = _tiny_model(seed=11, attention_bias=True, tie_embeddings=False)
    hf_sd = {
        _to_hf_name(k): v.clone() for k, v in model.state_dict().items()
    }
    # _to_hf_name only covers .weight; extend for .bias keys.
    hf_biased = {}
    for k, v in model.state_dict().items():
        if k.endswith(".bias"):
            parts = k.split(".")
            layer = parts[1]
            proj = parts[3]
            hf_biased[f"model.layers.{layer}.self_attn.{proj}.bias"] = v.clone()
        else:
            hf_biased[_to_hf_name(k)] = v.clone()
    assert "model.layers.0.self_attn.q_proj.bias" in hf_biased
    assert "model.layers.0.self_attn.o_proj.bias" not in hf_biased

    mapped = map_hf_state_dict(hf_biased)
    assert set(mapped) == set(model.state_dict())
    report = validate_state_dict(mapped, model)
    assert report == {"missing": [], "unexpected": [],
                      "shape_mismatch": [], "ok": True}
    fresh = _tiny_model(seed=99, attention_bias=True, tie_embeddings=False)
    assert fresh.load_state_dict(mapped, strict=True)
    idx = torch.randint(0, 48, (2, 6))
    with torch.no_grad():
        want, _ = model(idx)
        got, _ = fresh(idx)
    assert torch.equal(got, want)


def test_tied_checkpoint_without_lm_head_validates_and_loads():
    model = _tiny_model(seed=3, tie_embeddings=True)
    hf_sd = {
        _to_hf_name(k): v.clone()
        for k, v in model.state_dict().items()
        if k != "lm_head.weight"  # tied HF checkpoints omit the head
    }
    assert "lm_head.weight" not in hf_sd  # the omission being tested

    # Tie-aware validation: the embedding covers the shared head.
    plain = map_hf_state_dict(hf_sd)
    assert "lm_head.weight" not in plain
    assert validate_state_dict(plain, model)["ok"] is True

    # Loading materialises the alias (same object), then strict-loads.
    mapped = map_hf_state_dict(hf_sd, tie_embeddings=True)
    assert mapped["lm_head.weight"] is mapped["tok_embedding.weight"]
    fresh = _tiny_model(seed=42, tie_embeddings=True)
    assert fresh.load_state_dict(mapped, strict=True)
    idx = torch.randint(0, 48, (1, 5))
    with torch.no_grad():
        want, _ = model(idx)
        got, _ = fresh(idx)
    assert torch.equal(got, want)

