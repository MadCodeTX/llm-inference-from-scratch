"""Hybrid Qwen3.5/Qwen3.8 stack: gated full attention + gated delta rule.

Why this exists in inference
------------------------------
The Qwen3.8-27B-FP8 checkpoint (``Qwen3_5ForConditionalGeneration``) is
not the plain decoder of :mod:`model`: of its 64 text layers, 48 are
gated-delta-rule *linear* attention (:mod:`linear_attention`, ch 21) and
only every 4th layer (16 total) is softmax *full* attention -- but a
different full attention than ch 04's. This module builds that missing
piece and the stack that mixes the two:

* :class:`GatedFullAttention` -- full attention mirroring checkpoint
  layer 3 (the first full layer): a fused ``q_proj`` emitting query
  *and* gate, per-head Q/K RMSNorm, partial RoPE, GQA, a sigmoid output
  gate, then ``o_proj``.
* :class:`HybridConfig` -- the ``text_config`` of the real checkpoint
  as a dataclass (``ModelConfig`` is left untouched: it cannot even
  express this model, since here ``head_dim`` (256) is independent of
  ``hidden_size // num_attention_heads`` (5120/24 is not an integer)).
* :class:`HybridBlock` / :class:`TinyHybridLM` -- pre-norm blocks that
  dispatch on ``layer_types[i]``, and a tiny LM that prefills and
  decode-steps with *mixed* state: KV tensors for full layers,
  recurrent ``(S, conv_buffer)`` for linear layers, behind one
  snapshot/restore.
* :func:`list_hybrid_tensor_map` -- checkpoint name -> our name, built
  from safetensors *headers only* (names/shapes; weights never load).

Source of truth (read in ``.venv``, transformers 5.20.1)
---------------------------------------------------------
``transformers/models/qwen3_5/modeling_qwen3_5.py``:
``Qwen3_5Attention`` (fused q+gate projection and its split, Q/K norms,
sigmoid gate at the output), ``apply_rotary_pos_emb`` (rotate only the
first ``rotary_dim`` dims, tail passed through), ``Qwen3_5TextRotaryEmbedding``
(``rotary_dim = int(head_dim * partial_rotary_factor)``),
``Qwen3_5RMSNorm`` (zero-centred: gain is ``1 + weight``), and
``Qwen3_5DecoderLayer`` (``linear_attn`` vs ``self_attn`` dispatch);
``configuration_qwen3_5.py``: ``Qwen3_5TextConfig`` field names. The
real numbers in the docs/tests come from the checkpoint's own
``config.json`` and shard headers on disk, never from memory.

The q/gate split (the detail that silently corrupts if guessed)
----------------------------------------------------------------
HF computes ``q_proj(x).view(B, T, n_head, 2 * head_dim)`` and chunks
the *last* dim: within each head, the first ``head_dim`` rows are the
query and the second ``head_dim`` are the gate. It is a per-head
interleave, *not* "first half of the flat vector is all queries".
The gate then multiplies the attention output as
``attn_output * sigmoid(gate)`` before ``o_proj`` (modeling line 804;
the checkpoint config labels this ``output_gate_type="swish"`` --
``sigmoid`` is what the reference code applies, and is the default
here; ``gate_activation="silu"`` selects ``silu(gate)`` instead).

Partial RoPE and mrope
-----------------------
Only the first ``rotary_dim = head_dim * partial_rotary_factor`` dims
rotate (64 of 256 in the real model); the tail is identity. The real
config carries mrope sections ``[11, 11, 10]`` over the 32 rotary
*frequency pairs*: with interleaved mrope, the temporal/height/width
position axes pick different frequency slots. For text-only input all
three axes carry the same position, so the recomposition is the
identity and plain 1-D partial RoPE (this module) is exact; mrope
matters only once vision tokens supply distinct H/W positions, which
is the deferred vision tower's problem (see docs/22).

How production differs
------------------------
vLLM runs these layers with fused kernels, FP8 block-scaled weights
(every ``*.weight`` paired with a ``weight_scale_inv`` per [128, 128]
block), paged KV for the 16 full layers and a fixed recurrent state
for the 48 linear ones. Here every step is explicit PyTorch so
prefill-vs-decode and snapshot/restore equivalence are provable.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .attention import manual_scaled_dot_product_attention, repeat_kv
from .kv_cache import KVCache, StaticKVCache
from .layers import precompute_rope_cache, apply_rope
from .linear_attention import GatedDeltaNet, LinearAttentionState
from .mlp import SwiGLU
from .weights import read_safetensors_header

__all__ = [
    "GatedFullAttention",
    "GatedAttention",
    "HybridConfig",
    "HybridBlock",
    "HybridState",
    "HybridCache",
    "QwenRMSNorm",
    "ZeroCenteredRMSNorm",
    "TinyHybridLM",
    "apply_partial_rope",
    "partial_rope_dim",
    "precompute_partial_rope_cache",
    "list_hybrid_tensor_map",
    "hybrid_tensor_map",
]


# ---------------------------------------------------------------------------
# Zero-centred RMSNorm (Qwen3_5RMSNorm): gain is (1 + weight), weight init 0.
# ---------------------------------------------------------------------------


class QwenRMSNorm(nn.Module):
    """RMSNorm with the checkpoint's zero-centred gain.

    HF ``Qwen3_5RMSNorm`` stores ``weight`` initialised to *zeros* and
    multiplies by ``(1 + weight)`` -- a fresh norm is the identity
    rescale, and a checkpoint ``weight`` of ~0.23 means a gain of
    ~1.23. Loading those tensors into a ones-initialised norm (the
    repo's ch-03 :class:`~inference_from_scratch.layers.RMSNorm`)
    would silently shrink every normalised row, so the hybrid stack
    uses this class for block norms, the final norm, and the Q/K
    head norms. Shape: ``weight`` is ``[d]``; input/output ``[..., d]``.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.d = d
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(ms + self.eps)
        return (y * (1.0 + self.weight.float())).to(orig_dtype)

    def extra_repr(self) -> str:
        return f"d={self.d}, eps={self.eps}"


# Descriptive alias: same class, name says what is centred.
ZeroCenteredRMSNorm = QwenRMSNorm


# ---------------------------------------------------------------------------
# Partial RoPE: rotate the first rotary_dim dims, pass the tail through.
# ---------------------------------------------------------------------------


def partial_rope_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """Rotary width: ``int(head_dim * factor)`` rounded down to even.

    RoPE rotates 2-D planes, so the width must be even (real model:
    ``256 * 0.25 = 64``). A factor of 1.0 returns ``head_dim``.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if not (0.0 < partial_rotary_factor <= 1.0):
        raise ValueError(
            f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}"
        )
    dim = int(head_dim * partial_rotary_factor)
    if dim % 2:
        dim -= 1
    if dim <= 0:
        raise ValueError(
            f"partial_rotary_factor={partial_rotary_factor} on head_dim={head_dim} "
            "leaves no rotary dims"
        )
    return dim


def precompute_partial_rope_cache(
    max_seq_len: int,
    head_dim: int,
    partial_rotary_factor: float = 1.0,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """RoPE tables for the rotary prefix only: ``([T, R], [T, R], R)``.

    Frequencies are built over ``R = partial_rope_dim(...)`` exactly as
    HF's ``compute_default_rope_parameters`` does (``theta ** (-2i/R)``),
    *not* over the full head_dim -- the tail dims get no angles at all.
    """
    rotary_dim = partial_rope_dim(head_dim, partial_rotary_factor)
    cos, sin = precompute_rope_cache(
        max_seq_len, rotary_dim, theta=theta, device=device, dtype=dtype
    )
    return cos, sin, rotary_dim


def apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Rotate only the first ``rotary_dim`` dims of ``x`` ``[B, H, T, D]``.

    ``cos``/``sin`` are ``[T_max, R]`` tables (a wider table is sliced
    to ``R``). The tail ``x[..., R:]`` is returned bit-identical --
    that pass-through is the whole point of partial RoPE, and the
    tests assert it with ``torch.equal``.
    """
    if rotary_dim is None:
        rotary_dim = cos.shape[-1]
    if rotary_dim >= x.shape[-1]:
        return apply_rope(x, cos, sin, start_pos=start_pos)
    rotated = apply_rope(
        x[..., :rotary_dim],
        cos[..., :rotary_dim],
        sin[..., :rotary_dim],
        start_pos=start_pos,
    )
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


# ---------------------------------------------------------------------------
# HybridConfig: the real text_config as a dataclass.
# ---------------------------------------------------------------------------


def _first(d: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = d.get(name)
        if value is not None:
            return value
    return default


def _normalize_layer_type(value: str) -> str:
    v = str(value).strip().lower().replace("-", "_")
    if v in ("linear", "linear_attention", "gated_delta", "delta", "gdn"):
        return "linear_attention"
    if v in ("full", "full_attention", "attention", "softmax", "softmax_attention"):
        return "full_attention"
    raise ValueError(
        f"unknown layer type {value!r} (expected 'linear_attention' or 'full_attention')"
    )


@dataclass(init=False)
class HybridConfig:
    """Shape bundle for the hybrid stack (HF ``Qwen3_5TextConfig`` names).

    Canonical fields use the checkpoint's own names (``hidden_size``,
    ``num_attention_heads``, ...); the repo's ``ModelConfig`` spellings
    (``d_model``, ``n_layer``, ``n_head``, ``n_kv_head``, ``d_ff``,
    ``max_seq_len``, ``rms_eps``, ``tie_embeddings``) are read-only
    alias properties, and the constructor accepts either spelling for
    every field. ``head_dim`` is an independent field -- in this model
    family it does *not* equal ``hidden_size // num_attention_heads``.

    ``layer_types`` entries are normalised to ``"linear_attention"`` /
    ``"full_attention"`` (short forms ``"linear"`` / ``"full"`` are
    accepted). When omitted, the pattern is derived from
    ``full_attention_interval`` (every 4th layer full, HF's default).
    """

    vocab_size: int = 248320
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 17408
    max_position_embeddings: int = 262144
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    # Linear (gated delta rule) layers.
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Full-attention / RoPE.
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] | None = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True
    attn_output_gate: bool = True
    output_gate_type: str | None = "swish"
    # Shared.
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    mtp_num_hidden_layers: int = 1

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        max_position_embeddings: int | None = None,
        layer_types: Sequence[str] | None = None,
        full_attention_interval: int = 4,
        linear_num_key_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int | None = None,
        rope_theta: float | None = None,
        partial_rotary_factor: float | None = None,
        mrope_section: Sequence[int] | None = None,
        mrope_interleaved: bool = True,
        attn_output_gate: bool = True,
        output_gate_type: str | None = "swish",
        rms_norm_eps: float | None = None,
        attention_bias: bool = False,
        tie_word_embeddings: bool | None = None,
        mtp_num_hidden_layers: int = 1,
        # Repo-style (ModelConfig) aliases.
        d_model: int | None = None,
        n_layer: int | None = None,
        n_head: int | None = None,
        n_kv_head: int | None = None,
        d_ff: int | None = None,
        max_seq_len: int | None = None,
        rms_eps: float | None = None,
        tie_embeddings: bool | None = None,
        # Short linear-attention aliases.
        num_key_heads: int | None = None,
        key_dim: int | None = None,
        num_value_heads: int | None = None,
        value_dim: int | None = None,
        conv_kernel: int | None = None,
        conv_kernel_dim: int | None = None,
        **_ignored: Any,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size if hidden_size is not None else (d_model if d_model is not None else 5120))
        self.num_hidden_layers = int(
            num_hidden_layers if num_hidden_layers is not None else (n_layer if n_layer is not None else (len(layer_types) if layer_types else 64))
        )
        self.num_attention_heads = int(num_attention_heads if num_attention_heads is not None else (n_head if n_head is not None else 24))
        self.num_key_value_heads = int(num_key_value_heads if num_key_value_heads is not None else (n_kv_head if n_kv_head is not None else 4))
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.intermediate_size = int(intermediate_size if intermediate_size is not None else (d_ff if d_ff is not None else 17408))
        self.max_position_embeddings = int(max_position_embeddings if max_position_embeddings is not None else (max_seq_len if max_seq_len is not None else 262144))
        self.full_attention_interval = int(full_attention_interval)
        if layer_types:
            self.layer_types = [_normalize_layer_type(t) for t in layer_types]
            self.num_hidden_layers = len(self.layer_types)
        else:
            interval = max(self.full_attention_interval, 1)
            self.layer_types = [
                "linear_attention" if (i + 1) % interval else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.linear_num_key_heads = int(linear_num_key_heads if linear_num_key_heads is not None else (num_key_heads if num_key_heads is not None else 16))
        self.linear_key_head_dim = int(linear_key_head_dim if linear_key_head_dim is not None else (key_dim if key_dim is not None else 128))
        self.linear_num_value_heads = int(linear_num_value_heads if linear_num_value_heads is not None else (num_value_heads if num_value_heads is not None else 48))
        self.linear_value_head_dim = int(linear_value_head_dim if linear_value_head_dim is not None else (value_dim if value_dim is not None else 128))
        self.linear_conv_kernel_dim = int(
            linear_conv_kernel_dim if linear_conv_kernel_dim is not None else (conv_kernel_dim if conv_kernel_dim is not None else (conv_kernel if conv_kernel is not None else 4))
        )
        self.rope_theta = float(rope_theta) if rope_theta is not None else 10000000.0
        self.partial_rotary_factor = float(partial_rotary_factor) if partial_rotary_factor is not None else 0.25
        self.mrope_section = [int(v) for v in mrope_section] if mrope_section is not None else [11, 11, 10]
        self.mrope_interleaved = bool(mrope_interleaved)
        self.attn_output_gate = bool(attn_output_gate)
        self.output_gate_type = output_gate_type
        self.rms_norm_eps = float(rms_norm_eps if rms_norm_eps is not None else (rms_eps if rms_eps is not None else 1e-6))
        self.attention_bias = bool(attention_bias)
        self.tie_word_embeddings = bool(tie_word_embeddings if tie_word_embeddings is not None else (tie_embeddings if tie_embeddings is not None else False))
        self.mtp_num_hidden_layers = int(mtp_num_hidden_layers)
        self._validate()

    def _validate(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "intermediate_size", "max_position_embeddings",
                     "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
                     "linear_value_head_dim", "linear_conv_kernel_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads (GQA)")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        # Eagerly validates the factor range / even rotary width.
        partial_rope_dim(self.head_dim, self.partial_rotary_factor)

    # -- repo-style alias properties -------------------------------------
    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def n_layer(self) -> int:
        return self.num_hidden_layers

    @property
    def n_head(self) -> int:
        return self.num_attention_heads

    @property
    def n_kv_head(self) -> int:
        return self.num_key_value_heads

    @property
    def d_ff(self) -> int:
        return self.intermediate_size

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    @property
    def rms_eps(self) -> float:
        return self.rms_norm_eps

    @property
    def tie_embeddings(self) -> bool:
        return self.tie_word_embeddings

    @property
    def rotary_dim(self) -> int:
        """Dims actually rotated by RoPE (``head_dim * partial_rotary_factor``)."""
        return partial_rope_dim(self.head_dim, self.partial_rotary_factor)

    @property
    def n_rotary_dim(self) -> int:
        return self.rotary_dim

    @property
    def num_linear_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def num_full_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    # HF-count aliases.
    @property
    def n_linear_layers(self) -> int:
        return self.num_linear_layers

    @property
    def n_full_layers(self) -> int:
        return self.num_full_layers

    def layer_type(self, index: int) -> str:
        return self.layer_types[index]

    @classmethod
    def from_text_config(cls, source: "Mapping[str, Any] | str | Path | HybridConfig") -> "HybridConfig":
        """Parse a real ``text_config`` dict (or full ``config.json``).

        Accepts the ``text_config`` mapping itself, the full checkpoint
        ``config.json`` mapping (its ``text_config`` is used), a path to
        ``config.json``, or a checkpoint directory. Every number comes
        from the input: ``rope_theta`` / ``partial_rotary_factor`` /
        ``mrope_section`` are read from ``rope_parameters`` when nested
        there (the real file) and from top-level keys otherwise.
        """
        if isinstance(source, HybridConfig):
            return source
        if isinstance(source, Mapping):
            raw: dict[str, Any] = dict(source)
        else:
            p = Path(source)
            config_path = p / "config.json" if p.is_dir() else p
            if not config_path.is_file():
                raise FileNotFoundError(f"no config.json at {config_path}")
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError(f"{config_path}: config.json is not a JSON object")
            raw = parsed
        text = raw.get("text_config") if isinstance(raw.get("text_config"), Mapping) else raw
        rope_params = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), Mapping) else {}
        layer_types = text.get("layer_types")
        return cls(
            vocab_size=int(_first(text, "vocab_size", default=248320)),
            hidden_size=int(_first(text, "hidden_size", "d_model", default=5120)),
            num_hidden_layers=int(_first(text, "num_hidden_layers", "n_layer", default=len(layer_types) if layer_types else 64)),
            num_attention_heads=int(_first(text, "num_attention_heads", "n_head", default=24)),
            num_key_value_heads=int(_first(text, "num_key_value_heads", "n_kv_head", default=4)),
            head_dim=int(_first(text, "head_dim", default=0)) or None,
            intermediate_size=int(_first(text, "intermediate_size", "d_ff", default=17408)),
            max_position_embeddings=int(_first(text, "max_position_embeddings", "max_seq_len", default=262144)),
            layer_types=list(layer_types) if layer_types else None,
            full_attention_interval=int(_first(text, "full_attention_interval", default=4)),
            linear_num_key_heads=int(_first(text, "linear_num_key_heads", default=16)),
            linear_key_head_dim=int(_first(text, "linear_key_head_dim", default=128)),
            linear_num_value_heads=int(_first(text, "linear_num_value_heads", default=48)),
            linear_value_head_dim=int(_first(text, "linear_value_head_dim", default=128)),
            linear_conv_kernel_dim=int(_first(text, "linear_conv_kernel_dim", default=4)),
            rope_theta=float(_first(rope_params, "rope_theta", default=_first(text, "rope_theta", default=10000000.0))),
            partial_rotary_factor=float(
                _first(rope_params, "partial_rotary_factor", default=_first(text, "partial_rotary_factor", default=0.25))
            ),
            mrope_section=_first(rope_params, "mrope_section", default=_first(text, "mrope_section", default=None)),
            mrope_interleaved=bool(_first(rope_params, "mrope_interleaved", default=_first(text, "mrope_interleaved", default=True))),
            attn_output_gate=bool(_first(text, "attn_output_gate", default=True)),
            output_gate_type=_first(text, "output_gate_type", default=None),
            rms_norm_eps=float(_first(text, "rms_norm_eps", "rms_eps", default=1e-6)),
            attention_bias=bool(_first(text, "attention_bias", default=False)),
            tie_word_embeddings=bool(_first(text, "tie_word_embeddings", "tie_embeddings", default=False)),
            mtp_num_hidden_layers=int(_first(text, "mtp_num_hidden_layers", default=1)),
        )

    # Aliases for the same entry point.
    from_config_dict = from_text_config
    from_hf_config = from_text_config


# ---------------------------------------------------------------------------
# GatedFullAttention: checkpoint layer-3 style full attention.
# ---------------------------------------------------------------------------


def _cfg_get(config: Any, *names: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return _first(config, *names, default=default)
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


class GatedFullAttention(nn.Module):
    """Full softmax attention with Q/K norms, partial RoPE and a gate.

    Mirrors checkpoint layer 3 (``model.language_model.layers.3.self_attn``)
    and HF ``Qwen3_5Attention`` tensor-for-tensor::

        q_proj.weight  [2 * n_head * head_dim, d_model]  (query ++ gate)
        k_proj.weight  [n_kv_head * head_dim, d_model]
        v_proj.weight  [n_kv_head * head_dim, d_model]
        o_proj.weight  [d_model, n_head * head_dim]
        q_norm.weight  [head_dim]   (zero-centred RMSNorm, gain 1 + w)
        k_norm.weight  [head_dim]

    All projections are bias-free in the real checkpoint
    (``attention_bias=false``); ``attention_bias=True`` adds Q/K/V
    biases (``o_proj`` stays bias-free, as in this repo's ch-04).

    The constructor accepts a :class:`HybridConfig`, a repo
    ``ModelConfig``, a dict, or explicit dims (``d_model=..., n_head=...``
    in either naming family). ``attn_implementation`` selects the
    attention kernel: ``"manual"`` (repo's hand-written SDPA, default)
    or ``"sdpa"`` (``F.scaled_dot_product_attention``); both are causal
    and must agree numerically.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        d_model: int | None = None,
        hidden_size: int | None = None,
        n_head: int | None = None,
        num_attention_heads: int | None = None,
        n_kv_head: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        partial_rotary_factor: float | None = None,
        rope_theta: float | None = None,
        rms_eps: float | None = None,
        rms_norm_eps: float | None = None,
        eps: float | None = None,
        attention_bias: bool | None = None,
        bias: bool | None = None,
        attn_implementation: str = "manual",
        implementation: str | None = None,
        attn_impl: str | None = None,
        gate_activation: str | None = None,
        max_seq_len: int | None = None,
        max_position_embeddings: int | None = None,
        **_ignored: Any,
    ):
        super().__init__()
        self.d_model = int(
            d_model if d_model is not None else hidden_size if hidden_size is not None
            else _cfg_get(config, "hidden_size", "d_model", default=5120)
        )
        self.hidden_size = self.d_model
        self.n_head = int(
            n_head if n_head is not None else num_attention_heads if num_attention_heads is not None
            else _cfg_get(config, "num_attention_heads", "n_head", default=24)
        )
        self.num_attention_heads = self.n_head
        self.n_kv_head = int(
            n_kv_head if n_kv_head is not None else num_key_value_heads if num_key_value_heads is not None
            else _cfg_get(config, "num_key_value_heads", "n_kv_head", default=self.n_head)
        )
        self.num_key_value_heads = self.n_kv_head
        cfg_head_dim = _cfg_get(config, "head_dim", default=None)
        if head_dim is not None:
            self.head_dim = int(head_dim)
        elif cfg_head_dim:
            self.head_dim = int(cfg_head_dim)
        elif _cfg_get(config, "head_dim", default=None) is None and hasattr(config, "head_dim"):
            self.head_dim = int(config.head_dim)  # ModelConfig property
        else:
            self.head_dim = self.d_model // self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head (GQA)")
        self.n_rep = self.n_head // self.n_kv_head
        self.num_key_value_groups = self.n_rep
        self.partial_rotary_factor = float(
            partial_rotary_factor if partial_rotary_factor is not None
            else _cfg_get(config, "partial_rotary_factor", default=0.25)
        )
        self.rotary_dim = partial_rope_dim(self.head_dim, self.partial_rotary_factor)
        self.rope_theta = float(
            rope_theta if rope_theta is not None else _cfg_get(config, "rope_theta", default=10000.0)
        )
        self.rms_eps = float(
            rms_eps if rms_eps is not None else rms_norm_eps if rms_norm_eps is not None
            else eps if eps is not None else _cfg_get(config, "rms_norm_eps", "rms_eps", default=1e-6)
        )
        use_bias = attention_bias if attention_bias is not None else bias if bias is not None else bool(
            _cfg_get(config, "attention_bias", default=False)
        )
        self.attention_bias = use_bias
        self.attn_implementation = str(implementation or attn_impl or attn_implementation).lower()
        if self.attn_implementation in ("eager", "manual_sdpa"):
            self.attn_implementation = "manual"
        if self.attn_implementation not in ("manual", "sdpa"):
            raise ValueError(f"attn_implementation must be 'manual' or 'sdpa', got {self.attn_implementation!r}")
        self.gate_activation = str(gate_activation or "sigmoid").lower()
        if self.gate_activation in ("swish",):
            self.gate_activation = "silu"
        if self.gate_activation not in ("sigmoid", "silu", "none"):
            raise ValueError(f"gate_activation must be 'sigmoid', 'silu' or 'none', got {gate_activation!r}")
        self.scaling = self.head_dim ** -0.5

        D, H, Hkv, Dh = self.d_model, self.n_head, self.n_kv_head, self.head_dim
        # Fused query+gate: per head, [q (Dh) | gate (Dh)] -- see module docs.
        self.q_proj = nn.Linear(D, 2 * H * Dh, bias=use_bias)
        self.k_proj = nn.Linear(D, Hkv * Dh, bias=use_bias)
        self.v_proj = nn.Linear(D, Hkv * Dh, bias=use_bias)
        self.o_proj = nn.Linear(H * Dh, D, bias=False)
        self.q_norm = QwenRMSNorm(Dh, eps=self.rms_eps)
        self.k_norm = QwenRMSNorm(Dh, eps=self.rms_eps)

        # Own RoPE tables (used only when the caller passes no cos/sin).
        rope_len = int(
            max_seq_len if max_seq_len is not None else max_position_embeddings if max_position_embeddings is not None
            else _cfg_get(config, "max_position_embeddings", "max_seq_len", default=4096)
        )
        cos, sin, _ = precompute_partial_rope_cache(
            min(rope_len, 8192), Dh, self.partial_rotary_factor, theta=self.rope_theta
        )
        self.register_buffer("_cos", cos, persistent=False)
        self.register_buffer("_sin", sin, persistent=False)

    def _rope_tables(self, end: int, device: torch.device, dtype: torch.dtype):
        if end <= self._cos.shape[0] and self._cos.device == device:
            return self._cos, self._sin
        cos, sin, _ = precompute_partial_rope_cache(
            end, self.head_dim, self.partial_rotary_factor,
            theta=self.rope_theta, device=device, dtype=dtype,
        )
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        kv_cache: KVCache | StaticKVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """``x`` ``[B, T, D] -> [B, T, D]`` (causal, KV-cache aware).

        ``cos``/``sin`` are ``[T_max, R]`` partial-RoPE tables (wider
        full-head tables are sliced); when omitted, tables precomputed
        from the config are used. ``position_embeddings=(cos, sin)``
        is accepted as an alias (HF calling convention).
        """
        if cos is None and "position_embeddings" in kwargs:
            cos, sin = kwargs["position_embeddings"]
        if kv_cache is None and "past_key_values" in kwargs:
            kv_cache = kwargs["past_key_values"]
        B, T, D = x.shape
        H, Hkv, Dh = self.n_head, self.n_kv_head, self.head_dim

        # Fused Q+gate, split per head on the last dim (HF chunk).
        qg = self.q_proj(x).view(B, T, H, 2 * Dh)
        q = qg[..., :Dh]
        gate = qg[..., Dh:].reshape(B, T, H * Dh)
        k = self.k_proj(x).view(B, T, Hkv, Dh)
        v = self.v_proj(x).view(B, T, Hkv, Dh)

        # Per-head Q/K norms over head_dim, before RoPE (HF order).
        q = self.q_norm(q).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_norm(k).transpose(1, 2)  # [B, Hkv, T, Dh]
        v = v.transpose(1, 2)  # [B, Hkv, T, Dh]

        if cos is None or sin is None:
            cos, sin = self._rope_tables(start_pos + T, x.device, x.dtype)
        q = apply_partial_rope(q, cos, sin, start_pos=start_pos, rotary_dim=self.rotary_dim)
        k = apply_partial_rope(k, cos, sin, start_pos=start_pos, rotary_dim=self.rotary_dim)

        if kv_cache is not None:
            if isinstance(kv_cache, StaticKVCache):
                k, v = kv_cache.update(layer_idx, k, v, start_pos)
            else:
                k, v = kv_cache.update(layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if self.attn_implementation == "sdpa":
            attn = self._sdpa(q, k, v, start_pos=start_pos)
        else:
            attn = manual_scaled_dot_product_attention(q, k, v, is_causal=True)

        merged = attn.transpose(1, 2).contiguous().view(B, T, H * Dh)
        if self.gate_activation == "sigmoid":
            merged = merged * torch.sigmoid(gate)
        elif self.gate_activation == "silu":
            merged = merged * F.silu(gate)
        return self.o_proj(merged)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T_q, T_k = q.shape[2], k.shape[2]
        if T_q == T_k and start_pos == 0:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
        if T_q == 1:
            # A lone decode query sits at the last position: nothing is future.
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scaling)
        # Chunked continuation: explicit causal mask by absolute position.
        q_pos = torch.arange(T_q, device=q.device) + start_pos
        k_pos = torch.arange(T_k, device=q.device)
        future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        mask = torch.zeros(T_q, T_k, device=q.device, dtype=q.dtype).masked_fill(future, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=self.scaling)


# Short alias.
GatedAttention = GatedFullAttention


# ---------------------------------------------------------------------------
# Mixed per-layer state: KV tensors for full layers, recurrent for linear.
# ---------------------------------------------------------------------------


@dataclass
class HybridState:
    """Unified inference state for a hybrid stack.

    ``layer_states[i]`` is, by ``layer_types[i]``:

    * ``"full_attention"``: ``None`` before the first token, else the
      ``(k_all, v_all)`` pair, each ``[B, H_kv, T_filled, head_dim]``;
    * ``"linear_attention"``: ``None`` (meaning zero state) or a
      :data:`~inference_from_scratch.linear_attention.LinearAttentionState`
      ``(S [B, Hv, Dk, Dv], conv_buffer [B, conv_dim, K-1])``.

    ``seq_len`` is the number of positions already processed (the next
    token's absolute position). Snapshot/restore deep-copy every
    tensor, so a restored state can be continued repeatedly and
    independently -- the mixed-type analogue of ch-21's state pair.
    """

    layer_states: list[Any] = field(default_factory=list)
    seq_len: int = 0

    @property
    def position(self) -> int:
        return self.seq_len

    @property
    def states(self) -> list[Any]:
        return self.layer_states

    def snapshot(self) -> "HybridState":
        return _snapshot_hybrid_state(self)

    def restore(self) -> "HybridState":
        return _snapshot_hybrid_state(self)

    # clone() reads naturally at call sites that think in tensors.
    def clone(self) -> "HybridState":
        return _snapshot_hybrid_state(self)


# Name that reads as "the cache object" at model call sites.
HybridCache = HybridState


def _clone_layer_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) == 2:
        a, b = state
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return (a.clone(), b.clone())
    return state


def _snapshot_hybrid_state(state: HybridState) -> HybridState:
    return HybridState(
        layer_states=[_clone_layer_state(s) for s in state.layer_states],
        seq_len=int(state.seq_len),
    )


# ---------------------------------------------------------------------------
# HybridBlock: dispatch on layer type.
# ---------------------------------------------------------------------------


class HybridBlock(nn.Module):
    """One pre-norm hybrid block: token mixer (by type) + SwiGLU MLP.

    Structure mirrors HF ``Qwen3_5DecoderLayer``::

        x = x + mixer(input_layernorm(x))        # linear_attn | self_attn
        x = x + mlp(post_attention_layernorm(x)) # dense SwiGLU

    Attribute names mirror the checkpoint (``input_layernorm``,
    ``self_attn`` / ``linear_attn``, ``mlp.gate_proj`` ...); repo-style
    ``attn_norm`` / ``mlp_norm`` / ``attn`` are read-only aliases.
    Norms are :class:`QwenRMSNorm` (zero-centred) throughout, matching
    the tensors on disk. ``forward`` returns just the hidden tensor
    (full-sequence, zero-state for linear layers);
    :meth:`forward_with_state` threads mixed per-layer state and
    returns ``(hidden, new_layer_state)``.
    """

    def __init__(self, config: HybridConfig, layer_idx: int | str = 0, layer_type: str | None = None):
        super().__init__()
        if isinstance(layer_idx, str) and layer_type is None:
            layer_type, layer_idx = layer_idx, 0
        self.config = config
        self.layer_idx = int(layer_idx)  # type: ignore[arg-type]
        self.layer_type = _normalize_layer_type(
            layer_type if layer_type is not None else config.layer_types[self.layer_idx]
        )
        self.block_type = self.layer_type
        D = config.hidden_size
        self.input_layernorm = QwenRMSNorm(D, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(D, eps=config.rms_norm_eps)
        if self.layer_type == "full_attention":
            self.self_attn: GatedFullAttention | None = GatedFullAttention(config)
            self.linear_attn: GatedDeltaNet | None = None
        else:
            self.self_attn = None
            self.linear_attn = GatedDeltaNet(
                d_model=D,
                num_key_heads=config.linear_num_key_heads,
                key_dim=config.linear_key_head_dim,
                num_value_heads=config.linear_num_value_heads,
                value_dim=config.linear_value_head_dim,
                conv_kernel=config.linear_conv_kernel_dim,
                eps=config.rms_norm_eps,
            )
        self.mlp = SwiGLU(D, config.intermediate_size)

    @property
    def attn(self) -> nn.Module:
        m = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        assert m is not None
        return m

    # Repo-style norm names.
    @property
    def attn_norm(self) -> QwenRMSNorm:
        return self.input_layernorm

    @property
    def mlp_norm(self) -> QwenRMSNorm:
        return self.post_attention_layernorm

    def forward_with_state(
        self,
        x: torch.Tensor,
        layer_state: Any = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        start_pos: int = 0,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Run the block threading one layer's state.

        ``layer_state`` is ``(k_all, v_all)`` / ``None`` for full
        layers (fed through the repo :class:`KVCache` internally) and
        a linear-attention state / ``None`` for linear layers.
        Returns ``(out [B, T, D], new_layer_state)``; the input state
        tensors are never mutated.
        """
        idx = self.layer_idx if layer_idx is None else int(layer_idx)
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            cache = KVCache(1)
            if layer_state is not None:
                cache.keys[0], cache.values[0] = layer_state
            mixed = self.self_attn(h, cos, sin, kv_cache=cache, layer_idx=0, start_pos=start_pos)
            new_state = (cache.keys[0], cache.values[0])
        else:
            assert self.linear_attn is not None
            mixed, new_state = self.linear_attn.forward_prefill(h, state=layer_state)
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_state

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        kv_cache: KVCache | StaticKVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
        state: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Hidden-only forward, or ``(hidden, state)`` when state is used.

        With no state arguments this runs a stateless full-sequence
        pass and returns ``[B, T, D]``. Passing ``state=`` (a per-layer
        state) or a ``kv_cache`` for a full layer threads state and
        returns ``(hidden, new_layer_state)``.
        """
        if state is not None or kwargs.get("layer_state") is not None:
            return self.forward_with_state(
                x, state if state is not None else kwargs["layer_state"],
                cos=cos, sin=sin, start_pos=start_pos, layer_idx=layer_idx,
            )
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            mixed = self.self_attn(h, cos, sin, kv_cache=kv_cache, layer_idx=layer_idx, start_pos=start_pos)
        else:
            assert self.linear_attn is not None
            mixed = self.linear_attn(h)
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# TinyHybridLM: embed -> hybrid blocks -> final norm -> (untied) head.
# ---------------------------------------------------------------------------


class TinyHybridLM(nn.Module):
    """Tiny hybrid language model (same stack, tiny dims in tests).

    Names mirror the checkpoint: ``embed_tokens``, ``layers.{i}``,
    ``norm``, ``lm_head`` (repo aliases ``tok_embedding`` /
    ``final_norm`` are read-only properties). The head is *untied* by
    default (the real checkpoint ships a separate ``lm_head.weight``
    and ``tie_word_embeddings=false``); ``tie_word_embeddings=True``
    in the config shares the embedding matrix instead.

    Inference contract (prefill and decode are the same code path):

    * ``logits, state = model(idx)`` -- prefill ``[B, T]`` from scratch;
    * ``logits, state = model(next_ids, state)`` -- continue with any
      ``T >= 1`` (``T == 1`` is a decode step) at ``state.seq_len``;
    * :meth:`prefill` / :meth:`decode_step` (alias :meth:`step`) are
      the named forms of the two calls above;
    * :meth:`snapshot_state` / :meth:`restore_state` deep-copy the
      mixed state so a continuation can be replayed bit-identically.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, D)
        self.layers = nn.ModuleList(
            HybridBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        )
        self.norm = QwenRMSNorm(D, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin, _ = precompute_partial_rope_cache(
            min(config.max_position_embeddings, 8192),
            config.head_dim,
            config.partial_rotary_factor,
            theta=config.rope_theta,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def tok_embedding(self) -> nn.Embedding:
        return self.embed_tokens

    @property
    def final_norm(self) -> QwenRMSNorm:
        return self.norm

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def initial_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> HybridState:
        """Fresh empty mixed state (all layers ``None``, ``seq_len=0``)."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return HybridState(layer_states=[None] * len(self.layers), seq_len=0)

    def zero_state(self, *args: Any, **kwargs: Any) -> HybridState:
        return self.initial_state(*args, **kwargs)

    @staticmethod
    def snapshot_state(state: HybridState) -> HybridState:
        """Deep-copy a mixed state (KV tensors + recurrent states)."""
        return _snapshot_hybrid_state(state)

    @staticmethod
    def restore_state(snapshot: HybridState) -> HybridState:
        """Clone a snapshot into an independently continuable state."""
        return _snapshot_hybrid_state(snapshot)

    snapshot = snapshot_state
    restore = restore_state

    def _rope(self, end: int, device: torch.device, dtype: torch.dtype):
        if end <= self.cos.shape[0] and self.cos.device == device:
            return self.cos, self.sin
        cos, sin, _ = precompute_partial_rope_cache(
            end, self.config.head_dim, self.config.partial_rotary_factor,
            theta=self.config.rope_theta, device=device, dtype=dtype,
        )
        return cos, sin

    def forward(
        self,
        idx: torch.Tensor,
        state: HybridState | None = None,
        start_pos: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, HybridState]:
        """Token ids ``[B, T]`` -> ``(logits [B, T, V], new_state)``.

        ``state=None`` prefills from position 0; otherwise processing
        continues at ``state.seq_len`` (or explicit ``start_pos``).
        ``cache=`` / ``hybrid_state=`` / ``kv_caches=`` (when it holds a
        :class:`HybridState`) are accepted as aliases for ``state``.
        The input state is never mutated: the returned state owns the
        extended KV / recurrent tensors.
        """
        if state is None:
            for alias in ("cache", "hybrid_state", "kv_caches", "past_key_values"):
                candidate = kwargs.get(alias)
                if isinstance(candidate, HybridState):
                    state = candidate
                    break
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        if state is None:
            state = self.initial_state(B, device=idx.device)
        if len(state.layer_states) != len(self.layers):
            raise ValueError(
                f"state has {len(state.layer_states)} layers, model has {len(self.layers)}"
            )
        if start_pos is None:
            start_pos = int(state.seq_len)
        cos, sin = self._rope(start_pos + T, idx.device, torch.float32)
        x = self.embed_tokens(idx)
        new_layer_states: list[Any] = []
        for i, block in enumerate(self.layers):
            x, layer_state = block.forward_with_state(
                x, state.layer_states[i], cos=cos, sin=sin,
                start_pos=start_pos, layer_idx=i,
            )
            new_layer_states.append(layer_state)
        logits = self.lm_head(self.norm(x))
        return logits, HybridState(layer_states=new_layer_states, seq_len=start_pos + T)

    def prefill(
        self, idx: torch.Tensor, state: HybridState | None = None
    ) -> tuple[torch.Tensor, HybridState]:
        """Prefill ``idx`` ``[B, T]`` (from scratch unless ``state`` given)."""
        return self.forward(idx, state=state)

    forward_prefill = prefill

    def decode_step(
        self, token_ids: torch.Tensor, state: HybridState
    ) -> tuple[torch.Tensor, HybridState]:
        """One decode step: ``token_ids`` ``[B]`` or ``[B, 1]``.

        Returns ``(logits [B, 1, V], new_state)``.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)
        if token_ids.dim() != 2 or token_ids.shape[1] != 1:
            raise ValueError(f"token_ids must be [B] or [B, 1], got {tuple(token_ids.shape)}")
        return self.forward(token_ids, state=state)

    step = decode_step


# ---------------------------------------------------------------------------
# Checkpoint name map (names only; headers, never tensor bytes).
# ---------------------------------------------------------------------------

_RE_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_RE_PROJ = re.compile(
    r"^(self_attn|linear_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|"
    r"in_proj_a|in_proj_b|out_proj|gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)


def _map_hybrid_key(key: str) -> str | None:
    """One checkpoint name -> our name (``None``: not ours, e.g. vision)."""
    if key == "model.language_model.embed_tokens.weight":
        return "embed_tokens.weight"
    if key == "model.language_model.norm.weight":
        return "norm.weight"
    if key == "lm_head.weight":
        return "lm_head.weight"
    if key.startswith("mtp."):
        return key
    m = _RE_LAYER.match(key)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return f"layers.{i}.{rest}"
    pm = _RE_PROJ.match(rest)
    if pm:
        return f"layers.{i}.{pm.group(1)}.{pm.group(2)}.{pm.group(3)}"
    if rest in ("self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "linear_attn.norm.weight", "linear_attn.conv1d.weight",
                "linear_attn.A_log", "linear_attn.dt_bias"):
        return f"layers.{i}.{rest}"
    return None


def _synthesized_checkpoint_keys(config: HybridConfig) -> list[str]:
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i, layer_type in enumerate(config.layer_types):
        p = f"model.language_model.layers.{i}"
        keys += [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]
        if layer_type == "full_attention":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                keys += [f"{p}.self_attn.{proj}.weight", f"{p}.self_attn.{proj}.weight_scale_inv"]
            keys += [f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight"]
        else:
            for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
                keys += [f"{p}.linear_attn.{proj}.weight", f"{p}.linear_attn.{proj}.weight_scale_inv"]
            keys += [
                f"{p}.linear_attn.in_proj_a.weight", f"{p}.linear_attn.in_proj_b.weight",
                f"{p}.linear_attn.conv1d.weight", f"{p}.linear_attn.A_log",
                f"{p}.linear_attn.dt_bias", f"{p}.linear_attn.norm.weight",
            ]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys += [f"{p}.mlp.{proj}.weight", f"{p}.mlp.{proj}.weight_scale_inv"]
    for _ in range(config.mtp_num_hidden_layers):
        keys += ["mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                 "mtp.pre_fc_norm_hidden.weight"]
    return keys


def _collect_checkpoint_keys(source: Any) -> list[str] | None:
    """Tensor names from a checkpoint dir/file, headers, or a key list."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        if "weight_map" in source and isinstance(source["weight_map"], Mapping):
            return list(source["weight_map"].keys())
        return [k for k in source.keys() if k != "__metadata__"]
    if isinstance(source, (list, tuple, set)):
        return [str(k) for k in source]
    p = Path(source)
    if p.is_dir():
        index = p / "model.safetensors.index.json"
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(data.get("weight_map"), Mapping):
                return list(data["weight_map"].keys())
        keys: list[str] = []
        for shard in sorted(p.glob("*.safetensors")):
            keys.extend(k for k in read_safetensors_header(shard) if k != "__metadata__")
        return keys
    if p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("weight_map"), Mapping):
            return list(data["weight_map"].keys())
        return [k for k in data.keys() if k != "__metadata__"]
    if p.is_file():
        return [k for k in read_safetensors_header(p) if k != "__metadata__"]
    return None


def _find_default_checkpoint() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots"
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "config.json").is_file():
                return snap
    return None


def list_hybrid_tensor_map(
    source: Any = None,
    config: HybridConfig | None = None,
) -> dict[str, str]:
    """Checkpoint name -> our name, for ``model.language_model.*`` + ``mtp.*``.

    Names only: ``source`` may be a checkpoint directory (names come
    from ``model.safetensors.index.json`` or per-shard *headers* via
    :func:`~inference_from_scratch.weights.read_safetensors_header` --
    tensor bytes are never read), a single safetensors file, a parsed
    header/index mapping, or an explicit iterable of checkpoint names.
    With no ``source``, the local Qwen3.8 snapshot is located in the HF
    cache; failing that, names are synthesised from ``config``
    (default: the real 64-layer pattern). Vision-tower names
    (``model.visual.*``) are deliberately excluded -- that tower is
    deferred -- while FP8 ``weight_scale_inv`` companions map to the
    same suffix on our side, where a dequantising loader consumes them.
    """
    keys = _collect_checkpoint_keys(source)
    if keys is None:
        default_dir = _find_default_checkpoint()
        if default_dir is not None:
            keys = _collect_checkpoint_keys(default_dir)
    if keys is None:
        keys = _synthesized_checkpoint_keys(config or HybridConfig())
    out: dict[str, str] = {}
    for key in keys:
        mapped = _map_hybrid_key(key)
        if mapped is not None:
            out[key] = mapped
    return out


hybrid_tensor_map = list_hybrid_tensor_map
