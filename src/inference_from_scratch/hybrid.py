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
``sigmoid`` is what the reference code applies, so ``"swish"``
resolves to ``"sigmoid"`` here whether it arrives via the config or
via ``gate_activation=``. ``gate_activation="silu"`` remains an
explicit non-HF alternative: ``silu(g) = g * sigmoid(g)`` is not the
reference op and must never be selected by the config label).

Partial RoPE and mrope
-----------------------
Only the first ``rotary_dim = head_dim * partial_rotary_factor`` dims
rotate (64 of 256 in the real model); the tail is identity. The real
config carries mrope sections ``[11, 11, 10]`` over the 32 rotary
*frequency pairs*: with interleaved mrope, the temporal/height/width
position axes pick different frequency slots (H overwrites slots
``1, 4, 7, ...``, W slots ``2, 5, 8, ...``). For text-only input all
three axes carry the same position, so the recomposition is the
identity and plain 1-D partial RoPE is exact. The full-attention path
accepts explicit ``position_ids`` (``[T]`` or ``[3, T]``) and builds
tables with :func:`precompute_mrope_cos_sin`: a ``[T]`` input is the
text path repeated to three axes and reproduces the default exactly,
while a vision-shaped ``[3, T]`` with distinct H/W axes rotates only
the slots those axes own. No vision tensor is executed here; the
position seam is (see docs/22).

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
from .sampling import sample_next
from .weights import read_safetensors_header

__all__ = [
    "GatedFullAttention",
    "GatedAttention",
    "HybridConfig",
    "HybridBlock",
    "HybridState",
    "StaticHybridState",
    "HybridCache",
    "QwenRMSNorm",
    "ZeroCenteredRMSNorm",
    "TinyHybridLM",
    "HybridLM",
    "apply_partial_rope",
    "generate_hybrid",
    "pipeline_device_plan",
    "DeviceMappedHybridRunner",
    "partial_rope_dim",
    "precompute_mrope_cos_sin",
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
        # Opt-in fused path (F.rms_norm); off by default so dense and
        # pipeline runners keep the manual maths bit-for-bit. The TP2
        # runner enables it (see run_qwen38_tp2 --norm) after profiling
        # showed norm launches on the decode critical path.
        self.use_fused_rmsnorm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if self.use_fused_rmsnorm:
            return F.rms_norm(
                x.float(), [self.d], 1.0 + self.weight.float(), self.eps,
            ).to(orig_dtype)
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
# mRoPE: three position axes (T/H/W) recomposed over the rotary pairs.
# ---------------------------------------------------------------------------


def _normalize_position_ids(
    position_ids: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """Normalise caller positions to ``[3, T]`` or ``[3, B, T]``.

    Accepted shapes: ``[T]`` (text: the one axis, repeated to all three
    mRoPE axes), ``[3, T]`` (per-axis positions, shared across the
    batch), ``[B, T]`` (text positions per batch row, repeated to three
    axes) and ``[3, B, T]`` (HF's batched vision-aware form). Anything
    else -- wrong rank, or a last dim other than ``seq_len`` -- raises.
    A 2-D tensor whose first dim is 3 is read as ``[3, T]`` (the task's
    vision-shaped form); a genuine 3-row text batch must use the 3-D
    form to avoid that ambiguity.
    """
    if not isinstance(position_ids, torch.Tensor):
        raise ValueError(
            f"position_ids must be a tensor, got {type(position_ids).__name__}"
        )
    p = position_ids
    if p.dim() == 1:
        if p.shape[0] != seq_len:
            raise ValueError(
                f"position_ids [T] must have T={seq_len}, got {p.shape[0]}"
            )
        return p.unsqueeze(0).expand(3, seq_len)
    if p.dim() == 2:
        if p.shape[0] == 3 and p.shape[1] == seq_len:
            return p
        if p.shape[1] == seq_len:
            return p.unsqueeze(0).expand(3, p.shape[0], seq_len)
        raise ValueError(
            f"position_ids [3, T] / [B, T] must end with T={seq_len}, "
            f"got shape {tuple(p.shape)}"
        )
    if p.dim() == 3:
        if p.shape[0] == 3 and p.shape[2] == seq_len:
            return p
        raise ValueError(
            f"position_ids [3, B, T] must start with 3 and end with "
            f"T={seq_len}, got shape {tuple(p.shape)}"
        )
    raise ValueError(
        f"position_ids must be [T], [3, T], [B, T] or [3, B, T], "
        f"got shape {tuple(p.shape)}"
    )


def precompute_mrope_cos_sin(
    position_ids: torch.Tensor,
    rotary_dim: int,
    theta: float = 10000.0,
    mrope_section: Sequence[int] | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """mRoPE cos/sin for explicit positions: ``([T, R], [T, R])``.

    Mirrors HF ``Qwen3_5TextRotaryEmbedding`` (modeling_qwen3_5.py):
    frequencies ``pos * theta ** (-2i / R)`` are built per axis over
    the ``R // 2`` rotary pairs, then *interleaved* recomposition lets
    the H axis overwrite pair slots ``1, 4, 7, ...`` and the W axis
    slots ``2, 5, 8, ...`` (stride 3, bounded by
    ``mrope_section[axis] * 3``), with the T axis supplying every other
    slot. The result is duplicated to the rotate-half layout
    ``cat([half, half])`` that :func:`apply_partial_rope` consumes, so
    pair slot ``j`` lives at columns ``j`` and ``j + R // 2``.

    Input is ``[T]`` or ``[3, T]`` (batched ``[B, T]`` / ``[3, B, T]``
    also accepted, returning ``[B, T, R]``). For text -- all three axes
    equal -- every overwrite writes the value already there, so the
    recomposition is the identity and the tables equal the plain 1-D
    partial-RoPE tables for the same positions. HF's recomposition
    runs on the cos/sin values; recomposing the angles first and then
    taking cos/sin is the same thing elementwise.
    """
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be a positive even int, got {rotary_dim}")
    if not isinstance(position_ids, torch.Tensor) or position_ids.dim() == 0:
        raise ValueError("position_ids must be a tensor with at least one dim")
    seq_len = int(position_ids.shape[-1])
    pos3 = _normalize_position_ids(position_ids, seq_len)  # [3, T] | [3, B, T]
    section = [int(v) for v in (mrope_section if mrope_section is not None else [11, 11, 10])]
    if len(section) != 3:
        raise ValueError(f"mrope_section must have 3 entries, got {section}")
    n_pairs = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=pos3.device).float() / rotary_dim)
    )  # [R/2]
    angles = pos3.float().unsqueeze(-1) * inv_freq  # [3, (B,) T, R/2]
    recomposed = angles[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        idx = slice(offset, section[axis] * 3, 3)
        if idx.start < n_pairs:
            recomposed[..., idx] = angles[axis][..., idx]
    cos_half, sin_half = recomposed.cos(), recomposed.sin()
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def _apply_rope_with_tables(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, T, D]`` with per-position tables.

    ``cos`` / ``sin`` are ``[T, R]`` (shared over the batch) or
    ``[B, T, R]`` (per row), already in the duplicated rotate-half
    layout over the rotary prefix; dims past ``rotary_dim`` pass
    through untouched, exactly as in :func:`apply_partial_rope`.
    """
    prefix = x[..., :rotary_dim]
    if cos.dim() == 2:
        c = cos.view(1, 1, cos.shape[0], rotary_dim).to(dtype=x.dtype, device=x.device)
        s = sin.view(1, 1, sin.shape[0], rotary_dim).to(dtype=x.dtype, device=x.device)
    else:
        c = cos.unsqueeze(1).to(dtype=x.dtype, device=x.device)
        s = sin.unsqueeze(1).to(dtype=x.dtype, device=x.device)
    half = rotary_dim // 2
    rotated_half = torch.cat([-prefix[..., half:], prefix[..., :half]], dim=-1)
    rotated = prefix * c + rotated_half * s
    if rotary_dim >= x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def _is_capturing(tensor: torch.Tensor) -> bool:
    """True when ``tensor`` is CUDA and its stream is capturing a graph."""
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


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
        cfg_section = _cfg_get(config, "mrope_section", default=None)
        self.mrope_section: list[int] = (
            [int(v) for v in cfg_section] if cfg_section is not None else [11, 11, 10]
        )
        self.mrope_interleaved = bool(_cfg_get(config, "mrope_interleaved", default=True))
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
        # Gate resolution. HF Qwen3_5Attention.forward applies
        # ``attn_output * torch.sigmoid(gate)`` unconditionally
        # (modeling_qwen3_5.py line 804); the checkpoint config's
        # ``output_gate_type="swish"`` is only a label for that op --
        # SiLU (``x * sigmoid(x)``) is a *different* function and is
        # never what the reference computes. So "swish" resolves to
        # "sigmoid" here, whether it arrives via the explicit
        # ``gate_activation`` argument or via the config's
        # ``output_gate_type``; ``gate_activation="silu"`` remains an
        # explicit non-HF alternative, and ``attn_output_gate=False``
        # in the config disables the gate entirely.
        cfg_gate = _cfg_get(config, "output_gate_type", default=None)
        cfg_gate_enabled = _cfg_get(config, "attn_output_gate", default=True)
        if gate_activation is not None:
            requested_gate = str(gate_activation).lower()
        elif cfg_gate is not None:
            requested_gate = str(cfg_gate).lower()
        elif cfg_gate_enabled is False:
            requested_gate = "none"
        else:
            requested_gate = "sigmoid"
        if requested_gate in ("swish", "sigmoid"):
            requested_gate = "sigmoid"
        if requested_gate not in ("sigmoid", "silu", "none"):
            raise ValueError(
                f"gate_activation must be 'sigmoid', 'silu' or 'none', got {gate_activation!r}"
            )
        self.gate_activation = requested_gate
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
        start_pos: int | torch.Tensor = 0,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """``x`` ``[B, T, D] -> [B, T, D]`` (causal, KV-cache aware).

        ``cos``/``sin`` are ``[T_max, R]`` partial-RoPE tables (wider
        full-head tables are sliced); when omitted, tables precomputed
        from the config are used. ``position_embeddings=(cos, sin)``
        is accepted as an alias (HF calling convention).

        ``position_ids`` (``[T]`` or ``[3, T]``; batched ``[B, T]`` /
        ``[3, B, T]`` also accepted) gives explicit absolute positions
        and takes precedence over ``cos``/``sin``: tables are built
        with :func:`precompute_mrope_cos_sin`, so a ``[T]`` text input
        (repeated to all three mRoPE axes) reproduces the default
        ``start_pos`` path, while differing H/W axes in a ``[3, T]``
        input rotate only their interleaved pair slots. ``start_pos``
        does not offset explicit positions -- they are already
        absolute.
        """
        if position_ids is None:
            position_ids = kwargs.get("position_ids")
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

        # Graph-safe tensor position: 0-dim tensor read on-device, no
        # .item() and no Python-int slice baked into a captured graph.
        pos_tensor: torch.Tensor | None = None
        if isinstance(start_pos, torch.Tensor):
            if start_pos.dim() != 0:
                raise ValueError(
                    f"start_pos tensor must be 0-dim, got {tuple(start_pos.shape)}")
            if position_ids is not None:
                raise ValueError("pass only one of position_ids / tensor start_pos")
            if T != 1:
                raise ValueError("tensor start_pos supports decode T=1 only")
            pos_tensor = start_pos
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor) or position_ids.shape[-1] != T:
                got = tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else type(position_ids).__name__
                raise ValueError(
                    f"position_ids must end with T={T} (one position per token), got {got}"
                )
            pos_cos, pos_sin = precompute_mrope_cos_sin(
                position_ids, self.rotary_dim, theta=self.rope_theta,
                mrope_section=self.mrope_section,
            )
            q = _apply_rope_with_tables(q, pos_cos, pos_sin, self.rotary_dim)
            k = _apply_rope_with_tables(k, pos_cos, pos_sin, self.rotary_dim)
        elif pos_tensor is not None:
            if cos is None or sin is None:
                cos, sin = self._rope_tables(1, x.device, x.dtype)
            # Gather the single RoPE row on-device: [R] -> [1, R] tables.
            # index_select (not advanced indexing, which syncs during
            # capture) keeps this a single capturable kernel.
            pos1 = pos_tensor.reshape(1)
            row_cos = cos.index_select(0, pos1)
            row_sin = sin.index_select(0, pos1)
            q = _apply_rope_with_tables(q, row_cos, row_sin, self.rotary_dim)
            k = _apply_rope_with_tables(k, row_cos, row_sin, self.rotary_dim)
        else:
            if cos is None or sin is None:
                cos, sin = self._rope_tables(int(start_pos) + T, x.device, x.dtype)  # type: ignore[arg-type]
            q = apply_partial_rope(q, cos, sin, start_pos=int(start_pos), rotary_dim=self.rotary_dim)  # type: ignore[arg-type]
            k = apply_partial_rope(k, cos, sin, start_pos=int(start_pos), rotary_dim=self.rotary_dim)  # type: ignore[arg-type]

        if kv_cache is not None:
            if pos_tensor is not None:
                if not isinstance(kv_cache, StaticKVCache):
                    raise ValueError("tensor start_pos requires a StaticKVCache")
                k, v = kv_cache.update_tensor_position(layer_idx, k, v, pos_tensor)
            elif isinstance(kv_cache, StaticKVCache):
                k, v = kv_cache.update(layer_idx, k, v, int(start_pos))  # type: ignore[arg-type]
            else:
                k, v = kv_cache.update(layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if pos_tensor is not None:
            attn = self._decode_attn_tensor_pos(q, k, v, pos_tensor)
        elif self.attn_implementation == "sdpa":
            attn = self._sdpa(q, k, v, start_pos=int(start_pos))  # type: ignore[arg-type]
        else:
            attn = manual_scaled_dot_product_attention(q, k, v, is_causal=True)

        merged = attn.transpose(1, 2).contiguous().view(B, T, H * Dh)
        if self.gate_activation == "sigmoid":
            merged = merged * torch.sigmoid(gate)
        elif self.gate_activation == "silu":
            merged = merged * F.silu(gate)
        return self.o_proj(merged)

    def _decode_attn_tensor_pos(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Decode attention over the full static KV buffer (T_q=1).

        Keys past ``position`` are masked with ``-inf`` built from an
        on-device comparison, so the shape is fixed at ``T_max`` and no
        host sync is needed -- the graph-safe twin of the int-position
        prefix attention.
        """
        t_k = k.shape[2]
        if self.attn_implementation == "sdpa":
            k_pos = torch.arange(t_k, device=k.device)
            future = k_pos > position
            mask = torch.zeros(1, t_k, device=q.device, dtype=q.dtype)
            mask = mask.masked_fill(future, float("-inf"))
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.view(1, 1, 1, t_k),
                is_causal=False, scale=self.scaling)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        k_pos = torch.arange(t_k, device=k.device)
        future = k_pos > position
        scores = scores.masked_fill(future.view(1, 1, 1, t_k), float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v)

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


class HybridState:
    """Typed inference state for a hybrid stack.

    The canonical storage is ``entries``: one ``(tag, payload)`` pair
    per layer, tagged by the layer's kind:

    * ``("kv", KVCache)`` for ``full_attention`` layers -- a
      single-layer :class:`~inference_from_scratch.kv_cache.KVCache`
      holding that layer's ``k_all`` / ``v_all``
      (``[B, H_kv, T_filled, head_dim]`` each, empty until the first
      token);
    * ``("linear", (S, conv_buffer))`` for ``linear_attention``
      layers -- the gated-delta recurrent pair
      ``(S [B, Hv, Dk, Dv], conv_buffer [B, conv_dim, K-1])``, or
      ``("linear", None)`` for the lazy zero state (no token seen yet;
      the layer materialises zeros on first use).

    ``seq_len`` is the number of positions already processed (the next
    token's absolute position). ``clone()`` / ``snapshot()`` /
    ``restore()`` deep-copy every tensor, so a restored state can be
    continued repeatedly and independently -- the mixed-type analogue
    of ch-21's state pair. Branching is therefore cheap to express:
    snapshot once after a shared prompt, restore per branch, and each
    branch continues without seeing the others' writes.

    Backward compatibility: the constructor still accepts the legacy
    ``layer_states=[...]`` form (``None`` / ``(k, v)`` / ``(S, conv)``
    per layer, disambiguated by ``layer_types`` when given), and the
    :attr:`layer_states` property still reads/writes that legacy view
    -- full layers as ``(k_all, v_all)`` or ``None`` when empty,
    linear layers as ``(S, conv_buffer)`` or ``None``. The tensors in
    the legacy view are the live payload tensors (not copies), so
    in-place edits through it behave exactly as before.
    """

    def __init__(
        self,
        layer_states: Sequence[Any] | None = None,
        seq_len: int = 0,
        *,
        layer_types: Sequence[str] | None = None,
        entries: Sequence[tuple[str, Any]] | None = None,
    ):
        if entries is not None and layer_states is not None:
            raise ValueError("pass only one of entries / layer_states")
        self.seq_len = int(seq_len)
        self.layer_types: list[str] | None = (
            [_normalize_layer_type(t) for t in layer_types]
            if layer_types is not None
            else None
        )
        if entries is not None:
            self._entries: list[tuple[str, Any]] = [
                (str(tag), payload) for tag, payload in entries
            ]
        elif layer_states is not None:
            self._entries = [
                self._entry_from_legacy(i, s) for i, s in enumerate(layer_states)
            ]
        else:
            self._entries = []

    # -- construction helpers -------------------------------------------

    def _tag_for(self, index: int) -> str:
        if self.layer_types is not None and index < len(self.layer_types):
            return "linear" if self.layer_types[index] == "linear_attention" else "kv"
        return "kv"

    def _entry_from_legacy(self, index: int, state: Any) -> tuple[str, Any]:
        if isinstance(state, KVCache):
            return ("kv", state)
        tag = self._tag_for(index)
        if state is None:
            return ("linear", None) if tag == "linear" else ("kv", KVCache(1))
        if tag == "linear":
            s, conv = state
            return ("linear", (s, conv))
        cache = KVCache(1)
        cache.keys[0], cache.values[0] = state[0], state[1]
        return ("kv", cache)

    @classmethod
    def empty(
        cls, layer_types: Sequence[str], seq_len: int = 0
    ) -> "HybridState":
        """Fresh state for a stack: empty KV caches + zero linear states."""
        entries: list[tuple[str, Any]] = [
            ("linear", None) if t == "linear_attention" else ("kv", KVCache(1))
            for t in (_normalize_layer_type(x) for x in layer_types)
        ]
        return cls(entries=entries, seq_len=seq_len, layer_types=list(layer_types))

    # -- typed access -----------------------------------------------------

    @property
    def entries(self) -> list[tuple[str, Any]]:
        """The typed per-layer ``(tag, payload)`` pairs (list copy)."""
        return list(self._entries)

    def entry(self, index: int) -> tuple[str, Any]:
        return self._entries[index]

    def kv_cache(self, index: int) -> KVCache:
        """The :class:`KVCache` payload of a full-attention layer."""
        tag, payload = self._entries[index]
        if tag != "kv":
            raise ValueError(f"layer {index} holds a {tag!r} entry, not 'kv'")
        return payload  # type: ignore[return-value]

    def linear_state(self, index: int) -> LinearAttentionState | None:
        """The ``(S, conv_buffer)`` payload of a linear layer (or None)."""
        tag, payload = self._entries[index]
        if tag != "linear":
            raise ValueError(f"layer {index} holds a {tag!r} entry, not 'linear'")
        return payload

    def __len__(self) -> int:
        return len(self._entries)

    # -- legacy view ------------------------------------------------------

    @property
    def layer_states(self) -> list[Any]:
        out: list[Any] = []
        for tag, payload in self._entries:
            if tag == "kv":
                cache: KVCache = payload
                if cache.keys[0] is None:
                    out.append(None)
                else:
                    out.append((cache.keys[0], cache.values[0]))
            else:
                out.append(payload)
        return out

    @layer_states.setter
    def layer_states(self, states: Sequence[Any]) -> None:
        self._entries = [
            self._entry_from_legacy(i, s) for i, s in enumerate(states)
        ]

    @property
    def position(self) -> int:
        return self.seq_len

    @property
    def states(self) -> list[Any]:
        return self.layer_states

    # -- copying ------------------------------------------------------------

    def snapshot(self) -> "HybridState":
        return _snapshot_hybrid_state(self)

    def restore(self) -> "HybridState":
        return _snapshot_hybrid_state(self)

    # clone() reads naturally at call sites that think in tensors.
    def clone(self) -> "HybridState":
        return _snapshot_hybrid_state(self)

    def __repr__(self) -> str:
        tags = ",".join(tag for tag, _ in self._entries)
        return f"HybridState(seq_len={self.seq_len}, entries=[{tags}])"


# Name that reads as "the cache object" at model call sites.
HybridCache = HybridState


# ---------------------------------------------------------------------------
# StaticHybridState: preallocated, CUDA-graph-friendly mixed state.
# ---------------------------------------------------------------------------


class StaticHybridState:
    """Preallocated mixed state for static (graph-friendly) decode.

    Created once from ``(config, batch_size, max_seq_len, device, dtype)``
    and reused for every decode step:

    * full-attention layers share one :class:`StaticKVCache`
      (``kv_cache``) of shape ``[n_full, B, H_kv, max_seq_len, head_dim]``
      -- slice-written in place, never concatenated;
    * linear layers each own a fixed recurrent tensor
      ``S [B, Hv, Dk, Dv]`` and conv ring ``conv [B, C, K-1]``
      (``C = 2*Hk*Dk + Hv*Dv``), stored in ``linear_S`` /
      ``linear_conv_buffers`` (parallel lists, ``None`` at full layers).

    Position representation: ``position`` is a 0-dim ``torch.long``
    tensor on ``device`` (also exposed as ``position_tensor``). It is
    the graph-friendly choice -- a captured graph can read/update it
    without a Python-int shape change -- at the cost of a host sync
    when the Python ``seq_len`` property reads it (``int(position)``).
    Callers that prefer to thread position explicitly may pass a plain
    Python ``int`` (or a 0-dim tensor) as the ``position=`` argument of
    :meth:`TinyHybridLM.decode_token_static`; when given it must equal
    the state's current position. Either way position never changes any
    tensor *shape*, so no per-step shape data-dependence is introduced.

    ``reset()`` zeroes every buffer *in place* (``data_ptr`` stable --
    asserted in tests) and rewinds position to 0. ``snapshot()`` is the
    only copying operation and only runs when explicitly requested: it
    returns an independent :class:`StaticHybridState` whose tensors are
    clones. ``copy_from(snapshot)`` copies a snapshot back in place.
    No method appends to a list or concatenates state tensors.
    """

    def __init__(
        self,
        config: HybridConfig,
        batch_size: int = 1,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        batch: int | None = None,
    ):
        if batch is not None:
            batch_size = batch
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_seq_len is None:
            max_seq_len = int(config.max_position_embeddings)
        if int(max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.config = config
        self.batch_size = int(batch_size)
        self.batch = self.batch_size
        self.max_seq_len = int(max_seq_len)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.layer_types: list[str] = list(config.layer_types)
        B, Dtype, Dev = self.batch_size, self.dtype, self.device
        # Full layers: one shared StaticKVCache, slot per full layer.
        self._full_slot: dict[int, int] = {}
        n_full = 0
        for i, t in enumerate(self.layer_types):
            if t == "full_attention":
                self._full_slot[i] = n_full
                n_full += 1
        self.num_full_layers = n_full
        self.num_linear_layers = len(self.layer_types) - n_full
        if n_full > 0:
            self.kv_cache: StaticKVCache | None = StaticKVCache(
                n_layer=n_full, batch_size=B,
                n_kv_head=config.num_key_value_heads,
                max_seq_len=self.max_seq_len, head_dim=config.head_dim,
                device=Dev, dtype=Dtype,
            )
        else:
            self.kv_cache = None
        # Aliases for the same shared cache object.
        self.static_kv_cache = self.kv_cache
        self.full_kv_cache = self.kv_cache
        self.kv_caches = self.kv_cache
        # Linear layers: fixed S + conv ring per layer.
        conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        self.conv_dim = conv_dim
        self.linear_S: list[torch.Tensor | None] = []
        self.linear_conv_buffers: list[torch.Tensor | None] = []
        for t in self.layer_types:
            if t == "linear_attention":
                self.linear_S.append(torch.zeros(
                    B, config.linear_num_value_heads,
                    config.linear_key_head_dim, config.linear_value_head_dim,
                    device=Dev, dtype=Dtype))
                self.linear_conv_buffers.append(torch.zeros(
                    B, conv_dim, max(config.linear_conv_kernel_dim - 1, 0),
                    device=Dev, dtype=Dtype))
            else:
                self.linear_S.append(None)
                self.linear_conv_buffers.append(None)
        # Aliases (same tensor objects, not copies).
        self.S_list = self.linear_S
        self.conv_buffers = self.linear_conv_buffers
        self.linear_conv = self.linear_conv_buffers
        # 0-dim position tensor (graph-friendly; see class docstring).
        self.position = torch.zeros((), dtype=torch.long, device=Dev)
        self.position_tensor = self.position

    # -- position / length ------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Positions already processed (host sync via the 0-dim tensor)."""
        return int(self.position.item())

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        self.position.fill_(int(value))

    @property
    def pos(self) -> int:
        return self.seq_len

    # -- per-layer access ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.layer_types)

    def full_slot(self, layer_idx: int) -> int:
        """Slot of ``layer_idx`` inside the shared ``kv_cache``."""
        try:
            return self._full_slot[int(layer_idx)]
        except KeyError:
            raise ValueError(f"layer {layer_idx} is not a full_attention layer") from None

    def linear_state(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(S, conv_buffer)`` preallocated tensors of a linear layer."""
        S, conv = self.linear_S[layer_idx], self.linear_conv_buffers[layer_idx]
        if S is None or conv is None:
            raise ValueError(f"layer {layer_idx} is not a linear_attention layer")
        return (S, conv)

    # Compatibility: HybridState-style names.
    def linear_state_tensors(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear_state(layer_idx)

    @property
    def linear_states(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        return [
            (s, c) if s is not None and c is not None else None
            for s, c in zip(self.linear_S, self.linear_conv_buffers)
        ]

    def advance(self, n: int = 1) -> None:
        """Advance position in place (0-dim tensor, no shape change)."""
        self.position.add_(int(n))

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Zero all state in place; every ``data_ptr`` is preserved."""
        if self.kv_cache is not None:
            self.kv_cache.clear()  # zero_() in place, buffers kept
        for S in self.linear_S:
            if S is not None:
                S.zero_()
        for conv in self.linear_conv_buffers:
            if conv is not None:
                conv.zero_()
        self.position.zero_()

    def snapshot(self) -> "StaticHybridState":
        """Explicit deep copy (clones every tensor); the only copy path."""
        out = object.__new__(StaticHybridState)
        out.config = self.config
        out.batch_size = self.batch_size
        out.batch = self.batch
        out.max_seq_len = self.max_seq_len
        out.device = self.device
        out.dtype = self.dtype
        out.layer_types = list(self.layer_types)
        out._full_slot = dict(self._full_slot)
        out.num_full_layers = self.num_full_layers
        out.num_linear_layers = self.num_linear_layers
        out.conv_dim = self.conv_dim
        if self.kv_cache is not None:
            cache = StaticKVCache(
                n_layer=self.kv_cache.n_layer, batch_size=self.batch_size,
                n_kv_head=self.config.num_key_value_heads,
                max_seq_len=self.max_seq_len, head_dim=self.config.head_dim,
                device=self.device, dtype=self.dtype)
            cache.k_cache.copy_(self.kv_cache.k_cache)
            cache.v_cache.copy_(self.kv_cache.v_cache)
            cache._len = self.kv_cache._len
            out.kv_cache = cache
        else:
            out.kv_cache = None
        out.static_kv_cache = out.kv_cache
        out.full_kv_cache = out.kv_cache
        out.kv_caches = out.kv_cache
        out.linear_S = [s.clone() if s is not None else None for s in self.linear_S]
        out.linear_conv_buffers = [
            c.clone() if c is not None else None for c in self.linear_conv_buffers]
        out.S_list = out.linear_S
        out.conv_buffers = out.linear_conv_buffers
        out.linear_conv = out.linear_conv_buffers
        out.position = self.position.clone()
        out.position_tensor = out.position
        return out

    clone = snapshot

    def copy_from(self, snapshot: "StaticHybridState") -> None:
        """Copy ``snapshot`` back into this state in place (ptrs stable)."""
        if snapshot.batch_size != self.batch_size or snapshot.max_seq_len != self.max_seq_len:
            raise ValueError("snapshot shape (batch/max_seq_len) does not match this state")
        if self.kv_cache is not None and snapshot.kv_cache is not None:
            self.kv_cache.k_cache.copy_(snapshot.kv_cache.k_cache)
            self.kv_cache.v_cache.copy_(snapshot.kv_cache.v_cache)
            self.kv_cache._len = snapshot.kv_cache._len
        for dst, src in zip(self.linear_S, snapshot.linear_S):
            if dst is not None and src is not None:
                dst.copy_(src)
        for dst, src in zip(self.linear_conv_buffers, snapshot.linear_conv_buffers):
            if dst is not None and src is not None:
                dst.copy_(src)
        self.position.copy_(snapshot.position)

    restore_from = copy_from

    def data_ptrs(self) -> dict[str, list[int]]:
        """Storage pointers of every state tensor (for stability tests)."""
        ptrs: dict[str, list[int]] = {"linear_S": [], "linear_conv": [], "kv": []}
        for s in self.linear_S:
            if s is not None:
                ptrs["linear_S"].append(s.data_ptr())
        for c in self.linear_conv_buffers:
            if c is not None:
                ptrs["linear_conv"].append(c.data_ptr())
        if self.kv_cache is not None:
            ptrs["kv"] = [self.kv_cache.k_cache.data_ptr(), self.kv_cache.v_cache.data_ptr()]
        ptrs["position"] = [self.position.data_ptr()]
        return ptrs

    def __repr__(self) -> str:
        return (f"StaticHybridState(batch={self.batch_size}, "
                f"max_seq_len={self.max_seq_len}, seq_len={self.seq_len}, "
                f"layers={len(self.layer_types)})")


def _clone_layer_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) == 2:
        a, b = state
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return (a.clone(), b.clone())
    return state


def _clone_kv_cache(cache: KVCache) -> KVCache:
    """Deep-copy a single-layer KVCache (tensors cloned, None kept)."""
    out = KVCache(len(cache.keys))
    out.keys = [k.clone() if k is not None else None for k in cache.keys]
    out.values = [v.clone() if v is not None else None for v in cache.values]
    return out


def _snapshot_hybrid_state(state: HybridState) -> HybridState:
    entries: list[tuple[str, Any]] = []
    for tag, payload in state.entries:
        if tag == "kv":
            entries.append(("kv", _clone_kv_cache(payload)))
        elif payload is None:
            entries.append(("linear", None))
        else:
            s, conv = payload
            entries.append(("linear", (s.clone(), conv.clone())))
    return HybridState(
        entries=entries,
        seq_len=int(state.seq_len),
        layer_types=state.layer_types,
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
        position_ids: torch.Tensor | None = None,
        prefill_method: str = "sequential",
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, Any]:
        """Run the block threading one layer's state.

        ``layer_state`` is, for full layers, a single-layer
        :class:`KVCache`, a legacy ``(k_all, v_all)`` pair, or
        ``None`` (empty); for linear layers a
        ``(S, conv_buffer)`` state or ``None`` (zero state).
        ``position_ids`` (full layers only) and
        ``prefill_method`` / ``chunk_size`` (linear layers only) are
        forwarded to the token mixer. Returns
        ``(out [B, T, D], new_layer_state)`` with the legacy payload
        shapes -- ``(k_all, v_all)`` for full layers, ``(S, conv)``
        for linear ones; :class:`TinyHybridLM` wraps the full-layer
        pair into the typed ``("kv", KVCache)`` entry. The input state
        is never mutated (a passed ``KVCache`` is forked, not written
        through: ``KVCache.update`` only ever rebinds its slots to
        freshly concatenated tensors).
        """
        idx = self.layer_idx if layer_idx is None else int(layer_idx)
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            if isinstance(layer_state, KVCache):
                cache = KVCache(len(layer_state.keys))
                cache.keys = list(layer_state.keys)
                cache.values = list(layer_state.values)
            else:
                cache = KVCache(1)
                if layer_state is not None:
                    cache.keys[0], cache.values[0] = layer_state
            mixed = self.self_attn(
                h, cos, sin, kv_cache=cache, layer_idx=0,
                start_pos=start_pos, position_ids=position_ids,
            )
            new_state = (cache.keys[0], cache.values[0])
        else:
            assert self.linear_attn is not None
            mixed, new_state = self.linear_attn.forward_prefill(
                h, state=layer_state, method=prefill_method, chunk_size=chunk_size
            )
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_state

    def decode_token_static(
        self,
        x: torch.Tensor,
        state: "StaticHybridState",
        position: int | torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """In-place single-token block step over preallocated ``state``.

        ``x`` is hidden ``[B, D]`` or ``[B, 1, D]`` (rank preserved on
        return) -- a HybridBlock consumes hidden states, not token ids;
        token-id entry is :meth:`TinyHybridLM.decode_token_static`.
        For full layers the K/V for this token are slice-written into
        ``state.kv_cache`` at ``position``; for linear layers
        :meth:`GatedDeltaNet.step_inplace` mutates the layer's fixed
        ``S`` / conv ring. No list append / ``torch.cat`` touches state.
        ``position`` may be a Python int or a 0-dim tensor; when omitted
        the state's own position is used. The state's position is *not*
        advanced here -- the LM advances it once after all blocks run,
        so every layer sees the same absolute position for this token.
        """
        if not isinstance(state, StaticHybridState):
            raise ValueError(
                f"decode_token_static requires a StaticHybridState, got {type(state).__name__}")
        # Tensor position (or omitted position while capturing) stays a
        # tensor end-to-end: no .item(), so no host sync is captured.
        pos: int | torch.Tensor
        if isinstance(position, torch.Tensor):
            if position.dim() != 0:
                raise ValueError(f"position tensor must be 0-dim, got shape {tuple(position.shape)}")
            pos = position
        elif position is None and _is_capturing(state.position):
            pos = state.position
        elif position is None:
            pos = state.seq_len
        else:
            pos = int(position)  # type: ignore[arg-type]
        was_2d = x.dim() == 2
        if was_2d:
            x3 = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x3 = x
        else:
            raise ValueError(f"x must be [B, D] or [B, 1, D], got {tuple(x.shape)}")
        h = self.input_layernorm(x3)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None and state.kv_cache is not None
            mixed = self.self_attn(
                h, cos, sin, kv_cache=state.kv_cache,
                layer_idx=state.full_slot(self.layer_idx), start_pos=pos,
            )
        else:
            assert self.linear_attn is not None
            S, conv_buf = state.linear_state(self.layer_idx)
            mixed = self.linear_attn.step_inplace(h[:, 0, :], S, conv_buf).unsqueeze(1)
        out = x3 + mixed
        out = out + self.mlp(self.post_attention_layernorm(out))
        return out[:, 0, :] if was_2d else out

    # Aliases: same in-place block step under the other natural names.
    forward_static = decode_token_static
    decode_step_static = decode_token_static
    step_inplace = decode_token_static

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        kv_cache: KVCache | StaticKVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
        state: Any = None,
        position_ids: torch.Tensor | None = None,
        prefill_method: str = "sequential",
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> Any:
        """Hidden-only forward, or ``(hidden, state)`` when state is used.

        With no state arguments this runs a stateless full-sequence
        pass and returns ``[B, T, D]``. Passing ``state=`` (a per-layer
        state) or a ``kv_cache`` for a full layer threads state and
        returns ``(hidden, new_layer_state)``. ``position_ids`` /
        ``prefill_method`` / ``chunk_size`` are forwarded to the token
        mixer (also accepted as ``**kwargs`` for HF-style callers).
        """
        if position_ids is None:
            position_ids = kwargs.get("position_ids")
        if state is not None or kwargs.get("layer_state") is not None:
            return self.forward_with_state(
                x, state if state is not None else kwargs["layer_state"],
                cos=cos, sin=sin, start_pos=start_pos, layer_idx=layer_idx,
                position_ids=position_ids,
                prefill_method=prefill_method, chunk_size=chunk_size,
            )
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            mixed = self.self_attn(
                h, cos, sin, kv_cache=kv_cache, layer_idx=layer_idx,
                start_pos=start_pos, position_ids=position_ids,
            )
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
        """Fresh empty mixed state (typed entries, ``seq_len=0``).

        Full layers start as ``("kv", KVCache)`` with nothing cached;
        linear layers as the lazy zero state ``("linear", None)``.
        ``batch_size`` / ``device`` / ``dtype`` are accepted for
        signature compatibility with the per-layer state constructors;
        both payload kinds materialise lazily on the first forward, so
        they are not needed here.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return HybridState.empty(self.config.layer_types, seq_len=0)

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
        position_ids: torch.Tensor | None = None,
        prefill_method: str = "sequential",
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, HybridState]:
        """Token ids ``[B, T]`` -> ``(logits [B, T, V], new_state)``.

        ``state=None`` prefills from position 0; otherwise processing
        continues at ``state.seq_len`` (or explicit ``start_pos``).
        ``cache=`` / ``hybrid_state=`` / ``kv_caches=`` (when it holds a
        :class:`HybridState`) are accepted as aliases for ``state``.

        ``position_ids`` (``[T]`` or ``[3, T]``) supplies explicit
        absolute positions to the full-attention layers only (the
        gated-delta layers carry no position embedding); the default
        is the text path ``arange(start_pos, start_pos + T)`` repeated
        over the three mRoPE axes, which the mRoPE recomposition maps
        to exactly the default tables. ``prefill_method`` /
        ``chunk_size`` select the linear layers' prefill evaluation
        (``"sequential"`` by default, as in
        :meth:`GatedDeltaNet.forward_prefill`). The input state is
        never mutated: the returned state owns the extended KV /
        recurrent tensors.
        """
        if position_ids is None:
            position_ids = kwargs.get("position_ids")
        if state is None:
            for alias in ("cache", "hybrid_state", "kv_caches", "past_key_values"):
                candidate = kwargs.get(alias)
                if isinstance(candidate, HybridState):
                    state = candidate
                    break
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        if position_ids is not None and (
            not isinstance(position_ids, torch.Tensor) or position_ids.shape[-1] != T
        ):
            got = tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else type(position_ids).__name__
            raise ValueError(
                f"position_ids must end with T={T} (one position per token), got {got}"
            )
        if state is None:
            state = self.initial_state(B, device=idx.device)
        if len(state) != len(self.layers):
            raise ValueError(
                f"state has {len(state)} layers, model has {len(self.layers)}"
            )
        if start_pos is None:
            start_pos = int(state.seq_len)
        cos, sin = self._rope(start_pos + T, idx.device, torch.float32)
        x = self.embed_tokens(idx)
        new_entries: list[tuple[str, Any]] = []
        for i, block in enumerate(self.layers):
            tag, payload = state.entry(i)
            if block.layer_type == "full_attention":
                x, kv_pair = block.forward_with_state(
                    x, payload, cos=cos, sin=sin,
                    start_pos=start_pos, layer_idx=i,
                    position_ids=position_ids,
                )
                cache = KVCache(1)
                cache.keys[0], cache.values[0] = kv_pair
                new_entries.append(("kv", cache))
            else:
                # A legacy state built without layer_types may hold an
                # empty-KV placeholder here; it means the zero state.
                block_payload = payload
                if isinstance(block_payload, KVCache) and block_payload.keys[0] is None:
                    block_payload = None
                x, linear_payload = block.forward_with_state(
                    x, block_payload, cos=cos, sin=sin,
                    start_pos=start_pos, layer_idx=i,
                    prefill_method=prefill_method, chunk_size=chunk_size,
                )
                new_entries.append(("linear", linear_payload))
        logits = self.lm_head(self.norm(x))
        return logits, HybridState(
            entries=new_entries,
            seq_len=start_pos + T,
            layer_types=self.config.layer_types,
        )

    def prefill(
        self,
        idx: torch.Tensor,
        state: HybridState | None = None,
        *,
        prefill_method: str = "sequential",
        chunk_size: int = 64,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HybridState]:
        """Prefill ``idx`` ``[B, T]`` (from scratch unless ``state`` given)."""
        return self.forward(
            idx, state=state, prefill_method=prefill_method,
            chunk_size=chunk_size, position_ids=position_ids,
        )

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

    def static_state(
        self,
        batch_size: int = 1,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> StaticHybridState:
        """Allocate a :class:`StaticHybridState` for this model's config.

        ``device`` / ``dtype`` default to this model's own parameter
        device/dtype, so a model moved to CUDA (or cast to bf16) gets a
        state that can actually be stepped without the caller having
        to repeat the model's placement. Pass them explicitly to
        override (e.g. an fp32 state for a bf16 model).
        """
        if device is None or dtype is None:
            param = next(self.parameters(), None)
            if param is not None:
                if device is None:
                    device = param.device
                if dtype is None:
                    dtype = param.dtype
        return StaticHybridState(
            self.config, batch_size=batch_size,
            max_seq_len=max_seq_len if max_seq_len is not None else self.config.max_position_embeddings,
            device=device, dtype=dtype,
        )

    # Alias name for the same factory.
    initial_static_state = static_state

    def decode_token_static(
        self,
        token_id_1d: torch.Tensor,
        state: StaticHybridState,
        position: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode one token in place over a :class:`StaticHybridState`.

        ``token_id_1d`` is ``[B]`` (a ``[B, 1]`` column is also accepted);
        returns logits ``[B, vocab_size]`` for the next token. State is
        mutated in place -- full layers slice-write into the shared
        ``StaticKVCache``, linear layers update their fixed ``S`` / conv
        ring via :meth:`GatedDeltaNet.step_inplace` -- and the state's
        0-dim position tensor is advanced by one after all blocks run.
        No list append / ``torch.cat`` touches state in this path.

        ``position`` documents the integer-position choice: omit it to
        use the state's 0-dim position tensor (graph-friendly), or pass
        a Python ``int`` / 0-dim tensor that must equal the state's
        current position. Position never affects any tensor shape.

        Graph-safe form: when ``position`` is a 0-dim tensor (expected
        to be ``state.position`` itself), or when it is omitted while a
        CUDA graph capture is in progress, position stays a tensor
        end-to-end -- RoPE rows are gathered on-device, the KV write
        uses ``index_copy_`` at the tensor position, and full attention
        masks the fixed ``max_seq_len`` buffer on-device. No ``.item()``
        runs in that form (outside capture, the usual equality /
        overflow checks still read it once, before any kernel runs);
        during capture those Python checks cannot run, so the caller
        must guarantee ``steps <= max_seq_len`` before capturing.

        Raises ``ValueError`` with a clear message before mutating
        anything when the next position would reach ``max_seq_len``
        (overflow) or when shapes / state type do not match.
        """
        if not isinstance(state, StaticHybridState):
            raise ValueError(
                f"decode_token_static requires a StaticHybridState, got {type(state).__name__}")
        if token_id_1d.dim() == 2 and token_id_1d.shape[1] == 1:
            token_ids = token_id_1d[:, 0]
        elif token_id_1d.dim() == 1:
            token_ids = token_id_1d
        else:
            raise ValueError(
                f"token_id_1d must be [B] or [B, 1], got {tuple(token_id_1d.shape)}")
        if token_ids.shape[0] != state.batch_size:
            raise ValueError(
                f"token batch {token_ids.shape[0]} does not match state batch_size={state.batch_size}")
        pos: int | torch.Tensor
        capturing = _is_capturing(state.position)
        if isinstance(position, torch.Tensor):
            if position.dim() != 0:
                raise ValueError(f"position tensor must be 0-dim, got shape {tuple(position.shape)}")
            pos = position
        elif position is None and capturing:
            pos = state.position
        elif position is None:
            pos = state.seq_len
        else:
            pos = int(position)  # type: ignore[arg-type]
        if isinstance(pos, torch.Tensor):
            if not capturing:
                pos_int = int(pos.item())
                if pos_int != state.seq_len:
                    raise ValueError(
                        f"position={pos_int} does not match state position "
                        f"{state.seq_len} (static decode is strictly sequential)")
                if pos_int >= state.max_seq_len:
                    raise ValueError(
                        f"StaticHybridState overflow: position {pos_int} would exceed "
                        f"max_seq_len={state.max_seq_len} (state already holds "
                        f"{state.seq_len} tokens); allocate a larger StaticHybridState "
                        "or reset() it before decoding more tokens")
            cos, sin = self.cos, self.sin
        else:
            if pos != state.seq_len:
                raise ValueError(
                    f"position={pos} does not match state position {state.seq_len} "
                    "(static decode is strictly sequential)")
            if pos >= state.max_seq_len:
                raise ValueError(
                    f"StaticHybridState overflow: position {pos} would exceed "
                    f"max_seq_len={state.max_seq_len} (state already holds "
                    f"{state.seq_len} tokens); allocate a larger StaticHybridState "
                    "or reset() it before decoding more tokens")
            cos, sin = self._rope(pos + 1, token_ids.device, torch.float32)
        x = self.embed_tokens(token_ids).unsqueeze(1)  # [B, 1, D]
        for block in self.layers:
            x = block.decode_token_static(x, state, position=pos, cos=cos, sin=sin)
        logits = self.lm_head(self.norm(x))  # [B, 1, V]
        state.advance(1)
        return logits[:, 0, :]

    # Aliases: same static decode under other natural names.
    decode_static = decode_token_static
    step_static = decode_token_static


class HybridLM(TinyHybridLM):
    """General hybrid LM: the same stack as :class:`TinyHybridLM` at any size.

    ``TinyHybridLM`` was named for the tiny test fixtures, but its
    ``state_dict`` naming already matches the checkpoint target names
    produced by :func:`list_hybrid_tensor_map` / ``hybrid_checkpoint``
    (``embed_tokens.weight``, ``layers.{i}.*``, ``norm.weight``,
    ``lm_head.weight``) for both layer types, so no separate naming
    layer is needed. ``HybridLM`` is the general-purpose name for
    full-config construction (e.g. on the ``meta`` device for
    shape-only validation of the 27B checkpoint); behaviour and
    naming are identical, and ``TinyHybridLM`` is preserved.
    """


# ---------------------------------------------------------------------------
# Stateful generation over the hybrid stack.
# ---------------------------------------------------------------------------


def generate_hybrid(
    model: TinyHybridLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    prefill_method: str = "sequential",
    chunk_size: int = 64,
    *,
    eos: int | None = None,
) -> torch.Tensor:
    """Generate up to ``max_new_tokens`` tokens after prompt ``idx``.

    The hybrid analogue of :func:`~inference_from_scratch.generate.generate`:
    one stateful prefill over the prompt (``idx`` is ``[B, T_prompt]``),
    then one :meth:`TinyHybridLM.decode_step` per emitted token, with
    each step's ``logits[:, -1]`` sampled by
    :func:`~inference_from_scratch.sampling.sample_next`
    (``temperature=0`` is greedy; ``generator`` seeds stochastic runs).
    Returns ``[B, T_prompt + n]`` including the prompt, with
    ``n <= max_new_tokens`` -- smaller only when every batch row has
    emitted ``eos_id`` (``eos`` is accepted as an alias; the EOS token
    itself stays in the output, matching ``generate``).

    ``prefill_method`` selects how the linear layers evaluate the
    prompt prefill: ``"sequential"`` (default; the reference scan) or
    ``"chunkwise"`` (the chunked gated-delta form, equal to fp32
    tolerance). Decode steps are single tokens and are unaffected.
    Under greedy sampling the two prefill methods emit identical token
    ids on the tiny test models (asserted in the tests); under
    stochastic sampling they share the sampler and the seed but may
    diverge once a near-tie flips, as any re-associated arithmetic can.
    """
    if eos_id is None:
        eos_id = eos
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if idx.dim() != 2:
        raise ValueError(f"idx must be [B, T], got shape {tuple(idx.shape)}")
    if not isinstance(prefill_method, str) or prefill_method.strip().lower() not in (
        "sequential",
        "chunkwise",
    ):
        raise ValueError(
            f"prefill_method must be 'sequential' or 'chunkwise', got {prefill_method!r}"
        )
    if max_new_tokens == 0:
        return idx.clone()

    sample_kwargs = dict(
        temperature=temperature, top_k=top_k, top_p=top_p, generator=generator,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = idx.clone()
            logits, state = model.prefill(
                out, prefill_method=prefill_method.strip().lower(),
                chunk_size=chunk_size,
            )
            next_token = sample_next(logits[:, -1], **sample_kwargs)
            out = torch.cat([out, next_token], dim=1)
            finished = (
                (next_token.squeeze(1) == eos_id)
                if eos_id is not None
                else torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
            )
            if bool(finished.all()):
                return out
            for _ in range(max_new_tokens - 1):
                logits, state = model.decode_step(next_token, state)
                next_token = sample_next(logits[:, -1], **sample_kwargs)
                out = torch.cat([out, next_token], dim=1)
                if eos_id is not None:
                    finished = finished | (next_token.squeeze(1) == eos_id)
                    if bool(finished.all()):
                        break
            return out
    finally:
        if was_training:
            model.train()


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


# ---------------------------------------------------------------------------
# Pipeline-by-layers device mapping (two-GPU bridge; see scripts/run_qwen38_gpu.py).
# ---------------------------------------------------------------------------


def pipeline_device_plan(
    num_layers: int,
    split: int,
    device0: str | torch.device = "cuda:0",
    device1: str | torch.device = "cuda:1",
) -> dict[str, Any]:
    """Layer->device assignment for the pipeline-by-layers bridge.

    Layers ``0 .. split-1`` (with the embedding) live on ``device0``;
    layers ``split .. num_layers-1`` plus the final norm and ``lm_head``
    live on ``device1``. ``split=0`` puts every layer on ``device1``
    (embedding still enters on ``device0``); ``split=num_layers`` puts
    every layer on ``device0`` and only norm/head on ``device1``.
    """
    if int(num_layers) <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if not (0 <= int(split) <= int(num_layers)):
        raise ValueError(f"split must be in [0, {num_layers}], got {split}")
    d0, d1 = str(device0), str(device1)
    return {
        "device0": d0, "device1": d1, "split": int(split),
        "num_layers": int(num_layers), "embed": d0,
        "layers": [d0 if i < int(split) else d1 for i in range(int(num_layers))],
        "norm": d1, "lm_head": d1,
    }


class DeviceMappedHybridRunner:
    """Forward loop for a :class:`TinyHybridLM` split across two devices.

    The model's modules must already reside on the devices named by
    :func:`pipeline_device_plan` (embedding + early layers on
    ``device0``, late layers + norm + head on ``device1``); this runner
    only moves the hidden state across the boundary inside the loop.
    ``TinyHybridLM.forward`` is untouched, so single-device tests keep
    their semantics. ``rms_hook(layer_idx, hidden)``, when given, is
    called after every block (per-layer hidden-RMS bisect aid).
    """

    def __init__(self, model: TinyHybridLM, split: int,
                 device0: str | torch.device = "cuda:0",
                 device1: str | torch.device = "cuda:1",
                 rms_hook: Any = None):
        self.model = model
        self.plan = pipeline_device_plan(
            len(model.layers), split, device0, device1)
        self.device0 = torch.device(self.plan["device0"])
        self.device1 = torch.device(self.plan["device1"])
        self.split = int(split)
        self.rms_hook = rms_hook
        self._rope_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _tables(self, device: torch.device, end: int):
        cached = self._rope_cache.get(device)
        if cached is None or cached[0].shape[0] < end:
            cos, sin, _ = precompute_partial_rope_cache(
                max(int(end), 128), self.model.config.head_dim,
                self.model.config.partial_rotary_factor,
                theta=self.model.config.rope_theta,
                device=device, dtype=torch.float32)
            self._rope_cache[device] = (cos, sin)
            return cos, sin
        return cached

    def run(self, idx: torch.Tensor, state: HybridState | None = None,
            start_pos: int | None = None,
            prefill_method: str = "sequential", chunk_size: int = 64,
            ) -> tuple[torch.Tensor, HybridState]:
        """``idx [B, T]`` -> ``(logits [B, T, V] on device1, new_state)``."""
        model = self.model
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        idx = idx.to(self.device0)
        B, T = idx.shape
        if state is None:
            state = model.initial_state(B, device=self.device0)
        if len(state) != len(model.layers):
            raise ValueError(
                f"state has {len(state)} layers, model has {len(model.layers)}")
        if start_pos is None:
            start_pos = int(state.seq_len)
        x = model.embed_tokens(idx)
        new_entries: list[tuple[str, Any]] = []
        for i, block in enumerate(model.layers):
            dev = torch.device(self.plan["layers"][i])
            if x.device != dev:
                x = x.to(dev)
            cos, sin = self._tables(dev, start_pos + T)
            tag, payload = state.entry(i)
            if block.layer_type == "full_attention":
                x, kv_pair = block.forward_with_state(
                    x, payload, cos=cos, sin=sin, start_pos=start_pos,
                    layer_idx=i)
                cache = KVCache(1)
                cache.keys[0], cache.values[0] = kv_pair
                new_entries.append(("kv", cache))
            else:
                block_payload = payload
                if isinstance(block_payload, KVCache) and block_payload.keys[0] is None:
                    block_payload = None
                x, linear_payload = block.forward_with_state(
                    x, block_payload, cos=cos, sin=sin, start_pos=start_pos,
                    layer_idx=i, prefill_method=prefill_method,
                    chunk_size=chunk_size)
                new_entries.append(("linear", linear_payload))
            if self.rms_hook is not None:
                self.rms_hook(i, x)
        if x.device != self.device1:
            x = x.to(self.device1)
        logits = model.lm_head(model.norm(x))
        return logits, HybridState(
            entries=new_entries, seq_len=start_pos + T,
            layer_types=model.config.layer_types)

    @torch.no_grad()
    def generate_greedy(self, idx: torch.Tensor, max_new_tokens: int,
                        eos_id: int | None = None) -> list[int]:
        """Greedy-generate ``max_new_tokens`` ids for a single row ``[1, T]``."""
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        if idx.shape[0] != 1:
            raise ValueError("generate_greedy supports a single row [1, T]")
        logits, state = self.run(idx)
        out: list[int] = []
        nxt = int(logits[0, -1].argmax().item())
        out.append(nxt)
        for _ in range(int(max_new_tokens) - 1):
            if eos_id is not None and nxt == int(eos_id):
                break
            step = torch.tensor([[nxt]], dtype=torch.long, device=self.device1)
            logits, state = self.run(step, state=state)
            nxt = int(logits[0, -1].argmax().item())
            out.append(nxt)
        return out[: int(max_new_tokens)]
