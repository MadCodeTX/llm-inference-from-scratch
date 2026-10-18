"""Ch 02-03 -- Normalisation and RoPE: the small maths inside every block.

Why this exists in inference
------------------------------
Two pieces of per-token maths sit inside every Transformer block and
are worth dissecting once, by hand, before they hide inside attention:

* RMSNorm (Ch 02) rescales each residual-stream row ``[D]`` to unit
  root-mean-square. Llama / Qwen apply it before attention and before
  the MLP (pre-norm), so it runs ``2 * n_layer + 1`` times per token.
* RoPE (Ch 03) injects position by *rotating* Q/K vectors. It is the
  reason a cached prefix stays valid at decode time: the rotation for
  position ``t`` depends only on ``t``, never on neighbouring tokens,
  so keys computed during prefill need no update later.

How production differs
-----------------------
Fused kernels (FlashNorm, vLLM / TensorRT-LLM RoPE kernels) compute
exactly these formulae but keep intermediates in registers / SRAM and
fuse the rotation into the QKV projection epilogue. The maths per
element is identical to what is written out longhand below -- which is
the point of writing it longhand: ``F.rms_norm`` and friends are
treated as reference oracles in the tests, never called here.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalisation, from scratch (Ch 02).

    Shapes: input ``x`` is ``[..., D]`` (any leading batch / sequence
    dims), ``weight`` is ``[D]``, output is ``[..., D]`` (same shape).

    Formula per row ``x`` (over the last dim only)::

        rms(x) = sqrt(mean(x^2) + eps)
        y      = x / rms(x) * weight

    Unlike LayerNorm there is *no mean subtraction and no bias*: the
    row is only rescaled, so a constant row stays constant (just moved
    to RMS 1). That is one multiply-accumulate pass cheaper than
    LayerNorm and, empirically (Zhang & Sennrich 2019), loses nothing
    for pre-norm Transformers -- why Llama / Qwen use it everywhere.

    The reduction is computed in float32 even for half-precision
    inputs: ``mean(x^2)`` in fp16 overflows / loses precision fast,
    and production kernels do the same upcast internally.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.eps = eps
        # Gain only -- no bias term exists in RMSNorm. Initialised to
        # ones so a fresh norm is a pure rescale.
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        # mean of squares over the feature dim, kept rank for broadcast.
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(ms + self.eps)
        return (y * self.weight.float()).to(orig_dtype)

    def extra_repr(self) -> str:
        return f"d={self.d}, eps={self.eps}"


class LayerNorm(nn.Module):
    """Classic LayerNorm, from scratch, kept for comparison (Ch 02).

    Shapes: input ``[..., D]``, ``weight`` / ``bias`` each ``[D]``,
    output ``[..., D]``.

    Formula per row::

        mu  = mean(x)
        var = mean((x - mu)^2)          # biased variance (divide by D)
        y   = (x - mu) / sqrt(var + eps) * weight + bias

    Two statistics (mean *and* variance) and two affine params, vs
    RMSNorm's one of each. GPT-2 uses this; Llama / Qwen switched to
    RMSNorm. Keeping both lets the tests (and the reader) verify the
    manual maths against ``nn.LayerNorm`` and feel the difference:
    LayerNorm output rows have mean ~0, RMSNorm rows generally do not.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.d = d
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        mu = xf.mean(dim=-1, keepdim=True)
        xc = xf - mu
        var = xc.pow(2).mean(dim=-1, keepdim=True)
        y = xc * torch.rsqrt(var + self.eps)
        return (y * self.weight.float() + self.bias.float()).to(orig_dtype)

    def extra_repr(self) -> str:
        return f"d={self.d}, eps={self.eps}"


def precompute_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables: each ``[max_seq_len, head_dim]``.

    RoPE pairs up the ``head_dim`` features into ``head_dim // 2``
    2-D planes and rotates plane ``i`` at position ``t`` by angle
    ``t * theta ** (-2i / head_dim)``. Low planes spin fast (local
    detail), high planes barely move (long-range signal).

    The returned tables use the *rotate-half* layout consumed by
    :func:`apply_rope`: angles are duplicated so that
    ``cos[t] == cat([cos_half[t], cos_half[t]])``, letting the rotation
    be two elementwise multiplies against ``x`` and a swapped copy.

    Computed in float32 regardless of ``dtype`` (angles lose precision
    quickly in fp16 for large ``t``), then cast. Precomputing once for
    ``T_max`` costs ``T_max * head_dim`` floats -- trivial next to the
    KV cache -- and decode then needs only a row lookup per step.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE pairing, got {head_dim}")
    if dtype is None:
        dtype = torch.float32
    # inv_freq[i] = theta ** (-2i / D), i in [0, D/2) -> [D/2]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device).float()  # [T]
    freqs = torch.outer(t, inv_freq)  # [T, D/2] angles in radians
    # Rotate-half layout: duplicate so both halves share the angle.
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[.., D] -> cat(-x2, x1)`` where ``x1``/``x2`` are the halves.

    This is the linear partner of the duplicated cos/sin tables:
    ``x * cos + rotate_half(x) * sin`` rotates each pair
    ``(x_i, x_{i+D/2})`` -- note the pairing is half-split, *not*
    adjacent elements. (The GPT-NeoX convention; GPT-J interleaves.)
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape ``[B, H, T, D]``.

    ``cos`` / ``sin`` are the ``[T_max, D]`` tables from
    :func:`precompute_rope_cache`; rows
    ``start_pos : start_pos + T`` are used, so prefill passes
    ``start_pos=0`` and each decode step passes its absolute position.
    Using the *absolute* position is what makes cached keys reusable:
    a key rotated at prefill for position ``t`` is already correct
    when later queries attend to it.

    Properties the tests lean on (all consequences of rotation being
    orthogonal per 2-D plane): position 0 is the identity
    (angle 0), ``||rope(x)|| == ||x||``, and
    ``dot(rope(q, m), rope(k, n))`` depends only on ``m - n`` --
    attention scores see *relative* distance from absolute rotations.
    """
    T = x.shape[2]
    # [T, D] -> [1, 1, T, D] broadcast over batch and heads. Cast to
    # x's dtype/device: tables are typically kept fp32 on purpose.
    c = cos[start_pos : start_pos + T].to(dtype=x.dtype, device=x.device)
    s = sin[start_pos : start_pos + T].to(dtype=x.dtype, device=x.device)
    c = c.view(1, 1, T, -1)
    s = s.view(1, 1, T, -1)
    return x * c + _rotate_half(x) * s
