"""Ch 10 -- Quantization: fewer bytes per weight, bounded error.

Why this exists in inference
----------------------------
Decode is memory-bandwidth bound (Ch 00/05): every generated token
re-reads every weight from memory. Halving the bytes per weight
roughly halves decode time and doubles the model that fits. Weight
quantization stores weights in a low-precision integer grid plus a
small floating-point scale, and dequantizes on the fly (or in fused
kernels) for the matmul. This chapter dissects the simplest useful
scheme -- per-channel symmetric INT8 -- and the byte accounting that
decides whether a model + its KV cache fits at all.

Per-channel symmetric INT8
--------------------------
A weight matrix ``W`` is ``[out_features, in_features]``: each output
channel (row) is one dot-product unit, so each row gets its *own*
scale. For row ``i``::

    scale_i = max_j |W[i, j]| / 127
    Q[i, j] = clamp(round(W[i, j] / scale_i), -127, 127)   (int8)
    W'[i, j] = Q[i, j] * scale_i                            (dequant)

Symmetric means zero maps to zero exactly (no zero-point offset):
signed weights centred on 0 quantize without bias, and the multiply
stays a single scale per row. 127 (not 128) keeps the grid symmetric:
-127..127, so ``-W`` quantizes to exactly ``-Q``.

Error bound: rounding moves each element by at most half a grid step,
so ``|W - W'| <= scale_i / 2`` elementwise (up to float rounding in
the divide/multiply themselves). One outlier in a row stretches that
row's grid and coarsens every other element in it -- which is why
production schemes go finer (per-group scales, e.g. group-size 128
in GPTQ/AWQ/GGUF) or clip outliers. The tests hold the roundtrip to
exactly this bound.

Dtype bytes accounting
----------------------
Bytes are the budget everything else is spent from::

    dtype   bits   bytes/param   27e9 params (ASSUMED count, see below)
    fp32      32        4            108.0 GB
    fp16      16        2             54.0 GB
    bf16      16        2             54.0 GB
    int8       8        1             27.0 GB
    int4       4        0.5           13.5 GB

fp16 and bf16 cost the same bytes; they differ in *range vs
precision* (bf16 keeps fp32's 8 exponent bits, fp16 keeps more
mantissa). int4 packs two values per byte, so odd counts round up
to a whole byte. Scales add a small overhead on top (one fp32 per
output channel here -- negligible next to the matrix, and counted
honestly by callers that need it, not hidden in these helpers).

Worked 27B-scale EXAMPLE (ASSUMED dims -- NOT any real model's config)
----------------------------------------------------------------------
Every number below is computed from the formulas in this module via
:func:`worked_example`; the dims are round, plausible 27B-class
numbers chosen for the arithmetic, explicitly ASSUMED. They are
**not** Qwen's (or any shipped model's) real configuration -- real
configs are read from ``config.json`` by ``weights.load_hf_config``
(Ch 09), never copied from a worked example.

ASSUMED dims: 27e9 params; ``n_layer=64``, ``d_model=5120``,
``n_head=40``, ``n_kv_head=8`` (GQA), so ``head_dim=128``.

Weights: ``bytes_for_params(27_000_000_000, bits)`` gives the table
above -- 54 GB in fp16/bf16, 13.5 GB in int4. That gap is the whole
reason W4A16-style deployments exist.

KV cache (fp16, batch=1, seq_len=4096), from
:func:`kv_cache_bytes`::

    2 (K+V) * 64 layers * 1 * 8 KV heads * 4096 tokens * 128 * 2 B
    = 1,073,741,824 bytes  =  1 GiB   (256 KiB per token)

At batch=8 / seq=32768 that same formula gives 64 GiB -- the cache
outgrows int4 weights, which is why GQA, paged allocation (Ch 11)
and KV quantization exist. Run :func:`worked_example` to reproduce
every figure from the formulas.

How production differs
----------------------
GGUF/GPTQ/AWQ ship per-group scales (and sometimes zero-points,
i.e. asymmetric grids), activation-aware outlier handling, and
fused dequant+matmul kernels (Marlin, etc.) so the fp16 compute
never materialises a full dequantized matrix. KV caches are also
quantized (int8/fp8) in long-context serving. The dissection keeps
one scale per row and a separate dequantize step so the error and
the bytes can each be measured in isolation.
"""

from __future__ import annotations

import math

import torch

from .config import ModelConfig

# Bits per stored parameter for each dtype this repo talks about.
# bf16 and fp16 are both 16-bit floats (range/precision differ, bytes
# do not); int4 packs two values into one byte.
BITS_PER_PARAM: dict[str, int] = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
}

# Signed INT8 symmetric grid uses [-127, 127] (not -128) so the grid
# is symmetric about zero: quantize(-w) == -quantize(w).
_INT8_QMAX = 127


def quantize_int8_per_channel(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight matrix to per-channel symmetric INT8.

    ``w`` is ``[out_features, in_features]``; each row (output
    channel) gets one scale ``max|row| / 127``. Returns
    ``(q, scales)`` with ``q`` int8 of the same shape as ``w`` and
    ``scales`` fp32 of shape ``[out_features]``, such that
    ``dequantize_int8(q, scales)`` approximates ``w`` with per-element
    error at most ``scales[i] / 2`` (plus float rounding).

    Edge cases: an all-zero row has ``max|row| == 0``; its scale is
    set to 1.0 (any finite scale dequantizes ``q == 0`` back to exact
    zeros -- 1.0 simply avoids 0/0). Input is dequantized in fp32
    regardless of ``w.dtype`` so the roundtrip error is the grid's,
    not the input dtype's.
    """
    if w.dim() != 2:
        raise ValueError(
            f"quantize_int8_per_channel expects a 2D [out, in] matrix, "
            f"got shape {tuple(w.shape)}"
        )
    wf = w.detach().to(torch.float32)
    max_abs = wf.abs().amax(dim=1)  # [out_features]
    scales = max_abs / float(_INT8_QMAX)
    # Zero rows: scale 0 would make w/scale NaN. q is 0 there anyway,
    # so pin the scale to 1.0 -- dequant still returns exact zeros.
    scales = torch.where(max_abs == 0, torch.ones_like(scales), scales)
    q = torch.round(wf / scales.unsqueeze(1))
    q = torch.clamp(q, -_INT8_QMAX, _INT8_QMAX).to(torch.int8)
    return q, scales


def dequantize_int8(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_int8_per_channel`: ``q * scales``.

    ``q`` is int8 ``[out_features, in_features]``, ``scales`` is
    ``[out_features]``; returns fp32 ``[out_features, in_features]``
    with row ``i`` equal to ``q[i].float() * scales[i]``. This is an
    approximation of the original weights, never exact (except for
    values that happen to land on the grid, and zero rows).
    """
    if q.dim() != 2:
        raise ValueError(
            f"dequantize_int8 expects a 2D [out, in] matrix, "
            f"got shape {tuple(q.shape)}"
        )
    if scales.shape != (q.shape[0],):
        raise ValueError(
            f"scales must be [out_features]=[{q.shape[0]}], "
            f"got shape {tuple(scales.shape)}"
        )
    return q.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)


def quantization_error(
    w: torch.Tensor, w_dequant: torch.Tensor
) -> dict[str, float]:
    """Error metrics between original and dequantized weights.

    Returns ``{"max_abs": ..., "rmse": ...}`` (plus ``mean_abs``):
    ``max_abs`` is the worst single-element ``|w - w_dequant|`` --
    for :func:`quantize_int8_per_channel` output it must be at most
    ``max(scales) / 2``; ``rmse`` is the root-mean-square error over
    all elements, the typical-case companion to the worst case.
    Shapes must match exactly.
    """
    if w.shape != w_dequant.shape:
        raise ValueError(
            f"shape mismatch: {tuple(w.shape)} vs {tuple(w_dequant.shape)}"
        )
    err = (w.detach().to(torch.float32) - w_dequant.detach().to(torch.float32))
    if err.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rmse": 0.0}
    return {
        "max_abs": float(err.abs().max().item()),
        "mean_abs": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(err * err)).item()),
    }


def bytes_for_params(n_params: int, bits: int) -> int:
    """Bytes to store ``n_params`` values at ``bits`` bits each.

    ``ceil(n_params * bits / 8)``: sub-byte widths (int4) pack, so an
    odd int4 count rounds up to a whole byte. ``bits`` must be
    positive; the dtype table (fp32=32, fp16/bf16=16, int8=8, int4=4)
    lives in :data:`BITS_PER_PARAM` -- see :func:`bytes_for_dtype`.
    """
    if n_params < 0:
        raise ValueError(f"n_params must be >= 0, got {n_params}")
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    return (int(n_params) * int(bits) + 7) // 8


def bytes_for_dtype(n_params: int, dtype: str) -> int:
    """Bytes for ``n_params`` in a named dtype (``fp32``/``fp16``/
    ``bf16``/``int8``/``int4``); thin wrapper over
    :func:`bytes_for_params` via :data:`BITS_PER_PARAM`."""
    key = dtype.lower()
    if key not in BITS_PER_PARAM:
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of {sorted(BITS_PER_PARAM)}"
        )
    return bytes_for_params(n_params, BITS_PER_PARAM[key])


def kv_cache_bytes(
    config: ModelConfig,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 2,
) -> int:
    """Bytes of KV cache for ``batch`` sequences of ``seq_len`` tokens.

    Formula (per the layout in ``kv_cache.py``: K and V, each
    ``[B, H_kv, T, D_h]`` per layer)::

        2 (K and V) * n_layer * batch * n_kv_head * seq_len
            * head_dim * bytes_per_elem

    GQA is the lever: ``n_kv_head`` (not ``n_head``) appears, so
    halving KV heads halves the cache. ``bytes_per_elem`` is 2 for
    fp16/bf16, 4 for fp32, 1 for an int8-quantized cache.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    if batch < 0:
        raise ValueError(f"batch must be >= 0, got {batch}")
    if bytes_per_elem <= 0:
        raise ValueError(f"bytes_per_elem must be positive, got {bytes_per_elem}")
    return (
        2
        * config.n_layer
        * int(batch)
        * config.n_kv_head
        * int(seq_len)
        * config.head_dim
        * int(bytes_per_elem)
    )


def worked_example() -> dict[str, int]:
    """The module docstring's 27B-scale EXAMPLE, computed live.

    ASSUMED dims only -- round numbers chosen for the arithmetic,
    **not** any real model's config (real configs come from
    ``config.json`` via ``weights.load_hf_config``). Every value is
    produced by :func:`bytes_for_params` / :func:`kv_cache_bytes`,
    so the docstring figures can be checked against this function.
    """
    assumed_params = 27_000_000_000  # ASSUMED param count
    assumed_config = ModelConfig(  # ASSUMED shape, not a real config
        vocab_size=128_000,  # only used as a plausible placeholder
        d_model=5120,
        n_layer=64,
        n_head=40,
        n_kv_head=8,
        d_ff=13_824,
        max_seq_len=4096,
    )
    return {
        "assumed_params": assumed_params,
        "weights_fp32_bytes": bytes_for_params(assumed_params, 32),
        "weights_fp16_bytes": bytes_for_params(assumed_params, 16),
        "weights_bf16_bytes": bytes_for_params(assumed_params, 16),
        "weights_int8_bytes": bytes_for_params(assumed_params, 8),
        "weights_int4_bytes": bytes_for_params(assumed_params, 4),
        # fp16 KV, batch=1, seq=4096 under the ASSUMED config above.
        "kv_fp16_b1_seq4096_bytes": kv_cache_bytes(
            assumed_config, seq_len=4096, batch=1, bytes_per_elem=2
        ),
        "kv_fp16_per_token_bytes": kv_cache_bytes(
            assumed_config, seq_len=1, batch=1, bytes_per_elem=2
        ),
    }
