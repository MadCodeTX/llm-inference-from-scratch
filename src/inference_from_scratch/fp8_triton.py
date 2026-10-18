"""Triton fused dequant GEMV/GEMM-small-M for resident block-FP8 weights.

The reference path in :mod:`inference_from_scratch.fp8_linear`
materialises a full BF16 dequantised weight per forward (fp32 cast +
multiply + BF16 copy). That traffic dominates decode (see
``benchmarks/profile/SUMMARY.md``: ~243 ms/token/GPU of dequant vs
~25 ms of matmul). This module fuses dequantisation into the matmul:
each program streams FP8 weight tiles, applies the [128, 128] block
scale in-kernel, converts to BF16 (mirroring the reference rounding),
and accumulates in FP32. The FP8 weight is read once at 1 byte/param.

Import-guarded: on the CPU environment Triton is absent, so importing
this module always succeeds; :func:`triton_available` reports False
and callers keep the reference path.

``torch._scaled_mm`` probe (torch 2.11.0+cu128, RTX 4090, SM 8.9)
------------------------------------------------------------------
Scripted probes (see docs/10_quantization.md) show the checkpoint
layout cannot be expressed in ``_scaled_mm`` exactly:

* ``_scaled_mm`` 128x128 BlockWise mode requires float32 scales in a
  near-inner-dim-major, 16-byte-aligned (blocked) layout; handing it
  the checkpoint's plain contiguous BF16 ``[ceil(N/128), ceil(K/128)]``
  grid (or its float32 cast, transposed, or padded variants, at
  128/512-sized shapes) raises ``Invalid scaling configuration``.
* Even with a repacked scale layout, ``_scaled_mm`` needs FP8
  activations with matching block scales. Dynamically quantising
  BF16 activations to FP8 is lossy, so the result would not match
  the BF16-activation reference beyond FP8 activation rounding --
  not "exact within FP8 (weight) rounding".

Hence no ``_scaled_mm`` path is implemented; this Triton kernel,
which keeps BF16 activations, is the fast path.
"""

from __future__ import annotations

import os

import torch

try:  # Import-guarded so the CPU env imports this module fine.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised by CPU-env import
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

__all__ = [
    "triton_available",
    "choose_config",
    "m_bucket",
    "config_mode",
    "set_config_mode",
    "validate_config",
    "CONFIG_MODES",
    "TUNED_CONFIGS",
    "fp8_triton_linear",
]

CONFIG_MODES = ("old", "new")


def m_bucket(m: int) -> int:
    """M bucket for config lookup: 1 / 8 / 16 / 32 / 128 (upper bounds)."""
    m = int(m)
    if m <= 1:
        return 1
    if m <= 8:
        return 8
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    return 128


def _legacy_config(m: int, n: int, k: int) -> dict:
    m = int(m)
    if m <= 32:
        bm = 16 if m <= 16 else 32
        return {"BLOCK_M": bm, "BLOCK_N": 128, "BLOCK_K": 128,
                "num_warps": 4, "num_stages": 3}
    return {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128,
            "num_warps": 4, "num_stages": 3}


# Measured tuned configs keyed by (M bucket, N, K), from
# benchmarks/fp8_tile_sweep.py (RTX 4090, M=1, two runs, see docs/10):
# BLOCK_M=16/BLOCK_N=32/num_warps=4 beat the legacy BLOCK_N=128 config
# by 21-51% on every measured shape and was never worse, so it is also
# the M=1 family config for unlisted (N, K) in choose_config. Other
# buckets fall back to the legacy heuristic.
_M1_TUNED = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128,
             "num_warps": 4, "num_stages": 3}
TUNED_CONFIGS: dict[tuple[int, int, int], dict] = {
    (1, 5120, 17408): dict(_M1_TUNED),
    (1, 17408, 5120): dict(_M1_TUNED),
    (1, 5120, 12288): dict(_M1_TUNED),
    (1, 5120, 10240): dict(_M1_TUNED),
    (1, 8704, 5120): dict(_M1_TUNED),
    (1, 5120, 8704): dict(_M1_TUNED),
}

_CONFIG_MODE = os.environ.get("INFERENCE_FP8_TILE_CONFIG", "new").strip().lower()
if _CONFIG_MODE not in CONFIG_MODES:
    _CONFIG_MODE = "new"


def config_mode() -> str:
    return _CONFIG_MODE


def set_config_mode(mode: str) -> str:
    """Select 'old' (legacy heuristic) or 'new' (tuned map) configs.

    Fixed selection only -- no runtime autotune, so CUDA-graph capture
    sees one deterministic config per shape.
    """
    global _CONFIG_MODE
    if mode not in CONFIG_MODES:
        raise ValueError(f"unknown fp8 tile config mode {mode!r}, expected {CONFIG_MODES}")
    _CONFIG_MODE = str(mode)
    return _CONFIG_MODE


def validate_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"):
        if key not in cfg:
            raise ValueError(f"fp8 config missing {key}: {cfg}")
        cfg[key] = int(cfg[key])
    if cfg["BLOCK_K"] != 128:
        raise ValueError("BLOCK_K must be 128 (one block-scale column)")
    if cfg["BLOCK_N"] not in (32, 64, 128):
        raise ValueError("BLOCK_N must divide the 128 block (32/64/128)")
    if cfg["BLOCK_M"] < 16 or (cfg["BLOCK_M"] & (cfg["BLOCK_M"] - 1)) != 0:
        raise ValueError("BLOCK_M must be a power of two >= 16")
    if cfg["num_warps"] not in (4, 8):
        raise ValueError("num_warps must be 4 or 8 for measured configs")
    return cfg


def triton_available() -> bool:
    """True only when Triton imported and CUDA is present."""
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


def choose_config(m: int, n: int, k: int, mode: str | None = None) -> dict:
    """Fixed (non-autotuned) kernel config by (M bucket, N, K).

    ``BLOCK_K`` is always 128 so one K-tile is exactly one block-scale
    column; ``BLOCK_N`` divides 128 so one N-tile lies in one scale
    row. Mode 'old' is the legacy heuristic; mode 'new' (default)
    consults TUNED_CONFIGS keyed by (m_bucket, N, K) and falls back
    to the legacy heuristic for unmeasured shapes.
    """
    m, n, k = int(m), int(n), int(k)
    mode = config_mode() if mode is None else str(mode)
    if mode not in CONFIG_MODES:
        raise ValueError(f"unknown fp8 tile config mode {mode!r}, expected {CONFIG_MODES}")
    if mode == "new":
        tuned = TUNED_CONFIGS.get((m_bucket(m), n, k))
        if tuned is not None:
            return validate_config(tuned)
        if m_bucket(m) == 1:
            # M=1 family: measured winner on all six swept shapes.
            return dict(_M1_TUNED)
    return _legacy_config(m, n, k)


if _TRITON_AVAILABLE:

    @triton.jit
    def _fp8_dequant_gemm_kernel(
        x_ptr, w_ptr, s_ptr, b_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sk,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        # One scale row for this whole N-tile (BLOCK_N divides 128).
        scale_row = (pid_n * BLOCK_N) // 128
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_off = k_idx * BLOCK_K + offs_k
            mask_k = k_off < K
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            )
            w_tile = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + k_off[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0,
            )
            # BLOCK_K == 128 == block cols, so k_idx is the scale column.
            scale = tl.load(s_ptr + scale_row * stride_sn + k_idx * stride_sk)
            w_deq = (w_tile.to(tl.float32) * scale).to(tl.bfloat16)
            acc += tl.dot(x_tile, tl.trans(w_deq))
        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
            acc = acc + bias[None, :].to(tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def fp8_triton_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    block: tuple[int, int] = (128, 128),
    config: dict | None = None,
    config_mode: str | None = None,
) -> torch.Tensor:
    """Fused FP8 dequant matmul: ``x @ dequant(weight).T (+ bias)``.

    ``x`` is ``[M, K]`` BF16/FP32 (flattened by the caller if needed),
    ``weight`` is ``float8_e4m3fn`` ``[N, K]``, ``scale_inv`` is
    ``[ceil(N/128), ceil(K/128)]``. Returns ``[M, N]`` BF16.
    Supports any M (tiled; M=1..8 uses a padded BLOCK_M=16 tile) and
    K/N non-multiples of the tile via masks.
    """
    if not triton_available():
        raise RuntimeError("Triton/CUDA unavailable for fp8_triton_linear")
    if tuple(block) != (128, 128):
        raise ValueError(f"fp8_triton_linear supports block (128, 128), got {block}")
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must be 2D")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    m, k = x.shape
    n, kw = weight.shape
    if kw != k:
        raise ValueError(f"x K={k} != weight K={kw}")
    expected_scale = (-(-n // 128), -(-k // 128))
    if tuple(scale_inv.shape) != expected_scale:
        raise ValueError(
            f"scale_inv shape {tuple(scale_inv.shape)} != expected {expected_scale}"
        )
    x2 = x.to(torch.bfloat16).contiguous()
    w2 = weight.contiguous()
    s2 = scale_inv.to(torch.float32).contiguous()
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    cfg = validate_config(config) if config is not None else choose_config(m, n, k, mode=config_mode)
    grid = (
        triton.cdiv(m, cfg["BLOCK_M"]),
        triton.cdiv(n, cfg["BLOCK_N"]),
    )
    # Triton launches on the current CUDA device; the pipeline bridge
    # keeps half its weights on cuda:1, so launch on x's device.
    with torch.cuda.device(x.device):
        _fp8_dequant_gemm_kernel[grid](
            x2, w2, s2,
            bias if bias is not None else x2,
            out,
            m, n, k,
            x2.stride(0), x2.stride(1),
            w2.stride(0), w2.stride(1),
            s2.stride(0), s2.stride(1),
            out.stride(0), out.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )
    return out
