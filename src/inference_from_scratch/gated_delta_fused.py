"""Triton fused gated-delta decode recurrence (one kernel per layer).

The reference decode step in
:meth:`DistributedGatedDeltaNet.step_inplace_static` evaluates the
per-head recurrence with ~8 separate elementwise/reduction ops over
``S [B, Hv, Dk, Dv]`` (decay ``mul_``, ``kv_mem`` reduction, ``delta``,
rank-1 ``add_``, read reduction), each launching its own tiny kernel
(24 heads x [128, 128] per rank at TP2). This module fuses exactly
that recurrence into ONE Triton kernel per layer:

    S <- exp(g) * S
    delta = (v - S^T k) * beta
    S <- S + k (x) delta
    core = S^T q

One program per (batch, value head): it loads ``q``/``k`` once and
streams ``S`` in ``BLOCK_V`` column tiles, computing ``kv_mem``,
``delta``, the updated ``S`` tile (stored back in place) and ``core``
per tile. Fixed config (no autotune) so CUDA-graph capture sees one
deterministic launch.

Import-guarded like :mod:`inference_from_scratch.fp8_triton`: on CPU
(no Triton) :func:`fused_available` is False and the dispatch falls
back to the reference ops. The fused path is opt-in (flag-gated at
the call site) and requires fp32 CUDA tensors; anything else falls
back or raises on shape/dtype misuse, matching the reference maths
to fp32 reduction-order tolerance (not bit-exact: the tile reduction
order differs from ``Tensor.sum``).
"""

from __future__ import annotations

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
    "DELTA_MODES",
    "fused_available",
    "validate_delta_mode",
    "validate_recurrence_shapes",
    "reference_gated_delta_inplace",
    "fused_gated_delta_inplace",
    "gated_delta_inplace",
]

DELTA_MODES = ("ref", "fused")
BLOCK_V = 32


def fused_available() -> bool:
    """True only when Triton imported and CUDA is present."""
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


def validate_delta_mode(mode: str) -> str:
    if mode not in DELTA_MODES:
        raise ValueError(f"unknown delta mode {mode!r}, expected {DELTA_MODES}")
    return str(mode)


def validate_recurrence_shapes(q, k, v, beta, g, S) -> tuple[int, int, int]:
    """Validate recurrence inputs; returns ``(BH, Dk, Dv)``.

    ``q``/``k``: ``[BH, Dk]``; ``v``: ``[BH, Dv]``; ``beta``/``g``:
    ``[BH]``; ``S``: ``[BH, Dk, Dv]``. All tensors on one device.
    """
    for name, t, dims in (("q", q, 2), ("k", k, 2), ("v", v, 2),
                          ("beta", beta, 1), ("g", g, 1), ("S", S, 3)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a Tensor, got {type(t).__name__}")
        if t.dim() != dims:
            raise ValueError(f"{name} must be {dims}D, got shape {tuple(t.shape)}")
    bh, dk = q.shape
    if k.shape != (bh, dk):
        raise ValueError(f"k shape {tuple(k.shape)} != q shape {tuple(q.shape)}")
    dv = v.shape[1]
    if v.shape[0] != bh or beta.shape != (bh,) or g.shape != (bh,):
        raise ValueError(
            f"v/beta/g batch mismatch: v {tuple(v.shape)}, "
            f"beta {tuple(beta.shape)}, g {tuple(g.shape)} vs BH={bh}")
    if tuple(S.shape) != (bh, dk, dv):
        raise ValueError(f"S shape {tuple(S.shape)} != {(bh, dk, dv)}")
    if not (q.device == k.device == v.device == beta.device
            == g.device == S.device):
        raise ValueError("q/k/v/beta/g/S must share one device")
    return int(bh), int(dk), int(dv)


def reference_gated_delta_inplace(q, k, v, beta, g, S) -> torch.Tensor:
    """Reference recurrence (the ``step_inplace_static`` ops), in place on ``S``.

    ``q`` already scaled, ``q``/``k`` already L2-normalised by the
    caller. Returns ``core [BH, Dv]`` fp32. ``S`` is mutated exactly
    as the multi-op path mutates it (decay, then rank-1 write).
    """
    validate_recurrence_shapes(q, k, v, beta, g, S)
    S.mul_(torch.exp(g.float())[:, None, None].to(dtype=S.dtype))
    Sf = S.float()
    kv_mem = (Sf * k.float().unsqueeze(-1)).sum(dim=-2)
    delta = (v.float() - kv_mem) * beta.float().unsqueeze(-1)
    S.add_((k.float().unsqueeze(-1) * delta.unsqueeze(-2)).to(dtype=S.dtype))
    return (S.float() * q.float().unsqueeze(-1)).sum(dim=-2)


if _TRITON_AVAILABLE:

    @triton.jit
    def _gated_delta_kernel(
        q_ptr, k_ptr, v_ptr, beta_ptr, g_ptr, S_ptr, core_ptr,
        DK: tl.constexpr, DV: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        head = tl.program_id(0)
        offs_k = tl.arange(0, DK)
        q = tl.load(q_ptr + head * DK + offs_k).to(tl.float32)
        k = tl.load(k_ptr + head * DK + offs_k).to(tl.float32)
        beta = tl.load(beta_ptr + head).to(tl.float32)
        decay = tl.exp(tl.load(g_ptr + head).to(tl.float32))
        for v0 in range(0, DV, BLOCK_V):
            offs_v = v0 + tl.arange(0, BLOCK_V)
            vmask = offs_v < DV
            s = tl.load(S_ptr + head * DK * DV + offs_k[:, None] * DV
                        + offs_v[None, :],
                        mask=vmask[None, :], other=0.0).to(tl.float32)
            s = s * decay
            kv = tl.sum(s * k[:, None], axis=0)
            delta = (tl.load(v_ptr + head * DV + offs_v,
                             mask=vmask, other=0.0).to(tl.float32)
                     - kv) * beta
            s = s + k[:, None] * delta[None, :]
            tl.store(S_ptr + head * DK * DV + offs_k[:, None] * DV
                     + offs_v[None, :], s, mask=vmask[None, :])
            core = tl.sum(s * q[:, None], axis=0)
            tl.store(core_ptr + head * DV + offs_v, core, mask=vmask)


def fused_gated_delta_inplace(q, k, v, beta, g, S) -> torch.Tensor:
    """Fused Triton recurrence; mutates ``S`` in place, returns core.

    Requires fp32 CUDA tensors with power-of-two ``Dk``/``Dv`` <= 256
    (real shapes: 128/128). Raises on anything else -- callers that
    want a silent fallback use :func:`gated_delta_inplace`.
    """
    bh, dk, dv = validate_recurrence_shapes(q, k, v, beta, g, S)
    if not fused_available():
        raise RuntimeError("Triton/CUDA unavailable for fused gated-delta")
    for name, t in (("q", q), ("k", k), ("v", v), ("beta", beta),
                    ("g", g), ("S", S)):
        if not t.is_cuda:
            raise ValueError(f"{name} must be CUDA for the fused path")
        if t.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 for the fused path, got {t.dtype}")
    if dk > 256 or dv > 256 or (dk & (dk - 1)) or (dv & (dv - 1)):
        raise ValueError(f"Dk/Dv must be powers of two <= 256, got {dk}/{dv}")
    q2, k2, v2 = q.contiguous(), k.contiguous(), v.contiguous()
    beta2, g2, S2 = beta.contiguous(), g.contiguous(), S.contiguous()
    if S2.data_ptr() != S.data_ptr():
        raise ValueError("S must already be contiguous (in-place state)")
    core = torch.empty((bh, dv), device=S.device, dtype=torch.float32)
    with torch.cuda.device(S.device):
        _gated_delta_kernel[(bh,)](
            q2, k2, v2, beta2, g2, S2, core,
            DK=dk, DV=dv, BLOCK_V=BLOCK_V, num_warps=4, num_stages=2)
    return core


def gated_delta_inplace(q, k, v, beta, g, S, *, mode: str = "ref") -> torch.Tensor:
    """Dispatch: ``fused`` when available and eligible, else reference.

    The fused kernel is used only for fp32 CUDA inputs; every other
    case (CPU, non-fp32, unavailable Triton) falls back to
    :func:`reference_gated_delta_inplace`, so enabling the flag can
    never break a non-CUDA run.
    """
    validate_delta_mode(mode)
    validate_recurrence_shapes(q, k, v, beta, g, S)
    if (mode == "fused" and fused_available() and S.is_cuda
            and all(t.dtype == torch.float32
                    for t in (q, k, v, beta, g, S))):
        return fused_gated_delta_inplace(q, k, v, beta, g, S)
    return reference_gated_delta_inplace(q, k, v, beta, g, S)
