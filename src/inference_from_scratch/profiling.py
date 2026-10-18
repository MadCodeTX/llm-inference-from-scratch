"""Aten-op -> cost-category rollup for bridge profiling (off by default).

Used only by ``scripts/profile_qwen38_bridge.py``; never imported by the
generation path. Categories are honest about overlap: ``mlp`` module time
is a subset of ``matmul_linear`` time (MLP projections are linears), so
rollups report module totals as memo rows, not additive shares.
"""
from __future__ import annotations

CATEGORIES = ("dequant", "matmul_linear", "attention_core", "elementwise_norm", "device_transfer", "host_sync_other")

_RULES = (
    ("dequant", ("to", "mul", "reshape", "pad", "float", "copy_")),  # refined below by name match order
)


def categorize_op(op_name: str) -> str:
    """Map one aten/profiler op name to a cost category."""
    n = op_name.lower()
    if any(k in n for k in ("dequantize_block_fp8", "dequant")):
        return "dequant"
    if any(k in n for k in ("aten::mm", "aten::addmm", "aten::bmm", "aten::linear", "aten::matmul", "scaled_mm", "cublas", "cutlass")):
        return "matmul_linear"
    if any(k in n for k in ("scaled_dot_product", "aten::softmax", "_softmax", "flash", "mem_efficient_attention")):
        return "attention_core"
    if any(k in n for k in ("aten::to", "aten::_to_copy", "aten::copy_", "cudaMemcpy", "Memcpy")):
        return "device_transfer"
    if any(k in n for k in ("rms_norm", "layer_norm", "aten::rsqrt", "aten::pow", "aten::mean")):
        return "elementwise_norm"
    return "host_sync_other"


def rollup(events: list[dict]) -> dict:
    """Sum ``self_cuda_time_us`` per category; returns totals + shares."""
    totals: dict[str, float] = {c: 0.0 for c in CATEGORIES}
    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    for e in events:
        cat = categorize_op(str(e.get("name", "")))
        totals[cat] += float(e.get("self_cuda_time_us", 0.0) or 0.0)
        counts[cat] += 1
    total = sum(totals.values())
    return {"total_cuda_us": total,
            "by_category_us": totals,
            "by_category_count": counts,
            "share": {c: (totals[c] / total if total > 0 else 0.0) for c in CATEGORIES}}


def roofline_ms(bytes_moved: float, gbps: float = 920.0) -> float:
    """Minimum ms to move ``bytes_moved`` at ``gbps`` GB/s."""
    if gbps <= 0:
        raise ValueError("gbps must be positive")
    return bytes_moved / (gbps * 1e9) * 1e3
