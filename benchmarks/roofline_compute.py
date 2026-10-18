#!/usr/bin/env python3
"""Derive roofline ridge points and batch-1 decode ceilings.

Reads the raw JSONL written by ``benchmarks/microbench_roofline.py``
(under ``benchmarks/roofline/``) plus the checkpoint header manifest,
and applies the helpers in
``src/inference_from_scratch/bottlenecks.py`` (``ridge_point``,
``arithmetic_intensity``, ``roofline_time_lower_bound``).

Byte model (stated, not hidden)
--------------------------------
FP8-resident serving layout, language model + MTP only (vision tower
deferred, as elsewhere in this repo):

* FP8 E4M3 weight payload from the manifest headers:
  24,699,207,680 bytes over 407 tensors (1 byte/param). Under TP2
  these shard evenly: 12,349,603,840 bytes per rank.
* BF16 language tensors from the manifest: 5,242,980,352 bytes
  (embedding, untied lm_head, norms, conv/A_log/dt_bias, mtp.fc).
  In the TP2 layout of ``tensor_parallel.parameter_bytes_by_rank``
  most of this is *replicated* on every rank (embed + lm_head alone
  are 2 x 248320 x 5120 x 2 = 5,084,569,600 bytes); a small sharded
  remainder (in_proj_a/b) splits by value heads.
* ``weight_scale_inv`` companions (BF16, one per [128,128] block):
  3,015,040 bytes total.
* Per-rank decode step therefore re-reads 17,570,642,112 bytes of
  weights (formula per-rank total, cross-checked against the
  manifest). A hypothetical single-GPU resident layout would read
  29,945,203,072 bytes -- more than a 24 GiB card holds, which is
  why the incumbent serves TP2.
* KV read per decode token (16 full-attention layers only, FP8 KV
  as served): ``2 * 16 * 4 * 256 * S`` bytes at context ``S``;
  the 48 linear-attention layers keep fixed-size recurrent state
  and add no S-growing read. Activations are ignored, matching
  ``bottlenecks.py``'s stated lower bound.

FLOPs per decode token use the ``2 * params`` rule on the weights
that actually matmul per token (all FP8 weight elements plus the
BF16 lm_head; the embedding lookup is a gather and excluded), plus
the full-attention score/value term ``4 * 16 * 5120 * S``.

Every number below comes from a measurement file or a header
manifest read in this run; nothing is a spec-sheet figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from inference_from_scratch.bottlenecks import (  # noqa: E402
    HardwareSpec,
    arithmetic_intensity,
    ridge_point,
    roofline_time_lower_bound,
)
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tensor_parallel import (  # noqa: E402
    parameter_bytes_by_rank,
)

_ROOFLINE_DIR = _REPO / "benchmarks" / "roofline"
_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots"
    / "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def measured_peaks(roofline_dir: Path = _ROOFLINE_DIR) -> dict:
    """Peak achieved bandwidth / TFLOPs from the raw JSONL files."""
    bw_rows = []
    for name in ("bandwidth_gpu0.jsonl", "bandwidth_gpu1.jsonl"):
        bw_rows += [
            r
            for r in _read_jsonl(roofline_dir / name)
            if r.get("event") == "result"
        ]
    gemm_rows = [
        r
        for r in _read_jsonl(roofline_dir / "gemm_gpu0.jsonl")
        if r.get("event") == "result"
    ]
    ar_rows = [
        r
        for r in _read_jsonl(roofline_dir / "allreduce.jsonl")
        if r.get("event") == "result"
    ]
    # Latency/bandwidth split for all-reduce: least-squares fit of
    # seconds = latency + bytes / bus_bytes_s over the sweep.
    xs = [float(r["bytes"]) for r in ar_rows]
    ys = [float(r["seconds"]) for r in ar_rows]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return {
        "bandwidth_effective_gbytes_s": max(
            r["effective_gbytes_per_s"] for r in bw_rows
        ),
        "bandwidth_rows": bw_rows,
        "gemm_bf16_tflops": max(
            r["tflops"] for r in gemm_rows if r["dtype"] == "bf16"
        ),
        "gemm_fp8_tflops": max(
            r["tflops"]
            for r in gemm_rows
            if r["dtype"] == "fp8" and r.get("available")
        ),
        "gemm_rows": gemm_rows,
        "allreduce_latency_s": my - slope * mx,
        "allreduce_bus_gbytes_s": (1.0 / slope) / 1e9 if slope > 0 else None,
        "allreduce_rows": ar_rows,
    }


def byte_model(snapshot: Path = _SNAPSHOT) -> dict:
    """Weight bytes: manifest totals + TP2 per-rank split."""
    man = build_manifest(snapshot)
    by_kind_bytes: dict[str, int] = {}
    lang_bf16 = 0
    for e in man.entries:
        by_kind_bytes[e.kind] = by_kind_bytes.get(e.kind, 0) + int(e.bytes)
        if e.kind == "bf16" and e.target_name is not None:
            lang_bf16 += int(e.bytes)
    per_rank = parameter_bytes_by_rank(world=2)[0]
    return {
        "fp8_weight_bytes_total": by_kind_bytes["fp8_weight"],
        "bf16_language_bytes": lang_bf16,
        "scale_inv_bytes_total": by_kind_bytes["scale_inv"],
        "single_device_bytes": (
            by_kind_bytes["fp8_weight"]
            + lang_bf16
            + by_kind_bytes["scale_inv"]
        ),
        "per_rank_bytes": int(per_rank["total_bytes"]),
        "per_rank_detail": {
            k: int(v) for k, v in per_rank.items() if k.endswith("_bytes")
        },
    }


def decode_flops_per_token(fp8_params: int, seq_len: int) -> float:
    """2*params on per-token matmul weights + full-attn term."""
    lm_head_params = 248320 * 5120
    attn = 4.0 * 16 * 5120 * int(seq_len)
    return 2.0 * (int(fp8_params) + lm_head_params) + attn


def kv_read_bytes(seq_len: int) -> int:
    """FP8 KV read per decode token: 16 full layers only."""
    return 2 * 16 * 4 * 256 * int(seq_len) * 1


def ceiling(bandwidth_bytes_s: float, bytes_per_step: float) -> dict:
    """Batch-1 decode ceiling for one weight-resident layout."""
    secs = float(bytes_per_step) / float(bandwidth_bytes_s)
    return {"seconds_per_token": secs, "tokens_per_s": 1.0 / secs}


def compute(
    roofline_dir: Path = _ROOFLINE_DIR,
    snapshot: Path = _SNAPSHOT,
    seq_len: int = 2048,
) -> dict:
    peaks = measured_peaks(roofline_dir)
    model_bytes = byte_model(snapshot)
    bw = peaks["bandwidth_effective_gbytes_s"] * 1e9
    specs = {
        "bf16": HardwareSpec(
            name="RTX 4090 measured (bf16 peak / copy_ bandwidth)",
            peak_flops=peaks["gemm_bf16_tflops"] * 1e12,
            memory_bandwidth_bytes_s=bw,
            memory_bytes=24564 * 1024 * 1024,
        ),
        "fp8": HardwareSpec(
            name="RTX 4090 measured (fp8 scaled_mm peak / copy_ bandwidth)",
            peak_flops=peaks["gemm_fp8_tflops"] * 1e12,
            memory_bandwidth_bytes_s=bw,
            memory_bytes=24564 * 1024 * 1024,
        ),
    }
    flops = decode_flops_per_token(
        model_bytes["fp8_weight_bytes_total"], seq_len
    )
    kv = kv_read_bytes(seq_len)
    layouts = {}
    for label, weight_bytes in (
        ("tp2_per_rank", model_bytes["per_rank_bytes"]),
        ("single_device_hypothetical", model_bytes["single_device_bytes"]),
    ):
        moved = float(weight_bytes + kv)
        entry = ceiling(bw, moved)
        entry.update(
            {
                "weight_bytes": int(weight_bytes),
                "kv_read_bytes": int(kv),
                "bytes_per_token": moved,
                "intensity": arithmetic_intensity(flops, moved),
                "bound_bf16": (
                    "compute"
                    if arithmetic_intensity(flops, moved)
                    >= ridge_point(specs["bf16"])
                    else "memory"
                ),
                "roofline_lower_bound_s": roofline_time_lower_bound(
                    flops, moved, specs["bf16"]
                ),
            }
        )
        layouts[label] = entry
    return {
        "seq_len": int(seq_len),
        "peaks": {
            k: v
            for k, v in peaks.items()
            if not k.endswith("_rows")
        },
        "ridge_bf16_ops_per_byte": ridge_point(specs["bf16"]),
        "ridge_fp8_ops_per_byte": ridge_point(specs["fp8"]),
        "flops_per_token": flops,
        "byte_model": model_bytes,
        "layouts": layouts,
        "allreduce_per_token_note": {
            "collectives_per_token": 128,
            "payload_bytes_each_bf16": 2 * 5120 * 2,
            "fitted_latency_s_each": peaks["allreduce_latency_s"],
            "fitted_latency_total_s": 128 * peaks["allreduce_latency_s"],
        },
    }


def main() -> int:
    result = compute()
    out = _ROOFLINE_DIR / "roofline_derived.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
