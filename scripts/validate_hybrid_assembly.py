"""Meta-device validation of the real Qwen3.8 checkpoint (CPU, headers only).

Constructs the real-config HybridLM on the meta device (no allocation),
validates the snapshot manifest against its state_dict shapes, and prints
the result + parameter count computed from shapes.  No tensor data is
loaded; no GPU is touched.  Usage: .venv/bin/python scripts/validate_hybrid_assembly.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, HybridLM  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import (  # noqa: E402
    build_manifest, find_default_snapshot, validate_state_against_model,
)


def main() -> int:
    snap = find_default_snapshot()
    if snap is None:
        print("real snapshot not found in HF cache; skipping")
        return 0
    cfg = HybridConfig.from_text_config(snap)
    man = build_manifest(snap)
    with torch.device("meta"):
        model = HybridLM(cfg)
    res = validate_state_against_model(model, snap)
    print(f"snapshot: {snap}")
    print(f"manifest entries: {len(man.entries)}")
    print(f"model state_dict tensors: {res['counts']['expected']}")
    print(f"parameter_count: {res['parameter_count']}")
    print(f"missing: {res['missing']}")
    print(f"unexpected: {res['unexpected']}")
    print(f"shape_mismatch: {res['shape_mismatch']}")
    print(f"dtype_mismatch: {res['dtype_mismatch']}")
    print(f"counts: {res['counts']}")
    print(f"ok: {res['ok']}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
