"""Env-gated real-layer parity vs transformers (CPU, single layers).

Skipped by default: set RUN_REAL_LAYER_PARITY=1 to run. A skip is NOT
green evidence -- the evidence is the script run itself
(``scripts/parity_real_layer.py``), whose measured numbers are recorded
in docs/22. Each test loads only one real layer (layer 0 = linear,
layer 3 = full attention) from the local Qwen3.8 snapshot, dequantized
FP8->BF16, into both HF ``Qwen3_5DecoderLayer`` and our ``HybridBlock``.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "parity_real_layer.py"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_LAYER_PARITY") != "1",
    reason="real-layer parity is env-gated (RUN_REAL_LAYER_PARITY=1); "
           "skipped by default and not counted as green evidence")


def _run(layer, *extra):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--layer", str(layer),
         "--tokens", "4", "--seed", "0", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "missing_hf=[] missing_ours=[] extra=[]" in out
    m = re.search(r"whole_block: .* rel_rms=([0-9.e+-]+)", out)
    assert m, out
    return float(m.group(1)), out


def test_real_layer0_linear_parity_bf16():
    rel_rms, _ = _run(0)
    # bf16 dtype noise (chunkwise HF vs sequential ours): ~1.5e-3 measured.
    assert rel_rms < 0.02


def test_real_layer3_full_parity_bf16():
    rel_rms, _ = _run(3)
    assert rel_rms < 0.02


def test_real_layer0_linear_parity_fp32_control():
    # Same weights/inputs in fp32: dtype noise collapses; a maths error
    # would stay O(1). Measured rel_rms ~1.2e-7.
    rel_rms, _ = _run(0, "--fp32")
    assert rel_rms < 1e-5
