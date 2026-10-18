"""TP CUDA-graph check: CPU parser tests + skip-gated GPU equality proof.

GPU test runs scripts/tp2_graph_check.py under 2-process NCCL torchrun
(measured <60s on this machine); skipped without CUDA.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "tp2_graph_check.py"
_spec = importlib.util.spec_from_file_location("tp2_graph_check", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_graph_check_parser_defaults():
    args = _mod.build_parser().parse_args([])
    assert args.mode == "model" and args.tokens == 16
    assert args.max_seq_len == 32 and args.lm_head == "replicated"
    assert args.tokens <= args.max_seq_len
    args = _mod.build_parser().parse_args(["--mode", "collective", "--lm-head", "vocab"])
    assert args.mode == "collective" and args.lm_head == "vocab"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TP graph capture")
def test_tp2_graph_check_nccl_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(_SCRIPT), "--tokens", "8", "--repeats", "3"],
        capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_GRAPH_CHECK PASS" in proc.stdout
    assert "capture OK" in proc.stdout
    assert "tokens_equal=True" in proc.stdout
    assert "cleanup: done" in proc.stdout
