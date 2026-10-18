"""Tests for benchmarks/microbench_roofline.py plan-only guard.

No GPU is touched: without --i-understand-gpu-window every
subcommand must print a JSON plan line and allocate nothing.
"""

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "microbench_roofline", _REPO / "benchmarks" / "microbench_roofline.py"
)
micro = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(micro)


def _run(capsys, argv):
    rc = micro.main(argv)
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    return rc, lines


def test_bandwidth_plan_without_flag(capsys):
    rc, lines = _run(capsys, ["bandwidth"])
    assert rc == 0
    assert lines[0]["event"] == "plan"
    assert lines[0]["subcommand"] == "bandwidth"
    assert lines[0]["will_touch_gpu"] is False
    assert lines[0]["sizes_mib"] == micro.DEFAULT_BANDWIDTH_MIB


def test_gemm_plan_lists_27b_shapes(capsys):
    rc, lines = _run(capsys, ["gemm"])
    assert rc == 0
    plan = lines[0]
    assert plan["will_touch_gpu"] is False
    shapes = {(s["k"], s["n"]) for s in plan["shapes"]}
    assert (5120, 17408) in shapes
    assert (5120, 248320) in shapes


def test_allreduce_plan_documents_torchrun(capsys):
    rc, lines = _run(capsys, ["allreduce"])
    assert rc == 0
    plan = lines[0]
    assert plan["backend"] == "nccl"
    assert "torchrun --nproc_per_node=2" in plan["launch"]
    assert plan["will_touch_gpu"] is False
