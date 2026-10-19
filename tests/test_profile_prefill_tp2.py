"""CPU tests for profile_prefill_tp2 helpers (no CUDA needed)."""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_prefill_tp2.py"
_spec = importlib.util.spec_from_file_location("profile_prefill_tp2", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_is_device_kernel_excludes_runtime_events():
    # These inflated chunk512/1024 CUDA totals past wall time.
    assert not _mod.is_device_kernel("cudaLaunchKernel")
    assert not _mod.is_device_kernel("Command Buffer Full")
    assert not _mod.is_device_kernel("cudaMemcpyAsync")


def test_is_device_kernel_keeps_real_work():
    assert _mod.is_device_kernel("_fp8_dequant_gemm_kernel")
    assert _mod.is_device_kernel(
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)")
    assert _mod.is_device_kernel("Memcpy DtoD (Device -> Device)")
    assert _mod._cat("_fp8_dequant_gemm_kernel") == "triton_fp8"
