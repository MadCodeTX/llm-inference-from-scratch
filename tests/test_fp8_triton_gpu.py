"""Skip-gated GPU tests for the Triton fused FP8 kernel."""
import pytest
import torch
import torch.nn.functional as F

from inference_from_scratch.fp8_linear import BlockFP8Linear
from inference_from_scratch.fp8_triton import fp8_triton_linear, triton_available
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8

pytestmark = pytest.mark.skipif(
    not triton_available(), reason="Triton/CUDA not available"
)


@pytest.mark.parametrize("m,n,k", [(1, 512, 512), (4, 1024, 5120),
                                   (8, 200, 300), (16, 512, 512),
                                   (128, 512, 512), (1, 17408, 5120)])
def test_triton_matches_dequant_reference(m, n, k):
    torch.manual_seed(0)
    w = torch.randn(n, k) * 0.05
    wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
    wq, ws = wq.cuda(), ws.cuda().to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    ref = F.linear(x, dequantize_block_fp8(wq, ws, out_dtype=torch.bfloat16))
    got = fp8_triton_linear(x, wq, ws)
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    diff = (got.float() - ref.float()).abs().max().item()
    rel = diff / ref.float().abs().max().item()
    assert rel < 0.02, (m, n, k, diff, rel)


def test_triton_on_nondefault_device():
    """Pipeline bridge keeps half the model on cuda:1 (launch-device)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("need 2 CUDA devices")
    torch.manual_seed(0)
    w = torch.randn(512, 512) * 0.05
    wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
    wq, ws = wq.to("cuda:1"), ws.to("cuda:1").to(torch.bfloat16)
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda:1")
    got = fp8_triton_linear(x, wq, ws)
    ref = F.linear(x, dequantize_block_fp8(wq, ws, out_dtype=torch.bfloat16))
    assert got.device.type == "cuda" and got.device.index == 1
    assert (got.float() - ref.float()).abs().max().item() < 0.1


@pytest.mark.parametrize("m", [64, 128, 512, 1024])
def test_triton_prefill_configs_match_reference(m):
    """New prefill buckets (choose_config M>=64) vs dequant reference."""
    from inference_from_scratch.fp8_triton import choose_config
    torch.manual_seed(1)
    n, k = 2560, 8704
    w = torch.randn(n, k) * 0.05
    wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
    wq, ws = wq.cuda(), ws.cuda().to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    ref = F.linear(x, dequantize_block_fp8(wq, ws, out_dtype=torch.bfloat16))
    got = fp8_triton_linear(x, wq, ws)  # default choose_config path
    assert choose_config(m, n, k)["BLOCK_M"] == 64
    rel = (got.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 0.02, (m, rel)


def test_block_fp8_linear_triton_dispatch_cuda():
    torch.manual_seed(0)
    dense = torch.nn.Linear(512, 512, bias=True)
    mod = BlockFP8Linear.from_linear(dense, use_triton=True).to("cuda")
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        got = mod(x)
        ref = F.linear(
            x, dequantize_block_fp8(
                mod.weight, mod.weight_scale_inv, out_dtype=torch.bfloat16
            ), mod.bias.to(torch.bfloat16),
        )
    assert (got.float() - ref.float()).abs().max().item() < 0.1
    assert mod._triton_fallback_reason is None
