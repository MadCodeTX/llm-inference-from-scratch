"""Skip-gated CUDA proof for BlockFP8Linear (real proof: GPU script)."""
import pytest
import torch

from inference_from_scratch.fp8_linear import BlockFP8Linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def test_block_fp8_linear_cuda_matches_cpu_reference():
    torch.manual_seed(0)
    dense = torch.nn.Linear(256, 256, bias=True)
    cpu = BlockFP8Linear.from_linear(dense)
    gpu = BlockFP8Linear.from_linear(dense).to("cuda:0")
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = cpu(x)
        got = gpu(x.to("cuda:0")).cpu()
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max().item() < 0.05
