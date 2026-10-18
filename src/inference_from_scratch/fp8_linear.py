"""Resident block-FP8 linear: FP8 weights stay FP8 in memory (ch 10/22).

A 27B model in BF16 needs ~54 GB of weights -- more than one 24 GiB
card holds. The Qwen3.8 checkpoint already stores its large
projections as block FP8 (``float8_e4m3fn`` plus one BF16
``weight_scale_inv`` per [128, 128] block), so keeping those tensors
resident as FP8 instead of dequantising them at load time roughly
halves the weight bytes again versus BF16. :class:`BlockFP8Linear` is
that resident module: it *stores* the FP8 bits and scales, and its
reference ``forward`` dequantises with
:func:`inference_from_scratch.quantization.dequantize_block_fp8` to
the input dtype and applies :func:`torch.nn.functional.linear`. The
dequantised matrix is materialised per forward -- simple and exact
against the reference, not fast; a fused kernel is future work.

Scaled-mm fast path (probed, not usable as-is)
----------------------------------------------
``torch._scaled_mm`` on this box (torch 2.11.0+cu128, RTX 4090,
SM 8.9) accepts TensorWise, RowWise, BlockWise 1x128, BlockWise
128x128, and newer MX/NV formats -- but its 128x128 blockwise mode
requires *float* scales in a near-inner-dim-major, 16-byte-aligned
(blocked) layout, plus FP8-quantised activations carrying matching
block scales. The checkpoint's ``weight_scale_inv`` is a plain
contiguous BF16 ``[ceil(R/128), ceil(C/128)]`` grid, and this engine
runs BF16 activations, so the checkpoint scales cannot be handed to
``_scaled_mm`` directly (probe: ``Invalid scaling configuration``).
The module therefore exposes an opt-in ``use_fast_path`` hook that
attempts ``_scaled_mm`` behind a capability check and falls back
silently to the reference on any mismatch; the reference is the only
path exercised by the tests and the GPU proof.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .quantization import dequantize_block_fp8, quantize_block_fp8

__all__ = [
    "BlockFP8Linear",
    "replace_linears_with_fp8",
    "scaled_mm_block_supported",
]


def scaled_mm_block_supported(
    scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> tuple[bool, str]:
    """Whether ``_scaled_mm`` can consume these block scales directly.

    Returns ``(supported, reason)``. On current hardware/torch the
    checkpoint layout (plain contiguous BF16 block grid, BF16
    activations) is not a layout ``_scaled_mm`` accepts for 128x128
    blockwise scaling, so this returns ``False`` with the reason and
    callers keep the reference path.
    """
    if not torch.cuda.is_available():
        return False, "CUDA unavailable; reference dequantize path only"
    if scale_inv.dtype != torch.float32:
        return False, (
            "torch._scaled_mm blockwise scales must be float32 in a "
            "near-inner-dim-major 16-byte-aligned layout; checkpoint "
            f"weight_scale_inv is {scale_inv.dtype} in a plain "
            "contiguous block grid, so _scaled_mm cannot consume it "
            "directly (reference path kept)"
        )
    return False, (
        "checkpoint [128, 128] block scales do not match "
        "torch._scaled_mm's blocked scale layout and BF16 activations "
        "would need FP8 block quantisation first (reference path kept)"
    )


class BlockFP8Linear(nn.Module):
    """Linear layer storing block-FP8 weights instead of dense weights.

    Buffers (persistent, so ``state_dict`` keys are stable):

    * ``weight`` -- ``float8_e4m3fn`` ``[out_features, in_features]``
    * ``weight_scale_inv`` -- BF16 ``[ceil(out/br), ceil(in/bc)]``
      for ``block=(br, bc)`` (checkpoint convention: dequantisation
      *multiplies* by this scale)

    ``bias``, when present, is a dense ``nn.Parameter`` exactly as in
    ``nn.Linear``. ``forward`` dequantises to the input dtype and
    applies ``F.linear`` (reference path); see the module docstring
    for why the ``_scaled_mm`` fast path stays a guarded fallback.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block: tuple[int, int] = (128, 128),
        *,
        device: torch.device | str | None = None,
        use_fast_path: bool = False,
    ):
        super().__init__()
        if int(in_features) <= 0 or int(out_features) <= 0:
            raise ValueError(
                f"in/out features must be positive, got {in_features}, {out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block: tuple[int, int] = (int(block[0]), int(block[1]))
        self.use_fast_path = bool(use_fast_path)
        gr = -(-self.out_features // self.block[0])
        gc = -(-self.in_features // self.block[1])
        self.register_buffer(
            "weight",
            torch.zeros(
                (self.out_features, self.in_features),
                dtype=torch.float8_e4m3fn, device=device,
            ),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.ones((gr, gc), dtype=torch.bfloat16, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=device)
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block: tuple[int, int] = (128, 128),
        *,
        use_fast_path: bool = False,
    ) -> "BlockFP8Linear":
        """Quantise a dense ``nn.Linear`` into a resident FP8 module."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"from_linear expects nn.Linear, got {type(linear).__name__}")
        w = linear.weight.detach()
        q, scale = quantize_block_fp8(w.to(torch.float32), block=block)
        module = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None, block=block,
            device=w.device, use_fast_path=use_fast_path,
        )
        with torch.no_grad():
            module.weight.copy_(q.to(module.weight.device))
            module.weight_scale_inv.copy_(
                scale.to(dtype=torch.bfloat16, device=module.weight.device)
            )
            if module.bias is not None and linear.bias is not None:
                module.bias.copy_(linear.bias.detach().to(module.bias.dtype))
        return module

    def dequantized_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dense weight via the block-FP8 reference dequantisation."""
        out_dtype = dtype if dtype is not None else torch.bfloat16
        return dequantize_block_fp8(
            self.weight, self.weight_scale_inv,
            block=self.block, out_dtype=out_dtype,
        )

    def to_dense(self, dtype: torch.dtype = torch.float32) -> nn.Linear:
        """Materialise an equivalent dense ``nn.Linear`` (tests/debug)."""
        linear = nn.Linear(
            self.in_features, self.out_features,
            bias=self.bias is not None,
            device=self.weight.device, dtype=dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.dequantized_weight(dtype))
            if linear.bias is not None and self.bias is not None:
                linear.bias.copy_(self.bias.detach().to(dtype))
        return linear

    def _try_scaled_mm(self, x: torch.Tensor) -> torch.Tensor | None:
        """Attempt ``torch._scaled_mm``; ``None`` means use reference.

        Any capability or layout mismatch returns ``None`` so the
        caller falls back silently. On current torch/hardware the
        checkpoint scale layout is rejected (see module docstring),
        so this returns ``None`` in practice.
        """
        try:
            if x.dim() < 2 or not x.is_cuda:
                return None
            supported, _reason = scaled_mm_block_supported(
                self.weight_scale_inv, self.block
            )
            if not supported:
                return None
            return None  # No verified layout transform yet; stay exact.
        except Exception:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fast_path:
            fast = self._try_scaled_mm(x)
            if fast is not None:
                return fast
        out_dtype = x.dtype if x.is_floating_point() else torch.bfloat16
        w = dequantize_block_fp8(
            self.weight, self.weight_scale_inv,
            block=self.block, out_dtype=out_dtype,
        )
        bias = self.bias.to(out_dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, block={self.block}"
        )


def replace_linears_with_fp8(
    model: nn.Module,
    fp8_target_names: set[str],
    block: tuple[int, int] = (128, 128),
    *,
    use_fast_path: bool = False,
) -> nn.Module:
    """Convert named ``nn.Linear`` modules of ``model`` in place.

    ``fp8_target_names`` holds qualified names exactly as in
    ``model.named_modules()`` (e.g. ``layers.3.self_attn.q_proj``).
    Each named module that is an ``nn.Linear`` is replaced by a
    :class:`BlockFP8Linear` quantised from it, preserving device and
    bias. Names that are missing, already FP8, or point at non-Linear
    modules (norms, embeddings, ...) are skipped silently -- norms
    and embeddings are never converted unless they are ``nn.Linear``
    instances, which they are not. Returns ``model`` for chaining.
    """
    modules = dict(model.named_modules())
    for name in sorted(fp8_target_names):
        module = modules.get(name)
        if module is None or not isinstance(module, nn.Linear):
            continue
        if isinstance(module, BlockFP8Linear):
            continue
        replacement = BlockFP8Linear.from_linear(
            module, block=block, use_fast_path=use_fast_path
        )
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, replacement)
    return model
