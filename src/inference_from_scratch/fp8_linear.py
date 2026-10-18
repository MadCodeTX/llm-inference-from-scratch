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
against the reference, not fast. The opt-in Triton path
(``use_triton`` / ``INFERENCE_FP8_TRITON=1``) instead calls
:mod:`inference_from_scratch.fp8_triton`, which streams FP8 tiles
and applies block scales in-kernel; bench data in
``benchmarks/fp8_linear_bench.py`` shows it winning at every
measured M (see ``TRITON_MAX_M``).

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

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from . import fp8_triton as _fp8_triton_mod
from .quantization import dequantize_block_fp8, quantize_block_fp8

logger = logging.getLogger(__name__)

__all__ = [
    "BlockFP8Linear",
    "apply_scale_cache",
    "replace_linears_with_fp8",
    "scaled_mm_block_supported",
    "shard_fp8_pair",
    "TRITON_MAX_M",
]

# From benchmarks/fp8_linear_bench.py (RTX 4090, 2026-10-17): the
# Triton kernel beat the dequant reference at every measured shape
# and M in 1..128 (6.0x-27.0x). Above 128 rows is unmeasured, so the
# reference (cuBLAS on a materialised BF16 weight) is kept there.
TRITON_MAX_M = 128


def _triton_available() -> bool:
    """Monkeypatch seam for dispatch tests."""
    return _fp8_triton_mod.triton_available()


def _run_triton(x, weight, scale_inv, bias, block):
    """Monkeypatch seam for dispatch tests."""
    return _fp8_triton_mod.fp8_triton_linear(
        x, weight, scale_inv, bias=bias, block=block
    )


def _env_use_triton() -> bool:
    return os.environ.get("INFERENCE_FP8_TRITON", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


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
        use_triton: bool | None = None,
        use_scale_cache: bool = False,
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
        # Explicit flag wins; None falls back to INFERENCE_FP8_TRITON env.
        self.use_triton = _env_use_triton() if use_triton is None else bool(use_triton)
        self.use_scale_cache = bool(use_scale_cache)
        self._triton_fallback_reason: str | None = None
        self._scale_cache_key: tuple | None = None
        self._scale_cache_builds = 0
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
        # FP32 scale cache (persistent=False: never in state_dict).
        self.register_buffer(
            "_scale_f32_cache",
            torch.empty((0,), dtype=torch.float32, device=device),
            persistent=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=device)
            )
        else:
            self.register_parameter("bias", None)

    # -- cached FP32 scales -------------------------------------------------
    def _scale_cache_key_now(self) -> tuple:
        s = self.weight_scale_inv
        try:
            ptr = int(s.data_ptr())
        except Exception:
            ptr = -1
        try:
            ver = int(s._version)  # type: ignore[attr-defined]
        except Exception:
            ver = -1
        return (ptr, ver, tuple(s.shape), str(s.dtype), str(s.device))

    def refresh_scale_cache(self) -> torch.Tensor | None:
        """(Re)build the cached FP32 scale from ``weight_scale_inv``.

        Explicit invalidation entry point: loaders call this after
        assigning new scale tensors. Returns the cache, or ``None``
        on meta device / when caching is disabled and not forced by
        an explicit enable. Building increments ``_scale_cache_builds``
        so tests can prove it happens once per scale version.
        """
        if not self.use_scale_cache:
            return None
        s = self.weight_scale_inv
        if s.device.type == "meta":
            return None
        cache = s.detach().to(torch.float32).contiguous()
        # Assign via buffer object replacement keeps persistent=False.
        self._scale_f32_cache = cache
        self._scale_cache_key = self._scale_cache_key_now()
        self._scale_cache_builds += 1
        return cache

    def set_use_scale_cache(self, enabled: bool) -> bool:
        """Toggle the FP32 scale cache; builds eagerly when enabling."""
        self.use_scale_cache = bool(enabled)
        if self.use_scale_cache:
            self.refresh_scale_cache()
        else:
            self._scale_cache_key = None
            try:
                self._scale_f32_cache = torch.empty(
                    (0,), dtype=torch.float32, device=self.weight_scale_inv.device)
            except Exception:
                pass
        return self.use_scale_cache

    def cached_fp32_scale(self) -> torch.Tensor | None:
        """Cached FP32 scale, rebuilt iff storage/version changed.

        Honest invalidation: the key is (data_ptr, _version, shape,
        dtype, device) of ``weight_scale_inv``. In-place writes bump
        ``_version``; tensor replacement changes ``data_ptr``; either
        triggers exactly one rebuild on next use. Returns ``None``
        when caching is disabled.
        """
        if not self.use_scale_cache:
            return None
        if self.weight_scale_inv.device.type == "meta":
            return None
        if self._scale_cache_key != self._scale_cache_key_now() or \
                tuple(self._scale_f32_cache.shape) != tuple(self.weight_scale_inv.shape):
            return self.refresh_scale_cache()
        return self._scale_f32_cache

    # Back-compat alias used by tests/loaders.
    _cached_fp32_scale = cached_fp32_scale

    @classmethod
    def from_pre_sliced(
        cls,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None = None,
        block: tuple[int, int] = (128, 128),
        *,
        use_fast_path: bool = False,
        use_triton: bool | None = None,
        use_scale_cache: bool = False,
    ) -> "BlockFP8Linear":
        """Build from pre-sliced FP8 ``weight`` + ``weight_scale_inv``.

        For TP shards: ``weight`` is this rank's row/column slice of a
        dense FP8 matrix and ``weight_scale_inv`` the matching block-grid
        slice (see :func:`shard_fp8_pair`). Shapes are validated against
        the block grid exactly as the reference dequantiser requires.
        """
        if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
            raise ValueError(
                f"weight must be float8_e4m3fn [out, in], got {weight.dtype} {tuple(weight.shape)}")
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        gr, gc = -(-out_f // int(block[0])), -(-in_f // int(block[1]))
        if tuple(weight_scale_inv.shape) != (gr, gc):
            raise ValueError(
                f"weight_scale_inv {tuple(weight_scale_inv.shape)} != block grid {(gr, gc)} "
                f"for weight {(out_f, in_f)} block {tuple(block)}")
        module = cls(in_f, out_f, bias=bias is not None, block=block,
                     device=weight.device, use_fast_path=use_fast_path,
                     use_triton=use_triton, use_scale_cache=use_scale_cache)
        with torch.no_grad():
            module.weight.copy_(weight.to(module.weight.device))
            module.weight_scale_inv.copy_(
                weight_scale_inv.to(dtype=torch.bfloat16, device=module.weight.device))
            if module.bias is not None and bias is not None:
                module.bias.copy_(bias.detach().to(module.bias.dtype))
        if module.use_scale_cache:
            module.refresh_scale_cache()
        return module

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block: tuple[int, int] = (128, 128),
        *,
        use_fast_path: bool = False,
        use_triton: bool | None = None,
        use_scale_cache: bool = False,
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
            use_triton=use_triton, use_scale_cache=use_scale_cache,
        )
        with torch.no_grad():
            module.weight.copy_(q.to(module.weight.device))
            module.weight_scale_inv.copy_(
                scale.to(dtype=torch.bfloat16, device=module.weight.device)
            )
            if module.bias is not None and linear.bias is not None:
                module.bias.copy_(linear.bias.detach().to(module.bias.dtype))
        if module.use_scale_cache:
            module.refresh_scale_cache()
        return module

    def _load_from_state_dict(self, *args, **kwargs):  # type: ignore[override]
        super()._load_from_state_dict(*args, **kwargs)
        # Loaded scales may share storage/version coincidentally in
        # exotic cases; force one rebuild on next Triton use.
        self._scale_cache_key = None

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

    def _try_triton(self, x: torch.Tensor) -> torch.Tensor | None:
        """Triton fused path; ``None`` means use the reference.

        Only attempted for CUDA inputs with flattened M in
        ``1..TRITON_MAX_M`` (bench-measured winning range) and the
        checkpoint (128, 128) block. Any import/launch failure is
        logged (never silent) and recorded on
        ``_triton_fallback_reason`` before falling back.
        """
        if not self.use_triton or not x.is_cuda:
            return None
        if self.block != (128, 128) or x.dim() < 1:
            return None
        flat_m = x.numel() // x.shape[-1] if x.dim() >= 1 else 0
        if not (1 <= flat_m <= TRITON_MAX_M):
            return None
        if not _triton_available():
            self._triton_fallback_reason = "triton unavailable"
            logger.warning(
                "BlockFP8Linear: use_triton set but Triton unavailable; "
                "falling back to reference dequant path"
            )
            return None
        try:
            lead_shape = x.shape[:-1]
            x2 = x.reshape(flat_m, x.shape[-1])
            bias = self.bias.to(torch.bfloat16) if self.bias is not None else None
            scale = self.cached_fp32_scale()
            out = _run_triton(
                x2, self.weight,
                scale if scale is not None else self.weight_scale_inv,
                bias, self.block
            )
            return out.reshape(*lead_shape, self.out_features)
        except Exception as exc:  # Logged fallback, not silent.
            self._triton_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "BlockFP8Linear: Triton path failed (%s); "
                "falling back to reference dequant path", exc,
            )
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            triton_out = self._try_triton(x)
            if triton_out is not None:
                return triton_out
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



def apply_scale_cache(model: nn.Module, enabled: bool) -> int:
    """Set the FP32 scale cache on every BlockFP8Linear in ``model``."""
    n=0
    for mod in model.modules():
        if isinstance(mod, BlockFP8Linear):
            mod.set_use_scale_cache(bool(enabled)); n+=1
    return n

def shard_fp8_pair(
    name: str,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    rank: int,
    world: int,
    config=None,
    block: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's ``(weight, weight_scale_inv)`` shard of an FP8 matrix.

    The weight is sliced at the ``tensor_parallel.shard_spec_for``
    indices for ``name`` (head pairs, KV/value heads, segmented qkv
    groups, intermediate rows/columns). The scale grid is sliced by the
    *block* indices those weight indices fall in
    (``sorted(unique(idx // block))`` on the sharded axis), which is
    exactly the spec's scale rule (``weight_scale_inv`` follows its
    weight's axis, one scale per [128, 128] block) expressed on the
    grid -- including the segmented-qkv case, whose grid rows are the
    non-contiguous q/k/v block rows of this rank's weight rows. Raises
    ``ValueError`` when a shard boundary cuts a [128, 128] block (the
    sliced grid would not be the local block grid); every spec unit in
    the real checkpoint (head 256, head-pair 512, value-head groups,
    8704-row intermediate halves) is block-aligned, and a sub-block
    tiny tensor keeps the single shared block.
    """
    from .tensor_parallel import shard_spec_for

    if weight.dim() != 2 or weight_scale_inv.dim() != 2:
        raise ValueError("FP8 weight and scale must both be 2D")
    spec = shard_spec_for(name, tuple(weight.shape), config)
    if not spec.is_sharded or spec.dim is None:
        return weight.clone(), weight_scale_inv.clone()
    idx = spec.indices_for_rank(rank, world)
    w_shard = weight.index_select(
        spec.dim, torch.tensor(idx, dtype=torch.long, device=weight.device)).clone()
    blk = int(block[spec.dim])
    block_idx = sorted({int(i) // blk for i in idx})
    s_shard = weight_scale_inv.index_select(
        spec.dim, torch.tensor(block_idx, dtype=torch.long,
                               device=weight_scale_inv.device)).clone()
    gr = -(-int(w_shard.shape[0]) // int(block[0]))
    gc = -(-int(w_shard.shape[1]) // int(block[1]))
    if tuple(s_shard.shape) != (gr, gc):
        raise ValueError(
            f"{name}: shard boundary cuts a {tuple(block)} FP8 block "
            f"(sliced scale {tuple(s_shard.shape)} != local grid {(gr, gc)}); "
            "TP shard units must be block-aligned for resident FP8")
    return w_shard, s_shard


def replace_linears_with_fp8(
    model: nn.Module,
    fp8_target_names: set[str],
    block: tuple[int, int] = (128, 128),
    *,
    use_fast_path: bool = False,
    use_triton: bool | None = None,
    use_scale_cache: bool = False,
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
            module, block=block, use_fast_path=use_fast_path,
            use_triton=use_triton, use_scale_cache=use_scale_cache,
        )
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, replacement)
    return model
