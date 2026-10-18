"""Real two-process tensor parallelism via ``torch.distributed``.

This is the distributed counterpart of the in-process simulation in
:mod:`inference_from_scratch.tensor_parallel`. The simulated
``TPGatedDeltaNet.step`` / ``TPFullAttention.forward`` loop over ranks
inside one process; here each OS process owns exactly one rank's
shards and the per-rank partials combine through real collectives
(NCCL on GPU, gloo on CPU).

Shard construction reuses the specs in ``tensor_parallel``
(:func:`shard_spec_for` / :func:`shard_state_dict`), so a distributed
shard is byte-identical to the simulated shard for the same
``(name, rank, world, config)``.
"""
from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .fp8_linear import BlockFP8Linear
from .tensor_parallel import shard_spec_for, shard_state_dict

__all__ = [
    "DistributedTPContext",
    "TPColumnLinear",
    "TPRowLinear",
    "TPBlockFP8Column",
    "TPBlockFP8Row",
    "DistributedSwiGLU",
    "DistributedFullAttention",
    "DistributedGatedDeltaNet",
    "DistributedHybridBlock",
    "DistributedHybridLM",
    "TPState",
    "TPStaticState",
    "shard_weight_for_rank",
    "vocab_shard_layout",
    "LM_HEAD_MODES",
]


def shard_weight_for_rank(name: str, weight: torch.Tensor, rank: int,
                          world: int, config: Any = None) -> torch.Tensor:
    """This rank's shard of ``weight`` per ``shard_state_dict`` specs."""
    return shard_state_dict({name: weight}, rank, world, config)[name]


class DistributedTPContext:
    """One process's view of a real ``torch.distributed`` process group.

    Built from the torchrun environment (``RANK`` / ``WORLD_SIZE`` /
    ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT``). The device is
    ``cuda:LOCAL_RANK`` for NCCL and CPU for gloo unless overridden.
    """

    def __init__(self, backend: str, device: torch.device,
                 *, _owns_group: bool = True):
        if not dist.is_initialized():
            raise RuntimeError("process group not initialised; use from_env()")
        self.backend = backend
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self._owns_group = _owns_group

    @classmethod
    def from_env(cls, backend: str | None = None,
                 device_type: str | None = None) -> "DistributedTPContext":
        """Init the process group from torchrun env vars.

        ``backend`` defaults to ``nccl`` when CUDA is available else
        ``gloo``. ``device_type`` (``cuda``/``cpu``) picks the device;
        it defaults from the backend (nccl->cuda, gloo->cpu unless
        ``device_type="cuda"`` is requested for a gloo-on-GPU run).
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        if device_type is None:
            device_type = "cuda" if backend == "nccl" else "cpu"
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device_type='cuda' but CUDA unavailable")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, device_id=device if device_type == "cuda" else None)
        return cls(backend, device)

    # -- collectives -------------------------------------------------------
    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum ``tensor`` across ranks **in place**; returns the same tensor.

        In-place semantics: the caller's tensor object is overwritten
        with the cross-rank sum (``dist.all_reduce`` mutates its input).
        Callers must not reuse the pre-call value afterwards; clone
        first if the local partial is still needed.
        """
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather one tensor per rank (equal shapes) in rank order."""
        out = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(out, tensor)
        return out

    def all_gather_concat_dim0(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather and concatenate along dim 0 (column-output gather).

        Handles unequal dim-0 sizes across ranks by exchanging sizes
        first. Every rank returns the full concatenated tensor.
        """
        return self.all_gather_concat(tensor, dim=0)

    def all_gather_concat(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """All-gather and concatenate along ``dim`` in rank order."""
        if self.world_size == 1:
            return tensor
        # Exchange dim sizes (they may differ, e.g. uneven column shards).
        sizes = [torch.empty(1, dtype=torch.long, device=tensor.device)
                 for _ in range(self.world_size)]
        dist.all_gather(sizes, torch.tensor([tensor.shape[dim]], dtype=torch.long, device=tensor.device))
        size_list = [int(s.item()) for s in sizes]
        if len(set(size_list)) == 1:
            return torch.cat(self.all_gather(tensor), dim=dim)
        # Uneven sizes: pad to the max along dim, gather, then trim.
        max_sz = max(size_list)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_sz
        padded = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        sl = [slice(None)] * tensor.dim()
        sl[dim] = slice(0, tensor.shape[dim])
        padded[tuple(sl)] = tensor
        gathered = self.all_gather(padded)
        trimmed = []
        for buf, sz in zip(gathered, size_list):
            sl2 = [slice(None)] * buf.dim()
            sl2[dim] = slice(0, sz)
            trimmed.append(buf[tuple(sl2)])
        return torch.cat(trimmed, dim=dim)

    def barrier(self) -> None:
        dist.barrier()

    def destroy(self) -> None:
        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()


def _spec_or_even(name: str, shape: tuple[int, ...], config: Any):
    return shard_spec_for(name, shape, config)


class TPColumnLinear(nn.Module):
    """Column-parallel linear holding ONLY this rank's weight rows.

    ``forward`` is the local matmul only (no collective): the output
    is this rank's disjoint feature slice, consumed locally by the
    following row-parallel projection. :meth:`forward_gathered`
    all-gathers the slices when a caller needs the full output.
    """

    def __init__(self, weight_shard: torch.Tensor,
                 bias_shard: torch.Tensor | None, ctx: DistributedTPContext):
        super().__init__()
        self.ctx = ctx
        self.weight = nn.Parameter(weight_shard.detach().clone().to(ctx.device))
        self.bias = nn.Parameter(bias_shard.detach().clone().to(ctx.device)) if bias_shard is not None else None
        self.in_features = int(weight_shard.shape[1])
        self.out_features_local = int(weight_shard.shape[0])

    @classmethod
    def from_dense(cls, dense: nn.Linear, name: str, ctx: DistributedTPContext,
                   config: Any = None) -> "TPColumnLinear":
        spec = _spec_or_even(name, tuple(dense.weight.shape), config)
        if spec.dim != 0 or spec.kind not in ("column", "head_column"):
            raise ValueError(f"{name}: expected column/head_column dim=0 spec, got {spec.kind} dim={spec.dim}")
        w = shard_weight_for_rank(name, dense.weight.data, ctx.rank, ctx.world_size, config)
        b = None
        if dense.bias is not None:
            b = shard_weight_for_rank(name.replace(".weight", ".bias"), dense.bias.data,
                                      ctx.rank, ctx.world_size, config)
        return cls(w, b, ctx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def forward_gathered(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctx.all_gather_concat(self.forward(x), dim=-1)


class TPRowLinear(nn.Module):
    """Row-parallel linear holding ONLY this rank's weight columns.

    ``forward`` takes the replicated full input, multiplies this
    rank's column slice, and all-reduces the partials (bias, replicated,
    added once after the reduction). :meth:`forward_from_local` takes
    only this rank's input slice (e.g. a column-parallel output).
    """

    def __init__(self, weight_shard: torch.Tensor,
                 bias: torch.Tensor | None, ctx: DistributedTPContext,
                 col_start: int = 0):
        super().__init__()
        self.ctx = ctx
        self.weight = nn.Parameter(weight_shard.detach().clone().to(ctx.device))
        self.bias = nn.Parameter(bias.detach().clone().to(ctx.device)) if bias is not None else None
        self.out_features = int(weight_shard.shape[0])
        self.in_features_local = int(weight_shard.shape[1])
        self.col_start = int(col_start)

    @classmethod
    def from_dense(cls, dense: nn.Linear, name: str, ctx: DistributedTPContext,
                   config: Any = None) -> "TPRowLinear":
        spec = _spec_or_even(name, tuple(dense.weight.shape), config)
        if spec.dim != 1 or spec.kind != "row":
            raise ValueError(f"{name}: expected row dim=1 spec, got {spec.kind} dim={spec.dim}")
        w = shard_weight_for_rank(name, dense.weight.data, ctx.rank, ctx.world_size, config)
        idx = spec.indices_for_rank(ctx.rank, ctx.world_size)
        b = dense.bias.data if dense.bias is not None else None
        return cls(w, b, ctx, col_start=idx[0] if idx else 0)

    def forward_from_local(self, x_local: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x_local, self.weight)
        out = self.ctx.all_reduce_sum(partial)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.col_start
        return self.forward_from_local(x[..., s:s + self.in_features_local])


class _LocalContext:
    """Rank/world/device view for construction without a process group.

    Collectives are identity operations, valid only at ``world_size=1``
    (single-rank loads/tests); multi-rank forwards need a real
    :class:`DistributedTPContext` via ``DistributedHybridLM.set_ctx``.
    """

    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.backend = "none"

    def _check_single(self) -> None:
        if self.world_size != 1:
            raise RuntimeError("_LocalContext collectives need world_size=1; "
                               "use DistributedTPContext under torchrun")

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        self._check_single()
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        self._check_single()
        return [tensor]

    def all_gather_concat(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self._check_single()
        return tensor

    def barrier(self) -> None:
        return None


class TPBlockFP8Column(nn.Module):
    """Column-parallel resident-FP8 linear: this rank's output slice only.

    ``forward`` runs the local :class:`BlockFP8Linear` (reference or
    Triton per-rank on its own device) with no collective; the output
    stays sharded for a following row-parallel consumer.
    :meth:`forward_gathered` all-gathers only where semantics require
    the full output.
    """

    def __init__(self, local: BlockFP8Linear, ctx: Any):
        super().__init__()
        self.local = local
        self.ctx = ctx
        self.in_features = int(local.in_features)
        self.out_features_local = int(local.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x)

    def forward_gathered(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctx.all_gather_concat(self.forward(x), dim=-1)


class TPBlockFP8Row(nn.Module):
    """Row-parallel resident-FP8 linear: local partial + ONE all-reduce.

    The local :class:`BlockFP8Linear` must be bias-free (its forward
    would add bias per rank); the replicated bias is added once after
    the reduction, exactly as in :class:`TPRowLinear`.
    """

    def __init__(self, local: BlockFP8Linear, ctx: Any,
                 bias: torch.Tensor | None = None, col_start: int = 0):
        super().__init__()
        if local.bias is not None:
            raise ValueError("TPBlockFP8Row local module must be bias-free; "
                             "bias is added once after the all-reduce")
        self.local = local
        self.ctx = ctx
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.out_features = int(local.out_features)
        self.in_features_local = int(local.in_features)
        self.col_start = int(col_start)

    def forward_from_local(self, x_local: torch.Tensor) -> torch.Tensor:
        partial = self.local(x_local)
        out = self.ctx.all_reduce_sum(partial)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.col_start
        return self.forward_from_local(x[..., s:s + self.in_features_local])


class DistributedSwiGLU(nn.Module):
    """Distributed SwiGLU MLP: column gate/up shards, one all-reduce in down."""

    def __init__(self, ctx: DistributedTPContext, gate: TPColumnLinear,
                 up: TPColumnLinear, down: TPRowLinear):
        super().__init__()
        self.ctx = ctx
        self.gate_proj = gate
        self.up_proj = up
        self.down_proj = down

    @classmethod
    def from_dense(cls, dense: Any, ctx: DistributedTPContext,
                   config: Any = None) -> "DistributedSwiGLU":
        return cls(ctx,
                   TPColumnLinear.from_dense(dense.gate_proj, "mlp.gate_proj.weight", ctx, config),
                   TPColumnLinear.from_dense(dense.up_proj, "mlp.up_proj.weight", ctx, config),
                   TPRowLinear.from_dense(dense.down_proj, "mlp.down_proj.weight", ctx, config))

    @classmethod
    def empty(cls, ctx: Any, config: Any = None, fp8: bool = False,
              use_triton: bool | None = None) -> "DistributedSwiGLU":
        """Shard-shaped empty MLP on the meta device (loader fills it)."""
        from .hybrid import HybridConfig
        cfg = config if isinstance(config, HybridConfig) else HybridConfig()
        D, I = int(cfg.hidden_size), int(cfg.intermediate_size)
        world = int(ctx.world_size)
        if I % world != 0:
            raise ValueError(f"intermediate {I} not divisible by world {world}")
        per = I // world
        meta = torch.device("meta")
        if fp8:
            gate = TPBlockFP8Column(BlockFP8Linear(D, per, bias=False, device=meta, use_triton=use_triton), ctx)
            up = TPBlockFP8Column(BlockFP8Linear(D, per, bias=False, device=meta, use_triton=use_triton), ctx)
            down = TPBlockFP8Row(BlockFP8Linear(per, D, bias=False, device=meta, use_triton=use_triton),
                                 ctx, col_start=int(ctx.rank) * per)
        else:
            gate = TPColumnLinear(torch.empty((per, D), device=meta), None, ctx)
            up = TPColumnLinear(torch.empty((per, D), device=meta), None, ctx)
            down = TPRowLinear(torch.empty((D, per), device=meta), None, ctx,
                               col_start=int(ctx.rank) * per)
            # TP*Linear.__init__ clones onto ctx.device; empty() must stay meta.
            for mod, shape in ((gate, (per, D)), (up, (per, D)), (down, (D, per))):
                mod.weight = nn.Parameter(torch.empty(shape, device=meta))
        return cls(ctx, gate, up, down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj.forward_from_local(hidden)


class DistributedFullAttention(nn.Module):
    """Distributed gated full attention: this rank's Q/KV heads only.

    Mirrors one rank's branch of the simulated ``TPFullAttention``:
    local q/k/v projections, per-head norms + RoPE, attention over the
    local Q-head subset (local KV repeated by the GQA factor), gated
    output through the local ``o_proj`` column slice, one all-reduce.
    Norms are replicated (tiny); projection weights are shards only.
    """

    def __init__(self, ctx: DistributedTPContext, dense: Any,
                 config: Any = None):
        super().__init__()
        from .attention import manual_scaled_dot_product_attention, repeat_kv  # noqa: F401
        self.ctx = ctx
        cfg = config if config is not None else {
            "hidden_size": dense.d_model, "num_attention_heads": dense.n_head,
            "num_key_value_heads": dense.n_kv_head, "head_dim": dense.head_dim,
            "intermediate_size": dense.d_model}
        sd = {"self_attn.q_proj.weight": dense.q_proj.weight.data,
              "self_attn.k_proj.weight": dense.k_proj.weight.data,
              "self_attn.v_proj.weight": dense.v_proj.weight.data,
              "self_attn.o_proj.weight": dense.o_proj.weight.data}
        sh = shard_state_dict(sd, ctx.rank, ctx.world_size, cfg)
        dev = ctx.device
        self.q_weight = nn.Parameter(sh["self_attn.q_proj.weight"].to(dev))
        self.k_weight = nn.Parameter(sh["self_attn.k_proj.weight"].to(dev))
        self.v_weight = nn.Parameter(sh["self_attn.v_proj.weight"].to(dev))
        self.o_weight = nn.Parameter(sh["self_attn.o_proj.weight"].to(dev))
        self.q_bias = self._shard_bias(dense.q_proj, "self_attn.q_proj", cfg)
        self.k_bias = self._shard_bias(dense.k_proj, "self_attn.k_proj", cfg)
        self.v_bias = self._shard_bias(dense.v_proj, "self_attn.v_proj", cfg)
        self.q_norm = copy.deepcopy(dense.q_norm).to(dev)
        self.k_norm = copy.deepcopy(dense.k_norm).to(dev)
        self.n_head = int(dense.n_head); self.n_kv_head = int(dense.n_kv_head)
        self.head_dim = int(dense.head_dim); self.n_rep = int(dense.n_rep)
        self.rotary_dim = int(dense.rotary_dim); self.rope_theta = float(dense.rope_theta)
        self.partial_rotary_factor = float(dense.partial_rotary_factor)
        self.mrope_section: list[int] = (
            [int(v) for v in dense.mrope_section]
            if getattr(dense, "mrope_section", None) is not None else [11, 11, 10])
        self.attn_implementation = str(dense.attn_implementation)
        self.gate_activation = str(dense.gate_activation)
        self.scaling = float(getattr(dense, "scaling", self.head_dim ** -0.5))
        h_per = self.n_head // ctx.world_size
        kv_per = self.n_kv_head // ctx.world_size
        self.q_heads_local = list(range(ctx.rank * h_per, (ctx.rank + 1) * h_per))
        self.kv_heads_local = list(range(ctx.rank * kv_per, (ctx.rank + 1) * kv_per))
        # Resident-FP8 projections (None in dense mode; loader-built
        # FP8 models set these and clear the dense Parameters).
        self.q_fp8: BlockFP8Linear | None = None
        self.k_fp8: BlockFP8Linear | None = None
        self.v_fp8: BlockFP8Linear | None = None
        self.o_fp8: BlockFP8Linear | None = None

    @staticmethod
    def _linear(x: torch.Tensor, weight: Any, bias: Any, fp8: Any) -> torch.Tensor:
        return fp8(x) if fp8 is not None else F.linear(x, weight, bias)

    @classmethod
    def empty(cls, ctx: Any, config: Any = None, fp8: bool = False,
              use_triton: bool | None = None) -> "DistributedFullAttention":
        """Shard-shaped empty attention on meta (loader fills it)."""
        from .hybrid import HybridConfig, QwenRMSNorm, partial_rope_dim
        cfg = config if isinstance(config, HybridConfig) else HybridConfig()
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.ctx = ctx
        D, H, Hkv, Dh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        world, rank = int(ctx.world_size), int(ctx.rank)
        meta = torch.device("meta")
        q_shape = shard_spec_for("self_attn.q_proj.weight", (2 * H * Dh, D), cfg).shard_shape(rank, world)
        kv_shape = shard_spec_for("self_attn.k_proj.weight", (Hkv * Dh, D), cfg).shard_shape(rank, world)
        o_shape = shard_spec_for("self_attn.o_proj.weight", (D, H * Dh), cfg).shard_shape(rank, world)
        if fp8:
            obj.q_weight = None; obj.k_weight = None; obj.v_weight = None; obj.o_weight = None
            obj.q_fp8 = BlockFP8Linear(D, q_shape[0], bias=False, device=meta, use_triton=use_triton)
            obj.k_fp8 = BlockFP8Linear(D, kv_shape[0], bias=False, device=meta, use_triton=use_triton)
            obj.v_fp8 = BlockFP8Linear(D, kv_shape[0], bias=False, device=meta, use_triton=use_triton)
            obj.o_fp8 = BlockFP8Linear(o_shape[1], D, bias=False, device=meta, use_triton=use_triton)
        else:
            obj.q_weight = nn.Parameter(torch.empty(q_shape, device=meta))
            obj.k_weight = nn.Parameter(torch.empty(kv_shape, device=meta))
            obj.v_weight = nn.Parameter(torch.empty(kv_shape, device=meta))
            obj.o_weight = nn.Parameter(torch.empty(o_shape, device=meta))
            obj.q_fp8 = obj.k_fp8 = obj.v_fp8 = obj.o_fp8 = None
        obj.q_bias = obj.k_bias = obj.v_bias = None
        obj.q_norm = QwenRMSNorm(Dh, eps=cfg.rms_norm_eps).to(meta)
        obj.k_norm = QwenRMSNorm(Dh, eps=cfg.rms_norm_eps).to(meta)
        obj.n_head, obj.n_kv_head, obj.head_dim = int(H), int(Hkv), int(Dh)
        obj.n_rep = int(H) // int(Hkv)
        obj.rotary_dim = partial_rope_dim(int(Dh), float(cfg.partial_rotary_factor))
        obj.rope_theta = float(cfg.rope_theta)
        obj.partial_rotary_factor = float(cfg.partial_rotary_factor)
        obj.mrope_section = [int(v) for v in (cfg.mrope_section or [11, 11, 10])]
        obj.attn_implementation = "manual"
        gate = str(cfg.output_gate_type or "swish").lower() if cfg.attn_output_gate else "none"
        obj.gate_activation = {"swish": "sigmoid", "sigmoid": "sigmoid",
                               "silu": "silu", "none": "none"}[gate]
        obj.scaling = float(Dh) ** -0.5
        h_per, kv_per = int(H) // world, int(Hkv) // world
        obj.q_heads_local = list(range(rank * h_per, (rank + 1) * h_per))
        obj.kv_heads_local = list(range(rank * kv_per, (rank + 1) * kv_per))
        return obj

    def _shard_bias(self, linear: nn.Linear, prefix: str, cfg: Any):
        if linear.bias is None:
            return None
        return nn.Parameter(shard_weight_for_rank(f"{prefix}.bias", linear.bias.data,
                                                  self.ctx.rank, self.ctx.world_size, cfg).to(self.ctx.device))

    def _attend(self, q: torch.Tensor, k_rep: torch.Tensor,
                v_rep: torch.Tensor, start_pos: int) -> torch.Tensor:
        from .attention import manual_scaled_dot_product_attention
        if self.attn_implementation == "sdpa":
            t_q, t_k = q.shape[2], k_rep.shape[2]
            if t_q == t_k and start_pos == 0:
                return F.scaled_dot_product_attention(
                    q, k_rep, v_rep, is_causal=True, scale=self.scaling)
            if t_q == 1:
                return F.scaled_dot_product_attention(
                    q, k_rep, v_rep, is_causal=False, scale=self.scaling)
            q_pos = torch.arange(t_q, device=q.device) + int(start_pos)
            k_pos = torch.arange(t_k, device=q.device)
            future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            mask = torch.zeros(t_q, t_k, device=q.device, dtype=q.dtype)
            mask = mask.masked_fill(future, float("-inf"))
            return F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=mask, is_causal=False, scale=self.scaling)
        return manual_scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

    def forward_with_state(
        self, x: torch.Tensor,
        kv_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Cache-aware forward: ``x [B,T,D]`` -> ``(out, (k_all, v_all))``.

        ``kv_state`` is this rank's local cache ``(k, v)`` each
        ``[B, H_kv_local, T_past, Dh]`` (post-RoPE keys), or ``None``.
        New K/V are RoPE'd at ``start_pos`` (or ``position_ids``) and
        appended; attention reads the full local history. Exactly one
        all-reduce (the ``o_proj`` partial sum).
        """
        from .attention import repeat_kv
        from .hybrid import (_apply_rope_with_tables, apply_partial_rope,
                             precompute_mrope_cos_sin,
                             precompute_partial_rope_cache)
        B, T, _ = x.shape
        Dh = self.head_dim
        hs, kvs = self.q_heads_local, self.kv_heads_local
        qg = self._linear(x, self.q_weight, self.q_bias, self.q_fp8).view(B, T, len(hs), 2 * Dh)
        q = self.q_norm(qg[..., :Dh]).transpose(1, 2)
        gate = qg[..., Dh:].reshape(B, T, len(hs) * Dh)
        k = self._linear(x, self.k_weight, self.k_bias, self.k_fp8).view(B, T, len(kvs), Dh)
        v = self._linear(x, self.v_weight, self.v_bias, self.v_fp8).view(B, T, len(kvs), Dh)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor) or position_ids.shape[-1] != T:
                got = (tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor)
                       else type(position_ids).__name__)
                raise ValueError(f"position_ids must end with T={T}, got {got}")
            pos_cos, pos_sin = precompute_mrope_cos_sin(
                position_ids, self.rotary_dim, theta=self.rope_theta,
                mrope_section=self.mrope_section)
            q = _apply_rope_with_tables(q, pos_cos, pos_sin, self.rotary_dim)
            k = _apply_rope_with_tables(k, pos_cos, pos_sin, self.rotary_dim)
        else:
            cos, sin, _ = precompute_partial_rope_cache(
                int(start_pos) + T, Dh, self.partial_rotary_factor,
                theta=self.rope_theta, device=x.device, dtype=x.dtype)
            q = apply_partial_rope(q, cos, sin, start_pos=int(start_pos), rotary_dim=self.rotary_dim)
            k = apply_partial_rope(k, cos, sin, start_pos=int(start_pos), rotary_dim=self.rotary_dim)
        if kv_state is not None:
            k_all = torch.cat([kv_state[0], k], dim=2)
            v_all = torch.cat([kv_state[1], v], dim=2)
        else:
            k_all, v_all = k, v
        k_rep = repeat_kv(k_all, self.n_rep); v_rep = repeat_kv(v_all, self.n_rep)
        attn = self._attend(q, k_rep, v_rep, int(start_pos))
        merged = attn.transpose(1, 2).contiguous().view(B, T, -1)
        if self.gate_activation == "sigmoid":
            merged = merged * torch.sigmoid(gate)
        elif self.gate_activation == "silu":
            merged = merged * F.silu(gate)
        partial = self._linear(merged, self.o_weight, None, self.o_fp8)
        return self.ctx.all_reduce_sum(partial), (k_all, v_all)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        out, _ = self.forward_with_state(x, kv_state=None, start_pos=start_pos)
        return out

    def _static_rope_tables(self, max_seq_len: int, device: torch.device,
                            dtype: torch.dtype):
        """RoPE tables for static decode, built once per (len, dev, dtype)."""
        cache = getattr(self, "_static_rope_cache", None)
        if cache is None:
            cache = self._static_rope_cache = {}
        key = (int(max_seq_len), str(device), dtype)
        if key not in cache:
            from .hybrid import precompute_partial_rope_cache
            cos, sin, _ = precompute_partial_rope_cache(
                int(max_seq_len), self.head_dim, self.partial_rotary_factor,
                theta=self.rope_theta, device=device, dtype=dtype)
            cache[key] = (cos, sin)
        return cache[key]

    def step_static(self, x: torch.Tensor, k_buf: torch.Tensor,
                    v_buf: torch.Tensor, position: torch.Tensor,
                    rope_position: torch.Tensor | None = None) -> torch.Tensor:
        """One static decode token: ``x [B,1,D]`` -> ``[B,1,D]``.

        Mirrors :meth:`forward_with_state` at ``T=1`` step-for-step, but
        K/V are written in place into the preallocated local buffers
        ``k_buf`` / ``v_buf`` (``[B, H_kv_local, max_seq, Dh]``) with
        ``index_copy_`` at the 0-dim tensor ``position``, RoPE rows are
        gathered with ``index_select``, and attention runs over the
        full fixed-shape buffers with positions past ``position``
        masked on-device. No ``.item()``, ``torch.cat`` or shape change
        on the state. Exactly one all-reduce, as in the legacy step.

        ``rope_position`` (0-dim tensor) optionally decouples the RoPE
        row from the KV slot: K/V still land at ``position`` and the
        mask still keys on ``position``, but Q/K are rotated at
        ``rope_position``. The MTP draft needs this: its KV slots are
        sequential (one per committed draft) while its RoPE position
        is the target's absolute hidden position.
        """
        from .attention import repeat_kv
        from .hybrid import _apply_rope_with_tables
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"x must be [B, D] or [B, 1, D], got {tuple(x.shape)}")
        if not isinstance(position, torch.Tensor) or position.dim() != 0:
            raise ValueError("position must be a 0-dim tensor")
        B = int(x.shape[0])
        Dh = self.head_dim
        hs, kvs = self.q_heads_local, self.kv_heads_local
        qg = self._linear(x, self.q_weight, self.q_bias, self.q_fp8).view(B, 1, len(hs), 2 * Dh)
        q = self.q_norm(qg[..., :Dh]).transpose(1, 2)
        gate = qg[..., Dh:].reshape(B, 1, len(hs) * Dh)
        k = self._linear(x, self.k_weight, self.k_bias, self.k_fp8).view(B, 1, len(kvs), Dh)
        v = self._linear(x, self.v_weight, self.v_bias, self.v_fp8).view(B, 1, len(kvs), Dh)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self._static_rope_tables(int(k_buf.shape[2]), x.device, x.dtype)
        pos1 = position.reshape(1)
        rope1 = pos1 if rope_position is None else rope_position.reshape(1)
        row_cos = cos.index_select(0, rope1)
        row_sin = sin.index_select(0, rope1)
        q = _apply_rope_with_tables(q, row_cos, row_sin, self.rotary_dim)
        k = _apply_rope_with_tables(k, row_cos, row_sin, self.rotary_dim)
        k_buf.index_copy_(2, pos1, k)
        v_buf.index_copy_(2, pos1, v)
        k_rep = repeat_kv(k_buf, self.n_rep)
        v_rep = repeat_kv(v_buf, self.n_rep)
        t_k = int(k_buf.shape[2])
        if self.attn_implementation == "sdpa":
            k_pos = torch.arange(t_k, device=q.device)
            mask = torch.zeros(1, t_k, device=q.device, dtype=q.dtype)
            mask = mask.masked_fill(k_pos > position, float("-inf"))
            attn = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=mask.view(1, 1, 1, t_k),
                is_causal=False, scale=self.scaling)
        else:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (Dh ** 0.5)
            k_pos = torch.arange(t_k, device=q.device)
            scores = scores.masked_fill(
                (k_pos > position).view(1, 1, 1, t_k), float("-inf"))
            attn = torch.matmul(torch.softmax(scores, dim=-1), v_rep)
        merged = attn.transpose(1, 2).contiguous().view(B, 1, -1)
        if self.gate_activation == "sigmoid":
            merged = merged * torch.sigmoid(gate)
        elif self.gate_activation == "silu":
            merged = merged * F.silu(gate)
        partial = self._linear(merged, self.o_weight, None, self.o_fp8)
        return self.ctx.all_reduce_sum(partial)

    def step_chunk_static(self, x: torch.Tensor, k_buf: torch.Tensor,
                          v_buf: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Two-token chunk twin of :meth:`step_static`: ``x [B,2,D]``.

        Q/K/V/gate projections run ONCE over both tokens (one matmul
        each, M=2); K/V are written at ``position`` and ``position+1``;
        attention is causal per query (query 0 sees keys <= position,
        query 1 sees keys <= position+1, buffer tail masked). Exactly
        one all-reduce, as in the single-token step.
        """
        from .attention import repeat_kv
        from .hybrid import _apply_rope_with_tables
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(f"x must be [B, 2, D], got {tuple(x.shape)}")
        if not isinstance(position, torch.Tensor) or position.dim() != 0:
            raise ValueError("position must be a 0-dim tensor")
        B, T, _ = x.shape
        Dh = self.head_dim
        hs, kvs = self.q_heads_local, self.kv_heads_local
        qg = self._linear(x, self.q_weight, self.q_bias, self.q_fp8).view(B, T, len(hs), 2 * Dh)
        q = self.q_norm(qg[..., :Dh]).transpose(1, 2)
        gate = qg[..., Dh:].reshape(B, T, len(hs) * Dh)
        k = self._linear(x, self.k_weight, self.k_bias, self.k_fp8).view(B, T, len(kvs), Dh)
        v = self._linear(x, self.v_weight, self.v_bias, self.v_fp8).view(B, T, len(kvs), Dh)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self._static_rope_tables(int(k_buf.shape[2]), x.device, x.dtype)
        pos2 = torch.stack([position, position + 1])  # [2] absolute rows
        row_cos = cos.index_select(0, pos2)
        row_sin = sin.index_select(0, pos2)
        q = _apply_rope_with_tables(q, row_cos, row_sin, self.rotary_dim)
        k = _apply_rope_with_tables(k, row_cos, row_sin, self.rotary_dim)
        k_buf.index_copy_(2, pos2, k)
        v_buf.index_copy_(2, pos2, v)
        k_rep = repeat_kv(k_buf, self.n_rep)
        v_rep = repeat_kv(v_buf, self.n_rep)
        t_k = int(k_buf.shape[2])
        k_pos = torch.arange(t_k, device=q.device)
        q_pos = pos2  # [2] absolute query positions
        future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [2, T_k]
        if self.attn_implementation == "sdpa":
            mask = torch.zeros(T, t_k, device=q.device, dtype=q.dtype)
            mask = mask.masked_fill(future, float("-inf"))
            attn = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=mask.view(1, 1, T, t_k),
                is_causal=False, scale=self.scaling)
        else:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (Dh ** 0.5)
            scores = scores.masked_fill(future.view(1, 1, T, t_k), float("-inf"))
            attn = torch.matmul(torch.softmax(scores, dim=-1), v_rep)
        merged = attn.transpose(1, 2).contiguous().view(B, T, -1)
        if self.gate_activation == "sigmoid":
            merged = merged * torch.sigmoid(gate)
        elif self.gate_activation == "silu":
            merged = merged * F.silu(gate)
        partial = self._linear(merged, self.o_weight, None, self.o_fp8)
        return self.ctx.all_reduce_sum(partial)


class DistributedGatedDeltaNet(nn.Module):
    """Distributed gated-delta decode step: this rank's value heads only.

    This is the clean lift of one rank's branch of the simulated
    ``TPGatedDeltaNet.step``: identical per-rank maths (segmented qkv
    shard, conv step, gated-delta recurrence over local value heads,
    gated norm), with the final ``out_proj`` partial combined by a
    real all-reduce instead of an in-process sum. State is this rank's
    ``(S_r, conv_r)`` shard (see ``tensor_parallel.shard_linear_state``).
    """

    def __init__(self, ctx: DistributedTPContext, dense: Any,
                 config: Any = None):
        super().__init__()
        self.ctx = ctx
        cfg = config if config is not None else {
            "hidden_size": dense.d_model, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": dense.key_dim,
            "intermediate_size": dense.d_model,
            "linear_num_key_heads": dense.num_key_heads,
            "linear_key_head_dim": dense.key_dim,
            "linear_num_value_heads": dense.num_value_heads,
            "linear_value_head_dim": dense.value_dim}
        # Build segmented/head shards via the state-dict specs. Conv1d
        # weight is [C,1,K]; the segmented spec slices dim 0.
        sd = {"linear_attn.in_proj_qkv.weight": dense.in_proj_qkv.weight.data,
              "linear_attn.conv1d.weight": dense.conv1d.weight.data,
              "linear_attn.in_proj_z.weight": dense.in_proj_z.weight.data,
              "linear_attn.in_proj_a.weight": dense.in_proj_a.weight.data,
              "linear_attn.in_proj_b.weight": dense.in_proj_b.weight.data,
              "linear_attn.A_log": dense.A_log.data,
              "linear_attn.dt_bias": dense.dt_bias.data,
              "linear_attn.out_proj.weight": dense.out_proj.weight.data}
        sh = shard_state_dict(sd, ctx.rank, ctx.world_size, cfg)
        dev = ctx.device
        self.in_proj_qkv_w = nn.Parameter(sh["linear_attn.in_proj_qkv.weight"].to(dev))
        self.conv_weight = nn.Parameter(sh["linear_attn.conv1d.weight"].to(dev))
        self.in_proj_z_w = nn.Parameter(sh["linear_attn.in_proj_z.weight"].to(dev))
        self.in_proj_a_w = nn.Parameter(sh["linear_attn.in_proj_a.weight"].to(dev))
        self.in_proj_b_w = nn.Parameter(sh["linear_attn.in_proj_b.weight"].to(dev))
        self.A_log_r = nn.Parameter(sh["linear_attn.A_log"].to(dev))
        self.dt_bias_r = nn.Parameter(sh["linear_attn.dt_bias"].to(dev))
        self.out_proj_w = nn.Parameter(sh["linear_attn.out_proj.weight"].to(dev))
        self.norm = copy.deepcopy(dense.norm).to(dev)
        self.num_value_heads = int(dense.num_value_heads)
        self.vh_per = self.num_value_heads // ctx.world_size
        self.n_rep = int(dense.n_rep); self.key_dim = int(dense.key_dim)
        self.value_dim = int(dense.value_dim); self.conv_kernel = int(dense.conv_kernel)
        # Local (rank) shard shapes, for reporting/tests.
        self.shard_shapes = {k: tuple(v.shape) for k, v in sh.items()}
        self.in_proj_qkv_fp8: BlockFP8Linear | None = None
        self.in_proj_z_fp8: BlockFP8Linear | None = None
        self.out_proj_fp8: BlockFP8Linear | None = None

    @staticmethod
    def _linear(x: torch.Tensor, weight: Any, fp8: Any) -> torch.Tensor:
        return fp8(x) if fp8 is not None else F.linear(x, weight)

    @classmethod
    def empty(cls, ctx: Any, config: Any = None, fp8: bool = False,
              use_triton: bool | None = None) -> "DistributedGatedDeltaNet":
        """Shard-shaped empty gated-delta layer on meta (loader fills it)."""
        from .hybrid import HybridConfig
        from .linear_attention import RMSNormGated
        cfg = config if isinstance(config, HybridConfig) else HybridConfig()
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.ctx = ctx
        D = int(cfg.hidden_size)
        world, rank = int(ctx.world_size), int(ctx.rank)
        meta = torch.device("meta")
        kt = int(cfg.linear_num_key_heads) * int(cfg.linear_key_head_dim)
        vt = int(cfg.linear_num_value_heads) * int(cfg.linear_value_head_dim)
        qkv_shape = shard_spec_for("linear_attn.in_proj_qkv.weight", (2 * kt + vt, D), cfg).shard_shape(rank, world)
        z_shape = shard_spec_for("linear_attn.in_proj_z.weight", (vt, D), cfg).shard_shape(rank, world)
        out_shape = shard_spec_for("linear_attn.out_proj.weight", (D, vt), cfg).shard_shape(rank, world)
        a_shape = shard_spec_for("linear_attn.in_proj_a.weight", (int(cfg.linear_num_value_heads), D), cfg).shard_shape(rank, world)
        conv_shape = shard_spec_for("linear_attn.conv1d.weight", (2 * kt + vt, 1, int(cfg.linear_conv_kernel_dim)), cfg).shard_shape(rank, world)
        if fp8:
            obj.in_proj_qkv_w = None; obj.in_proj_z_w = None; obj.out_proj_w = None
            obj.in_proj_qkv_fp8 = BlockFP8Linear(D, qkv_shape[0], bias=False, device=meta, use_triton=use_triton)
            obj.in_proj_z_fp8 = BlockFP8Linear(D, z_shape[0], bias=False, device=meta, use_triton=use_triton)
            obj.out_proj_fp8 = BlockFP8Linear(out_shape[1], D, bias=False, device=meta, use_triton=use_triton)
        else:
            obj.in_proj_qkv_w = nn.Parameter(torch.empty(qkv_shape, device=meta))
            obj.in_proj_z_w = nn.Parameter(torch.empty(z_shape, device=meta))
            obj.out_proj_w = nn.Parameter(torch.empty(out_shape, device=meta))
            obj.in_proj_qkv_fp8 = obj.in_proj_z_fp8 = obj.out_proj_fp8 = None
        obj.conv_weight = nn.Parameter(torch.empty(conv_shape, device=meta))
        obj.in_proj_a_w = nn.Parameter(torch.empty(a_shape, device=meta))
        obj.in_proj_b_w = nn.Parameter(torch.empty(a_shape, device=meta))
        vh_per = int(cfg.linear_num_value_heads) // world
        obj.A_log_r = nn.Parameter(torch.empty((vh_per,), device=meta))
        obj.dt_bias_r = nn.Parameter(torch.empty((vh_per,), device=meta))
        obj.norm = RMSNormGated(int(cfg.linear_value_head_dim), eps=float(cfg.rms_norm_eps)).to(meta)
        obj.num_value_heads = int(cfg.linear_num_value_heads)
        obj.vh_per = vh_per
        obj.n_rep = int(cfg.linear_num_value_heads) // int(cfg.linear_num_key_heads)
        obj.key_dim = int(cfg.linear_key_head_dim)
        obj.value_dim = int(cfg.linear_value_head_dim)
        obj.conv_kernel = int(cfg.linear_conv_kernel_dim)
        obj.shard_shapes = {"linear_attn.in_proj_qkv.weight": qkv_shape,
                            "linear_attn.in_proj_z.weight": z_shape,
                            "linear_attn.out_proj.weight": out_shape}
        return obj

    def step(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
             ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One decode token. ``state`` is this rank's ``(S_r, conv_r)``."""
        from .linear_attention import _l2norm
        squeeze = False
        if x_t.dim() == 2:
            x2 = x_t
        elif x_t.dim() == 3 and x_t.shape[1] == 1:
            x2, squeeze = x_t[:, 0, :], True
        else:
            raise ValueError(f"x_t must be [B, D] or [B, 1, D], got {tuple(x_t.shape)}")
        B = int(x2.shape[0])
        S_r, conv_r = state
        Dk, Dv = self.key_dim, self.value_dim
        mixed = self._linear(x2, self.in_proj_qkv_w, self.in_proj_qkv_fp8)
        z = self._linear(x2, self.in_proj_z_w, self.in_proj_z_fp8).reshape(B, self.vh_per, Dv)
        beta = torch.sigmoid(F.linear(x2, self.in_proj_b_w))
        a = F.linear(x2, self.in_proj_a_w)
        g = -self.A_log_r.float().exp() * F.softplus(a.float() + self.dt_bias_r.float())
        if self.conv_kernel == 1:
            window = mixed.unsqueeze(-1); new_buf = conv_r
        else:
            buf = conv_r.to(dtype=mixed.dtype)
            window = torch.cat([buf, mixed.unsqueeze(-1)], dim=-1)
            new_buf = window[:, :, 1:]
        w = self.conv_weight[:, 0, :].to(dtype=window.dtype)
        mixed_act = F.silu((window * w.unsqueeze(0)).sum(dim=-1))
        kh_per = self.vh_per // self.n_rep
        q_sz, v_sz = kh_per * Dk, self.vh_per * Dv
        q_raw, k_raw, v = torch.split(mixed_act, [q_sz, q_sz, v_sz], dim=-1)
        q = q_raw.reshape(B, kh_per, Dk); k = k_raw.reshape(B, kh_per, Dk)
        v = v.reshape(B, self.vh_per, Dv).float()
        if self.n_rep > 1:
            q = q.repeat_interleave(self.n_rep, dim=1)
            k = k.repeat_interleave(self.n_rep, dim=1)
        q = _l2norm(q) * (Dk ** -0.5); k = _l2norm(k)
        S = S_r.float() * torch.exp(g.float())[:, :, None, None]
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.float().unsqueeze(-1)
        S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
        core = (S * q.unsqueeze(-1)).sum(dim=-2)
        core = self.norm(core.reshape(B * self.vh_per, Dv), z.reshape(B * self.vh_per, Dv))
        core = core.reshape(B, v_sz)
        partial = (self.out_proj_fp8(core) if self.out_proj_fp8 is not None
                   else F.linear(core.to(self.out_proj_w.dtype), self.out_proj_w))
        out = self.ctx.all_reduce_sum(partial).to(x_t.dtype)
        if squeeze:
            out = out.unsqueeze(1)
        return out, (S, new_buf.to(dtype=mixed.dtype))

    def zero_state(self, batch_size: int, device: torch.device | None = None,
                   dtype: torch.dtype = torch.float32,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's zero ``(S_r, conv_r)`` for a fresh sequence."""
        dev = device if device is not None else self.ctx.device
        conv_dim_local = int(self.in_proj_qkv_fp8.out_features
                             if self.in_proj_qkv_fp8 is not None
                             else self.in_proj_qkv_w.shape[0])
        S = torch.zeros(batch_size, self.vh_per, self.key_dim, self.value_dim,
                        device=dev, dtype=torch.float32)
        conv = torch.zeros(batch_size, conv_dim_local,
                           max(self.conv_kernel - 1, 0), device=dev, dtype=dtype)
        return (S, conv)

    def step_inplace_static(self, x_t: torch.Tensor, S: torch.Tensor,
                            conv_buf: torch.Tensor) -> torch.Tensor:
        """In-place twin of :meth:`step`: mutate ``S`` / ``conv_buf``.

        Same maths and reduction order as :meth:`step`, but the conv
        ring is shifted with ``copy_`` and the gated-delta update is
        ``S.mul_`` then ``S.add_`` on the caller's preallocated tensors
        (the dense ``GatedDeltaNet.step_inplace`` pattern, lifted to
        this rank's head slices). No ``torch.cat``, no new state
        tensor. Returns the all-reduced output, rank preserved.
        """
        from .linear_attention import _l2norm
        squeeze = False
        if x_t.dim() == 2:
            x2 = x_t
        elif x_t.dim() == 3 and x_t.shape[1] == 1:
            x2, squeeze = x_t[:, 0, :], True
        else:
            raise ValueError(f"x_t must be [B, D] or [B, 1, D], got {tuple(x_t.shape)}")
        B = int(x2.shape[0])
        Dk, Dv = self.key_dim, self.value_dim
        if tuple(S.shape) != (B, self.vh_per, Dk, Dv):
            raise ValueError(f"S must be {(B, self.vh_per, Dk, Dv)}, got {tuple(S.shape)}")
        mixed = self._linear(x2, self.in_proj_qkv_w, self.in_proj_qkv_fp8)
        z = self._linear(x2, self.in_proj_z_w, self.in_proj_z_fp8).reshape(B, self.vh_per, Dv)
        beta = torch.sigmoid(F.linear(x2, self.in_proj_b_w))
        a = F.linear(x2, self.in_proj_a_w)
        g = -self.A_log_r.float().exp() * F.softplus(a.float() + self.dt_bias_r.float())
        w = self.conv_weight[:, 0, :]
        if self.conv_kernel == 1:
            mixed_act = F.silu(mixed * w[:, 0].to(dtype=mixed.dtype).unsqueeze(0))
        else:
            if tuple(conv_buf.shape) != (B, int(mixed.shape[-1]), self.conv_kernel - 1):
                raise ValueError(
                    f"conv_buf must be {(B, int(mixed.shape[-1]), self.conv_kernel - 1)}, "
                    f"got {tuple(conv_buf.shape)}")
            window = torch.empty(B, int(mixed.shape[-1]), self.conv_kernel,
                                 device=mixed.device, dtype=mixed.dtype)
            window[:, :, :-1].copy_(conv_buf.to(dtype=mixed.dtype))
            window[:, :, -1].copy_(mixed)
            mixed_act = F.silu(
                (window * w.to(dtype=mixed.dtype).unsqueeze(0)).sum(dim=-1))
            for i in range(conv_buf.shape[-1] - 1):
                conv_buf[:, :, i].copy_(conv_buf[:, :, i + 1])
            if conv_buf.shape[-1] > 0:
                conv_buf[:, :, -1].copy_(mixed.to(dtype=conv_buf.dtype))
        kh_per = self.vh_per // self.n_rep
        q_sz, v_sz = kh_per * Dk, self.vh_per * Dv
        q_raw, k_raw, v = torch.split(mixed_act, [q_sz, q_sz, v_sz], dim=-1)
        q = q_raw.reshape(B, kh_per, Dk)
        k = k_raw.reshape(B, kh_per, Dk)
        v = v.reshape(B, self.vh_per, Dv).float()
        if self.n_rep > 1:
            q = q.repeat_interleave(self.n_rep, dim=1)
            k = k.repeat_interleave(self.n_rep, dim=1)
        q = _l2norm(q) * (Dk ** -0.5)
        k = _l2norm(k)
        S.mul_(torch.exp(g.float())[:, :, None, None].to(dtype=S.dtype))
        kv_mem = (S.float() * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.float().unsqueeze(-1)
        S.add_((k.unsqueeze(-1) * delta.unsqueeze(-2)).to(dtype=S.dtype))
        core = (S.float() * q.unsqueeze(-1)).sum(dim=-2)
        core = self.norm(core.reshape(B * self.vh_per, Dv), z.reshape(B * self.vh_per, Dv))
        core = core.reshape(B, v_sz)
        partial = (self.out_proj_fp8(core) if self.out_proj_fp8 is not None
                   else F.linear(core.to(self.out_proj_w.dtype), self.out_proj_w))
        out = self.ctx.all_reduce_sum(partial).to(x_t.dtype)
        if squeeze:
            out = out.unsqueeze(1)
        return out

    def step_chunk_inplace_static(self, x: torch.Tensor, S: torch.Tensor,
                                  conv_buf: torch.Tensor) -> torch.Tensor:
        """Two-token chunk twin of :meth:`step_inplace_static`.

        ``x [B, 2, D]`` -> ``[B, 2, D]``. Every projection weight
        (``in_proj_qkv/z/a/b`` and ``out_proj``) is read ONCE for both
        tokens: the input projections run as single M=2 matmuls and
        the output projection as one M=2 matmul over the stacked
        per-token cores (one all-reduce total). Only the conv ring and
        the gated-delta recurrence -- inherently sequential -- step
        token by token over the precomputed projections, mutating
        ``S`` / ``conv_buf`` in place exactly as two single steps.
        """
        from .linear_attention import _l2norm
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(f"x must be [B, 2, D], got {tuple(x.shape)}")
        B, T, _ = x.shape
        Dk, Dv = self.key_dim, self.value_dim
        if tuple(S.shape) != (B, self.vh_per, Dk, Dv):
            raise ValueError(f"S must be {(B, self.vh_per, Dk, Dv)}, got {tuple(S.shape)}")
        # One M=2 matmul per input projection (weights read once).
        mixed = self._linear(x, self.in_proj_qkv_w, self.in_proj_qkv_fp8)  # [B,2,C]
        z = self._linear(x, self.in_proj_z_w, self.in_proj_z_fp8).reshape(B, T, self.vh_per, Dv)
        beta = torch.sigmoid(F.linear(x, self.in_proj_b_w))  # [B,2,vh]
        a = F.linear(x, self.in_proj_a_w)
        g = -self.A_log_r.float().exp() * F.softplus(a.float() + self.dt_bias_r.float())
        w = self.conv_weight[:, 0, :]
        kh_per = self.vh_per // self.n_rep
        q_sz, v_sz = kh_per * Dk, self.vh_per * Dv
        cores: list[torch.Tensor] = []
        for t in range(T):
            mixed_t = mixed[:, t, :]
            if self.conv_kernel == 1:
                mixed_act = F.silu(mixed_t * w[:, 0].to(dtype=mixed.dtype).unsqueeze(0))
            else:
                if tuple(conv_buf.shape) != (B, int(mixed.shape[-1]), self.conv_kernel - 1):
                    raise ValueError(
                        f"conv_buf must be {(B, int(mixed.shape[-1]), self.conv_kernel - 1)}, "
                        f"got {tuple(conv_buf.shape)}")
                window = torch.empty(B, int(mixed.shape[-1]), self.conv_kernel,
                                     device=mixed.device, dtype=mixed.dtype)
                window[:, :, :-1].copy_(conv_buf.to(dtype=mixed.dtype))
                window[:, :, -1].copy_(mixed_t)
                mixed_act = F.silu(
                    (window * w.to(dtype=mixed.dtype).unsqueeze(0)).sum(dim=-1))
                for i in range(conv_buf.shape[-1] - 1):
                    conv_buf[:, :, i].copy_(conv_buf[:, :, i + 1])
                if conv_buf.shape[-1] > 0:
                    conv_buf[:, :, -1].copy_(mixed_t.to(dtype=conv_buf.dtype))
            q_raw, k_raw, v = torch.split(mixed_act, [q_sz, q_sz, v_sz], dim=-1)
            q = q_raw.reshape(B, kh_per, Dk)
            k = k_raw.reshape(B, kh_per, Dk)
            v = v.reshape(B, self.vh_per, Dv).float()
            if self.n_rep > 1:
                q = q.repeat_interleave(self.n_rep, dim=1)
                k = k.repeat_interleave(self.n_rep, dim=1)
            q = _l2norm(q) * (Dk ** -0.5)
            k = _l2norm(k)
            S.mul_(torch.exp(g[:, t].float())[:, :, None, None].to(dtype=S.dtype))
            kv_mem = (S.float() * k.unsqueeze(-1)).sum(dim=-2)
            delta = (v - kv_mem) * beta[:, t].float().unsqueeze(-1)
            S.add_((k.unsqueeze(-1) * delta.unsqueeze(-2)).to(dtype=S.dtype))
            core = (S.float() * q.unsqueeze(-1)).sum(dim=-2)
            core = self.norm(core.reshape(B * self.vh_per, Dv),
                             z[:, t].reshape(B * self.vh_per, Dv))
            cores.append(core.reshape(B, v_sz))
        stacked = torch.stack(cores, dim=1)  # [B, 2, v_sz]
        partial = (self.out_proj_fp8(stacked) if self.out_proj_fp8 is not None
                   else F.linear(stacked.to(self.out_proj_w.dtype), self.out_proj_w))
        return self.ctx.all_reduce_sum(partial).to(x.dtype)

    def forward_prefill_v1(
        self, x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """v1 prefill: token-by-token step loop over ``x [B, T, D]``.

        Chunkwise TP prefill is deliberately *not* implemented here
        (later task); this loop reuses :meth:`step` exactly, so each
        token costs one all-reduce in ``out_proj`` (``T`` mixer
        all-reduces per prefill, plus the block's single MLP
        all-reduce over the full ``[B, T, D]`` output).
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        B, T, _ = x.shape
        cur = state if state is not None else self.zero_state(B, device=x.device, dtype=x.dtype)
        outs: list[torch.Tensor] = []
        for t in range(T):
            out_t, cur = self.step(x[:, t, :], cur)
            outs.append(out_t)
        out = torch.stack(outs, dim=1) if T > 0 else torch.empty_like(x)
        return out, cur

    # v1 is the only prefill for now; the alias keeps the block code short.
    forward_prefill = forward_prefill_v1


# ---------------------------------------------------------------------------
# Full distributed hybrid model: blocks + LM (tiny-proven, 27B not yet).
# Collective count per block (dense maths implies exactly these):
#   token mixer: 1 all-reduce (o_proj / out_proj row-parallel partial sum)
#   MLP:         1 all-reduce (down_proj row-parallel partial sum)
#   => 2 all-reduces per block per decode step, and per full-attention
#   prefill call. Linear-attention v1 prefill loops step() per token, so
#   its mixer costs T all-reduces (one per token) + 1 MLP all-reduce.
#   Norms, embeddings, residual adds and lm_head are replicated: 0.
# ---------------------------------------------------------------------------


class TPStaticState:
    """Preallocated per-rank state for static (graph-ready) TP decode.

    Allocated once from ``(config, batch_size, max_seq_len, device,
    dtype, ctx)`` and reused for every decode step:

    * full layers: local KV buffers ``kv_k`` / ``kv_v`` (parallel
      lists, ``None`` at linear layers), each
      ``[B, H_kv_local, max_seq_len, Dh]`` -- this rank's KV-head
      slice, slice/``index_copy_``-written in place, never
      concatenated;
    * linear layers: fixed ``S`` (``[B, Hv_local, Dk, Dv]``, fp32 as
      in :meth:`DistributedGatedDeltaNet.zero_state`) and conv ring
      (``[B, C_local, K-1]``, ``C_local`` = this rank's segmented
      qkv rows) in ``linear_S`` / ``linear_conv_buffers``.

    ``position`` is a 0-dim ``torch.long`` tensor on ``device``,
    advanced in place (``add_``) inside
    :meth:`DistributedHybridLM.decode_token_static`; the Python
    ``seq_len`` property reads it (a host sync, for use outside any
    captured graph only). ``reset()`` zeroes every buffer in place,
    so ``data_ptr()`` is stable across steps and across reset.
    """

    def __init__(self, config: Any, batch_size: int = 1,
                 max_seq_len: int | None = None,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None,
                 ctx: Any = None, *, batch: int | None = None):
        if batch is not None:
            batch_size = batch
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_seq_len is None:
            max_seq_len = int(config.max_position_embeddings)
        if int(max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        world = int(ctx.world_size) if ctx is not None else 1
        rank = int(ctx.rank) if ctx is not None else 0
        if device is None:
            device = ctx.device if ctx is not None else torch.device("cpu")
        self.config = config
        self.batch_size = int(batch_size)
        self.batch = self.batch_size
        self.max_seq_len = int(max_seq_len)
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.float32
        self.ctx = ctx
        self.layer_types: list[str] = list(config.layer_types)
        B, Dev, Dtype = self.batch_size, self.device, self.dtype
        Hkv, Dh = int(config.num_key_value_heads), int(config.head_dim)
        if Hkv % world != 0:
            raise ValueError(f"num_key_value_heads {Hkv} not divisible by world {world}")
        hkv_local = Hkv // world
        self.hkv_local = hkv_local
        Hv = int(config.linear_num_value_heads)
        if Hv % world != 0:
            raise ValueError(f"linear_num_value_heads {Hv} not divisible by world {world}")
        vh_per = Hv // world
        self.vh_per = vh_per
        qkv_shape = shard_spec_for(
            "linear_attn.in_proj_qkv.weight",
            (2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
             + Hv * int(config.linear_value_head_dim), int(config.hidden_size)),
            config).shard_shape(rank, world)
        conv_dim_local = int(qkv_shape[0])
        self.conv_dim_local = conv_dim_local
        self.kv_k: list[torch.Tensor | None] = []
        self.kv_v: list[torch.Tensor | None] = []
        self.linear_S: list[torch.Tensor | None] = []
        self.linear_conv_buffers: list[torch.Tensor | None] = []
        for t in self.layer_types:
            if t == "full_attention":
                self.kv_k.append(torch.zeros(B, hkv_local, self.max_seq_len, Dh,
                                             device=Dev, dtype=Dtype))
                self.kv_v.append(torch.zeros(B, hkv_local, self.max_seq_len, Dh,
                                             device=Dev, dtype=Dtype))
                self.linear_S.append(None)
                self.linear_conv_buffers.append(None)
            else:
                self.kv_k.append(None)
                self.kv_v.append(None)
                self.linear_S.append(torch.zeros(
                    B, vh_per, int(config.linear_key_head_dim),
                    int(config.linear_value_head_dim), device=Dev, dtype=torch.float32))
                self.linear_conv_buffers.append(torch.zeros(
                    B, conv_dim_local, max(int(config.linear_conv_kernel_dim) - 1, 0),
                    device=Dev, dtype=Dtype))
        self.position = torch.zeros((), dtype=torch.long, device=Dev)
        self.position_tensor = self.position

    @property
    def seq_len(self) -> int:
        """Positions processed (host sync; read only outside a graph)."""
        return int(self.position.item())

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        self.position.fill_(int(value))

    def __len__(self) -> int:
        return len(self.layer_types)

    def reset(self) -> None:
        """Zero all buffers in place; every ``data_ptr`` is preserved."""
        for t in self.kv_k:
            if t is not None:
                t.zero_()
        for t in self.kv_v:
            if t is not None:
                t.zero_()
        for t in self.linear_S:
            if t is not None:
                t.zero_()
        for t in self.linear_conv_buffers:
            if t is not None:
                t.zero_()
        self.position.zero_()

    def snapshot(self) -> "TPStaticState":
        """Independent deep copy (cloned buffers; the only copy path).

        Used by speculative verification: the pre-cycle snapshot is
        restored with :meth:`copy_from` when a draft is rejected, so
        the linear recurrent/conv state that cannot be truncated by
        position is rewound exactly (full-attention KV in the snapshot
        is copied too, keeping one uniform restore).
        """
        out = object.__new__(TPStaticState)
        out.config, out.batch_size, out.batch = self.config, self.batch_size, self.batch
        out.max_seq_len, out.device, out.dtype = self.max_seq_len, self.device, self.dtype
        out.ctx, out.layer_types = self.ctx, list(self.layer_types)
        out.hkv_local, out.vh_per = self.hkv_local, self.vh_per
        out.conv_dim_local = self.conv_dim_local
        out.kv_k = [t.clone() if t is not None else None for t in self.kv_k]
        out.kv_v = [t.clone() if t is not None else None for t in self.kv_v]
        out.linear_S = [t.clone() if t is not None else None for t in self.linear_S]
        out.linear_conv_buffers = [
            t.clone() if t is not None else None for t in self.linear_conv_buffers]
        out.position = self.position.clone()
        out.position_tensor = out.position
        return out

    clone = snapshot

    def alloc_snapshot(self) -> "TPStaticState":
        """Preallocated snapshot buffers (same shapes/ptrs each cycle)."""
        return self.snapshot()

    def snapshot_into(self, dst: "TPStaticState") -> None:
        """Copy only rollback-relevant state into preallocated ``dst``.

        Linear recurrent (``linear_S``) + conv buffers and ``position``
        are copied with ``copy_`` (no allocation, CUDA-graph capturable).
        Full-attention KV is deliberately skipped: rewinding
        ``position`` makes the next chunk overwrite the stale KV slots
        before they can be read, so copying KV is wasted work.
        """
        if dst.batch_size != self.batch_size or dst.max_seq_len != self.max_seq_len:
            raise ValueError("snapshot shape (batch/max_seq_len) does not match this state")
        if len(dst.layer_types) != len(self.layer_types):
            raise ValueError("snapshot layer count does not match this state")
        for d, s in zip(dst.linear_S, self.linear_S):
            if (d is None) != (s is None):
                raise ValueError("snapshot linear_S layout does not match this state")
            if d is not None:
                d.copy_(s)
        for d, s in zip(dst.linear_conv_buffers, self.linear_conv_buffers):
            if (d is None) != (s is None):
                raise ValueError("snapshot linear_conv layout does not match this state")
            if d is not None:
                d.copy_(s)
        dst.position.copy_(self.position)

    def restore_linear_from(self, src: "TPStaticState") -> None:
        """Restore linear/conv/position from ``src`` (KV left as-is)."""
        src.snapshot_into(self)

    def copy_from(self, snapshot: "TPStaticState") -> None:
        """Copy ``snapshot`` back into this state in place (ptrs stable)."""
        if snapshot.batch_size != self.batch_size or snapshot.max_seq_len != self.max_seq_len:
            raise ValueError("snapshot shape (batch/max_seq_len) does not match this state")
        for dst, src in zip(self.kv_k, snapshot.kv_k):
            if dst is not None and src is not None:
                dst.copy_(src)
        for dst, src in zip(self.kv_v, snapshot.kv_v):
            if dst is not None and src is not None:
                dst.copy_(src)
        for dst, src in zip(self.linear_S, snapshot.linear_S):
            if dst is not None and src is not None:
                dst.copy_(src)
        for dst, src in zip(self.linear_conv_buffers, snapshot.linear_conv_buffers):
            if dst is not None and src is not None:
                dst.copy_(src)
        self.position.copy_(snapshot.position)

    restore_from = copy_from

    def data_ptrs(self) -> dict[str, list[int]]:
        ptrs: dict[str, list[int]] = {"kv_k": [], "kv_v": [], "linear_S": [],
                                      "linear_conv": [], "position": [self.position.data_ptr()]}
        for t in self.kv_k:
            if t is not None:
                ptrs["kv_k"].append(t.data_ptr())
        for t in self.kv_v:
            if t is not None:
                ptrs["kv_v"].append(t.data_ptr())
        for t in self.linear_S:
            if t is not None:
                ptrs["linear_S"].append(t.data_ptr())
        for t in self.linear_conv_buffers:
            if t is not None:
                ptrs["linear_conv"].append(t.data_ptr())
        return ptrs

    def __repr__(self) -> str:
        return (f"TPStaticState(batch={self.batch_size}, "
                f"max_seq_len={self.max_seq_len}, seq_len={self.seq_len}, "
                f"layers={len(self.layer_types)})")


class TPState:
    """Per-rank inference state for :class:`DistributedHybridLM`.

    ``entries`` holds one ``(tag, payload)`` per layer:
    ``("kv", (k_local, v_local))`` for full layers, where each tensor is
    ``[B, H_kv_local, T_filled, head_dim]`` this rank's KV-head slice
    (``None`` payload = nothing cached yet), and
    ``("linear", (S_r, conv_r))`` for linear layers, this rank's
    value-head shard (``None`` = lazy zero state). ``seq_len`` is the
    number of positions already processed.
    """

    def __init__(self, entries: list[tuple[str, Any]] | None = None,
                 seq_len: int = 0, layer_types: list[str] | None = None):
        self.entries: list[tuple[str, Any]] = list(entries) if entries is not None else []
        self.seq_len = int(seq_len)
        self.layer_types: list[str] | None = list(layer_types) if layer_types is not None else None

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, index: int) -> tuple[str, Any]:
        return self.entries[index]

    @classmethod
    def empty(cls, layer_types: list[str], seq_len: int = 0) -> "TPState":
        entries: list[tuple[str, Any]] = [
            ("linear", None) if t == "linear_attention" else ("kv", None)
            for t in layer_types]
        return cls(entries=entries, seq_len=seq_len, layer_types=list(layer_types))

    def clone(self) -> "TPState":
        out: list[tuple[str, Any]] = []
        for tag, payload in self.entries:
            if payload is None:
                out.append((tag, None))
            else:
                out.append((tag, (payload[0].clone(), payload[1].clone())))
        return TPState(entries=out, seq_len=self.seq_len, layer_types=self.layer_types)

    snapshot = clone

    def __repr__(self) -> str:
        tags = ",".join(t for t, _ in self.entries)
        return f"TPState(seq_len={self.seq_len}, entries=[{tags}])"


class DistributedHybridBlock(nn.Module):
    """One distributed hybrid block: replicated norms, sharded mixer+MLP.

    ``input_layernorm`` / ``post_attention_layernorm`` are replicated
    :class:`~inference_from_scratch.hybrid.QwenRMSNorm` copies; the token
    mixer is :class:`DistributedFullAttention` or
    :class:`DistributedGatedDeltaNet` (one all-reduce) and ``mlp`` is
    :class:`DistributedSwiGLU` (one all-reduce). See the collective
    count documented above this section.
    """

    def __init__(self, ctx: DistributedTPContext, dense: Any, config: Any = None):
        super().__init__()
        self.ctx = ctx
        self.layer_type = str(dense.layer_type)
        self.layer_idx = int(dense.layer_idx)
        dev = ctx.device
        self.input_layernorm = copy.deepcopy(dense.input_layernorm).to(dev)
        self.post_attention_layernorm = copy.deepcopy(dense.post_attention_layernorm).to(dev)
        if self.layer_type == "full_attention":
            self.self_attn: DistributedFullAttention | None = DistributedFullAttention(
                ctx, dense.self_attn, config=config)
            self.linear_attn: DistributedGatedDeltaNet | None = None
        else:
            self.self_attn = None
            self.linear_attn = DistributedGatedDeltaNet(ctx, dense.linear_attn, config=config)
        self.mlp = DistributedSwiGLU.from_dense(dense.mlp, ctx, config)

    @classmethod
    def empty(cls, ctx: Any, config: Any, layer_idx: int, fp8: bool = False,
              use_triton: bool | None = None) -> "DistributedHybridBlock":
        """Shard-shaped empty block on meta (loader fills it)."""
        from .hybrid import QwenRMSNorm
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.ctx = ctx
        obj.layer_type = str(config.layer_types[int(layer_idx)])
        obj.layer_idx = int(layer_idx)
        meta = torch.device("meta")
        obj.input_layernorm = QwenRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps)).to(meta)
        obj.post_attention_layernorm = QwenRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps)).to(meta)
        if obj.layer_type == "full_attention":
            obj.self_attn = DistributedFullAttention.empty(ctx, config, fp8=fp8, use_triton=use_triton)
            obj.linear_attn = None
        else:
            obj.self_attn = None
            obj.linear_attn = DistributedGatedDeltaNet.empty(ctx, config, fp8=fp8, use_triton=use_triton)
        obj.mlp = DistributedSwiGLU.empty(ctx, config, fp8=fp8, use_triton=use_triton)
        return obj

    @property
    def mixer(self) -> nn.Module:
        m = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        assert m is not None
        return m

    def forward_prefill(self, x: torch.Tensor, layer_state: Any = None,
                        start_pos: int = 0,
                        position_ids: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, Any]:
        """Prefill hidden ``x [B,T,D]`` -> ``(out, new_layer_state)``."""
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            mixed, new_state = self.self_attn.forward_with_state(
                h, kv_state=layer_state, start_pos=start_pos, position_ids=position_ids)
        else:
            assert self.linear_attn is not None
            mixed, new_state = self.linear_attn.forward_prefill_v1(h, state=layer_state)
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_state

    def step(self, x: torch.Tensor, layer_state: Any = None, start_pos: int = 0,
             ) -> tuple[torch.Tensor, Any]:
        """Decode one token: hidden ``x [B,1,D]`` -> ``(out, new_state)``."""
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            mixed, new_state = self.self_attn.forward_with_state(
                h, kv_state=layer_state, start_pos=start_pos)
        else:
            assert self.linear_attn is not None
            cur = layer_state
            if cur is None:
                cur = self.linear_attn.zero_state(x.shape[0], device=x.device, dtype=x.dtype)
            mixed3, new_state = self.linear_attn.step(h, cur)
            mixed = mixed3
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_state

    def forward(self, x: torch.Tensor, layer_state: Any = None, start_pos: int = 0,
                position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, Any]:
        return self.forward_prefill(x, layer_state=layer_state, start_pos=start_pos,
                                    position_ids=position_ids)

    def step_static(self, x: torch.Tensor, state: "TPStaticState",
                    rope_position: torch.Tensor | None = None) -> torch.Tensor:
        """Static decode: hidden ``x [B,1,D]`` over preallocated state.

        Same norms/residuals/MLP as :meth:`step`; the mixer writes
        into ``state`` in place at ``state.position`` (not advanced
        here -- the LM advances it once after all blocks).
        ``rope_position`` is forwarded to the full-attention mixer
        (see :meth:`DistributedFullAttention.step_static`).
        """
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            k_buf, v_buf = state.kv_k[self.layer_idx], state.kv_v[self.layer_idx]
            assert k_buf is not None and v_buf is not None
            mixed = self.self_attn.step_static(h, k_buf, v_buf, state.position,
                                               rope_position=rope_position)
        else:
            assert self.linear_attn is not None
            S, conv = state.linear_S[self.layer_idx], state.linear_conv_buffers[self.layer_idx]
            assert S is not None and conv is not None
            mixed = self.linear_attn.step_inplace_static(h, S, conv)
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def step_chunk_static(self, x: torch.Tensor, state: "TPStaticState") -> torch.Tensor:
        """Static 2-token chunk: hidden ``x [B,2,D]`` over preallocated state.

        Chunk twin of :meth:`step_static` (position not advanced here):
        full layers write both K/V and attend causally, linear layers
        read each projection weight once over M=2 and step only the
        conv/recurrence sequentially, the MLP runs once over M=2.
        """
        h = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            k_buf, v_buf = state.kv_k[self.layer_idx], state.kv_v[self.layer_idx]
            assert k_buf is not None and v_buf is not None
            mixed = self.self_attn.step_chunk_static(h, k_buf, v_buf, state.position)
        else:
            assert self.linear_attn is not None
            S, conv = state.linear_S[self.layer_idx], state.linear_conv_buffers[self.layer_idx]
            assert S is not None and conv is not None
            mixed = self.linear_attn.step_chunk_inplace_static(h, S, conv)
        x = x + mixed
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x



LM_HEAD_MODES = ("replicated", "vocab")


def vocab_shard_layout(vocab_size: int, world: int) -> tuple[int, int]:
    """(padded_vocab, shard_rows) for vocab-parallel lm_head.

    Vocab is padded up to a multiple of ``world`` so every rank holds
    exactly ``shard_rows`` rows (last rank zero-pads past ``vocab_size``;
    padded logits are trimmed / masked to -inf for argmax). Real config
    248320 % 2 == 0, so no padding there.
    """
    v, w = int(vocab_size), int(world)
    if v <= 0 or w <= 0:
        raise ValueError(f"vocab/world must be positive, got {v}/{w}")
    shard = -(-v // w)
    return shard * w, shard


def _vocab_range(vocab_size: int, rank: int, world: int) -> tuple[int, int, int]:
    padded, shard = vocab_shard_layout(vocab_size, world)
    start = int(rank) * shard
    return start, min(start + shard, int(vocab_size)), shard


class DistributedHybridLM(nn.Module):
    """Distributed hybrid LM mirroring ``TinyHybridLM``'s API.

    Replicated: ``embed_tokens``, per-block norms, final ``norm`` and
    ``lm_head`` (vocab-parallel embedding/head explicitly deferred).
    Sharded per rank: every mixer and MLP projection (via
    :class:`DistributedHybridBlock`). Each rank holds ONLY its shards
    plus the replicated tensors.

    API: ``prefill(idx, position_ids=None) -> (logits, TPState)`` and
    ``step(token, state) -> (logits, state)``, where ``TPState`` holds
    per-rank local KV (full layers) and local ``(S, conv)`` (linear
    layers). Linear prefill is v1 (token-by-token step loop, labelled
    in :meth:`DistributedGatedDeltaNet.forward_prefill_v1`); chunkwise
    TP prefill is a later task.
    """

    def __init__(self, ctx: DistributedTPContext, dense: Any, config: Any = None,
                 lm_head_mode: str = "replicated"):
        super().__init__()
        if lm_head_mode not in LM_HEAD_MODES:
            raise ValueError(f"lm_head_mode must be one of {LM_HEAD_MODES}, got {lm_head_mode!r}")
        if lm_head_mode == "vocab" and bool(getattr(config if config is not None else dense.config, "tie_word_embeddings", False)):
            raise ValueError("lm_head_mode='vocab' requires untied embeddings "
                             "(tied lm_head would need the full replicated weight)")
        self.ctx = ctx
        self.config = config if config is not None else dense.config
        self.lm_head_mode = str(lm_head_mode)
        dev = ctx.device
        cfg = self.config
        self._init_vocab_layout()
        # Replicated embedding (whole copy per rank; stays replicated
        # in vocab mode -- only lm_head rows are sharded).
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(dense.embed_tokens.weight.data.to(dev))
        self.embed_tokens = self.embed_tokens.to(dev)
        self.layers = nn.ModuleList(
            [DistributedHybridBlock(ctx, block, config=cfg) for block in dense.layers])
        self.norm = copy.deepcopy(dense.norm).to(dev)
        if self.lm_head_mode == "vocab":
            start, end, shard = _vocab_range(int(cfg.vocab_size), int(ctx.rank), int(ctx.world_size))
            self.lm_head = nn.Linear(cfg.hidden_size, shard, bias=False)
            with torch.no_grad():
                w = torch.zeros((shard, int(cfg.hidden_size)), dtype=dense.lm_head.weight.dtype)
                if end > start:
                    w[: end - start].copy_(dense.lm_head.weight.data[start:end].to("cpu"))
                self.lm_head.weight.copy_(w.to(dev))
            self.lm_head = self.lm_head.to(dev)
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            with torch.no_grad():
                if dense.lm_head.weight.data_ptr() == dense.embed_tokens.weight.data_ptr():
                    self.lm_head.weight = self.embed_tokens.weight  # tied
                else:
                    self.lm_head.weight.copy_(dense.lm_head.weight.data.to(dev))
            self.lm_head = self.lm_head.to(dev) if not cfg.tie_word_embeddings else self.lm_head
        self.mtp_draft = None  # attached by empty(include_mtp=True) / loader

    @classmethod
    def empty(cls, config: Any, rank: int, world: int, fp8_mode: str = "resident",
              use_triton: bool | None = None, lm_head_mode: str = "replicated",
              include_mtp: bool = False) -> "DistributedHybridLM":
        """Shard-shaped empty LM on meta for rank ``rank`` (loader fills it).

        ``fp8_mode="resident"`` builds resident-FP8 projections
        (:class:`BlockFP8Linear` shards); ``"dequantize"`` builds dense
        shard Parameters. No tensor data is allocated (meta device).
        Call :meth:`set_ctx` with a real context before forwarding.
        """
        if fp8_mode not in ("resident", "dequantize"):
            raise ValueError(f"fp8_mode must be 'resident' or 'dequantize', got {fp8_mode!r}")
        if lm_head_mode not in LM_HEAD_MODES:
            raise ValueError(f"lm_head_mode must be one of {LM_HEAD_MODES}, got {lm_head_mode!r}")
        if lm_head_mode == "vocab" and bool(getattr(config, "tie_word_embeddings", False)):
            raise ValueError("lm_head_mode='vocab' requires untied embeddings")
        from .hybrid import QwenRMSNorm
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        ctx = _LocalContext(int(rank), int(world), torch.device("meta"))
        obj.ctx = ctx  # type: ignore[assignment]
        obj.config = config
        obj.lm_head_mode = str(lm_head_mode)
        obj._init_vocab_layout()
        obj.fp8_mode = fp8_mode
        fp8 = fp8_mode == "resident"
        meta = torch.device("meta")
        obj.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        obj.embed_tokens.weight = nn.Parameter(
            torch.empty((int(config.vocab_size), int(config.hidden_size)), device=meta))
        obj.layers = nn.ModuleList([
            DistributedHybridBlock.empty(ctx, config, i, fp8=fp8, use_triton=use_triton)
            for i in range(len(config.layer_types))])
        obj.norm = QwenRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps)).to(meta)
        if obj.lm_head_mode == "vocab":
            _, _, shard = _vocab_range(int(config.vocab_size), int(rank), int(world))
            obj.lm_head = nn.Linear(int(config.hidden_size), shard, bias=False)
            obj.lm_head.weight = nn.Parameter(
                torch.empty((shard, int(config.hidden_size)), device=meta))
        else:
            obj.lm_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)
            obj.lm_head.weight = nn.Parameter(
                torch.empty((int(config.vocab_size), int(config.hidden_size)), device=meta))
            if config.tie_word_embeddings:
                obj.lm_head.weight = obj.embed_tokens.weight
        obj.mtp_draft = None
        if include_mtp:
            if int(getattr(config, "mtp_num_hidden_layers", 0)) <= 0:
                raise ValueError("include_mtp=True but config has no MTP layers")
            from .mtp import MTPConfig, TPMTPDraftModule
            obj.mtp_draft = TPMTPDraftModule.empty(
                ctx, MTPConfig.from_config(config), fp8=fp8)
        return obj

    def set_ctx(self, ctx: Any) -> "DistributedHybridLM":
        """Point every submodule's context at ``ctx`` (post-load)."""
        self.ctx = ctx
        for module in self.modules():
            if "ctx" in module.__dict__:
                module.ctx = ctx
        return self

    @classmethod
    def from_dense(cls, model: Any, ctx: DistributedTPContext,
                   config: Any = None, lm_head_mode: str = "replicated") -> "DistributedHybridLM":
        """Build this rank's model by slicing ``model`` (dense TinyHybridLM).

        Every sharded weight is cut via ``shard_state_dict`` specs (the
        same specs as the simulated runtime), so this rank stores only
        its slice; replicated tensors are cloned whole.
        """
        return cls(ctx, model, config=config, lm_head_mode=lm_head_mode)

    def local_parameter_bytes(self) -> int:
        """Bytes held on THIS rank (params + FP8 buffers, shards+replicated)."""
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        total += sum(b.numel() * b.element_size() for b in self.buffers())
        return int(total)

    @staticmethod
    def dense_parameter_bytes(model: Any) -> int:
        return int(sum(p.numel() * p.element_size() for p in model.parameters()))


    def _init_vocab_layout(self) -> None:
        cfg = self.config
        world = int(getattr(self.ctx, "world_size", 1))
        rank = int(getattr(self.ctx, "rank", 0))
        padded, shard = vocab_shard_layout(int(cfg.vocab_size), world)
        start, end, _ = _vocab_range(int(cfg.vocab_size), rank, world)
        self.vocab_padded = padded
        self.vocab_shard_size = shard
        self.vocab_start = start
        self.vocab_end = end

    def _all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        ctx = self.ctx
        if int(getattr(ctx, "world_size", 1)) == 1:
            return [tensor]
        if hasattr(ctx, "all_gather"):
            return ctx.all_gather(tensor)  # type: ignore[no-any-return]
        out = [torch.empty_like(tensor) for _ in range(int(ctx.world_size))]
        dist.all_gather(out, tensor)
        return out

    def _lm_head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full logits from normed hidden ``[..., H]``.

        Replicated: local ``lm_head`` (no collective). Vocab: local
        shard matmul ``[..., shard]`` then ONE all_gather over the
        vocab dim and trim of padding. Collective bytes per call:
        each rank sends ``numel(hidden[..., :1]) * shard * elem_size``
        and receives ``world`` x that (e.g. decode B=1: 124160 BF16
        values sent, 248320 gathered pre-trim at the real config).
        """
        if getattr(self, "lm_head_mode", "replicated") != "vocab":
            return self.lm_head(hidden)
        local = F.linear(hidden, self.lm_head.weight)
        if int(self.ctx.world_size) == 1:
            return local[..., : int(self.config.vocab_size)]
        gathered = self._all_gather(local.contiguous())
        full = torch.cat(gathered, dim=-1)
        return full[..., : int(self.config.vocab_size)]

    def greedy_next_token(self, hidden_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax token from normed last-token hidden ``[B, H]``.

        Vocab mode fast path: local shard logits, local max, then ONE
        small all_gather of ``[B, 2]`` FP32 ``(value, global_index)``
        pairs (8*B bytes/rank vs B*vocab for full logits). Ties go to
        the smallest global index (matches dense ``argmax``). Padded
        rows are masked to -inf. Replicated mode: plain local argmax
        (no collective). Returns ``(tokens [B] int64, top_values [B])``.
        """
        if hidden_last.dim() == 3:
            hidden_last = hidden_last[:, -1, :]
        if hidden_last.dim() != 2:
            raise ValueError(f"hidden_last must be [B, H] or [B, T, H], got {tuple(hidden_last.shape)}")
        if getattr(self, "lm_head_mode", "replicated") != "vocab":
            logits = self.lm_head(hidden_last)
            vals, idx = logits.float().max(dim=-1)
            return idx.to(torch.long), vals
        local = F.linear(hidden_last, self.lm_head.weight).float()
        valid = int(self.vocab_end - self.vocab_start)
        if valid < local.shape[-1]:
            local = local.clone()
            local[:, valid:] = float("-inf")
        vals, local_idx = local.max(dim=-1)
        global_idx = (local_idx + int(self.vocab_start)).to(torch.float32)
        pairs = torch.stack([vals, global_idx], dim=-1)  # [B, 2]
        if int(self.ctx.world_size) == 1:
            return global_idx.to(torch.long), vals
        gathered = self._all_gather(pairs.contiguous())
        best_val = gathered[0][:, 0].clone()
        best_idx = gathered[0][:, 1].clone()
        for g in gathered[1:]:
            better = (g[:, 0] > best_val) | ((g[:, 0] == best_val) & (g[:, 1] < best_idx))
            best_val = torch.where(better, g[:, 0], best_val)
            best_idx = torch.where(better, g[:, 1], best_idx)
        return best_idx.to(torch.long), best_val

    def set_scale_cache(self, enabled: bool) -> int:
        """Toggle FP32 scale cache on every resident FP8 linear."""
        from .fp8_linear import apply_scale_cache
        return apply_scale_cache(self, bool(enabled))

    def initial_state(self) -> TPState:
        return TPState.empty(list(self.config.layer_types), seq_len=0)

    def prefill(self, idx: torch.Tensor, position_ids: torch.Tensor | None = None,
                state: TPState | None = None, return_hidden: bool = False,
                ) -> tuple[torch.Tensor, TPState] | tuple[torch.Tensor, TPState, torch.Tensor]:
        """Prefill ``idx [B,T]`` -> ``(logits [B,T,V], TPState)``.

        With ``return_hidden=True`` the pre-final-norm hidden
        ``[B, T, D]`` (the final norm's input) is appended; default
        off returns exactly the 2-tuple.
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        if position_ids is not None and (
                not isinstance(position_ids, torch.Tensor) or position_ids.shape[-1] != T):
            got = (tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor)
                   else type(position_ids).__name__)
            raise ValueError(f"position_ids must end with T={T}, got {got}")
        if state is None:
            state = self.initial_state()
        if len(state) != len(self.layers):
            raise ValueError(f"state has {len(state)} layers, model has {len(self.layers)}")
        start_pos = int(state.seq_len)
        x = self.embed_tokens(idx.to(self.ctx.device))
        new_entries: list[tuple[str, Any]] = []
        for i, block in enumerate(self.layers):
            tag, payload = state.entry(i)
            x, new_payload = block.forward_prefill(
                x, layer_state=payload, start_pos=start_pos, position_ids=position_ids)
            new_entries.append((tag, new_payload))
        new_state = TPState(entries=new_entries, seq_len=start_pos + T,
                            layer_types=list(self.config.layer_types))
        logits = self._lm_head_logits(self.norm(x))
        if return_hidden:
            return logits, new_state, x
        return logits, new_state

    def forward_with_hidden(self, idx: torch.Tensor,
                            state: TPState | None = None,
                            position_ids: torch.Tensor | None = None,
                            ) -> tuple[torch.Tensor, TPState, torch.Tensor, torch.Tensor]:
        """``(logits, state, hidden_pre_norm, hidden_post_norm)``.

        Hook-free target-hidden capture mirroring
        ``TinyHybridLM.forward_with_hidden``: prefill (or one step
        when ``state`` is given and ``idx`` has one column) plus the
        final norm's input and its output.
        """
        if state is not None and (idx.dim() == 1 or (idx.dim() == 2 and idx.shape[1] == 1)):
            logits, new_state, hidden_pre = self.step(idx, state, return_hidden=True)
        else:
            logits, new_state, hidden_pre = self.prefill(
                idx, position_ids=position_ids, state=state, return_hidden=True)
        return logits, new_state, hidden_pre, self.norm(hidden_pre)

    def step(self, token: torch.Tensor, state: TPState,
             return_hidden: bool = False,
             ) -> tuple[torch.Tensor, TPState] | tuple[torch.Tensor, TPState, torch.Tensor]:
        """Decode one token: ``token [B]`` or ``[B,1]`` -> ``(logits [B,1,V], state)``.

        With ``return_hidden=True`` the pre-final-norm hidden
        ``[B, 1, D]`` is appended as a third value.
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        if token.dim() != 2 or token.shape[1] != 1:
            raise ValueError(f"token must be [B] or [B, 1], got {tuple(token.shape)}")
        if len(state) != len(self.layers):
            raise ValueError(f"state has {len(state)} layers, model has {len(self.layers)}")
        start_pos = int(state.seq_len)
        x = self.embed_tokens(token.to(self.ctx.device))
        new_entries: list[tuple[str, Any]] = []
        for i, block in enumerate(self.layers):
            tag, payload = state.entry(i)
            x, new_payload = block.step(x, layer_state=payload, start_pos=start_pos)
            new_entries.append((tag, new_payload))
        new_state = TPState(entries=new_entries, seq_len=start_pos + 1,
                            layer_types=list(self.config.layer_types))
        logits = self._lm_head_logits(self.norm(x))
        if return_hidden:
            return logits, new_state, x
        return logits, new_state

    # TinyHybridLM spelling for the decode step.
    decode_step = step

    def prefill_greedy(self, idx: torch.Tensor, position_ids: torch.Tensor | None = None,
                       state: TPState | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor, TPState]:
        """Prefill + greedy next token without materialising full logits.

        Returns ``(tokens [B], top_values [B], state)``. In vocab mode
        this uses :meth:`greedy_next_token` (one small all_gather);
        the hidden run is identical to :meth:`prefill`.
        """
        logits_state: TPState
        # Reuse prefill's block run by capturing normed hidden: run
        # prefill normally in replicated mode; in vocab mode avoid
        # the full-vocab gather by re-running via hidden helper.
        if getattr(self, "lm_head_mode", "replicated") != "vocab":
            logits, new_state = self.prefill(idx, position_ids=position_ids, state=state)
            vals, tok = logits[:, -1].float().max(dim=-1)
            return tok.to(torch.long), vals, new_state
        hidden, new_state = self._hidden_prefill(idx, position_ids=position_ids, state=state)
        tok, vals = self.greedy_next_token(hidden[:, -1, :])
        return tok, vals, new_state

    def step_greedy(self, token: torch.Tensor, state: TPState,
                    ) -> tuple[torch.Tensor, torch.Tensor, TPState]:
        """Decode step + greedy token without full logits (see prefill_greedy)."""
        if getattr(self, "lm_head_mode", "replicated") != "vocab":
            logits, new_state = self.step(token, state)
            vals, tok = logits[:, -1].float().max(dim=-1)
            return tok.to(torch.long), vals, new_state
        hidden, new_state = self._hidden_step(token, state)
        tok, vals = self.greedy_next_token(hidden)
        return tok, vals, new_state

    def _hidden_prefill(self, idx: torch.Tensor, position_ids=None, state=None):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        if state is None:
            state = self.initial_state()
        start_pos = int(state.seq_len)
        x = self.embed_tokens(idx.to(self.ctx.device))
        new_entries: list[tuple[str, Any]] = []
        for i, block in enumerate(self.layers):
            tag, payload = state.entry(i)
            x, new_payload = block.forward_prefill(
                x, layer_state=payload, start_pos=start_pos, position_ids=position_ids)
            new_entries.append((tag, new_payload))
        new_state = TPState(entries=new_entries, seq_len=start_pos + T,
                            layer_types=list(self.config.layer_types))
        return self.norm(x), new_state

    def _hidden_step(self, token: torch.Tensor, state: TPState):
        if token.dim() == 1:
            token = token.unsqueeze(1)
        if token.dim() != 2 or token.shape[1] != 1:
            raise ValueError(f"token must be [B] or [B, 1], got {tuple(token.shape)}")
        start_pos = int(state.seq_len)
        x = self.embed_tokens(token.to(self.ctx.device))
        new_entries: list[tuple[str, Any]] = []
        for i, block in enumerate(self.layers):
            tag, payload = state.entry(i)
            x, new_payload = block.step(x, layer_state=payload, start_pos=start_pos)
            new_entries.append((tag, new_payload))
        new_state = TPState(entries=new_entries, seq_len=start_pos + 1,
                            layer_types=list(self.config.layer_types))
        return self.norm(x)[:, 0, :], new_state

    def static_state(self, batch_size: int = 1, max_seq_len: int | None = None,
                     device: torch.device | str | None = None,
                     dtype: torch.dtype | None = None) -> TPStaticState:
        """Allocate a :class:`TPStaticState` for this rank's shards."""
        if device is None:
            device = self.ctx.device
        if dtype is None:
            param = next(self.parameters(), None)
            dtype = param.dtype if param is not None else torch.float32
        return TPStaticState(
            self.config, batch_size=batch_size,
            max_seq_len=(max_seq_len if max_seq_len is not None
                         else int(self.config.max_position_embeddings)),
            device=device, dtype=dtype, ctx=self.ctx)

    initial_static_state = static_state

    def decode_token_static(self, token_ids: torch.Tensor, state: TPStaticState,
                            out: torch.Tensor | None = None,
                            return_hidden: bool = False):
        """Decode one token in place over a :class:`TPStaticState`.

        ``token_ids`` is ``[B]`` or ``[B, 1]``; returns logits
        ``[B, vocab_size]`` -- written into the preallocated ``out``
        buffer (and returned) when given, else a fresh tensor. State
        is mutated in place and ``state.position`` is advanced by one
        (``add_``) after all blocks run. Overflow past ``max_seq_len``
        raises ``ValueError`` *before* any mutation (checked via one
        ``.item()`` read outside capture; during CUDA-graph capture
        the check is skipped and the caller must guarantee the bound).

        Allocation audit (hot path): no ``torch.cat``, list append,
        ``.item()`` (outside capture) or state clone touches state.
        Remaining per-step allocations are activations only --
        projection/attention/MLP intermediates, the ``repeat_kv``
        expansion and attention mask/scores, and the fresh logits
        when ``out`` is not provided -- plus the collectives' own
        buffers inside ``torch.distributed``.
        """
        from .hybrid import _is_capturing
        if not isinstance(state, TPStaticState):
            raise ValueError(
                f"decode_token_static requires a TPStaticState, got {type(state).__name__}")
        if token_ids.dim() == 2 and token_ids.shape[1] == 1:
            toks = token_ids[:, 0]
        elif token_ids.dim() == 1:
            toks = token_ids
        else:
            raise ValueError(f"token_ids must be [B] or [B, 1], got {tuple(token_ids.shape)}")
        if int(toks.shape[0]) != state.batch_size:
            raise ValueError(
                f"token batch {int(toks.shape[0])} does not match state batch_size={state.batch_size}")
        if len(state) != len(self.layers):
            raise ValueError(f"state has {len(state)} layers, model has {len(self.layers)}")
        if out is not None and tuple(out.shape) != (state.batch_size, int(self.config.vocab_size)):
            raise ValueError(
                f"out must be {(state.batch_size, int(self.config.vocab_size))}, got {tuple(out.shape)}")
        capturing = _is_capturing(state.position)
        if not capturing:
            pos = int(state.position.item())
            if pos >= state.max_seq_len:
                raise ValueError(
                    f"TPStaticState overflow: position {pos} would exceed "
                    f"max_seq_len={state.max_seq_len} (state already holds "
                    f"{pos} tokens); allocate a larger TPStaticState "
                    "or reset() it before decoding more tokens")
        x = self.embed_tokens(toks.to(self.ctx.device)).unsqueeze(1)
        for block in self.layers:
            x = block.step_static(x, state)
        logits = self._lm_head_logits(self.norm(x))[:, 0, :]
        state.position.add_(1)
        if return_hidden:
            return logits, x  # x is pre-final-norm hidden [B, 1, D]
        if out is not None:
            out.copy_(logits)
            return out
        return logits

    def decode_chunk_static(self, token_ids: torch.Tensor, state: TPStaticState,
                            return_hidden: bool = False,
                            out: torch.Tensor | None = None,
                            hidden_out: torch.Tensor | None = None):
        """Decode a 2-token chunk in place: ``token_ids [B, 2]``.

        Verification forward for speculative decoding: returns logits
        ``[B, 2, V]`` (position 0 predicts the token after
        ``token_ids[:, 0]``, position 1 the token after
        ``token_ids[:, 1]``) and advances ``state.position`` by 2.
        With ``return_hidden=True`` also returns the pre-final-norm
        hidden ``[B, 2, D]``. Overflow raises before any mutation.
        Every block reads each projection weight once over M=2 (see
        ``step_chunk_static``); no module is left at T=1 only.

        ``out`` / ``hidden_out`` are preallocated buffers the logits /
        hidden are copied into (and returned), mirroring
        :meth:`decode_token_static`'s ``out`` -- the form a CUDA-graph
        capture replays into. Default ``None`` keeps fresh returns.
        """
        if not isinstance(state, TPStaticState):
            raise ValueError(
                f"decode_chunk_static requires a TPStaticState, got {type(state).__name__}")
        if token_ids.dim() != 2 or token_ids.shape[1] != 2:
            raise ValueError(f"token_ids must be [B, 2], got {tuple(token_ids.shape)}")
        if int(token_ids.shape[0]) != state.batch_size:
            raise ValueError(
                f"token batch {int(token_ids.shape[0])} does not match state batch_size={state.batch_size}")
        if len(state) != len(self.layers):
            raise ValueError(f"state has {len(state)} layers, model has {len(self.layers)}")
        from .hybrid import _is_capturing
        if not _is_capturing(state.position):
            pos = int(state.position.item())
            if pos + 2 > state.max_seq_len:
                raise ValueError(
                    f"TPStaticState overflow: chunk of 2 at position {pos} would exceed "
                    f"max_seq_len={state.max_seq_len}; allocate a larger TPStaticState "
                    "or reset() it first")
        if out is not None and tuple(out.shape) != (
                state.batch_size, 2, int(self.config.vocab_size)):
            raise ValueError(
                f"out must be {(state.batch_size, 2, int(self.config.vocab_size))}, "
                f"got {tuple(out.shape)}")
        if hidden_out is not None and tuple(hidden_out.shape) != (
                state.batch_size, 2, int(self.config.hidden_size)):
            raise ValueError(
                f"hidden_out must be {(state.batch_size, 2, int(self.config.hidden_size))}, "
                f"got {tuple(hidden_out.shape)}")
        x = self.embed_tokens(token_ids.to(self.ctx.device))
        for block in self.layers:
            x = block.step_chunk_static(x, state)
        logits = self._lm_head_logits(self.norm(x))
        state.position.add_(2)
        if out is not None:
            out.copy_(logits)
            logits = out
        if hidden_out is not None:
            hidden_out.copy_(x)
            x = hidden_out
        if return_hidden:
            return logits, x
        return logits

    def prefill_static(self, idx: torch.Tensor, state: TPStaticState,
                       return_hidden: bool = False):
        """Prefill into static state (v1: token-by-token decode loop).

        ``idx [B, T]`` -> stacked logits ``[B, T, V]``. Chunkwise TP
        prefill remains future work; this loop reuses
        :meth:`decode_token_static` exactly (one mixer all-reduce per
        token per linear layer, as in the legacy v1 prefill). Raises
        before any mutation when the prompt would overflow the state.
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        if idx.dim() != 2:
            raise ValueError(f"idx must be [B, T], got {tuple(idx.shape)}")
        if not isinstance(state, TPStaticState):
            raise ValueError(
                f"prefill_static requires a TPStaticState, got {type(state).__name__}")
        if int(idx.shape[0]) != state.batch_size:
            raise ValueError(
                f"idx batch {int(idx.shape[0])} does not match state batch_size={state.batch_size}")
        from .hybrid import _is_capturing
        if not _is_capturing(state.position):
            pos = int(state.position.item())
            if pos + int(idx.shape[1]) > state.max_seq_len:
                raise ValueError(
                    f"TPStaticState overflow: prefill of {int(idx.shape[1])} tokens at "
                    f"position {pos} would exceed max_seq_len={state.max_seq_len}; "
                    "allocate a larger TPStaticState or reset() it first")
        outs: list[torch.Tensor] = []
        hids: list[torch.Tensor] = []
        for t in range(int(idx.shape[1])):
            if return_hidden:
                lg, hd = self.decode_token_static(idx[:, t], state, return_hidden=True)
                outs.append(lg)
                hids.append(hd)
            else:
                outs.append(self.decode_token_static(idx[:, t], state))
        if not outs:
            empty = torch.empty(int(idx.shape[0]), 0, int(self.config.vocab_size),
                                device=self.ctx.device)
            if return_hidden:
                return empty, torch.empty(int(idx.shape[0]), 0,
                                          int(self.config.hidden_size), device=self.ctx.device)
            return empty
        logits = torch.stack(outs, dim=1)
        if return_hidden:
            return logits, torch.cat(hids, dim=1)
        return logits

    def forward(self, idx: torch.Tensor, state: TPState | None = None,
                position_ids: torch.Tensor | None = None,
                return_hidden: bool = False,
                ) -> tuple[torch.Tensor, TPState] | tuple[torch.Tensor, TPState, torch.Tensor]:
        if state is None:
            return self.prefill(idx, position_ids=position_ids, return_hidden=return_hidden)
        if idx.dim() == 2 and idx.shape[1] == 1 or idx.dim() == 1:
            return self.step(idx, state, return_hidden=return_hidden)
        return self.prefill(idx, position_ids=position_ids, state=state,
                            return_hidden=return_hidden)
