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
    "shard_weight_for_rank",
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

    def __init__(self, ctx: DistributedTPContext, dense: Any, config: Any = None):
        super().__init__()
        self.ctx = ctx
        self.config = config if config is not None else dense.config
        dev = ctx.device
        cfg = self.config
        # Replicated embedding (whole copy per rank).
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(dense.embed_tokens.weight.data.to(dev))
        self.embed_tokens = self.embed_tokens.to(dev)
        self.layers = nn.ModuleList(
            [DistributedHybridBlock(ctx, block, config=cfg) for block in dense.layers])
        # Replicated final norm + lm_head (vocab-parallel deferred).
        self.norm = copy.deepcopy(dense.norm).to(dev)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        with torch.no_grad():
            if dense.lm_head.weight.data_ptr() == dense.embed_tokens.weight.data_ptr():
                self.lm_head.weight = self.embed_tokens.weight  # tied
            else:
                self.lm_head.weight.copy_(dense.lm_head.weight.data.to(dev))
        self.lm_head = self.lm_head.to(dev) if not cfg.tie_word_embeddings else self.lm_head

    @classmethod
    def empty(cls, config: Any, rank: int, world: int, fp8_mode: str = "resident",
              use_triton: bool | None = None) -> "DistributedHybridLM":
        """Shard-shaped empty LM on meta for rank ``rank`` (loader fills it).

        ``fp8_mode="resident"`` builds resident-FP8 projections
        (:class:`BlockFP8Linear` shards); ``"dequantize"`` builds dense
        shard Parameters. No tensor data is allocated (meta device).
        Call :meth:`set_ctx` with a real context before forwarding.
        """
        if fp8_mode not in ("resident", "dequantize"):
            raise ValueError(f"fp8_mode must be 'resident' or 'dequantize', got {fp8_mode!r}")
        from .hybrid import QwenRMSNorm
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        ctx = _LocalContext(int(rank), int(world), torch.device("meta"))
        obj.ctx = ctx  # type: ignore[assignment]
        obj.config = config
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
        obj.lm_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)
        obj.lm_head.weight = nn.Parameter(
            torch.empty((int(config.vocab_size), int(config.hidden_size)), device=meta))
        if config.tie_word_embeddings:
            obj.lm_head.weight = obj.embed_tokens.weight
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
                   config: Any = None) -> "DistributedHybridLM":
        """Build this rank's model by slicing ``model`` (dense TinyHybridLM).

        Every sharded weight is cut via ``shard_state_dict`` specs (the
        same specs as the simulated runtime), so this rank stores only
        its slice; replicated tensors are cloned whole.
        """
        return cls(ctx, model, config=config)

    def local_parameter_bytes(self) -> int:
        """Bytes held on THIS rank (params + FP8 buffers, shards+replicated)."""
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        total += sum(b.numel() * b.element_size() for b in self.buffers())
        return int(total)

    @staticmethod
    def dense_parameter_bytes(model: Any) -> int:
        return int(sum(p.numel() * p.element_size() for p in model.parameters()))

    def initial_state(self) -> TPState:
        return TPState.empty(list(self.config.layer_types), seq_len=0)

    def prefill(self, idx: torch.Tensor, position_ids: torch.Tensor | None = None,
                state: TPState | None = None,
                ) -> tuple[torch.Tensor, TPState]:
        """Prefill ``idx [B,T]`` -> ``(logits [B,T,V], TPState)``."""
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
        logits = self.lm_head(self.norm(x))
        return logits, TPState(entries=new_entries, seq_len=start_pos + T,
                               layer_types=list(self.config.layer_types))

    def step(self, token: torch.Tensor, state: TPState,
             ) -> tuple[torch.Tensor, TPState]:
        """Decode one token: ``token [B]`` or ``[B,1]`` -> ``(logits [B,1,V], state)``."""
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
        logits = self.lm_head(self.norm(x))
        return logits, TPState(entries=new_entries, seq_len=start_pos + 1,
                               layer_types=list(self.config.layer_types))

    # TinyHybridLM spelling for the decode step.
    decode_step = step

    def forward(self, idx: torch.Tensor, state: TPState | None = None,
                position_ids: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, TPState]:
        if state is None:
            return self.prefill(idx, position_ids=position_ids)
        if idx.dim() == 2 and idx.shape[1] == 1 or idx.dim() == 1:
            return self.step(idx, state)
        return self.prefill(idx, position_ids=position_ids, state=state)
