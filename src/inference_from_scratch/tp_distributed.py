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

from .tensor_parallel import shard_spec_for, shard_state_dict

__all__ = [
    "DistributedTPContext",
    "TPColumnLinear",
    "TPRowLinear",
    "DistributedSwiGLU",
    "DistributedFullAttention",
    "DistributedGatedDeltaNet",
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
        self.attn_implementation = str(dense.attn_implementation)
        self.gate_activation = str(dense.gate_activation)
        self.scaling = float(getattr(dense, "scaling", self.head_dim ** -0.5))
        h_per = self.n_head // ctx.world_size
        kv_per = self.n_kv_head // ctx.world_size
        self.q_heads_local = list(range(ctx.rank * h_per, (ctx.rank + 1) * h_per))
        self.kv_heads_local = list(range(ctx.rank * kv_per, (ctx.rank + 1) * kv_per))

    def _shard_bias(self, linear: nn.Linear, prefix: str, cfg: Any):
        if linear.bias is None:
            return None
        return nn.Parameter(shard_weight_for_rank(f"{prefix}.bias", linear.bias.data,
                                                  self.ctx.rank, self.ctx.world_size, cfg).to(self.ctx.device))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        from .attention import manual_scaled_dot_product_attention, repeat_kv
        from .hybrid import apply_partial_rope, precompute_partial_rope_cache
        B, T, _ = x.shape
        Dh = self.head_dim
        hs, kvs = self.q_heads_local, self.kv_heads_local
        qg = F.linear(x, self.q_weight, self.q_bias).view(B, T, len(hs), 2 * Dh)
        q = self.q_norm(qg[..., :Dh]).transpose(1, 2)
        gate = qg[..., Dh:].reshape(B, T, len(hs) * Dh)
        k = F.linear(x, self.k_weight, self.k_bias).view(B, T, len(kvs), Dh)
        v = F.linear(x, self.v_weight, self.v_bias).view(B, T, len(kvs), Dh)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin, _ = precompute_partial_rope_cache(
            start_pos + T, Dh, self.partial_rotary_factor,
            theta=self.rope_theta, device=x.device, dtype=x.dtype)
        q = apply_partial_rope(q, cos, sin, start_pos=start_pos, rotary_dim=self.rotary_dim)
        k = apply_partial_rope(k, cos, sin, start_pos=start_pos, rotary_dim=self.rotary_dim)
        k_rep = repeat_kv(k, self.n_rep); v_rep = repeat_kv(v, self.n_rep)
        if self.attn_implementation == "sdpa":
            attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True, scale=self.scaling)
        else:
            attn = manual_scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        merged = attn.transpose(1, 2).contiguous().view(B, T, -1)
        if self.gate_activation == "sigmoid":
            merged = merged * torch.sigmoid(gate)
        elif self.gate_activation == "silu":
            merged = merged * F.silu(gate)
        partial = F.linear(merged, self.o_weight)
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
        mixed = F.linear(x2, self.in_proj_qkv_w)
        z = F.linear(x2, self.in_proj_z_w).reshape(B, self.vh_per, Dv)
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
        partial = F.linear(core.to(self.out_proj_w.dtype), self.out_proj_w)
        out = self.ctx.all_reduce_sum(partial).to(x_t.dtype)
        if squeeze:
            out = out.unsqueeze(1)
        return out, (S, new_buf.to(dtype=mixed.dtype))
