"""Linear attention (gated delta rule) -- the Qwen3.5 hybrid layer.

Why this exists in inference
------------------------------
Most Qwen3.5/Qwen3.8 layers do not use softmax attention. They use a
*gated delta rule* (DeltaNet family): each token writes into a fixed-size
recurrent matrix state ``S`` and reads its output from that same state.
Decode therefore costs O(1) memory per layer in sequence length -- ``S``
never grows -- unlike a KV cache that grows with ``T``. Prefill is still
sequential in the reference form dissected here; production kernels
re-associate the same recurrence into chunkwise matmuls.

Source of truth
---------------
Read directly in ``.venv`` (transformers 5.20.1):

* ``transformers/models/qwen3_5/modeling_qwen3_5.py`` --
  ``Qwen3_5GatedDeltaNet`` (projections, depthwise causal conv, beta/gate,
  repeat-interleave of Q/K, output norm + gating),
  ``torch_recurrent_gated_delta_rule`` (the per-token recurrence this
  module mirrors step-for-step), ``torch_chunk_gated_delta_rule`` (the
  chunkwise production form, *not* implemented here),
  ``causal_conv1d_fn`` / ``causal_conv1d_update``, ``l2norm`` and
  ``Qwen3_5RMSNormGated``.
* ``transformers/models/qwen3_5/configuration_qwen3_5.py`` --
  ``Qwen3_5TextConfig`` field names ``linear_num_key_heads``,
  ``linear_key_head_dim``, ``linear_num_value_heads``,
  ``linear_value_head_dim`` and ``linear_conv_kernel_dim``.

Parameter names/shapes mirror the checkpoint exactly (an HF
``Qwen3_5GatedDeltaNet`` state dict loads into this module with
``strict=True`` at matching dims): ``in_proj_qkv.weight``
``[2*Hk*Dk + Hv*Dv, D]``, ``in_proj_z.weight`` ``[Hv*Dv, D]``,
``in_proj_a.weight`` / ``in_proj_b.weight`` ``[Hv, D]``,
``conv1d.weight`` ``[2*Hk*Dk + Hv*Dv, 1, K]``, ``A_log`` / ``dt_bias``
``[Hv]``, ``norm.weight`` ``[Dv]``, ``out_proj.weight`` ``[D, Hv*Dv]``.
All projections are bias-free.

The recurrence (per value head, per token ``t``)
-------------------------------------------------
With ``S`` of shape ``[Dk, Dv]`` (one per value head)::

    S <- S * exp(g_t)                                # gated decay
    delta = (v_t - S^T-applied-to-k_t) * beta_t      # delta-rule error
    S <- S + k_t (x) delta                           # rank-1 write
    o_t = S^T-applied-to-q_t                         # read (after write)

where ``(S * k)`` contracts over ``Dk``. Queries/keys are L2-normalised
and queries scaled by ``Dk**-0.5`` before the recurrence; ``beta`` is a
sigmoid and ``g`` is a negative log-decay (see :meth:`GatedDeltaNet.step`).

How production differs
-----------------------
FLA / vLLM evaluate the identical recurrence with chunkwise kernels
(``torch_chunk_gated_delta_rule`` above): inside a chunk the delta-rule
updates are condensed via a triangular (UT) solve into matmuls, and only
the chunk-boundary state is carried sequentially. Decode uses the fused
recurrent kernel and updates the conv state in place. This module keeps
the explicit scan so prefill-vs-decode equivalence is provable by eye.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# State carried between tokens: (recurrent S, conv buffer).
#   S:           [B, Hv, Dk, Dv]  fp32 recurrent matrix per value head
#   conv_buffer: [B, conv_dim, K-1]  last K-1 raw (pre-conv) mixed_qkv
#                vectors in chronological order (zeros at sequence start)
LinearAttentionState = tuple[torch.Tensor, torch.Tensor]


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise over the last dim, as in HF ``l2norm`` (modeling_qwen3_5.py).

    Computed in float32: ``x * rsqrt(sum(x^2) + eps)``. Note the eps sits
    *inside* the rsqrt and there is no ``max(norm, eps)`` clamp -- matching
    the FLA-aligned HF helper exactly matters for parity.
    """
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)


class RMSNormGated(nn.Module):
    """RMSNorm followed by a SiLU gate, mirroring ``Qwen3_5RMSNormGated``.

    Per row (over ``Dv``): ``rms_norm(core) * weight * silu(z)``, norm
    computed in float32, norm *before* the gate (HF comment, ibid.).
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = core.dtype
        xf = core.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.eps)
        y = (y * self.weight.float()).to(input_dtype)
        y = y * F.silu(gate.float())
        return y.to(input_dtype)

    def extra_repr(self) -> str:
        return f"d={self.d}, eps={self.eps}"


class GatedDeltaNet(nn.Module):
    """Gated delta-rule linear attention layer (tiny-dims CPU reference).

    Args mirror ``Qwen3_5TextConfig`` linear-attention fields (with the
    task's naming): ``key_dim`` / ``value_dim`` are *per-head* dims
    (HF ``linear_key_head_dim`` / ``linear_value_head_dim``, 128 in the
    real 27B checkpoint), ``num_key_heads`` / ``num_value_heads`` are HF
    ``linear_num_key_heads`` / ``linear_num_value_heads`` (16 / 48 real).
    ``num_value_heads`` must be an integer multiple of
    ``num_key_heads``: Q/K heads are ``repeat_interleave``-expanded by
    ``n_rep = Hv // Hk`` to meet the value heads (HF
    ``Qwen3_5GatedDeltaNet.forward``), so with ``n_rep = 3`` Q/K head
    ``g`` serves value heads ``3g, 3g+1, 3g+2``.

    Shapes: input ``x`` is ``[B, T, d_model]`` for prefill and
    ``[B, d_model]`` (or ``[B, 1, d_model]``) for a decode step; output
    has the same leading shape with last dim ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        num_key_heads: int,
        key_dim: int,
        num_value_heads: int,
        value_dim: int,
        conv_kernel: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if d_model <= 0 or num_key_heads <= 0 or key_dim <= 0:
            raise ValueError("d_model, num_key_heads and key_dim must be positive")
        if num_value_heads <= 0 or value_dim <= 0 or conv_kernel <= 0:
            raise ValueError("num_value_heads, value_dim and conv_kernel must be positive")
        if num_value_heads % num_key_heads != 0:
            raise ValueError(
                f"num_value_heads={num_value_heads} must be divisible by "
                f"num_key_heads={num_key_heads} (Q/K repeat_interleave factor)"
            )
        self.d_model = d_model
        self.hidden_size = d_model  # HF alias
        self.num_key_heads = num_key_heads
        self.num_k_heads = num_key_heads  # HF alias
        self.num_value_heads = num_value_heads
        self.num_v_heads = num_value_heads  # HF alias
        self.key_dim = key_dim  # per-head (task naming)
        self.head_k_dim = key_dim  # HF alias
        self.value_dim = value_dim  # per-head (task naming)
        self.head_v_dim = value_dim  # HF alias
        self.conv_kernel = conv_kernel
        self.conv_kernel_size = conv_kernel  # HF alias
        self.eps = eps
        self.n_rep = num_value_heads // num_key_heads

        self.key_total = num_key_heads * key_dim
        self.value_total = num_value_heads * value_dim
        self.conv_dim = self.key_total * 2 + self.value_total

        # Checkpoint-mirroring parameters (all bias-free, as in HF).
        self.in_proj_qkv = nn.Linear(d_model, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d_model, self.value_total, bias=False)
        self.in_proj_a = nn.Linear(d_model, num_value_heads, bias=False)
        self.in_proj_b = nn.Linear(d_model, num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            bias=False,
            padding=conv_kernel - 1,
        )
        # HF init: dt_bias ones; A uniform in [0.01, 16), stored as log so
        # exp(A_log) is the (positive) decay rate scale.
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        A = torch.empty(num_value_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(value_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_total, d_model, bias=False)

    # -- state handling ----------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> LinearAttentionState:
        """Zero state for a fresh sequence: ``(S, conv_buffer)``.

        ``S`` is ``[B, Hv, Dk, Dv]``; ``conv_buffer`` is
        ``[B, conv_dim, K-1]`` (empty last dim when ``K == 1``), holding
        the raw pre-conv inputs the causal conv still needs to see.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        S = torch.zeros(
            batch_size, self.num_value_heads, self.head_k_dim, self.head_v_dim,
            device=device, dtype=dtype,
        )
        conv_buffer = torch.zeros(
            batch_size, self.conv_dim, max(self.conv_kernel - 1, 0),
            device=device, dtype=dtype,
        )
        return (S, conv_buffer)

    # Alias: some callers name the zero-state constructor this way.
    def zero_state(self, *args, **kwargs) -> LinearAttentionState:  # type: ignore[no-untyped-def]
        return self.initial_state(*args, **kwargs)

    @staticmethod
    def snapshot_state(state: LinearAttentionState) -> LinearAttentionState:
        """Deep-copy ``state`` (both tensors cloned).

        The snapshot is independent: continuing decode from the original
        state, or mutating the snapshot, never affects the other side.
        This is the hook a future prefix cache stores.
        """
        S, conv_buffer = state
        return (S.clone(), conv_buffer.clone())

    @staticmethod
    def restore_state(snapshot: LinearAttentionState) -> LinearAttentionState:
        """Deep-copy a snapshot back into a usable, independent state.

        Cloning again (rather than returning the snapshot itself) means
        the same snapshot can be restored repeatedly and each restored
        state can be mutated/continued without corrupting the snapshot.
        """
        S, conv_buffer = snapshot
        return (S.clone(), conv_buffer.clone())

    # -- per-token core ------------------------------------------------------

    def _conv_step(
        self, raw_t: torch.Tensor, conv_buffer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One causal depthwise-conv step + SiLU.

        ``raw_t``: ``[B, conv_dim]`` pre-conv vector for this token;
        ``conv_buffer``: ``[B, conv_dim, K-1]`` previous raw vectors.
        Returns ``(activated [B, conv_dim], new_buffer)``. The window is
        ``[prev_{K-1} ... prev_1, raw_t]`` cross-correlated with
        ``conv1d.weight[:, 0, :]`` -- exactly ``F.conv1d`` with left
        zero-padding ``K-1`` evaluated at the last position (HF
        ``causal_conv1d_fn`` / ``causal_conv1d_update``).
        """
        if self.conv_kernel == 1:
            window = raw_t.unsqueeze(-1)  # [B, C, 1]
            new_buffer = conv_buffer  # [B, C, 0], stays empty
        else:
            window = torch.cat([conv_buffer, raw_t.unsqueeze(-1)], dim=-1)
            new_buffer = window[:, :, 1:]
        w = self.conv1d.weight[:, 0, :].to(dtype=window.dtype)  # [C, K]
        conv_out = (window * w.unsqueeze(0)).sum(dim=-1)  # [B, C]
        return F.silu(conv_out), new_buffer

    def _recurrent_token(
        self,
        mixed_t: torch.Tensor,
        z_t: torch.Tensor,
        beta_t: torch.Tensor,
        g_t: torch.Tensor,
        S: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated-delta update + read for one token (HF recurrent rule).

        ``mixed_t``: ``[B, conv_dim]`` post-conv/SiLU; ``z_t``:
        ``[B, Hv, Dv]`` gate; ``beta_t`` / ``g_t``: ``[B, Hv]``;
        ``S``: ``[B, Hv, Dk, Dv]``. Returns ``(out_t [B, d_model], S_new)``.
        No in-place ops: the caller's ``S`` is never mutated.
        """
        B = mixed_t.shape[0]
        q_raw, k_raw, v = torch.split(
            mixed_t, [self.key_total, self.key_total, self.value_total], dim=-1
        )
        q = q_raw.reshape(B, self.num_key_heads, self.head_k_dim)
        k = k_raw.reshape(B, self.num_key_heads, self.head_k_dim)
        v = v.reshape(B, self.num_value_heads, self.head_v_dim).float()
        if self.n_rep > 1:
            # Grouped repeat: Q/K head g -> value heads g*n_rep..(g+1)*n_rep-1.
            q = q.repeat_interleave(self.n_rep, dim=1)
            k = k.repeat_interleave(self.n_rep, dim=1)
        q = _l2norm(q) * (self.head_k_dim ** -0.5)
        k = _l2norm(k)

        # 1) Gated decay: g_t <= 0, so exp(g_t) in (0, 1].
        S = S.float() * torch.exp(g_t.float())[:, :, None, None]
        # 2) Delta rule: what S already predicts for k_t is subtracted;
        #    only the error, scaled by beta in (0, 1), is written.
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)  # [B, Hv, Dv]
        delta = (v - kv_mem) * beta_t.float().unsqueeze(-1)  # [B, Hv, Dv]
        S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)  # rank-1 write
        # 3) Read AFTER the write: the current token sees itself.
        core = (S * q.unsqueeze(-1)).sum(dim=-2)  # [B, Hv, Dv]

        core = self.norm(core.reshape(B * self.num_value_heads, self.head_v_dim),
                         z_t.reshape(B * self.num_value_heads, self.head_v_dim))
        core = core.reshape(B, self.value_total)
        out = F.linear(core.to(self.out_proj.weight.dtype), self.out_proj.weight)
        return out, S

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token projections shared by prefill and step.

        ``x``: ``[B, d_model]`` -> ``(mixed [B, conv_dim], z [B, Hv, Dv],
        beta [B, Hv], g [B, Hv])`` with ``beta = sigmoid(b)`` and
        ``g = -exp(A_log) * softplus(a + dt_bias)`` (HF forward verbatim;
        the float32 upcast keeps ``A_log`` finite in half precision).
        """
        mixed = F.linear(x, self.in_proj_qkv.weight)
        z = F.linear(x, self.in_proj_z.weight).reshape(
            x.shape[0], self.num_value_heads, self.head_v_dim
        )
        b = F.linear(x, self.in_proj_b.weight)
        a = F.linear(x, self.in_proj_a.weight)
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return mixed, z, beta, g

    # -- public API ------------------------------------------------------------

    def step(
        self, x_t: torch.Tensor, state: LinearAttentionState | None = None
    ) -> tuple[torch.Tensor, LinearAttentionState]:
        """Decode one token: ``x_t`` ``[B, D]`` or ``[B, 1, D]``.

        Returns ``(out_t, new_state)`` with ``out_t`` in the same rank as
        the input. ``state=None`` starts from a zero state. The input
        state is never mutated, so a snapshot taken before the step can
        still be restored and re-stepped identically.
        """
        if x_t.dim() == 2:
            x2 = x_t
        elif x_t.dim() == 3 and x_t.shape[1] == 1:
            x2 = x_t[:, 0, :]  # rank restored to [B, 1, D] on the way out
        else:
            raise ValueError(f"x_t must be [B, D] or [B, 1, D], got {tuple(x_t.shape)}")
        if x2.shape[-1] != self.d_model:
            raise ValueError(f"x_t last dim must be d_model={self.d_model}, got {x2.shape[-1]}")
        B = x2.shape[0]
        if state is None:
            S, conv_buffer = self.initial_state(B, device=x2.device, dtype=torch.float32)
        else:
            S, conv_buffer = state
        mixed, z, beta, g = self._project(x2)
        mixed_act, new_buffer = self._conv_step(
            mixed, conv_buffer.to(dtype=mixed.dtype)
        )
        out, S_new = self._recurrent_token(mixed_act, z, beta, g, S)
        out = out.to(x_t.dtype)
        if x_t.dim() == 3:
            out = out.unsqueeze(1)
        return out, (S_new, new_buffer.to(dtype=mixed.dtype))

    def forward_prefill(
        self,
        x: torch.Tensor,
        state: LinearAttentionState | None = None,
    ) -> tuple[torch.Tensor, LinearAttentionState]:
        """Prefill ``x`` ``[B, T, D]`` as an explicit sequential scan.

        Loops over ``T`` calling the same conv + recurrence step as
        decode (projections are computed once -- they are per-token
        linears with no cross-token mixing, so this changes nothing
        mathematically). Returns ``(out [B, T, D], final_state)``; the
        final state continues seamlessly into :meth:`step`.
        """
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must be [B, T, d_model={self.d_model}], got {tuple(x.shape)}")
        B, T, _ = x.shape
        if state is None:
            S, conv_buffer = self.initial_state(B, device=x.device, dtype=torch.float32)
        else:
            S, conv_buffer = state
        # Per-token projections for the whole sequence up front.
        mixed_all = F.linear(x, self.in_proj_qkv.weight)  # [B, T, C]
        z_all = F.linear(x, self.in_proj_z.weight).reshape(
            B, T, self.num_value_heads, self.head_v_dim
        )
        b_all = F.linear(x, self.in_proj_b.weight)
        a_all = F.linear(x, self.in_proj_a.weight)
        beta_all = torch.sigmoid(b_all)
        g_all = -self.A_log.float().exp() * F.softplus(
            a_all.float() + self.dt_bias.float()
        )
        outs: list[torch.Tensor] = []
        buf = conv_buffer.to(dtype=mixed_all.dtype)
        for t in range(T):
            mixed_act, buf = self._conv_step(mixed_all[:, t, :], buf)
            out_t, S = self._recurrent_token(
                mixed_act, z_all[:, t], beta_all[:, t], g_all[:, t], S
            )
            outs.append(out_t)
        out = torch.stack(outs, dim=1).to(x.dtype) if T > 0 else torch.empty_like(x)
        return out, (S, buf)

    def forward(
        self, x: torch.Tensor, state: LinearAttentionState | None = None
    ) -> torch.Tensor:
        """HF-style forward: ``[B, T, D] -> [B, T, D]`` (output only).

        Use :meth:`forward_prefill` when the final state is needed.
        A length-1 sequence with a state is a decode step in disguise;
        it still routes through the sequential scan and stays exact.
        """
        out, _ = self.forward_prefill(x, state=state)
        return out
