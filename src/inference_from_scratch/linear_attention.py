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
  chunkwise production form, also implemented here as
  :meth:`GatedDeltaNet.forward_prefill_chunkwise`),
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
the explicit scan as the reference and adds the same chunkwise form
(:meth:`GatedDeltaNet.forward_prefill_chunkwise`) so the two can be
checked against each other. ``forward_prefill`` keeps
``method="sequential"`` as its default for compatibility: existing
callers get bit-for-bit the reference reduction order, and chunkwise
(which re-associates the arithmetic, so only agrees to fp32 tolerance)
is an explicit opt-in.
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


def _linear(module: nn.Module, x: torch.Tensor, *, cast_to_weight: bool = False) -> torch.Tensor:
    """Apply a projection module, dense or resident-FP8.

    Dense ``nn.Linear`` keeps the historic ``F.linear`` form (with the
    optional cast of ``x`` to the weight dtype the out-projection
    sites relied on); any other module -- notably
    :class:`~inference_from_scratch.fp8_linear.BlockFP8Linear`, whose
    FP8 ``weight`` buffer must never reach ``F.linear`` directly --
    is called, so it can dequantise first.
    """
    if isinstance(module, nn.Linear):
        if cast_to_weight:
            x = x.to(module.weight.dtype)
        return F.linear(x, module.weight, module.bias)
    return module(x)


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
        # Fused path (F.rms_norm); off at module level so direct
        # callers keep the manual maths. The TP2 runner enables it
        # by default via --gated-norm fused (shipped, decode round 5;
        # --gated-norm manual is the A/B baseline).
        self.use_fused_rmsnorm = False
        # Eager-call counter proving the fused branch executed (graph
        # replays do not re-run Python, so this counts captures/eager
        # calls, not replays).
        self.fused_rmsnorm_calls = 0

    def forward(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = core.dtype
        if self.use_fused_rmsnorm:
            self.fused_rmsnorm_calls += 1
            # Same maths and intermediate dtype round-trip as the
            # manual path: rms*weight is cast to input_dtype before
            # the float SiLU-gate multiply, then cast back.
            y = F.rms_norm(
                core.float(), [self.d], self.weight.float(), self.eps,
            ).to(input_dtype)
            y = y * F.silu(gate.float())
            return y.to(input_dtype)
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
        out = _linear(self.out_proj, core, cast_to_weight=True)
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
        mixed = _linear(self.in_proj_qkv, x)
        z = _linear(self.in_proj_z, x).reshape(
            x.shape[0], self.num_value_heads, self.head_v_dim
        )
        b = _linear(self.in_proj_b, x)
        a = _linear(self.in_proj_a, x)
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return mixed, z, beta, g

    # -- chunkwise pieces ----------------------------------------------------

    def _causal_conv_sequence(
        self, mixed_all: torch.Tensor, conv_buffer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal depthwise conv over a whole sequence + SiLU.

        ``mixed_all``: ``[B, T, C]`` raw projections; ``conv_buffer``:
        ``[B, C, K-1]`` raw vectors preceding the sequence (chronological).
        Returns ``(activated [B, T, C], new_buffer [B, C, K-1])`` where the
        new buffer holds the last ``K-1`` raw vectors of the extended
        (buffer + sequence) stream -- exactly the buffer the sequential
        ``_conv_step`` loop ends with. One grouped ``F.conv1d`` over the
        extended sequence, no per-token loop.
        """
        B, T, C = mixed_all.shape
        K = self.conv_kernel
        if T == 0:
            return mixed_all.clone(), conv_buffer.clone()
        mixed_t = mixed_all.transpose(1, 2)  # [B, C, T]
        if K == 1:
            extended = mixed_t
            new_buffer = conv_buffer.clone()  # [B, C, 0]
        else:
            extended = torch.cat(
                [conv_buffer.to(dtype=mixed_all.dtype), mixed_t], dim=-1
            )  # [B, C, K-1+T]
            new_buffer = extended[:, :, -(K - 1):].clone()
        weight = self.conv1d.weight.to(dtype=mixed_all.dtype)  # [C, 1, K]
        conv_out = F.conv1d(extended, weight, bias=None, padding=0, groups=C)
        # conv_out length is T in both branches (K==1: T-K+1 == T).
        return F.silu(conv_out.transpose(1, 2)), new_buffer

    @staticmethod
    def _chunk_gated_delta_core(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        S0: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise gated delta rule (HF ``torch_chunk_gated_delta_rule``).

        ``q``/``k``: ``[B, T, Hv, Dk]`` fp32, already L2-normalised with
        ``q`` scaled by ``Dk**-0.5`` and Q/K already expanded to value
        heads; ``v``: ``[B, T, Hv, Dv]`` fp32; ``beta``/``g``:
        ``[B, T, Hv]`` fp32 (``g <= 0``, log-decay); ``S0``:
        ``[B, Hv, Dk, Dv]`` fp32. Returns ``(core [B, T, Hv, Dv], S_final)``.

        Per chunk of ``C`` tokens the ``C`` sequential delta updates are
        re-associated (all shapes below per batch/head/chunk):

        * ``cum[i] = sum_{j<=i} g_j`` -- cumulative log-decay in the chunk;
          ``pairwise[i, j] = exp(cum[i] - cum[j])`` for ``i >= j`` (strictly
          upper masked to ``exp(-inf) = 0``) is the decay from ``j`` to ``i``.
        * The intra-chunk delta interactions form a unit-lower-triangular
          system ``A[i, j] = beta_i * (k_i . k_j) * pairwise[i, j]``.
          Solving ``A u = beta * v`` (forward substitution via
          ``solve_triangular(..., unitriangular=True)``; the diagonal of
          ``A`` is ignored, the implied ``I`` is the write of a token that
          no earlier chunk-mate predicted) condenses the chunk's writes
          into ``u`` (HF ``new_values`` before the state correction), and
          solving the same system against ``beta * k * exp(cum)`` gives the
          decayed keys re-expressed in that condensed basis.
        * Inter-chunk: with ``S`` the state at the chunk boundary,
          ``v_new = u - k_cumdecay @ S`` subtracts what the old state
          already predicts (the delta rule, once per chunk instead of
          once per token), ``core = (q * exp(cum)) @ S + intra @ v_new``
          adds the within-chunk reads ``intra[i, j] = (q_i . k_j) *
          pairwise[i, j]``, and the boundary state advances as
          ``S <- exp(cum[-1]) * S + (k * exp(cum[-1] - cum))^T @ v_new``.

        The only Python loop is over chunks; everything inside a chunk is
        a matmul / triangular solve over the ``C`` axis. Padding tokens
        (zero ``q``/``k``/``v``/``beta``/``g``) contribute no write and no
        read, so a padded tail chunk leaves ``S`` unchanged.
        """
        B, T, Hv, Dk = q.shape
        Dv = v.shape[-1]
        # [B, Hv, T, D] fp32 layout, as in the HF kernel.
        qh = q.transpose(1, 2).to(torch.float32).contiguous()
        kh = k.transpose(1, 2).to(torch.float32).contiguous()
        vh = v.transpose(1, 2).to(torch.float32).contiguous()
        beta_h = beta.transpose(1, 2).to(torch.float32).contiguous()  # [B, Hv, T]
        decay_h = g.transpose(1, 2).to(torch.float32).contiguous()

        pad_size = (chunk_size - T % chunk_size) % chunk_size
        if pad_size:
            qh, kh, vh = (F.pad(t, (0, 0, 0, pad_size)) for t in (qh, kh, vh))
            beta_h, decay_h = (F.pad(t, (0, pad_size)) for t in (beta_h, decay_h))
        T_pad = T + pad_size
        n_chunks = T_pad // chunk_size
        C = chunk_size

        v_beta = vh * beta_h.unsqueeze(-1)
        k_beta = kh * beta_h.unsqueeze(-1)

        # Chunk axis: [B, Hv, n_chunks, C, D] / [B, Hv, n_chunks, C].
        q_c, k_c, kb_c, vb_c = (
            t.reshape(B, Hv, n_chunks, C, t.shape[-1]) for t in (qh, kh, k_beta, v_beta)
        )
        decay_c = decay_h.reshape(B, Hv, n_chunks, C)
        cum = decay_c.cumsum(dim=3)  # cumulative log-decay within chunk

        strictly_upper = torch.ones(C, C, dtype=torch.bool, device=q.device).triu(1)
        pairwise = cum.unsqueeze(4) - cum.unsqueeze(3)  # [.., C_i, C_j]
        pairwise = pairwise.masked_fill(strictly_upper, float("-inf")).exp()

        # Intra-chunk triangular (UT) system and QK interactions.
        ut_system = (kb_c @ k_c.transpose(-1, -2)) * pairwise
        intra = (q_c @ k_c.transpose(-1, -2)) * pairwise
        decayed_k_beta = kb_c * cum.exp().unsqueeze(-1)
        new_values = torch.linalg.solve_triangular(
            ut_system, vb_c, upper=False, unitriangular=True
        )
        k_cumdecay = torch.linalg.solve_triangular(
            ut_system, decayed_k_beta, upper=False, unitriangular=True
        )

        # Decay-adjusted reads/writes against the boundary state.
        q_decayed = q_c * cum.exp().unsqueeze(-1)
        k_decayed = k_c * (cum[..., -1:] - cum).exp().unsqueeze(-1)
        chunk_decay = cum[..., -1].exp()  # [B, Hv, n_chunks]

        S = S0.to(torch.float32).clone()
        core_c = torch.empty_like(new_values)
        for i in range(n_chunks):
            v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ S
            core_c[:, :, i] = q_decayed[:, :, i] @ S + intra[:, :, i] @ v_new
            S = (
                S * chunk_decay[:, :, i][:, :, None, None]
                + k_decayed[:, :, i].transpose(-1, -2) @ v_new
            )

        core = core_c.reshape(B, Hv, T_pad, Dv)[:, :, :T]
        return core.transpose(1, 2).contiguous(), S

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

    # -- in-place decode (CUDA-graph-friendly semantics, CPU-provable) -----

    def _conv_step_inplace(
        self, raw_t: torch.Tensor, conv_buffer: torch.Tensor
    ) -> torch.Tensor:
        """In-place version of :meth:`_conv_step`.

        ``raw_t`` is ``[B, conv_dim]``; ``conv_buffer`` is the preallocated
        ``[B, conv_dim, K-1]`` ring (chronological). The activated conv
        output ``[B, conv_dim]`` is returned (a fresh intermediate), but
        ``conv_buffer`` itself is shifted and overwritten in place: no
        ``torch.cat`` and no new buffer tensor is allocated for the state.
        The conv sum is evaluated *before* the shift so the arithmetic
        order matches :meth:`_conv_step` exactly.
        """
        w = self.conv1d.weight[:, 0, :].to(dtype=raw_t.dtype)  # [C, K]
        if self.conv_kernel == 1:
            conv_out = raw_t * w[:, 0].unsqueeze(0)
            return F.silu(conv_out)
        # Build the same [B, C, K] window the legacy cat builds (an
        # activation temporary, not state) so the conv reduction order
        # is identical; the ring itself is then shifted in place below.
        window = torch.empty(
            raw_t.shape[0], self.conv_dim, self.conv_kernel,
            device=raw_t.device, dtype=raw_t.dtype)
        window[:, :, :-1].copy_(conv_buffer.to(dtype=raw_t.dtype))
        window[:, :, -1].copy_(raw_t)
        conv_out = (window * w.unsqueeze(0)).sum(dim=-1)
        # Shift the ring left by one in place (forward copy is safe: each
        # destination index is read as a source only in the previous iter
        # for a strictly earlier destination, and K-1 <= 3 here).
        for i in range(conv_buffer.shape[-1] - 1):
            conv_buffer[:, :, i].copy_(conv_buffer[:, :, i + 1])
        if conv_buffer.shape[-1] > 0:
            conv_buffer[:, :, -1].copy_(raw_t.to(dtype=conv_buffer.dtype))
        return F.silu(conv_out)

    def _recurrent_token_inplace(
        self,
        mixed_t: torch.Tensor,
        z_t: torch.Tensor,
        beta_t: torch.Tensor,
        g_t: torch.Tensor,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """In-place gated-delta update: mutate ``S`` and return ``out_t``.

        Numerically mirrors :meth:`_recurrent_token` step-for-step, but
        the decay and rank-1 write are ``mul_`` / ``add_`` on the caller's
        preallocated ``S`` instead of allocating a new state tensor.
        Intermediates (``kv_mem``, ``delta``, ``core``) are still fresh
        temporaries -- they are activations, not state.
        """
        B = mixed_t.shape[0]
        q_raw, k_raw, v = torch.split(
            mixed_t, [self.key_total, self.key_total, self.value_total], dim=-1
        )
        q = q_raw.reshape(B, self.num_key_heads, self.head_k_dim)
        k = k_raw.reshape(B, self.num_key_heads, self.head_k_dim)
        v = v.reshape(B, self.num_value_heads, self.head_v_dim).float()
        if self.n_rep > 1:
            q = q.repeat_interleave(self.n_rep, dim=1)
            k = k.repeat_interleave(self.n_rep, dim=1)
        q = _l2norm(q) * (self.head_k_dim ** -0.5)
        k = _l2norm(k)
        # 1) Gated decay in place.
        S.mul_(torch.exp(g_t.float())[:, :, None, None].to(dtype=S.dtype))
        # 2) Delta rule in place.
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta_t.float().unsqueeze(-1)
        S.add_(k.unsqueeze(-1) * delta.unsqueeze(-2))
        # 3) Read after write.
        core = (S * q.unsqueeze(-1)).sum(dim=-2)
        core = self.norm(core.reshape(B * self.num_value_heads, self.head_v_dim),
                         z_t.reshape(B * self.num_value_heads, self.head_v_dim))
        core = core.reshape(B, self.value_total)
        out = _linear(self.out_proj, core, cast_to_weight=True)
        return out

    def step_inplace(
        self, x_t: torch.Tensor, S: torch.Tensor, conv_buf: torch.Tensor
    ) -> torch.Tensor:
        """Decode one token in place: mutate ``S`` / ``conv_buf``, return out.

        ``x_t`` is ``[B, D]`` or ``[B, 1, D]`` (output rank matches input);
        ``S`` is the preallocated ``[B, Hv, Dk, Dv]`` recurrent state and
        ``conv_buf`` the preallocated ``[B, conv_dim, K-1]`` ring. Both are
        updated in place -- their ``data_ptr`` / storage are unchanged --
        unlike :meth:`step`, which never mutates its input state and
        returns a fresh ``(S_new, new_buffer)`` pair. No list append or
        ``torch.cat`` is used for state. Position is not part of this
        layer (gated-delta carries no position embedding); sequence
        position lives in the caller's static state (see
        ``hybrid.StaticHybridState``), as a Python ``int`` or 0-dim tensor
        documented there. Raises ``ValueError`` on shape mismatch.
        """
        if x_t.dim() == 2:
            x2 = x_t
            squeeze = False
        elif x_t.dim() == 3 and x_t.shape[1] == 1:
            x2 = x_t[:, 0, :]
            squeeze = True
        else:
            raise ValueError(f"x_t must be [B, D] or [B, 1, D], got {tuple(x_t.shape)}")
        if x2.shape[-1] != self.d_model:
            raise ValueError(f"x_t last dim must be d_model={self.d_model}, got {x2.shape[-1]}")
        B = x2.shape[0]
        expected_S = (B, self.num_value_heads, self.head_k_dim, self.head_v_dim)
        if tuple(S.shape) != expected_S:
            raise ValueError(f"S must be {expected_S}, got {tuple(S.shape)}")
        expected_conv = (B, self.conv_dim, max(self.conv_kernel - 1, 0))
        if tuple(conv_buf.shape) != expected_conv:
            raise ValueError(f"conv_buf must be {expected_conv}, got {tuple(conv_buf.shape)}")
        mixed, z, beta, g = self._project(x2)
        mixed_act = self._conv_step_inplace(mixed, conv_buf)
        out = self._recurrent_token_inplace(mixed_act, z, beta, g, S)
        out = out.to(x_t.dtype)
        if squeeze:
            out = out.unsqueeze(1)
        return out

    # Spelling alias (in_place with underscore).
    step_in_place = step_inplace

    def forward_prefill_chunkwise(
        self,
        x: torch.Tensor,
        chunk_size: int = 64,
        initial_state: LinearAttentionState | None = None,
        state: LinearAttentionState | None = None,
    ) -> tuple[torch.Tensor, LinearAttentionState]:
        """Chunkwise prefill: same result as the sequential scan, by chunks.

        ``x`` is ``[B, T, d_model]``; ``chunk_size`` is the chunk length
        ``C`` (default 64, as in HF ``torch_chunk_gated_delta_rule``);
        ``initial_state`` is the ``(S, conv_buffer)`` pair to continue
        from (``state`` is accepted as an alias keyword; passing both is
        an error). Returns ``(out [B, T, D], final_state)`` whose final
        state continues into :meth:`step` exactly as the sequential
        final state does. The input state is never mutated.

        Structure (no per-token Python loop anywhere): projections for
        the whole sequence at once, one grouped causal ``conv1d`` seeded
        with the incoming conv buffer, then
        :meth:`_chunk_gated_delta_core` (intra-chunk triangular solve +
        inter-chunk boundary state, gates/decay in fp32), then the same
        gated RMSNorm and ``out_proj`` as the sequential path.
        """
        if state is not None and initial_state is not None:
            raise ValueError("pass only one of initial_state / state")
        if initial_state is None:
            initial_state = state
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must be [B, T, d_model={self.d_model}], got {tuple(x.shape)}")
        B, T, _ = x.shape
        if initial_state is None:
            S0, conv_buffer = self.initial_state(B, device=x.device, dtype=torch.float32)
        else:
            S0, conv_buffer = initial_state
        if T == 0:
            return torch.empty_like(x), (S0.clone(), conv_buffer.clone())

        # Per-token projections for the whole sequence up front (fp32 gates).
        mixed_all = _linear(self.in_proj_qkv, x)  # [B, T, C]
        z_all = _linear(self.in_proj_z, x).reshape(
            B, T, self.num_value_heads, self.head_v_dim
        )
        b_all = _linear(self.in_proj_b, x)
        a_all = _linear(self.in_proj_a, x)
        beta_all = torch.sigmoid(b_all.float())  # [B, T, Hv] fp32
        g_all = -self.A_log.float().exp() * F.softplus(
            a_all.float() + self.dt_bias.float()
        )  # [B, T, Hv] fp32, <= 0

        # Causal conv over the sequence, seeded with the incoming buffer.
        mixed_act, new_buffer = self._causal_conv_sequence(
            mixed_all, conv_buffer.to(dtype=mixed_all.dtype)
        )

        q_raw, k_raw, v = torch.split(
            mixed_act, [self.key_total, self.key_total, self.value_total], dim=-1
        )
        q = q_raw.reshape(B, T, self.num_key_heads, self.head_k_dim)
        k = k_raw.reshape(B, T, self.num_key_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_value_heads, self.head_v_dim).float()
        if self.n_rep > 1:
            q = q.repeat_interleave(self.n_rep, dim=2)
            k = k.repeat_interleave(self.n_rep, dim=2)
        q = _l2norm(q) * (self.head_k_dim ** -0.5)
        k = _l2norm(k)

        core, S_final = self._chunk_gated_delta_core(
            q, k, v, beta_all, g_all, S0, chunk_size
        )  # core [B, T, Hv, Dv] fp32

        # Final gated norm + output projection (same tail as _recurrent_token).
        core = self.norm(
            core.reshape(B * T * self.num_value_heads, self.head_v_dim),
            z_all.reshape(B * T * self.num_value_heads, self.head_v_dim),
        )
        core = core.reshape(B, T, self.value_total)
        out = _linear(self.out_proj, core, cast_to_weight=True)
        return out.to(x.dtype), (S_final, new_buffer.to(dtype=mixed_all.dtype))

    def forward_prefill(
        self,
        x: torch.Tensor,
        state: LinearAttentionState | None = None,
        method: str = "sequential",
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, LinearAttentionState]:
        """Prefill ``x`` ``[B, T, D]``; ``method`` selects the evaluation.

        ``method="sequential"`` (default) is the explicit per-token scan:
        loops over ``T`` calling the same conv + recurrence step as
        decode (projections are computed once -- they are per-token
        linears with no cross-token mixing, so this changes nothing
        mathematically). ``method="chunkwise"`` delegates to
        :meth:`forward_prefill_chunkwise` with ``chunk_size``.

        The default stays ``"sequential"`` for compatibility: every
        existing caller keeps the reference reduction order bit-for-bit,
        while chunkwise re-associates the arithmetic and is therefore
        only equal to fp32 tolerance -- an opt-in, not a silent swap.
        Returns ``(out [B, T, D], final_state)``; the final state
        continues seamlessly into :meth:`step` under either method.
        """
        if not isinstance(method, str) or method.strip().lower() not in (
            "sequential",
            "chunkwise",
        ):
            raise ValueError(
                f"method must be 'sequential' or 'chunkwise', got {method!r}"
            )
        if method.strip().lower() == "chunkwise":
            return self.forward_prefill_chunkwise(
                x, chunk_size=chunk_size, initial_state=state
            )
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must be [B, T, d_model={self.d_model}], got {tuple(x.shape)}")
        B, T, _ = x.shape
        if state is None:
            S, conv_buffer = self.initial_state(B, device=x.device, dtype=torch.float32)
        else:
            S, conv_buffer = state
        # Per-token projections for the whole sequence up front.
        mixed_all = _linear(self.in_proj_qkv, x)  # [B, T, C]
        z_all = _linear(self.in_proj_z, x).reshape(
            B, T, self.num_value_heads, self.head_v_dim
        )
        b_all = _linear(self.in_proj_b, x)
        a_all = _linear(self.in_proj_a, x)
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
        self,
        x: torch.Tensor,
        state: LinearAttentionState | None = None,
        method: str = "sequential",
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """HF-style forward: ``[B, T, D] -> [B, T, D]`` (output only).

        Use :meth:`forward_prefill` when the final state is needed.
        A length-1 sequence with a state is a decode step in disguise;
        it still routes through the sequential scan and stays exact.
        ``method`` / ``chunk_size`` are forwarded to
        :meth:`forward_prefill` (default stays sequential).
        """
        out, _ = self.forward_prefill(x, state=state, method=method, chunk_size=chunk_size)
        return out
