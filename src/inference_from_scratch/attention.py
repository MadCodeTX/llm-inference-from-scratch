"""Ch 04 -- Attention: manual scaled dot-product, causal mask, GQA.

Why this exists in inference
------------------------------
Attention is the only place tokens talk to each other. During prefill
it is a big parallel matmul over ``T`` queries; during decode it is one
query dotted against every cached key. Writing it out longhand --
scores, scale, mask, softmax, weighted sum -- makes both regimes (and
the KV-cache contract between them) visible instead of hiding them
inside ``F.scaled_dot_product_attention``.

How production differs
-----------------------
FlashAttention / ``F.scaled_dot_product_attention`` compute exactly
this maths but tile the ``[T, T]`` score matrix through SRAM so it is
never materialised in HBM, and fuse scale+mask+softmax. The manual
version here materialises scores on purpose: it is the reference the
tests (and Ch 12) compare the fast path against.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig
from .kv_cache import KVCache, StaticKVCache
from .layers import apply_rope


def manual_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention built from matmul + softmax only.

    Shapes:
        q: ``[B, H, T_q, D_h]`` -- queries
        k: ``[B, H, T_k, D_h]`` -- keys (same head count / head_dim as q;
           expand KV heads with :func:`repeat_kv` *before* calling)
        v: ``[B, H, T_k, D_v]`` -- values (usually ``D_v == D_h``)

    Returns ``[B, H, T_q, D_v]``.

    The steps, in order (this ordering is the whole chapter):

    1. scores  = q @ k^T            -- raw affinities
    2. scores /= sqrt(D_h)          -- scale, keeps softmax out of its
                                       saturated tails as D_h grows
    3. causal mask (optional)       -- future positions set to -inf so
                                       softmax gives them exactly 0
    4. weights = softmax(scores)    -- rows sum to 1 over keys
    5. out     = weights @ v        -- weighted sum of values

    Causal alignment when ``T_q != T_k``: queries are assumed to be the
    *last* ``T_q`` positions of the key sequence (the decode case:
    one new query at absolute position ``T_k - 1`` may see every key,
    including itself). Query row ``i`` therefore sits at absolute
    position ``i + (T_k - T_q)`` and may attend keys ``j <= that``.
    For the usual prefill case ``T_q == T_k`` this is the plain
    lower-triangular mask.
    """
    # q: [B, H, T_q, D_h]; k: [B, H, T_k, D_h]; v: [B, H, T_k, D_v]
    d_h = q.shape[-1]  # scalar: per-head width drives the scale

    # Step 1+2: scores [B, H, T_q, D_h] @ [B, H, D_h, T_k] -> [B, H, T_q, T_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

    if is_causal:
        t_q, t_k = scores.shape[-2], scores.shape[-1]  # scalars
        # Absolute position of each query row: [T_q] (0-based, offset by
        # T_k - T_q so a lone decode query lands on the last position).
        q_pos = torch.arange(t_q, device=q.device) + (t_k - t_q)  # [T_q]
        k_pos = torch.arange(t_k, device=q.device)  # [T_k]
        # future [T_q, T_k]: True where key position > query position.
        future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [T_q, T_k]
        # -inf (not a large negative) so softmax yields exact zeros and
        # masked positions contribute nothing -- not even rounding.
        scores = scores.masked_fill(future, float("-inf"))  # [B, H, T_q, T_k]

    # Step 4: attn_weights [B, H, T_q, T_k], each row sums to 1.
    attn_weights = torch.softmax(scores, dim=-1)  # [B, H, T_q, T_k]

    # Step 5: [B, H, T_q, T_k] @ [B, H, T_k, D_v] -> [B, H, T_q, D_v]
    return torch.matmul(attn_weights, v)  # [B, H, T_q, D_v]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat each KV head ``n_rep`` times (GQA -> MHA head layout).

    ``x``: ``[B, H_kv, T, D_h]`` -> ``[B, H_kv * n_rep, T, D_h]``.

    Head ordering is *grouped*, matching how query heads are laid out:
    KV head ``g`` serves query heads ``g*n_rep .. (g+1)*n_rep - 1``, so
    the output head order is ``[kv0, kv0, ..., kv1, kv1, ...]`` -- each
    KV head's copies are consecutive. (Interleaving instead would pair
    every query head with the wrong KV head and silently corrupt GQA.)

    This materialises the expanded tensor for clarity. Production GQA
    kernels index the shared KV head without copying; the maths (and
    the KV-cache saving: only ``H_kv`` heads are ever stored) is the
    same either way.
    """
    if n_rep < 1:
        raise ValueError(f"n_rep must be >= 1, got {n_rep}")
    if n_rep == 1:
        return x  # [B, H_kv, T, D_h] already has the right head count
    B, H_kv, T, D_h = x.shape  # scalars unpacked from [B, H_kv, T, D_h]
    # Insert a repeat axis: [B, H_kv, T, D_h] -> [B, H_kv, 1, T, D_h]
    expanded = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, D_h)
    # Merge (H_kv, n_rep): [B, H_kv, n_rep, T, D_h] -> [B, H_kv*n_rep, T, D_h]
    return expanded.reshape(B, H_kv * n_rep, T, D_h)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with GQA, RoPE and KV-cache support.

    Separate ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` Linear
    layers (all bias-free, Llama/Qwen style, and *no dropout anywhere*:
    inference is deterministic and the cache-equivalence tests require
    prefill and decode to agree exactly).

    Widths from ``config``: queries span all ``n_head`` heads, keys and
    values only ``n_kv_head`` heads, so the projections are::

        q_proj: [d_model -> n_head    * head_dim]   (= d_model)
        k_proj: [d_model -> n_kv_head * head_dim]
        v_proj: [d_model -> n_kv_head * head_dim]
        o_proj: [n_head * head_dim -> d_model]      (= d_model)

    With ``n_kv_head < n_head`` this is grouped-query attention: the
    KV cache stores ``n_kv_head`` heads and :func:`repeat_kv` expands
    them to meet the queries at attention time.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = config.n_head // config.n_kv_head  # Q heads per KV head

        D = config.d_model  # residual width
        Dh = config.head_dim  # per-head width
        # x [B, T, D] -> q flat [B, T, H * Dh]
        self.q_proj = nn.Linear(D, config.n_head * Dh, bias=False)
        # x [B, T, D] -> k/v flat [B, T, H_kv * Dh] (narrower under GQA)
        self.k_proj = nn.Linear(D, config.n_kv_head * Dh, bias=False)
        self.v_proj = nn.Linear(D, config.n_kv_head * Dh, bias=False)
        # merged heads [B, T, H * Dh] -> y [B, T, D]
        self.o_proj = nn.Linear(config.n_head * Dh, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        kv_cache: KVCache | StaticKVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Run attention over ``x``: ``[B, T, D] -> [B, T, D]``.

        ``cos`` / ``sin`` are the ``[T_max, D_h]`` RoPE tables from
        ``precompute_rope_cache`` (pass ``None`` to skip RoPE, e.g. the
        learned-position variant). ``start_pos`` is the absolute
        position of ``x``'s first token: 0 for prefill, >0 for decode.

        Cache contract: with ``kv_cache`` given, this step's K/V are
        appended (at ``start_pos`` for :class:`StaticKVCache`) and
        attention reads back *all* cached positions. Without a cache,
        attention is over ``x`` alone. Either way causality is by
        absolute position, so a decode step (``T == 1``,
        ``start_pos > 0``) attends to every cached token and itself,
        and chunked prefill (``T > 1``, ``start_pos > 0``) masks only
        genuinely future keys.
        """
        B, T, D = x.shape  # x: [B, T, d_model]
        H, H_kv, Dh = self.n_head, self.n_kv_head, self.head_dim  # scalars

        # --- Project: [B, T, D] -> per-head [B, H, T, Dh] --------------
        # q flat [B, T, H*Dh] -> view [B, T, H, Dh] -> transpose [B, H, T, Dh]
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        # k flat [B, T, H_kv*Dh] -> [B, H_kv, T, Dh] (fewer heads: GQA)
        k = self.k_proj(x).view(B, T, H_kv, Dh).transpose(1, 2)  # [B, H_kv, T, Dh]
        # v flat [B, T, H_kv*Dh] -> [B, H_kv, T, Dh]
        v = self.v_proj(x).view(B, T, H_kv, Dh).transpose(1, 2)  # [B, H_kv, T, Dh]

        # --- RoPE on q/k only (values carry no position) ----------------
        if cos is not None and sin is not None:
            # [B, H, T, Dh] rotated at absolute positions start_pos..+T
            q = apply_rope(q, cos, sin, start_pos=start_pos)  # [B, H, T, Dh]
            # [B, H_kv, T, Dh]: cached keys are rotated once, here,
            # before being stored -- decode never re-rotates them.
            k = apply_rope(k, cos, sin, start_pos=start_pos)  # [B, H_kv, T, Dh]
            # v deliberately untouched: [B, H_kv, T, Dh]

        # --- KV cache: append this step, read back everything -----------
        if kv_cache is not None:
            if isinstance(kv_cache, StaticKVCache):
                # Slice-write at start_pos; returns filled prefix only.
                k, v = kv_cache.update(layer_idx, k, v, start_pos)  # [B, H_kv, T_all, Dh]
            else:
                # Concat cache appends on the T dim (dim 2).
                k, v = kv_cache.update(layer_idx, k, v)  # [B, H_kv, T_all, Dh]
            # T_all == start_pos + T in normal (contiguous) use.

        # --- GQA: expand KV heads to meet the query heads ---------------
        # [B, H_kv, T_all, Dh] -> [B, H, T_all, Dh] (no-op when MHA)
        k = repeat_kv(k, self.n_rep)  # [B, H, T_all, Dh]
        v = repeat_kv(v, self.n_rep)  # [B, H, T_all, Dh]

        # --- Manual attention (causal by absolute position) -------------
        # scores inside: [B, H, T, T_all]; out: [B, H, T, Dh].
        # is_causal=True is correct for decode too: the manual op
        # aligns queries to the last T key positions (= start_pos..),
        # so a T==1 query at start_pos>0 masks nothing.
        attn = manual_scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, Dh]

        # --- Merge heads and output-project -----------------------------
        # [B, H, T, Dh] -> transpose [B, T, H, Dh] -> flatten [B, T, H*Dh]
        merged = attn.transpose(1, 2).contiguous().view(B, T, H * Dh)  # [B, T, d_model]
        # [B, T, d_model] -> [B, T, d_model]
        return self.o_proj(merged)  # [B, T, d_model]
