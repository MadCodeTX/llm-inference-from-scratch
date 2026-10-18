"""Ch 05 -- KV cache: never recompute keys/values for old tokens.

Why this exists in inference
------------------------------
Autoregressive decode emits one token at a time, and token ``t``'s
attention needs the keys and values of *all* previous tokens. Keys and
values are pure functions of their own token (plus its RoPE position),
so recomputing them every step would turn decode into O(T^2) work per
token. The KV cache stores them once: prefill writes ``T`` entries in
one shot, each decode step appends exactly one entry per layer.

Shapes / memory layout
-----------------------
Per layer, K and V are each ``[B, H_kv, T, D_h]`` -- batch, KV heads,
time, head_dim. Time is the append axis (dim 2): new entries are
logically "to the right" of old ones, and attention reads the whole
``[.., :T_filled, :]`` prefix. Total bytes are
``2 (K and V) * n_layer * B * H_kv * T * D_h * bytes_per_elem`` --
for long contexts this dwarfs the weights, which is why GQA
(``H_kv < H``), paged allocation (see ``serving.py``) and KV
quantization exist. ``quantization.kv_cache_bytes`` does this maths.

Why decode is memory-bound
--------------------------
Each decode step does O(T) arithmetic per head (one query dot every
cached key) but must *read* the entire ``[H_kv, T, D_h]`` K and V
blocks from HBM to do it: arithmetic intensity ~1 flop per byte.
Prefill is compute-bound (big matmuls reuse data); decode streams the
cache (plus all weights) for a single token. That asymmetry -- not
FLOPs -- is why batching (amortising weight reads) and shrinking the
cache (GQA, int8 KV) are the main decode optimizations.

How production differs
-----------------------
vLLM never concatenates and never keeps one contiguous ``[B,..,T,..]``
tensor per request: PagedAttention carves the cache into fixed blocks
scattered across GPU memory, addressed by a block table (see
``serving.BlockAllocator``). The two classes below are the textbook
versions that make the trade-off visible: :class:`KVCache` is simple
but copies on every append; :class:`StaticKVCache` preallocates the
worst case once and writes in place.
"""

from __future__ import annotations

import torch


class KVCache:
    """Concatenating KV cache: one growing tensor per layer.

    ``update`` appends along the time dim (dim 2) with ``torch.cat``
    and returns the full ``(k_all, v_all)`` so attention can consume
    them directly. Dead simple and obviously correct -- but ``cat``
    allocates a fresh tensor and copies all previous entries on every
    decode step, making total decode copying O(T^2). Fine for a
    dissection (and for prefill, which cats exactly once); production
    uses :class:`StaticKVCache`-style preallocation or paged blocks.
    """

    def __init__(self, n_layer: int):
        self.n_layer = n_layer
        # None until first update: layers fill lazily and (in this
        # model) all in the same step, but nothing assumes that.
        self.keys: list[torch.Tensor | None] = [None] * n_layer
        self.values: list[torch.Tensor | None] = [None] * n_layer

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``k``/``v`` ``[B, H_kv, T_new, D_h]`` for one layer.

        Returns the full cached ``(k_all, v_all)`` with
        ``T_all == T_old + T_new``, old entries first (positions are
        order: attention's causal mask assumes index == position).
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], k], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    @property
    def seq_len(self) -> int:
        """Filled time length (0 before any update). All layers agree
        in normal use; the first non-empty layer is authoritative."""
        for k in self.keys:
            if k is not None:
                return k.shape[2]
        return 0

    def clear(self) -> None:
        """Drop all entries (new request / sequence reuse of the
        object). Tensors are released, not zeroed -- concat cache owns
        no buffer worth keeping."""
        self.keys = [None] * self.n_layer
        self.values = [None] * self.n_layer


class StaticKVCache:
    """Preallocated KV cache: allocate ``T_max`` once, write in place.

    Buffers are ``k_cache`` / ``v_cache`` of shape
    ``[n_layer, B, H_kv, T_max, D_h]``. ``update`` slice-writes the new
    entries at ``start_pos`` and returns *views of only the filled
    prefix* ``[..., :start_pos + T_new, :]`` -- the unfilled tail is
    uninitialised capacity, never valid attention input. No copying on
    append (O(1) per decode step beyond the write itself), at the cost
    of reserving worst-case memory up front: exactly the trade
    production systems refine with paged / dynamic allocation.

    ``_len`` tracks the filled length per the usual invariant that
    every layer advances together; it is kept as a single counter
    because a forward pass always writes all layers for the same
    positions.
    """

    def __init__(
        self,
        n_layer: int,
        batch_size: int = 1,
        n_kv_head: int = 1,
        max_seq_len: int = 128,
        head_dim: int = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        if dtype is None:
            dtype = torch.float32
        shape = (n_layer, batch_size, n_kv_head, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self._len = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``k``/``v`` ``[B, H_kv, T_new, D_h]`` at ``start_pos``.

        Returns views ``[B, H_kv, start_pos + T_new, D_h]`` covering
        only filled positions. Raises on overflow rather than silently
        truncating: an over-long sequence is a caller bug (it should
        have been rejected against ``max_seq_len`` upstream).
        """
        T_new = k.shape[2]
        end = start_pos + T_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: positions [{start_pos}, {end}) exceed "
                f"max_seq_len={self.max_seq_len}"
            )
        self.k_cache[layer_idx, :, :, start_pos:end] = k
        self.v_cache[layer_idx, :, :, start_pos:end] = v
        self._len = max(self._len, end)
        return (
            self.k_cache[layer_idx, :, :, :end],
            self.v_cache[layer_idx, :, :, :end],
        )

    @property
    def seq_len(self) -> int:
        """Filled time length (high-water mark of written positions)."""
        return self._len

    def clear(self) -> None:
        """Reset to empty. Buffers are kept (that is the point of
        preallocation) and zeroed so stale entries can never leak into
        a later request that reads past its own filled prefix."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._len = 0
