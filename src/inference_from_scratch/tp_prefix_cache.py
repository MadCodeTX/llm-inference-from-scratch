"""Block prefix caching for TPStaticState (Kiely sec 5.3 style).

Caches a snapshot of the FULL TPStaticState at each block boundary
(block_size tokens, default 16). A later prompt sharing a block-aligned
prefix restores the snapshot instead of re-prefilling those tokens.

Rank consistency: the cache logic is identical on every rank. Keys are
derived only from token ids (identical on all ranks) via hashlib over
int64 bytes -- no Python hash(), no collectives. Each rank stores /
restores only its own shard tensors from its local cache instance;
because every rank computes the same hits, all ranks restore the same
logical position and proceed in lockstep through the same collectives
in the suffix prefill. No communication is needed for the cache itself.

No aliasing: store clones (KV only up to filled length), restore uses
copy_ into the provided state. Mutating live state never corrupts cache.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

# Root parent hash: hash of empty prefix.
ROOT_HASH = hashlib.sha256(b"tp-prefix-cache-root-v1").digest()


class BlockHasher:
    """Chain hash: parent_hash + block token tuple -> stable hash."""

    def __init__(self, block_size: int = 16):
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size must be positive int, got {block_size!r}")
        self.block_size = int(block_size)

    def hash_block(self, parent_hash: bytes, tokens) -> bytes:
        toks = [int(t) for t in tokens]
        if len(toks) != self.block_size:
            raise ValueError(f"block must have {self.block_size} tokens, got {len(toks)}")
        arr = torch.tensor(toks, dtype=torch.int64)
        h = hashlib.sha256()
        h.update(bytes(parent_hash))
        h.update(arr.numpy().tobytes())
        return h.digest()

    def chain_hashes(self, token_ids) -> list[bytes]:
        """One chained hash per full block, in order."""
        toks = [int(t) for t in token_ids]
        out, parent = [], ROOT_HASH
        bs = self.block_size
        for i in range(0, len(toks) - bs + 1, bs):
            parent = self.hash_block(parent, toks[i:i + bs])
            out.append(parent)
        return out

    @staticmethod
    def key(h: bytes) -> str:
        return h.hex()


def _tbytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def snapshot_bytes_of(kv_k, kv_v, linear_S, conv, position: int) -> int:
    total = 8  # position int64
    for lst in (kv_k, kv_v, linear_S, conv):
        for t in lst:
            if t is not None:
                total += _tbytes(t)
    return total


def state_snapshot_bytes(state: Any, filled: int | None = None) -> int:
    """Measured bytes a snapshot of state at `filled` would occupy."""
    if filled is None:
        filled = int(state.position.item()) if hasattr(state, "position") else state.seq_len
    total = 8
    for i in range(len(state.layer_types)):
        if state.kv_k[i] is not None:
            total += _tbytes(state.kv_k[i][:, :, :filled])
            total += _tbytes(state.kv_v[i][:, :, :filled])
        if state.linear_S[i] is not None:
            total += _tbytes(state.linear_S[i])
            total += _tbytes(state.linear_conv_buffers[i])
    return total


# Alias named in task spec: state_bytes() helper using numel*itemsize.
def state_bytes(state: Any, filled: int | None = None) -> int:
    return state_snapshot_bytes(state, filled)


@dataclass
class _Entry:
    kv_k: list
    kv_v: list
    linear_S: list
    linear_conv: list
    position: int
    nbytes: int


class TPBlockPrefixCache:
    def __init__(self, block_size: int = 16, max_bytes: int | None = None,
                 max_entries: int | None = None, device: str = "cpu",
                 storage: str | None = None):
        # storage alias: task says device storage='cpu'|'gpu'
        if storage is not None:
            device = storage
        if device not in ("cpu", "gpu", "cuda"):
            raise ValueError(f"storage device must be cpu|gpu, got {device!r}")
        self.block_size = int(BlockHasher(block_size).block_size)
        self.hasher = BlockHasher(self.block_size)
        self.max_bytes = int(max_bytes) if max_bytes is not None else None
        self.max_entries = int(max_entries) if max_entries is not None else None
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.storage_device = torch.device("cuda" if device in ("gpu", "cuda") else "cpu")
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    # -- snapshot / restore -------------------------------------------------
    def _snapshot(self, state) -> _Entry:
        filled = int(state.position.item())
        dev = self.storage_device
        kv_k, kv_v, lin, conv = [], [], [], []
        for i in range(len(state.layer_types)):
            if state.kv_k[i] is not None:
                kv_k.append(state.kv_k[i][:, :, :filled].clone().to(dev))
                kv_v.append(state.kv_v[i][:, :, :filled].clone().to(dev))
            else:
                kv_k.append(None); kv_v.append(None)
            if state.linear_S[i] is not None:
                lin.append(state.linear_S[i].clone().to(dev))
                conv.append(state.linear_conv_buffers[i].clone().to(dev))
            else:
                lin.append(None); conv.append(None)
        nb = snapshot_bytes_of(kv_k, kv_v, lin, conv, filled)
        return _Entry(kv_k, kv_v, lin, conv, filled, nb)

    def store(self, key_hash: bytes, state) -> None:
        key = BlockHasher.key(key_hash) if isinstance(key_hash, (bytes, bytearray)) else str(key_hash)
        entry = self._snapshot(state)
        if key in self._entries:
            self._total_bytes -= self._entries[key].nbytes
            del self._entries[key]
        self._entries[key] = entry
        self._total_bytes += entry.nbytes
        self._evict()

    put = store

    def get(self, key_hash) -> _Entry | None:
        key = BlockHasher.key(key_hash) if isinstance(key_hash, (bytes, bytearray)) else str(key_hash)
        e = self._entries.get(key)
        if e is not None:
            self._entries.move_to_end(key)
        return e

    def restore(self, key_hash, state) -> bool:
        e = self.get(key_hash)
        if e is None:
            return False
        self.restore_entry(e, state)
        return True

    def restore_entry(self, entry: _Entry, state) -> None:
        filled = entry.position
        for i in range(len(state.layer_types)):
            if state.kv_k[i] is not None:
                state.kv_k[i].zero_()
                state.kv_v[i].zero_()
                if filled:
                    state.kv_k[i][:, :, :filled].copy_(entry.kv_k[i])
                    state.kv_v[i][:, :, :filled].copy_(entry.kv_v[i])
            if state.linear_S[i] is not None:
                state.linear_S[i].copy_(entry.linear_S[i])
                state.linear_conv_buffers[i].copy_(entry.linear_conv[i])
        state.position.fill_(int(filled))

    def _evict(self) -> None:
        while self._entries and (
                (self.max_bytes is not None and self._total_bytes > self.max_bytes)
                or (self.max_entries is not None and len(self._entries) > self.max_entries)):
            _, old = self._entries.popitem(last=False)
            self._total_bytes -= old.nbytes

    # -- lookup -------------------------------------------------------------
    def find_longest_prefix(self, token_ids):
        """Return (n_reused, last_hash_or_None, entry_or_None). Does not count stats."""
        hashes = self.hasher.chain_hashes(token_ids)
        n, last, entry = 0, None, None
        for h in hashes:
            e = self.get(h)
            if e is None:
                break
            n += self.block_size
            last, entry = h, e
        return n, last, entry

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses,
                "blocks": len(self._entries), "entries": len(self._entries),
                "tokens_saved": self.tokens_saved, "bytes": self._total_bytes}


def prefill_with_prefix_cache(model, ids, state, cache: TPBlockPrefixCache,
                              chunk_size: int = 128):
    """Prefill ids ([T] or [B,T]) reusing cached block prefix.

    Returns (logits [B,V], n_reused, stats_delta). State is reset first,
    longest block-aligned prefix restored, suffix prefilled via
    prefill_chunkwise_static in block-aligned pieces (snapshotting each
    newly completed block boundary). Block-aligned piece splitting keeps
    chunk boundaries at identical absolute positions on cold and warm
    runs, so warm logits are bit-exact vs a cold call with same args.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive int, got {chunk_size!r}")
    idx = ids
    if isinstance(idx, (list, tuple)):
        idx = torch.tensor(idx, dtype=torch.long)
    if idx.dim() == 1:
        idx2 = idx.unsqueeze(0)
    elif idx.dim() == 2:
        idx2 = idx
    else:
        raise ValueError(f"ids must be [T] or [B,T], got {tuple(idx.shape)}")
    B, T = int(idx2.shape[0]), int(idx2.shape[1])
    # Prefix is defined by token sequence; multi-row needs identical rows.
    row = [int(t) for t in idx2[0].tolist()] if T else []
    if B > 1 and T:
        for r in range(1, B):
            if [int(t) for t in idx2[r].tolist()] != row:
                raise ValueError("prefix cache requires identical rows for B>1")
    if T and (int(state.position.item()) + 0) and False:
        pass
    if T > state.max_seq_len:
        raise ValueError(f"prompt {T} exceeds max_seq_len={state.max_seq_len}")
    hits0, miss0, saved0 = cache.hits, cache.misses, cache.tokens_saved

    hashes = cache.hasher.chain_hashes(row)
    # Longest prefix (count stats: hit per found block, one miss if stopped early)
    n_reused, last_entry = 0, None
    for h in hashes:
        e = cache.get(h)
        if e is None:
            cache.misses += 1
            break
        cache.hits += 1
        n_reused += cache.block_size
        last_entry = e
    else:
        if not hashes:
            cache.misses += 1
    # Full-hit back-off one block so last-token logits are computed exactly.
    backed_off = False
    if n_reused == T and T >= cache.block_size and n_reused > 0:
        n_reused -= cache.block_size
        backed_off = True
        # recompute entry for reduced prefix
        last_entry = cache.get(hashes[n_reused // cache.block_size - 1]) if n_reused else None
    cache.tokens_saved += n_reused

    state.reset()
    if n_reused and last_entry is not None:
        cache.restore_entry(last_entry, state)

    if T == 0:
        empty = torch.empty(B, 0, int(model.config.vocab_size), device=model.ctx.device)
        delta = {"hits": cache.hits - hits0, "misses": cache.misses - miss0,
                 "tokens_saved": n_reused, "n_reused": n_reused, "blocks_stored": 0}
        return empty, 0, delta

    # Prefill suffix in block-aligned pieces, snapshot at boundaries.
    logits = None
    blocks_stored = 0
    pos = n_reused
    while pos < T:
        next_boundary = ((pos // cache.block_size) + 1) * cache.block_size
        end = min(next_boundary, T)
        logits = model.prefill_chunkwise_static(idx2[:, pos:end], state, chunk_size=chunk_size)
        pos = end
        if pos % cache.block_size == 0 and pos <= len(hashes) * cache.block_size:
            cache.store(hashes[pos // cache.block_size - 1], state)
            blocks_stored += 1
    delta = {"hits": cache.hits - hits0, "misses": cache.misses - miss0,
             "tokens_saved": n_reused, "n_reused": n_reused,
             "blocks_stored": blocks_stored, "blocks": len(cache),
             "bytes": cache.total_bytes}
    return logits, n_reused, delta
