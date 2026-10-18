"""Block prefix caching for TPStaticState (Kiely sec 5.3 style).

Caches a snapshot of the FULL TPStaticState at sparse snapshot
boundaries. Hashing remains at ``block_size`` granularity (default 16),
but snapshots are stored only every ``snapshot_interval_tokens``
(default 256, a multiple of ``block_size``) plus optionally at the final
prompt end. A later prompt sharing a prefix restores the longest
STORED boundary <= the shared prefix; the unstored tail of the shared
prefix (< one interval) is recomputed. Entry keys encode the token
count: ``(chain_hash_at_boundary, boundary_len)``.

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


def _uniform_filled(state: Any) -> int:
    """Common per-row filled length of a TPStaticState (host sync).

    Prefix caching is defined for rows in lockstep, so divergent
    per-row positions (batched state after a selective reset) raise.
    """
    if hasattr(state, "position"):
        vals = [int(v) for v in state.position.reshape(-1).tolist()]
        if any(v != vals[0] for v in vals):
            raise ValueError(f"prefix cache requires uniform per-row positions, got {vals}")
        return int(vals[0])
    return int(state.seq_len)


def state_snapshot_bytes(state: Any, filled: int | None = None) -> int:
    """Measured bytes a snapshot of state at `filled` would occupy."""
    if filled is None:
        filled = _uniform_filled(state)
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


def _validate_interval(snapshot_interval_tokens, block_size) -> int:
    if (isinstance(snapshot_interval_tokens, bool)
            or not isinstance(snapshot_interval_tokens, int)
            or snapshot_interval_tokens <= 0):
        raise ValueError(
            f"snapshot_interval_tokens must be positive int, got {snapshot_interval_tokens!r}")
    if snapshot_interval_tokens % int(block_size) != 0:
        raise ValueError(
            f"snapshot_interval_tokens={snapshot_interval_tokens} must be a multiple "
            f"of block_size={block_size} (boundaries must be block-aligned)")
    return int(snapshot_interval_tokens)


def entry_key(chain_hash: bytes, boundary_len: int) -> str:
    """Cache key encoding both the chain hash and the token count."""
    return f"{BlockHasher.key(chain_hash)}@{int(boundary_len)}"


@dataclass
class _Entry:
    kv_k: list
    kv_v: list
    linear_S: list
    linear_conv: list
    position: int
    nbytes: int
    chain_hash: bytes | None = None


class TPBlockPrefixCache:
    def __init__(self, block_size: int = 16, max_bytes: int | None = None,
                 max_entries: int | None = None, device: str = "cpu",
                 storage: str | None = None,
                 snapshot_interval_tokens: int = 256,
                 store_final: bool = True):
        # storage alias: task says device storage='cpu'|'gpu'
        if storage is not None:
            device = storage
        if device not in ("cpu", "gpu", "cuda"):
            raise ValueError(f"storage device must be cpu|gpu, got {device!r}")
        self.block_size = int(BlockHasher(block_size).block_size)
        self.hasher = BlockHasher(self.block_size)
        self.snapshot_interval_tokens = _validate_interval(
            snapshot_interval_tokens, self.block_size)
        self.store_final = bool(store_final)
        self.snapshots_taken = 0
        self.snapshot_bytes_copied = 0
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
        filled = _uniform_filled(state)
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

    def store(self, key_hash: bytes, state, boundary_len: int | None = None) -> None:
        """Store a snapshot. Key encodes (chain hash, token count).

        ``boundary_len`` defaults to the state's filled length. A store
        whose key already exists is deduplicated (no second clone/copy)
        but still counts as a snapshot event in ``snapshots_taken``.
        """
        self.snapshots_taken += 1
        filled = _uniform_filled(state)
        length = int(filled if boundary_len is None else boundary_len)
        if length != filled:
            raise ValueError(f"boundary_len={length} != state filled={filled}")
        ch = bytes(key_hash) if isinstance(key_hash, (bytes, bytearray)) else None
        key = entry_key(ch, length) if ch is not None else str(key_hash)
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        entry = self._snapshot(state)
        entry.chain_hash = ch
        self._entries[key] = entry
        self._total_bytes += entry.nbytes
        self.snapshot_bytes_copied += entry.nbytes
        self._evict()

    put = store

    def get(self, key_hash, boundary_len: int | None = None) -> _Entry | None:
        if isinstance(key_hash, (bytes, bytearray)):
            if boundary_len is not None:
                key = entry_key(bytes(key_hash), int(boundary_len))
                e = self._entries.get(key)
                if e is not None:
                    self._entries.move_to_end(key)
                return e
            # Legacy bare-hash lookup: longest entry carrying this hash.
            best_key, best = None, None
            for k, e in self._entries.items():
                if e.chain_hash is not None and bytes(e.chain_hash) == bytes(key_hash):
                    if best is None or e.position >= best.position:
                        best_key, best = k, e
            if best_key is not None:
                self._entries.move_to_end(best_key)
            return best
        key = str(key_hash)
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
    def _hash_at(self, hashes: list[bytes], length: int, row=None) -> bytes:
        """Hash covering exactly the first ``length`` tokens.

        Block-aligned lengths use the chain hash; a partial final block
        hashes the parent chain hash plus the partial int64 tokens, so
        two prompts sharing only the block floor never collide.
        """
        if length < self.block_size:
            parent, part = ROOT_HASH, (row[:length] if row is not None else [])
            if not part:
                return ROOT_HASH
            h = hashlib.sha256(); h.update(parent)
            h.update(torch.tensor([int(t) for t in part], dtype=torch.int64).numpy().tobytes())
            h.update(int(length).to_bytes(8, "little")); return h.digest()
        if length % self.block_size == 0:
            return hashes[length // self.block_size - 1]
        floor = (length // self.block_size) * self.block_size
        parent = hashes[floor // self.block_size - 1] if floor >= self.block_size else ROOT_HASH
        part = row[floor:length] if row is not None else []
        h = hashlib.sha256(); h.update(parent)
        h.update(torch.tensor([int(t) for t in part], dtype=torch.int64).numpy().tobytes())
        h.update(int(length).to_bytes(8, "little")); return h.digest()

    def stored_lengths(self) -> list[int]:
        return sorted({int(e.position) for e in self._entries.values()})

    def find_longest_prefix(self, token_ids):
        """Return (n_reused, last_hash_or_None, entry_or_None).

        Longest STORED boundary <= len(token_ids) whose chain hash
        matches this prompt's prefix at that length. Does not count stats.
        """
        toks = [int(t) for t in token_ids]
        hashes = self.hasher.chain_hashes(toks)
        for length in sorted(self.stored_lengths(), reverse=True):
            if length <= 0 or length > len(toks):
                continue
            h = self._hash_at(hashes, length, toks)
            e = self.get(h, boundary_len=length)
            if e is not None and int(e.position) == length:
                return length, h, e
        return 0, None, None

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses,
                "blocks": len(self._entries), "entries": len(self._entries),
                "tokens_saved": self.tokens_saved, "bytes": self._total_bytes,
                "snapshots_taken": self.snapshots_taken,
                "snapshot_bytes_copied": self.snapshot_bytes_copied,
                "snapshot_interval_tokens": self.snapshot_interval_tokens}


def prefill_with_prefix_cache(model, ids, state, cache: TPBlockPrefixCache,
                              chunk_size: int = 128,
                              snapshot_interval_tokens: int | None = None,
                              store_final: bool | None = None):
    """Prefill ids ([T] or [B,T]) reusing cached prefix.

    Returns (logits [B,V], n_reused, stats_delta). State is reset first,
    longest STORED boundary <= shared prefix restored, suffix prefilled
    via prefill_chunkwise_static in block-aligned pieces. Snapshots are
    taken only when a piece crosses a ``snapshot_interval_tokens``
    boundary (default from cache, 256) and, when ``store_final``, at the
    final prompt end (deduplicated if that end is an interval boundary,
    so for T=2304/interval=256 snapshot events = 9 interval + 1 final
    = 10, distinct entries = 9). Block-aligned piece splitting keeps
    chunk boundaries at identical absolute positions on cold and warm
    runs, so warm logits are bit-exact vs a cold call with same args.
    Tradeoff: a shared prefix whose length is not stored is reused only
    up to the previous stored boundary; the suffix recompute grows by
    less than one interval.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive int, got {chunk_size!r}")
    interval = (cache.snapshot_interval_tokens if snapshot_interval_tokens is None
                else _validate_interval(snapshot_interval_tokens, cache.block_size))
    do_final = cache.store_final if store_final is None else bool(store_final)
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
    if T > state.max_seq_len:
        raise ValueError(f"prompt {T} exceeds max_seq_len={state.max_seq_len}")
    hits0, miss0, saved0 = cache.hits, cache.misses, cache.tokens_saved

    hashes = cache.hasher.chain_hashes(row)
    snaps0 = cache.snapshots_taken
    n_reused, _h, last_entry = cache.find_longest_prefix(row)
    if n_reused:
        cache.hits += n_reused // cache.block_size
    else:
        cache.misses += 1
    # Full hit: back off to the previous STORED boundary so last-token
    # logits are computed exactly (not merely one block).
    if n_reused == T and T > 0 and n_reused > 0:
        prev = [ln for ln in cache.stored_lengths() if ln < T]
        # Verify candidate matches this prompt's prefix.
        backed = 0
        last_entry = None
        for ln in sorted(prev, reverse=True):
            e = cache.get(cache._hash_at(hashes, ln, row), boundary_len=ln)
            if e is not None:
                backed, last_entry = ln, e
                break
        n_reused = backed
    cache.tokens_saved += n_reused

    state.reset()
    if n_reused and last_entry is not None:
        cache.restore_entry(last_entry, state)

    if T == 0:
        empty = torch.empty(B, 0, int(model.config.vocab_size), device=model.ctx.device)
        delta = {"hits": cache.hits - hits0, "misses": cache.misses - miss0,
                 "tokens_saved": n_reused, "n_reused": n_reused, "blocks_stored": 0,
                 "snapshots": 0, "snapshot_interval_tokens": interval}
        return empty, 0, delta

    # Prefill suffix in interval-sized pieces; snapshot sparsely.
    # Piece boundaries at absolute interval multiples keep internal
    # chunk boundaries on the same absolute grid cold and warm when
    # interval is a multiple of chunk_size (the tested configurations).
    logits = None
    pos = n_reused
    while pos < T:
        next_boundary = ((pos // interval) + 1) * interval
        end = min(next_boundary, T)
        logits = model.prefill_chunkwise_static(idx2[:, pos:end], state, chunk_size=chunk_size)
        pos = end
        if pos % interval == 0:
            cache.store(cache._hash_at(hashes, pos, row), state, boundary_len=pos)
    if do_final and T > 0:
        cache.store(cache._hash_at(hashes, T, row), state, boundary_len=T)
    blocks_stored = cache.snapshots_taken - snaps0
    delta = {"hits": cache.hits - hits0, "misses": cache.misses - miss0,
             "tokens_saved": n_reused, "n_reused": n_reused,
             "blocks_stored": blocks_stored, "snapshots": blocks_stored,
             "entries": len(cache), "blocks": len(cache),
             "snapshot_interval_tokens": interval,
             "bytes": cache.total_bytes}
    return logits, n_reused, delta
