"""Ch 14 -- Prefix KV cache: reuse the prefill of a shared prefix.

Framing follows Kiely, *Inference Engineering* (2026), sec. 5.3.1.

Why this exists
---------------
Many requests share a long opening: a system prompt, a document,
few-shot examples. Prefilling that opening again for every request
repeats identical arithmetic and writes identical keys/values.
A prefix cache stores the KV state produced for a token prefix so a
later request beginning with the same tokens can start its forward
part-way through the sequence instead of at position zero.

What this dissection does (and does not) do
--------------------------------------------
Keys are exact tuples of token ids, and an entry is created only when
the caller invokes :meth:`PrefixKVCache.put`. A lookup therefore
reuses only a whole previously stored prefix; it never invents a
reuse point inside a stored sequence. Production engines generalise
this with a tree over shared openings. Their KV is also chopped into
fixed-size blocks (compare the paged allocator in ``serving.py``),
so sharing can stop at block boundaries, blocks can be shared between
requests, and eviction can drop individual blocks. Here every entry
holds a complete cloned :class:`~inference_from_scratch.kv_cache.KVCache`
for one prefix -- simple enough to test exactly, wasteful in the ways
the docstring for the class names.

Clone discipline
----------------
KV tensors are mutable working state: the next forward appends to the
cache object it is given. Stored entries are therefore cloned on the
way in (``put``) and cloned again on the way out (``lookup``). Mutating
a returned cache, or continuing the cache that was handed to ``put``,
must never change what a later lookup returns.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from .kv_cache import KVCache


class _StatsDict(dict):
    """Dict of cache counters that is also callable (``stats()``)."""

    def __call__(self):  # type: ignore[no-untyped-def]
        return dict(self)


def _tokens_tuple(tokens) -> tuple[int, ...]:  # type: ignore[no-untyped-def]
    """Normalise a token sequence to ``tuple[int, ...]``.

    Accepts a list/tuple of ints or a ``torch.Tensor`` of shape ``[T]``
    or ``[1, T]``. A batched tensor with batch > 1 is rejected: this
    cache is keyed by one sequence's prefix, never a batch's.
    """
    if isinstance(tokens, torch.Tensor):
        t = tokens.detach().cpu()
        if t.dim() == 2:
            if t.shape[0] != 1:
                raise ValueError(
                    f"prefix cache keys are single sequences, got batch {t.shape[0]}"
                )
            t = t[0]
        elif t.dim() != 1:
            raise ValueError(f"tokens must be [T] or [1, T], got shape {tuple(t.shape)}")
        return tuple(int(x) for x in t.tolist())
    return tuple(int(x) for x in tokens)


def _model_n_layer(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "n_layer"):
        return int(config.n_layer)
    layers = getattr(model, "layers", None)
    if layers is not None:
        try:
            return len(layers)
        except TypeError:
            pass
    return 1


def _clone_kv_cache(cache: KVCache, upto: int | None = None) -> KVCache:
    """Deep-clone a concat ``KVCache``, optionally truncated to ``upto``."""
    cloned = KVCache(n_layer=cache.n_layer)
    for i in range(cache.n_layer):
        k = cache.keys[i] if i < len(cache.keys) else None
        v = cache.values[i] if i < len(cache.values) else None
        if k is None or v is None:
            continue
        if upto is not None:
            k = k[:, :, :upto, :]
            v = v[:, :, :upto, :]
        cloned.keys[i] = k.clone()
        cloned.values[i] = v.clone()
    return cloned


def _forward(model, idx, kv_caches=None, start_pos=0):  # type: ignore[no-untyped-def]
    out = model(idx, kv_caches=kv_caches, start_pos=start_pos)
    if isinstance(out, tuple):
        logits, caches = out
        return logits, caches
    return out, kv_caches


class PrefixKVCache:
    """Exact-prefix store of cloned KV caches, with LRU eviction.

    ``max_entries`` bounds how many stored prefixes are kept. When a
    ``put`` would exceed the bound, the least recently used entry is
    dropped (a successful lookup counts as use). Entries are stored
    only at ``put`` boundaries: putting a 6-token prefix makes exactly
    that 6-token key reusable, not its 3-token opening.

    Counters are exposed three ways for convenience: ``.hits``,
    ``.misses`` and ``.tokens_saved`` attributes, and ``.stats`` -- a
    dict ``{"hits": ..., "misses": ..., "tokens_saved": ...}`` (also
    callable as ``.stats()``). A lookup with ``matched_len > 0`` is a
    hit and adds ``matched_len`` to ``tokens_saved``; anything else is
    a miss. ``tokens_saved`` counts prefill tokens a caller could skip,
    not a measured time saving.
    """

    def __init__(self, max_entries: int):
        if int(max_entries) <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[int, ...], KVCache] = OrderedDict()
        self._n_layer: int | None = None
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    # -- counters ------------------------------------------------------
    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def tokens_saved(self) -> int:
        return self._tokens_saved

    @property
    def stats(self) -> _StatsDict:  # type: ignore[override]
        return _StatsDict(
            hits=self._hits, misses=self._misses, tokens_saved=self._tokens_saved
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tokens) -> bool:  # type: ignore[no-untyped-def]
        try:
            return _tokens_tuple(tokens) in self._entries
        except Exception:
            return False

    # -- core API ------------------------------------------------------
    def put(self, tokens, kv_cache: KVCache) -> None:  # type: ignore[no-untyped-def]
        """Store a clone of ``kv_cache`` under the exact ``tokens`` key.

        If the cache holds more positions than ``len(tokens)``, only
        the first ``len(tokens)`` positions are stored -- the entry must
        describe exactly the keyed prefix. If it holds fewer, that is a
        caller bug and ``ValueError`` is raised rather than storing a
        cache that would silently mis-position the next forward.
        """
        key = _tokens_tuple(tokens)
        if not isinstance(kv_cache, KVCache):
            raise TypeError(
                "PrefixKVCache stores KVCache (concat) objects; "
                f"got {type(kv_cache).__name__}"
            )
        if kv_cache.seq_len < len(key):
            raise ValueError(
                f"kv_cache holds {kv_cache.seq_len} positions but key has "
                f"{len(key)} tokens"
            )
        stored = _clone_kv_cache(kv_cache, upto=len(key))
        self._n_layer = kv_cache.n_layer
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = stored
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)  # evict least recently used

    def lookup(self, tokens):  # type: ignore[no-untyped-def]
        """Return ``(matched_len, kv_cache_clone)`` for ``tokens``.

        Finds the longest stored key that is a prefix of ``tokens``.
        On a hit the returned cache is a fresh clone (safe to append
        to) and the counters record the hit. On a miss returns
        ``(0, empty_cache)`` where the empty cache has the stored
        layer count when that is known, else a zero-layer placeholder
        whose ``seq_len`` is 0 (callers that need a usable cache on a
        miss, such as :func:`prefill_with_prefix`, replace it).
        """
        query = _tokens_tuple(tokens)
        best: tuple[int, ...] | None = None
        for key in self._entries:
            if len(key) <= len(query) and query[: len(key)] == key:
                if best is None or len(key) > len(best):
                    best = key
        if best is None or len(best) == 0:
            self._misses += 1
            empty = KVCache(n_layer=self._n_layer or 0)
            return 0, empty
        self._entries.move_to_end(best)
        self._hits += 1
        self._tokens_saved += len(best)
        return len(best), _clone_kv_cache(self._entries[best])

    # Compatibility alias: some callers phrase lookup as ``get``.
    get = lookup


def prefill_with_prefix(model, idx, prefix_cache: PrefixKVCache):  # type: ignore[no-untyped-def]
    """Prefill ``idx`` reusing a stored prefix; return last logits.

    ``idx`` is ``[1, T]`` (batch 1, as elsewhere in this repo's helpers).
    Returns ``(logits_last, kv_cache, n_reused)`` where ``logits_last``
    is ``[1, V]`` (the distribution that predicts the token after
    ``idx``), ``kv_cache`` is the now-complete cache for all of ``idx``,
    and ``n_reused`` is how many leading positions came from
    ``prefix_cache`` instead of being forwarded.

    Only the uncached suffix is run through the model, with
    ``start_pos=n_reused`` so RoPE / learned positions and the causal
    alignment line up with a full prefill. This helper deliberately
    does *not* store its result: the caller chooses reuse boundaries by
    calling ``prefix_cache.put(prefix_tokens, prefix_kv)`` -- typically
    once after prefilling a shared opening. When the whole sequence is
    already stored, the final token is re-forwarded from a cache
    trimmed by one position, because the cache holds keys/values but
    not the logits that token produced.
    """
    if not isinstance(idx, torch.Tensor) or idx.dim() != 2:
        raise ValueError("idx must be a [B, T] tensor")
    if idx.shape[0] != 1:
        raise ValueError(f"prefill_with_prefix is batch=1, got batch {idx.shape[0]}")
    if idx.shape[1] == 0:
        raise ValueError("idx must contain at least one token")

    tokens = _tokens_tuple(idx)
    matched, cached = prefix_cache.lookup(tokens)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if matched == 0:
                cache = KVCache(n_layer=_model_n_layer(model))
                logits, cache = _forward(model, idx, kv_caches=cache, start_pos=0)
                return logits[:, -1], cache, 0

            cache = cached
            start = matched
            if matched == len(tokens):
                # Need logits for the last position, which the cache
                # does not hold: trim one position and re-forward it.
                cache = _clone_kv_cache(cached, upto=matched - 1)
                start = matched - 1

            suffix = idx[:, start:]
            logits, cache = _forward(
                model, suffix, kv_caches=cache, start_pos=start
            )
            return logits[:, -1], cache, start
    finally:
        if was_training:
            model.train()
