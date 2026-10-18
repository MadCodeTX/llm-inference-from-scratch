# 27 — Block prefix caching for TPStaticState

## Design (`tp_prefix_cache.py`)

- `BlockHasher(block_size=16)`: chained hash per block —
  `sha256(parent_hash_bytes || int64 token bytes)` via hashlib (never
  Python `hash()`), deterministic across ranks/processes. A shared
  prefix yields identical hash chains; a divergent block changes all
  later hashes.
- `TPBlockPrefixCache(block_size=16, max_bytes|max_entries, device
  storage='cpu'|'gpu')`: maps chained hash → snapshot of the FULL
  `TPStaticState` at that block boundary. KV tensors are cloned only up
  to the filled length (`[:, :, :filled]`); linear `S`/conv are cloned
  whole; `position` stored as int. Restore uses `copy_` into the
  provided state (KV tail zeroed first). Store clones, so mutating live
  state never corrupts the cache (aliasing-tested).
- LRU eviction by measured bytes: `state_bytes(state, filled)` sums
  `numel*itemsize` over snapshot tensors (+8 B position). Stats:
  hits/misses/blocks/tokens_saved (+ bytes/entries).
- Rank consistency: every rank runs identical cache logic from the
  same token ids and hits identically; each rank stores only its own
  shards. No collectives are needed for the cache itself — suffix
  prefill collectives stay in lockstep because all ranks restore the
  same logical position. This is per-rank replicated *logic*, sharded
  *data*.

## Integration

`prefill_with_prefix_cache(model, ids, state, cache, chunk_size)`:
resets state, restores the longest block-aligned cached prefix, then
prefills the suffix with `prefill_chunkwise_static` in block-aligned
pieces, snapshotting each newly completed block boundary. Block-aligned
piece splitting keeps chunk boundaries at identical absolute positions
on cold and warm runs, so warm logits are **bit-exact** vs a cold call
with the same arguments (verified, incl. 4 greedy decode tokens). A
full-cache-hit prompt backs off one block so last-token logits are
computed exactly. Partial-block suffixes are handled (only full blocks
are cached/reused). Returns `(logits [B,V], n_reused, stats_delta)`.

## Byte accounting (real Qwen3.8 config, world=2, `tensor_parallel.state_bytes`)

- Per-16-token-block marginal (KV growth per snapshot step):
  524,288 B/rank (16 full layers × 2 × 16 tok × 2 KV-heads/rank × 256 × 2 B).
- Per-snapshot linear-state bytes: 76,972,032 B/rank (48 linear layers,
  seq-independent S+conv) — this dominates: every block snapshot costs
  ~77 MB/rank in linear state plus 0.5 MB per 16 tokens of KV. Prefix
  caching on this hybrid stack is therefore memory-heavy per block;
  `max_bytes` budgeting is essential, and CPU storage is the default.

## Tests

`tests/test_tp_prefix_cache.py` (tiny world=1 + 2-process gloo via
`scripts/tp2_prefix_cache_check.py`): shared 32-token prefix →
`n_reused==32`, warm logits + 4 greedy tokens exactly equal fresh cold
runs, partial-block suffix, LRU eviction (entries and bytes), aliasing,
hasher chaining, byte accounting vs the formula above.

## GPU micro-bench

`scripts/prefix_ttft_bench.py` (TP2, real snapshot, scaled agent shape
2048 shared + 256 unique, median of 3): see
`benchmarks/baseline/ours_prefix_ttft.json`: cold median 10.66 s, warm
median 1.31 s (8.1×, ours-vs-ours), `n_reused=2048`, last argmax equal.
Raw warm reps include one 1340 s first-rep outlier (median unaffected;
retained in the JSON for honesty). Cold includes per-block snapshot
store overhead. Comparison is only against our own cold path. Arm A
agent-shared-prefix TTFT 1.81 s at 8704 tokens is context only
(different shape; no win claim).
