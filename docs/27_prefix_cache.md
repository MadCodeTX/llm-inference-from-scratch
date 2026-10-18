# 27 — Block prefix caching for TPStaticState

## Design (`tp_prefix_cache.py`)

- `BlockHasher(block_size=16)`: chained hash per block —
  `sha256(parent_hash_bytes || int64 token bytes)` via hashlib (never
  Python `hash()`), deterministic across ranks/processes. A shared
  prefix yields identical hash chains; a divergent block changes all
  later hashes.
- `TPBlockPrefixCache(block_size=16, max_bytes|max_entries, device
  storage='cpu'|'gpu', snapshot_interval_tokens=256, store_final=True)`:
  maps `(chain_hash_at_boundary, boundary_len)` → snapshot of the FULL
  `TPStaticState` at that stored boundary (see `entry_key`). Hashing
  stays at `block_size`; snapshots are stored only at interval
  boundaries plus the final prompt end. `snapshot_interval_tokens`
  must be a positive multiple of `block_size`. KV tensors are cloned
  only up to the filled length (`[:, :, :filled]`); linear `S`/conv
  are cloned whole; `position` stored as int. Restore uses `copy_`
  into the provided state (KV tail zeroed first). Store clones, so
  mutating live state never corrupts the cache (aliasing-tested).
  A final store coinciding with an interval boundary is deduplicated
  (counts as a snapshot event, no second copy).
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

`prefill_with_prefix_cache(model, ids, state, cache, chunk_size,
snapshot_interval_tokens=None, store_final=None)` (defaults from the
cache): resets state, restores the longest STORED boundary <= the
shared prefix, then prefills the suffix with
`prefill_chunkwise_static` in interval-sized pieces. Piece boundaries
at absolute interval multiples keep internal chunk boundaries on the
same absolute grid cold and warm when the interval is a multiple of
`chunk_size` (all tested configurations), so warm logits are
**bit-exact** vs a cold call with same args (verified, incl. 4 greedy
decode tokens). A full-cache-hit prompt backs off to the previous
stored boundary so last-token logits are computed exactly.
Returns `(logits, n_reused, stats_delta)`; delta includes
`snapshots`/`blocks_stored` (snapshot events), `entries`, and
`snapshot_interval_tokens`.

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

## GPU micro-bench (v1, superseded by v2 below)

`scripts/prefix_ttft_bench.py` (TP2, real snapshot, scaled agent shape
2048 shared + 256 unique, median of 3): see
`benchmarks/baseline/ours_prefix_ttft.json`: cold median 10.66 s, warm
median 1.31 s (8.1×, ours-vs-ours), `n_reused=2048`, last argmax equal.
Raw warm reps include one 1340 s first-rep outlier (median unaffected;
retained in the JSON for honesty). Cold includes per-block snapshot
store overhead. Comparison is only against our own cold path. Arm A
agent-shared-prefix TTFT 1.81 s at 8704 tokens is context only
(different shape; no win claim).

## Snapshot-overhead diagnosis (v1 cold 10.66 s at 2304 tokens)

Code audit of the v1 path (`block_size=16`, default `storage='cpu'`):

- Snapshots per cold prefill: 2304/16 = **144**. Each `_snapshot`
  clones 128 tensors (16 full layers x K/V + 48 linear layers x
  S/conv), each `.clone().to(cpu)` — a device->host copy per tensor —
  plus one host sync in `_uniform_filled` (`position.tolist()`) per
  snapshot. Bench storage device in v1 was CPU (cache default).
- Bytes per snapshot at filled `f`: 76,972,032 B linear (constant,
  48 layers) + 32,768 B/token x `f` KV. First snapshot 77,496,320 B,
  last (f=2304) 152,469,504 B; total copied over a cold prefill
  **15.42 GiB/rank** (10.32 GiB linear + 5.10 GiB KV).
- Secondary cost found by A/B: v1 also prefilled in 16-token pieces,
  so weights were re-read every 16 tokens. Keeping 16-token pieces
  but storing only at interval 256 gave cold 8.12 s (cpu) / 7.94 s
  (gpu); moving pieces to interval boundaries gave 1.93 s / 1.64 s.

## Cost model and interval tradeoff (2304-token prompt, per rank)

Snapshot bytes = 76,972,032 (linear S+conv) + 32,768 x tokens (KV:
16 full layers x 2 x 2 KV-heads/rank x 256 x 2 B). Tradeoff: larger
interval = fewer/cheaper snapshots, but a shared prefix is reused
only up to the previous stored boundary, so suffix recompute grows
by < interval tokens (worst extra vs block=16 shown).

| interval | distinct snapshots | bytes copied | worst extra recompute |
|---:|---:|---:|---:|
| 16 | 144 | 15.42 GiB | 0 tok |
| 64 | 36 | 3.88 GiB | 48 tok |
| 128 | 18 | 1.96 GiB | 112 tok |
| 256 | 9 (+1 final event, deduped) | 1.00 GiB | 240 tok |
| 512 | 5 (incl. final 2304) | 0.59 GiB | 496 tok |
| 1024 | 3 (incl. final 2304) | 0.38 GiB | 1008 tok |

Default 256 balances copy cost against at most ~240 extra recomputed
tokens on a partial-prefix hit.

## GPU bench v2 (`benchmarks/baseline/ours_prefix_ttft_v2.json`)

`scripts/prefix_ttft_bench.py` (TP2, 2048 shared + 256 unique,
interval 256, median of 3, all reps retained, per-rep wall timestamps
and cache stats in the JSON):

| storage | cold median | warm median | warm n_reused |
|---|---:|---:|---:|
| cpu | 1.926 s | 0.252 s | 2048 |
| gpu | 1.643 s | 0.184 s | 2048 |

Last argmax cold == warm == 12356 in both configs; no rep exceeded
60 s. Cold cache stats: 10 snapshot events, 9 distinct entries,
1,070,235,720 B copied (vs 15.42 GiB modelled for v1).

Prior v1 outlier: the 1339.99 s first warm rep in
`ours_prefix_ttft.json` has no per-rep timestamps or GPU-utilisation
log, so external GPU contention during that rep can neither be
confirmed nor excluded from available data; the file mtime only
dates the completed run. It is not reproduced in v2 (all reps
0.18-1.96 s with an idle-GPU gate checked before the run).
