# 28 — Batched TP static decode (per-row positions)

## Problem

`TPStaticState.position` was a shared 0-dim tensor: every row of a
`[B, ...]` static decode was forced to the same sequence position.
`DistributedFullAttention.step_static` gathered one shared RoPE row,
`index_copy_`-ed K/V at that one slot for all rows, and masked keys
past that one position. A batch could only ever be rows in lockstep
from position 0 — no staggered starts, no resetting one finished row
while others keep decoding, i.e. no real continuous batching.

## Change (`tp_distributed.py`)

- `TPStaticState.position` is now int64 `[B]`, one position per row.
  `seq_len` keeps its old meaning when all rows agree (returns the
  common int; the old B=1 callers and tests are untouched) and raises
  with the divergent values otherwise — read `state.positions` /
  `state.position` then. `scalar_position` is the same value under a
  name that says what it is. The `seq_len` setter accepts an int
  (all rows) or a per-row sequence.
- `DistributedFullAttention.step_static` / `step_chunk_static` /
  `prefill_chunk_static`: positions normalised by `_row_positions`
  (accepts `[B]`, and legacy scalar / `[1]` which broadcast — B=1
  CUDA-graph and MTP-draft callers passing a scalar `rope_position`
  keep working). RoPE cos/sin are gathered per row (`cos[pos]` →
  `[B, T, R]`, fed to `_apply_rope_with_tables` in its per-row form).
  K/V are written loop-free with `scatter_` along the sequence dim
  at each row's own slots (`_write_kv_at`, index expanded over heads
  and head-dim). The attention mask is built per row:
  `future[b, t, k] = k > pos[b, t]`, for both the SDPA and manual
  score paths. Linear (gated-delta) layers needed no change: their
  `S` / conv state was already `[B, ...]` and position-free.
- Overflow is checked **per row before any mutation**
  (`DistributedHybridLM._raise_if_static_overflow`): any row whose
  `position + n` would exceed `max_seq_len` raises `ValueError`
  naming the offending rows, with all buffers and positions provably
  unchanged (tested). The check is skipped during CUDA-graph capture,
  exactly as before.
- `TPStaticState.reset(rows=None)`: `None` resets everything (old
  behaviour, pointers stable). `rows` (int, index sequence/tensor,
  bool mask, or slice) zeroes only those rows' KV, linear `S`, conv
  buffers and `position` entries; other rows' buffers are byte-equal
  afterwards (tested). Invalid rows raise before any mutation.
- Graph/spec compatibility: B=1 states now carry `position` of shape
  `[1]`, which works everywhere the 0-dim tensor did (`add_`,
  `fill_`, `copy_`, `.item()` on a single element, graph capture of
  `decode_token_static`). `_MergedSpecGraph`'s MTP position snapshot
  buffer is now `zeros_like(mtp_state.position)` instead of a 0-dim
  tensor. `tp_prefix_cache` reads the filled length via a helper that
  requires uniform per-row positions (prefix caching is a lockstep
  operation; divergent rows raise instead of silently picking one).

## Batched greedy helper

`DistributedHybridLM.generate_static_batched(idx[B, T], max_new_tokens, state, chunk_size=128)`
returns generated ids `[B, max_new_tokens]` on device (prompt not
included, EOS ignored — a fixed greedy count).

- Prefill: the whole batch at once via `prefill_chunkwise_static`.
  Prompts are equal-length because `idx` is a dense `[B, T]` tensor;
  there is deliberately no per-row prompt length. Because positions
  are per-row, rows starting at different offsets (one row reset,
  others mid-sequence) still prefill at their own correct slots.
- Decode: first token = argmax of the prefill's last logits, then one
  `decode_token_static` step per further token, each fed the previous
  **on-device** argmax. In eager mode there is no per-token host sync:
  ids are stacked on device at the end and the caller pays one final
  transfer only if it reads them on the host.
- Combined overflow (prompt + fed-back tokens per row) is checked
  before the prefill, so a too-small state fails without mutation.

## Proofs (durable tests)

`tests/test_tp_batched_decode.py` (world=1) plus the 2-process gloo
proof `scripts/tp2_batched_check.py` (`TP2_BATCHED_CHECK PASS`,
run under torchrun by the test):

- B=3 different prompts: `generate_static_batched` tokens exactly
  equal three independent B=1 runs (also on TP2/gloo).
- Staggered reset: after 3 decode steps, `reset(rows=[1])`; rows 0/2
  logits continue **exactly** as an uninterrupted batched run for
  every later step, row 1 re-fed its prompt token-by-token matches a
  fresh B=1 prefill + greedy continuation, and rows 0/2's buffers are
  byte-equal across the reset.
- Per-row overflow raises before mutation, including when only one
  row would overflow; selective-reset buffer/position isolation;
  `seq_len` divergence contract; B=1 backward compatibility.

## GPU bench

`scripts/batched_decode_bench.py` (torchrun TP2, real snapshot):
capital quality-probe prompt replicated per row with a distinct
suffix token, B ∈ {1,2,4,8,16}, 64 fixed greedy decode tokens,
ignore-EOS, eager mode. Reports step ms, aggregate tok/s + tok/min,
per-seq tok/s, peak memory per rank, and B=1 row-0 ids vs a
single-row run. Results in
`benchmarks/baseline/ours_tp2_batched_decode.json`.

Measured 2026-10-18 (GPU gate clear: no compute processes, no
vllm-*/mp-* containers; 2× RTX 4090, torch 2.11.0+cu128, eager):

| B | step ms | aggregate tok/s | aggregate tok/min | per-seq tok/s | peak GiB/rank |
|---|---------|-----------------|-------------------|---------------|---------------|
| 1 | 43.68 | 22.8 | 1,367 | 22.78 | 15.36 |
| 2 | 43.91 | 45.3 | 2,721 | 22.67 | 15.59 |
| 4 | 44.57 | 89.3 | 5,360 | 22.33 | 15.90 |
| 8 | 44.41 | 179.1 | 10,746 | 22.39 | 16.52 |
| 16 | 47.28 | 333.3 | 20,001 | 20.83 | 17.79 |

Reading: per-step time is almost flat in B (the batch is nearly free
in eager mode: same weight reads serve every row), so aggregate
throughput scales ~linearly — B=8 already passes the vLLM Arm A c4
short bar (8,530 tok/min) and B=16 passes Arm B c4 (12,113 tok/min)
in aggregate, while per-seq latency (~44 ms/token eager) stays well
behind the graphed B=1 path (23.67 ms, docs/25). Row 0's 64 ids are
identical at every B and equal the single-row run
(`b1_row0_eq_single_run: true`). Next lever is a batched CUDA-graph
capture, not more eager batching.
## Continuous-batching engine core (`tp_engine.py`)

`TPEngineCore(model, ctx, max_slots, max_seq_len, token_budget, eos_id=None)`
manages fixed slots over ONE batched `TPStaticState`
(`batch_size == max_slots`), non-HTTP, greedy only in v1:

- `submit(prompt_ids, max_new, sampling="greedy") -> request_id`
  (any other sampling raises; prompt/max_new/max_seq_len validated
  before any state is touched). `poll(request_id)` returns
  `{status, token_ids, ttft_steps, finish_step}` with status in
  `waiting|prefilling|decoding|finished`; unknown ids raise `KeyError`.
- `step()`: (a) admit waiting FIFO into the lowest free slots,
  `reset(rows=[slot])` at admission (a finished row is freed at
  completion but reset deferred to this next admission); (b) prefill
  chunkwise under the step's `token_budget`, shared with decode by
  decode-first reservation -- each chunk is one row-sliced
  `prefill_chunkwise_static` call (B=1 view of that slot's buffers and
  `position` entry), so staggered rows prefill at their own offsets;
  when a prompt completes, its last logits' argmax is the first token
  that step (`ttft_steps` counts arrival step through that step);
  (c) ONE batched `decode_token_static` over all slots for the
  decoding rows planned at step start. Non-planned rows (prefilling,
  free, budget-starved) are snapshotted before the call and restored
  after it, so the batched step cannot corrupt a mid-prefill row;
  rows finishing prefill this step join decode from the next step.
  EOS (when `eos_id` set) or `max_new` finishes a request and frees
  its slot. Every step returns an `EngineStepRecord`
  (`prefill_tokens + decode_tokens <= token_budget`, asserted) and
  appends it to `engine.history` -- the instrumented step token counts.
- Bookkeeping mirrors `serving.ChunkedPrefillScheduler`'s accounting
  concepts (decode-first budget, FIFO chunks, step-index TTFT/finish)
  rather than importing it: that scheduler is a token-count simulation
  whose `step()` invents tokens with no model, no slots and no real
  EOS, so reusing it would double-track state. Prefix caching is not
  integrated: the seam is admission, where a per-row prefix restore
  would replace the full chunkwise prefill and seed `prefill_done`.

Proofs: `tests/test_tp_engine.py` -- 5 staggered requests on 2 slots
produce per-request ids exactly equal to independent B=1
`generate_static_batched` runs (world=1 and, via
`scripts/tp_engine_smoke.py` under torchrun, real 2-process gloo;
the same script passes tiny NCCL on 2x RTX 4090,
`TP_ENGINE_SMOKE PASS`, `staggered_exact=True`,
`budget_respected=True`, 18 steps), slot reuse after EOS on a single
slot, small-budget chunked prefill exactness, determinism replay, and
validation. No throughput claim yet: eager per-step latency is the
docs/28 bench above; a batched CUDA-graph decode remains the lever.

## Graphed batched decode (`decode_mode="graph"`, default on CUDA)

`TPEngineCore(..., decode_mode="graph")` (the default resolves to
`"graph"` on a CUDA context and `"eager"` elsewhere; explicit
`"eager"` is the unchanged v1 path) captures ONE CUDA graph over the fixed
`max_slots`-wide `decode_token_static` -- static input/output
buffers, side-stream warmup on a scratch state and a TP barrier
before capture, exactly the runner's B=1 pattern
(`scripts/run_qwen38_tp2.py:greedy_generate_graph_tp`). A step
replays the graph only when every occupied slot is decode-planned;
free slots are replayed with dummy input and reset at admission
(their positions are also re-zeroed before they could leave the
buffers), and any step with a prefilling or just-finished row uses
the eager snapshot/restore step instead. Graph mode fails closed at
construction off CUDA rather than silently falling back. The flag
changes only launch mechanics: same greedy argmax, same EOS/finish
rules, so token ids are identical to eager -- proven on tiny NCCL
TP2 (`scripts/tp_engine_smoke.py --decode-mode graph`:
`staggered_exact=True`, `graph_captures=1`, `graph_replays=13`,
`TP_ENGINE_SMOKE PASS`) and by the CPU validation/eligibility tests
in `tests/test_tp_engine.py`. Measured effect on the real 27B
shape, suite-like TTFT/TPOT (steps are indices, no wall clock in
the engine), and any throughput claim are deliberately not made
here; the caller must still apply the adopted fused/scale-cache
model tweaks (`run_qwen38_tp2.py:apply_*`) to the model it hands
the engine, as before. Round-1 measurement on the real 27B at the
suite-short geometry (`benchmarks/harness/throughput_round1.json`):
graph vs eager median aggregate 2,334 vs 1,397 tok/min at c1,
5,109 vs 4,462 at c4, 7,298 vs 6,979 at c8, token-identical on all
14 runs (captures=1, replays=511 per graph run) -- exact and faster
at every measured concurrency with no shape regression, which is
why the CUDA default is now `graph`. That harness is the local
engine, not `run_bench.py` serving: no vLLM win is claimed. Note GEMV (`gemv-bk512`) is an M=1 kernel:
batched decode at `B > 1` takes the M=B FP8 path regardless, so
the comparable-throughput lever at concurrency is this graph, not
the B=1 GEMV default.
