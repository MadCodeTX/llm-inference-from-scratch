# mtp1-graph host_overhead — diagnosis (measured, no fix applied)

Source: [profile script](/home/nick/muse/llm-inference-from-scratch/scripts/profile_spec_tp2.py) → [spec_host_diagnosis.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/profile/spec_host_diagnosis.json). Capital probe, 32 tokens, TP2 mtp1-graph, one warm run then one `torch.profiler` (CPU+CUDA) run, plus a 1-token profiled baseline (prefill+capture, 0 cycles) subtracted for loop-marginal profiler counts. Generated ids exact vs `mtp1-graph` in `ours_tp2_mtp_matrix.json`: **true** (16 cycles, 15/16 accepted, 1 rollback). Three runs agreed within ~0.3 ms/cycle on every segment.

## What `host_overhead_ms` actually is

`_PhaseTimer.finish` ([speculation.py:361](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:361)) defines it as wall minus CUDA-event phase times. It is **not** pure host CPU work. The segment split (loop wall **47.04 ms/cycle**) shows most of it is the host *waiting at synchronisations for GPU work that is itself the cost*, plus a smaller slice of genuine host launch work while the GPU is idle:

| Segment (wall, per cycle) | ms | Kind |
|---|---|---|
| argmax pair readback (`.tolist()`, [speculation.py:727](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:727)) | 28.96 | sync wait ≈ verify GPU (35.68 event ms, partly elapsed during launch) |
| verify replay launch ([speculation.py:488](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:488)) | 6.62 | **host CPU work, GPU idle** |
| draft token readback (`.item()`, [speculation.py:686](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:686)) | 5.65 | sync wait ≈ draft GPU (5.85 event ms) |
| rollback/refeed (amortised; 69.93 ms per rollback × 1/16) | 4.37 | eager launch-bound refeed |
| snapshot (every cycle, [speculation.py:706](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:706)) | 0.87 | host launch + GPU clones, serialised before verify |
| loop bookkeeping (residual) | 0.29 | host CPU |
| draft replay launch | 0.25 | host CPU |
| verify input fill (2× `__setitem__`) | 0.03 | host CPU |

Sync-wait segments total **34.62 ms/cycle**; host-launch segments total **8.05 ms/cycle**. Profiler corroboration (loop-marginal, baseline-subtracted): `cudaGraphLaunch` exactly 2.0/cycle at **6.52 ms/cycle CPU** — matching the two replay-launch segments (6.62 + 0.25) and showing nearly all of it is the *verify* graph launch; `aten::item`/`_local_scalar_dense` 1.1/cycle at 6.6 ms/cycle (the draft readback wait); `aten::clone` 139/cycle at 0.86 ms/cycle (the snapshot). Runtime sync counters per cycle: `item` 1, `tolist` 1, `copy_` 1 (draft hidden), `fill_` 2.06, `setitem` 2, snapshot 1, `copy_from` 0.0625. Nested profiler rows (`aten::to`/`_to_copy`/`copy_`/`cudaMemcpyAsync`) triple-count the same calls and are inflated by profiler per-op overhead — counts are used, their CPU totals are not summed.

## Ranked removable overhead (estimates, not yet fixed)

| # | Item | Est. ms/cycle | Location |
|---|---|---|---|
| 1 | Verify-graph `cudaGraphLaunch` host cost while GPU idle (draft graph launch is only 0.25 ms; the verify graph carries the captured NCCL all-reduces) | up to ~6.6 | `_ChunkGraph.run` [speculation.py:476](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:476) |
| 2 | Reject-path eager refeed: one non-graph `decode_token_static` costs 69.93 ms wall per rollback (launch-bound), vs ~24.5 ms/token for graph decode | ~4.4 at this 1/16 rollback rate; ~40+ per rollback avoided/cheapened | refeed [speculation.py:766](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:766), `copy_from` [speculation.py:757](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:757) |
| 3 | Full-state snapshot every cycle (139 clones/cycle: all KV + linear state) although only recurrent/conv state + position need restore (KV slots are overwritten by position); also queues ~0.88 ms GPU before verify | ~0.9–1.7 (host+GPU serial) | `TPStaticState.snapshot` [tp_distributed.py:1198](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/tp_distributed.py:1198), call [speculation.py:706](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:706) |
| 4 | Two serialising syncs per cycle (`.item()` draft, `.tolist()` pair) prevent any overlap; each is individually required by the data flow (draft token feeds verify input; pair feeds accept branch), so the win is only from shrinking the idle launch window in #1, not from removing a sync | 0 directly | [speculation.py:686](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:686), [speculation.py:727](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:727) |
| 5 | Bookkeeping + draft launch + input fill | ~0.6 total, negligible | loop body [speculation.py:668](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/speculation.py:668) |

## Answer in one line

The slower-than-nonspec result is mostly **not** removable host work: ~35 ms/cycle is GPU draft+verify execution surfaced as sync waits, and the verify GPU time (35.7 ms for a 2-token chunk vs 24.5 ms/token non-spec graph) is the real gap. Genuine host-side removable overhead is ~**8 ms/cycle** (verify graph launch ~6.6, snapshot ~0.9, bookkeeping ~0.6) plus ~**4.4 ms/cycle** amortised eager-refeed cost at the measured rollback rate. No behaviour change was made; instrumentation is off by default (`segment_profile=None`).
