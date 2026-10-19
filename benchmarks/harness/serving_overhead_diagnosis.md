# Serving overhead diagnosis — why real TPOT ≈ 113.9 ms vs local graphed ≈ 21.9 ms

Head: `7fbb4a7` (tree clean before and after; nothing committed, no source edited).
Model: Qwen/Qwen3.8-27B-FP8, TP2, 2× RTX 4090 24 GiB. Server defaults: 8×8192,
`token_budget=8192`, tp-sync `object`, `TPEngineCore` graph on CUDA, SSE `Connection: close`.
Greedy exactness is untouched by every measurement below (scheduling/transport only).

Raw evidence: [serving_overhead_diagnosis.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/harness/serving_overhead_diagnosis.json) in this directory.
Prior context: [serving_readiness_round1.md](/home/nick/muse/llm-inference-from-scratch/benchmarks/harness/serving_readiness_round1.md),
[throughput_round1_summary.md](/home/nick/muse/llm-inference-from-scratch/benchmarks/harness/throughput_round1_summary.md),
[decode_round5.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/harness/decode_round5.json) (source of the 21.867 ms local figure).

## Verdict (attribution closes to within ~0.5 ms)

Serving c1 short TPOT p50 reproduced at **114.06 ms** (mean 115.12 ms, n=511 inter-token
gaps, 512-token run; prior round: 113.82–113.89 ms). The ~92 ms gap over the 21.867 ms
local runner figure decomposes into two dominant, independently measured buckets:

| Bucket | Measured ms/token | Share of 114 ms |
|---|---|---|
| Driver loop poll quantum (`BatchedEngineBackend._drive`) | 50.3 | 44% |
| Engine decode step at serving geometry (8 slots × 8192, B=8 graph replay) | 63.6 | 56% |
| TP sync_hook (object broadcast per step) | 0.5 in-server (0.08 isolated) | <1% |
| Tokenizer full re-decode per token | 0.02–0.04 | <1% |
| HTTP/SSE per-token JSON + flush + bench client | <0.2 (unresolved below noise) | <1% |
| **Sum** | **≈114.5** | vs observed p50 114.06 |

Comparison baselines are not like-for-like, which is most of the "mystery":

- 21.867 ms = B=1 runner (`greedy_generate_graph_tp` in [run_qwen38_tp2.py](/home/nick/muse/llm-inference-from-scratch/scripts/run_qwen38_tp2.py)), batch 1, no engine, no HTTP.
- 38.9 tok/s (25.55 ms/step) = throughput-round engine at `max_slots=1`, `max_seq_len=2560`, `token_budget=512`, in-process wall/step, no driver/HTTP.
- 113.9 ms = serving at `max_slots=8`, `max_seq_len=8192`, through the driver thread + TP lockstep + SSE.

## Bucket evidence

### 1. Driver poll quantum: +50.27 ms/token (dominant, exact)

[BatchedEngineBackend._drive](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/openai_server.py:238) runs
`self._pending.get(timeout=0.05)` at the top of **every** loop iteration — including
mid-generation, when the pending queue is empty by construction. Each engine step is
therefore preceded by a full 50 ms empty-queue wait.

Fake/stub timing (CPU, instant fake engine, real `BatchedEngineBackend`):

- step cost 0 ms → inter-token latency p50 **50.25 ms**, mean 50.27 ms (64 tokens, 3.17 s wall).
- step cost 22 ms (sleep) → p50 **72.52 ms** — exactly additive (50 + 22), so the poll is serial per token, not overlapped.
- Full HTTP loopback (real `make_server` + `run_bench.one_request`, instant engine): ITL p50 **50.18 ms** — HTTP/SSE adds nothing resolvable on top of the driver quantum.

The consumer side (`stream_token_ids`, `job.tokens.get(timeout=0.05)`) is *not* additive:
a queued token wakes the getter immediately.

### 2. Engine step at serving geometry: 63.6 ms vs 23.1 ms at local geometry

Engine-only GPU measurement (real weights, `TPEngineCore(decode_mode="graph")`, one
2033-token request, 64 decode steps, per-step CUDA-synced wall; script in /tmp, not committed):

| max_slots | max_seq_len | decode step p50 |
|---|---|---|
| 1 | 2560 (throughput geometry) | 23.06 ms |
| 1 | 8192 | 25.53 ms |
| 8 | 2560 | 43.19 ms |
| 8 | 8192 (serving geometry) | 63.61 ms |

So batch geometry costs +20 ms, seq-len +2.5 ms, and the combination a further ~+18 ms
(superadditive). Mechanisms, from [tp_engine.py](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/tp_engine.py:326):

- The captured graph always replays the full `max_slots` batch; 7 of 8 rows are free/dummy at c1 (`_decode_graphed` replays free rows and discards their state at next admission).
- M=8 decode cannot use the M=1 GEMV FP8 config (`gemv-bk512` is M=1-only); it falls back to the batched FP8 GEMM path, and the lm_head produces `[8, 248320]` logits.
- Every graphed step ends with `_graph_out.float().argmax(dim=-1).tolist()` — a full-vocab fp32 conversion, argmax, and host sync per step.
- Larger static state (2.56 GiB vs 0.35 GiB) worsens memory locality for the replay.

In-server corroboration (instrumented /tmp copy of `serve_qwen38_tp2.py`, counters only):
100 steps = 1 prefill step (2.05 s) + 99 decode steps totalling 6.84 s (**≈69 ms/step mean**,
outliers included), `graph_captures=1`, `graph_replays=99`, `graphed_steps=99`.

### 3. Graph eligibility under serving: steps ARE graph-replayed at c1

At c1 steady state every occupied slot is decode-planned and the rest are free, so
`_graph_eligible` holds: in-server counters show 99/99 decode steps graphed, 1 capture.
The gap is **not** an eager-fallback problem at c1. Eager fallback (with snapshot/restore)
only occurs on mixed prefill+decode steps — i.e. at admission bursts and c≥4 arrivals —
where it becomes a memory problem instead (see c16 below). Sampled requests force eager
by design (`_graph_eligible` rejects sampled rows; greedy exactness preserved).

### 4. TP sync (object vs compact): ≤0.5 ms/step — not the gap

Isolated NCCL micro-bench (2 ranks, 300 calls, this machine):

- object step-only `broadcast_object_list`: **0.080 ms/call**
- object with one 2048-id submit: **0.089 ms/call**
- compact (2 tensor broadcasts): **0.036 ms/call**

In-server object sync_hook mean: 0.54 ms/step (includes driver-thread GIL contention).
This explains the prior ABBA result (object 113.89 vs compact 113.82–113.84 ms): the
transport difference (~0.04 ms isolated) is three orders of magnitude below the TPOT.
Compact remains correctly opt-in with no measured benefit.

### 5. Tokenizer decode per token: 0.02 ms/token now, O(n²) by construction

The HTTP handler's `gen()` calls `backend.decode(list(gen_ids))` on the **full** id list
for every token ([openai_server.py](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/openai_server.py:677)).
Measured with the real Qwen tokenizer (CPU): per-token decode mean 0.004 ms at n=32,
0.008 ms at n=128, 0.016 ms at n=512; in-server 99 calls = 3.5 ms total. Negligible at
512 output tokens, but total cost grows quadratically (8.2 ms per 512-token request;
~0.5 s per 8k-token request) and it also re-runs the stop-sequence scan over full text per token.

### 6. HTTP/SSE flush and run_bench semantics: <0.2 ms/token

Per token the handler does one `json.dumps`, one `wfile.write`, one `flush`
([openai_server.py](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/openai_server.py:752)).
The loopback test above bounds the whole HTTP+SSE+client path at <0.2 ms/token.
`run_bench` TPOT is SSE-chunk inter-arrival at the client (`token_times` on chunks with
text), so it faithfully reports server pacing; it adds no pacing of its own. Note
`run_bench` aggregate tok/min (504 at c1 here) is wall-clock based and is not comparable
to the throughput round's in-process 38.9 tok/s per-stream figure.

### 7. Prefill / token_budget / TTFT

Serving `token_budget=8192` admits the whole 2033-token prompt in one chunkwise prefill
step: measured prefill step 2.05–2.21 s, bench TTFT 2.10 s. Prefill does not interleave
with decode at c1 after admission, so it does not enter TPOT. (Throughput round used
`token_budget=512`, hence 4 prefill steps — a TTFT/accounting difference only.)

## Root cause: 16×4096 c16 — 15/16 HTTP 500

Reproduced live on the stock server (`--max-slots 16 --max-seq-len 4096 --tp-sync compact`, port 18435):

1. First c16 batch (16× 2048→512): **16/16 HTTP 200**, peak 20,474 MiB.
2. Second c16 batch via `run_bench`: **1/16 OK, 15× HTTP 500**, wall 9.3 s, GPU 23,702 MiB.
3. Third batch with body capture: 4× HTTP 500 with body
   `generation failed: CUDA out of memory. Tried to allocate 40.00 MiB. GPU 0 has a total
   capacity of 23.52 GiB of which 9.75 MiB is free… 22.37 GiB is allocated by PyTorch,
   with 783.88 MiB allocated in private pools (e.g., CUDA Graphs), and 516.00 MiB is
   reserved by PyTorch but unallocated`, plus 12 client timeouts (server wedged;
   prior round needed SIGKILL — same signature).

Mechanism (code + memory math):

- Static state at 16×4096 = 3.12 GiB/rank (KV 2.00 GiB: 16 full layers × K+V × 2 local
  KV heads × 256 × bf16 = 32,768 B/token; linear `S` 1.12 GiB: 48 layers × 24 heads ×
  128×128 × fp32 = 72 MiB/row). Weights resident 14.91 GiB. Graph capture adds a second
  full state as scratch ([tp_engine.py](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/tp_engine.py:339) `_capture_graph`) plus a
  784 MiB CUDA-graph private pool that is never released.
- At c16 the 8192 budget prefills only ~4 rows/step, so several steps mix decoding and
  prefilling rows. Those steps are graph-ineligible and take the eager path, which first
  **clones the full state of every non-decoding row** (`_snapshot_rows`): ~200 MiB/row at
  seq 4096 (128 MiB KV + 72 MiB `S` + conv), up to ~3 GiB transient per mixed step.
- Whether a given batch OOMs depends on allocator history (first batch passed at
  20,474 MiB; later batches fragment/retain and a 20–40 MiB allocation fails at
  23.5 GiB in use). The OOM inside `engine.step()` makes the driver fail **all** active
  jobs (`_drive` exception path); jobs that have not yet received a first token surface
  as HTTP 500 before SSE headers, the rest hang when rank 1 is left blocked in a
  collective — hence 15/16 500s in `run_bench` (which does not capture bodies) and the
  timeout/SIGKILL tail.

## Root cause: graph-capture scratch OOM at 8×16384

Deterministic, same mechanism without the snapshot term: weights 14.91 GiB + resident
state 4.56 GiB (KV 4.00 + `S` 0.56) + capture scratch second state 4.56 GiB + graph pool
and activations exceeds 24 GiB. Prior round's log: first graphed decode tried to allocate
128 MiB with 31.75 MiB free; the 16-token smoke passes because it never reaches a
decode-eligible step that triggers capture at full batch geometry in that probe's path,
and the first 32-token greedy probe returns HTTP 500. At 8×8192 the state is 2.56 GiB
and the scratch fits (observed peak 23,304 MiB). This is why 8×8192 is the adopted default.

## Ranked fix candidates (not implemented; diagnosis lane only)

| # | Fix | Expected gain | Risk / notes |
|---|---|---|---|
| 1 | Driver: block on `_pending` only when idle; when active, `get_nowait` (or an event with ≤1 ms timeout) before `sync_hook`/`step` | −50 ms/token → TPOT ≈64 ms | Scheduling only; greedy ids unchanged. Avoid pure busy-spin when idle. |
| 2 | Reduce graphed batch cost at low occupancy: capture per-occupancy graphs, or right-size `max_slots` per deployment, or keep an M=1 GEMV graph for the c1 case | −20 to −41 ms/token at c1 | Per-occupancy capture multiplies capture memory/time; needs the scratch fix (#4) first at large geometries. |
| 3 | Kill eager snapshot clones: snapshot only rows a decode step can mutate (none outside planned rows if decode is row-isolated), or restore via saved row views instead of full clones | Fixes c16-class OOM; removes mixed-step stalls at c≥4 | Must prove `decode_token_static` never writes non-planned rows (it replays/writes all rows in eager too — the clone exists because eager decode mutates free rows; resetting free rows instead of restoring them is the cheap exact variant). |
| 4 | Capture without a second full state: warm up on the real state then `reset(rows=...)`, or scratch at reduced seq len | Unblocks 8×16384; −3–4.5 GiB peak at capture | Warmup mutates real state; reset must be proven complete (KV, `S`, conv, position). |
| 5 | Incremental detokenization (decode only the new suffix / streaming decoder) + incremental stop-scan | −0.02 ms/token now; removes O(n²) at long outputs | Must preserve exact delta semantics for multi-byte splits and stop truncation. |
| 6 | SSE coalescing / fewer flushes | <0.2 ms/token | Skip unless #1–#2 land and profiling says otherwise. |
| 7 | tp-sync compact as default | ~0.04 ms/token | Rejected again: no measured benefit; keep opt-in. |

Projected serving TPOT after #1+#2 at c1: ≈23–26 ms/token (local geometry parity),
before any kernel work. c4 throughput is additionally bounded by mixed-step eager
fallbacks until #3 lands.

## Method and cleanup

- Fake/stub: instant/sleeping fake engines through the real `BatchedEngineBackend` and
  real HTTP server; real Qwen tokenizer decode timing; 2-rank NCCL sync micro-bench.
- Real GPU: instrumented /tmp copy of the server (counters only, repo untouched) at
  8×8192 on port 18433 (c1 128-token bench: TTFT 2.10 s, graphed 99/99); stock server
  8×8192 on port 18431 (c1 512-token bench: TPOT p50 114.06 ms, 504 tok/min); engine-only
  geometry sweep; stock 16×4096 compact server on port 18435 for the c16 repro.
- All servers were shut down (SIGTERM, SIGKILL escalation only on the wedged c16 server);
  GPUs verified at 1 MiB, ports 18431/18433/18435 free, no server processes remain, and
  `git status` is clean. No vLLM or external process was started or stopped.
