# What counts as a win

This document defines the metrics, the incumbent reference numbers, and
the thresholds a candidate engine must clear. Nothing here is a new
measurement; live comparisons happen only inside an approved GPU window
(see the protocol at the end).

## Metrics

All timings are end-to-end wall time measured by the client
(`time.perf_counter` around the streamed request), so they include
queueing, network loopback, and client overhead. They are not on-GPU
kernel times; a profiler would report smaller numbers for the same work.

- **TTFT (time to first token, seconds):** request start → first
  streamed chunk carrying text. Prefill-dominated. Reported as a
  distribution (`metrics.summarize`: n/mean/p50/p90/p99/min/max), never
  as a bare mean.
- **TPOT / TBT (time per output token, seconds):** inter-token
  latencies between consecutive streamed chunks after the first.
  Decode-dominated. Lower is better. The incumbent benches did not
  record TPOT, so TPOT wins are judged only between `run_bench.py`
  runs made with the same harness.
- **Throughput:** `aggregate_output_tps = total_output_tokens / wall_s`
  for a batch; `tokens_per_minute = aggregate_output_tps × 60`;
  `per_stream_tps = aggregate_output_tps / concurrency`. `wall_s` is
  first-thread-start → last-thread-join for the batch.
- **Percentiles:** `src/inference_from_scratch/metrics.py:summarize`
  (linear interpolation, same convention as the incumbent means plus
  p50/p90/p99 tails). End-to-end latency percentiles use per-request
  `latency_s`; TPOT percentiles pool all inter-token gaps in the batch.
- **Cold vs warm:** "cold" = first batch after server start (no prefix
  in cache, no warm allocator); "warm" = subsequent repeats. Compare
  like with like.

## Incumbent numbers (pre-existing, this box)

From `benchmarks/baseline/incumbent_existing.json` (vLLM serving
`Qwen/Qwen3.8-27B-FP8`; see `benchmarks/baseline/README.md` for
provenance). One `run=0` row per cell; `baseline` had two runs.

| Source | Shape (in→out tok) | Conc. | Prompt tok | Agg TPS | Per-stream TPS | Mean TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| baseline | chart_summary 16384→1024 | 1 | 16241 | 34.4 | 34.4 | 10.03 | 29.75 |
| baseline | patient_gen 2048→2048 | 1 | 2023 | 52.4 | 52.4 | 0.23 | 39.06 |
| baseline | research_synth 49152→4096 | 1 | 48812 | 41.0 | 41.0 | 18.55 | 100.01 |
| baseline | chart_summary 16384→1024 | 4 | 16241 | 47.4 | 11.9 | 32.79 | 86.36 |
| baseline | patient_gen 2048→2048 | 4 | 2120 | 51.9 | 13.0 | 59.40 | 157.71 |
| S5-nativefp8 | patient_gen 2048→2048 | 4 | 2023 | 138.3 | 34.6 | 1.66 | 59.25 |
| S5-nativefp8 | chart_summary 16384→1024 | 4 | 16241 | 109.8 | 27.4 | 5.55 | 37.31 |
| S9-FINAL | patient_gen 2048→2048 | 1 | 2023 | 52.3 | 52.3 | 0.78 | 39.14 |
| S9-FINAL | patient_gen 2048→2048 | 4 | 2023 | 173.9 | 43.5 | 1.99 | 47.10 |
| S9-FINAL | chart_summary 16384→1024 | 1 | 16241 | 39.7 | 39.7 | 6.23 | 25.77 |
| S9-FINAL | chart_summary 16384→1024 | 4 | 16241 | 126.0 | 31.5 | 6.53 | 32.51 |

Read the spread honestly: the early `baseline` sweep queued badly at
concurrency (mean TTFT 59 s at c=4 on the short shape), while S5/S9
show what the tuned serving config reaches. The bar below is the best
incumbent cell for the same shape/concurrency, not the worst.

## Win thresholds

A candidate wins a cell only if it beats **both** arms below, measured
with `benchmarks/run_bench.py` on `benchmarks/suite.json` (same shapes,
same arms, same box):

- **Arm A — incumbent as-is:** vLLM serving `Qwen/Qwen3.8-27B-FP8` with
  its current config (the `incumbent_existing.json` reference).
- **Arm B — vLLM + MTP:** the same vLLM with multi-token prediction
  enabled, once that row has actually been measured. Until then Arm B
  is "not yet measured" and no win may be claimed against it.

Thresholds (all must hold for the claimed cells):

1. **Throughput + decode speed (concurrency 1–4, short/medium):**
   ≥1.25× the incumbent's `tokens_per_minute` **and** ≥20% lower p50
   TPOT than the incumbent's `run_bench.py` run.
2. **Shared prefix (`agent-shared-prefix`):** ≥30% lower TTFT
   (p50) than the incumbent — this is the prefix-cache claim.
3. **Cold long (`long`, first/cold batch):** ≥15% lower TTFT (p50)
   than the incumbent cold run.

## Quality gate

Speed without correctness is not a win. Before any timing claim:

- Greedy outputs on the suite prompts must be token-identical between
  candidate and incumbent for the short shape, or any divergence must
  be explained (numerics/quantization) and bounded.
- No failed requests (`failed == 0`), no truncated outputs
  (`completion_tokens` equals the requested budget under greedy).
- Sampled-arm runs use the fixed seed in `suite.json` so reruns are
  comparable.

## GPU-window protocol (run only when the user approves a pause)

The box normally serves production traffic on both GPUs. Do not stop,
restart, or load it outside an approved window. When the user approves:

1. Confirm the window and the exact incumbent config to restore after.
2. Incumbent arm: `vllm bench serve` (or the incumbent's own bench)
   against the live server, then
   `python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --output benchmarks/results-incumbent.json`.
3. Candidate arm: start the candidate server on a free port, run the
   same `run_bench.py` command against it, then the microbench
   (`scripts/benchmark.py` for the tiny model, or the candidate's own).
4. MTP arm: enable MTP in vLLM, restart once inside the window, rerun
   step 2's `run_bench.py` command.
5. Restore the original serving config, verify `/v1/models` answers,
   and record every output file with its command line. Outside the
   window, the only permitted live call is a single tiny smoke
   (`run_bench.py --smoke`, 16 tokens, greedy).
