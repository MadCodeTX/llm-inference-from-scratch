# What counts as a win

This document defines the metrics, the incumbent reference numbers, and
the thresholds a candidate engine must clear. Nothing here is a new
measurement; live comparisons happen only inside an approved GPU window
(see the protocol at the end).

## Status: ours-bridge first full-model run (measured)

Our pipeline-by-layers bridge ran the FULL 27B model across both GPUs and generated tokens: 32/32 token ids exact vs the recorded vLLM probes on all 3 quality prompts, decode 1.83 tok/s (unoptimized reference-dequant bridge), peak ~15.9 GB/GPU. Raw record: `benchmarks/baseline/ours_pipeline_first_run.json`; write-up in `docs/22_hybrid_qwen38.md`. Correctness milestone met; the win bar below (throughput vs incumbent) is not yet attempted.

Integration smoke re-check (this window, incumbent stopped first via `launch.sh --stop`): `scripts/gpu_smoke_hybrid.py` → `RESULT: PASS` (fp32 prefill/decode max_abs ≤1.5e-07, static vs legacy 0.0, chunkwise 1.19e-07; peak 9,043,968 B on cuda:0); `scripts/run_qwen38_gpu.py --max-new-tokens 8` on the single `capital` probe → 8/8 generated ids exact vs the recorded vLLM prefix (`first_divergence: null`, finite logits, prefill 0.92 s, decode 1.82 tok/s, load 27.4 s, peak 15,885,218,304 B / 15,931,358,720 B). Smoke re-check only; no new throughput claim. Restore the incumbent with `/home/nick/llm/vllm/launch.sh qwen3.8-27b-fp8 --no-cc`.

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

## Arm A measured baseline (exact-token, this window)

Measured live with `benchmarks/run_bench.py` against the incumbent
(`vllm/vllm-openai:v0.30.0`, TP2; full flags in
`benchmarks/baseline/arm_a_live_config.json`) using
`--tokenizer-dir` exact-token prompts. **Subset:** greedy arm only,
concurrency 1 and 4, all four shapes, 2 repeats per cell (16 batches,
`failed == 0` throughout). Concurrency 16 was not run, so this is not
the full suite. The sampled arm at c1/c4 was also measured
(`benchmarks/baseline/arm_a_exact_sampled_c1_c4.json`, 16 batches,
`failed == 0`). Values below are per-cell medians across the 2 runs;
full per-run distributions are in
`benchmarks/baseline/arm_a_exact_greedy_c1_c4.json`. Server-reported
`prompt_tokens` ran slightly below target (e.g. 2033 vs 2048) because
prompts are sent as decoded text and re-tokenized by the server.
Greedy `agent-shared-prefix` outputs ended early on EOS in some
requests (430–512 of 512 tokens at c1), so its throughput mixes
shorter decodes.

| Shape (in→out tok) | Conc. | TTFT p50 (s) | TTFT p90 (s) | TPOT p50 (s) | Tokens/min |
|---|---|---|---|---|---|
| short 2048→512 | 1 | 0.79 | 0.79 | 0.0203 | 2745 |
| short 2048→512 | 4 | 2.49 | 2.49 | 0.0233 | 8530 |
| medium 16384→1024 | 1 | 5.69 | 5.69 | 0.0207 | 2284 |
| medium 16384→1024 | 4 | 11.46 | 16.40 | 0.0244 | 5800 |
| long 49152→1024 | 1 | 14.34 | 14.34 | 0.0216 | 1687 |
| long 49152→1024 | 4 | 30.14 | 49.78 | 0.0266 | 3017 |
| agent-shared-prefix 8704→512 | 1 | 1.81 | 1.81 | 0.0205 | 2523 |
| agent-shared-prefix 8704→512 | 4 | 1.29 | 1.29 | 0.0234 | 8274 |

Quality probes (temperature 0, 32 tokens) are in
`benchmarks/baseline/arm_a_quality_probes.json` via
`benchmarks/quality_probe.py`; output-token ids there are labelled
`retokenized_local` because the API returned no server token ids.

## Measured roofline (this GPU window)

Measured with `benchmarks/microbench_roofline.py` in `.venv-gpu`
(defaults: warmup 2, repeats 5) after stopping the incumbent via
`launch.sh --stop`; raw JSONL and full tables in
`benchmarks/roofline/SUMMARY.md`, derived figures in
`benchmarks/roofline/roofline_derived.json` (via
`benchmarks/roofline_compute.py`). Achieved copy_ bandwidth peaked
at 920.0 GB/s per GPU; GEMM peaks on GPU0 were 149.42 TFLOP/s BF16
and 320.61 TFLOP/s FP8 (`torch._scaled_mm`, ran at all shapes);
2-rank NCCL all-reduce over PCIe fit to 8.67 us fixed latency plus
6.81 GB/s bus. Ridge points: 162.4 ops/byte (BF16), 348.5 (FP8).
Batch-1 decode at S=2048 under the FP8-resident byte model
(FP8 24,699,207,680 B + BF16 language 5,242,980,352 B + scales;
TP2 per-rank step 17,570,642,112 B + 67,108,864 B FP8 KV read) has
intensity 2.98 ops/byte (memory-bound) and a ceiling of
52.2 tok/s per stream (TPOT floor 19.17 ms). The incumbent's
measured c1 TPOT p50 (0.0203–0.0216 s) sits ~6–13% above that
floor. A single-device resident layout (29,945,203,072 B) does not
fit a 24 GiB card; its 30.7 tok/s ceiling is hypothetical only.
Roofline bounds are not candidate measurements; no win is claimed.

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

## Phase status

- **Phase 1b-prep (CPU foundations): done.** The streaming hybrid
  checkpoint loader, chunkwise gated-delta prefill, hybrid runtime
  state with `generate_hybrid`, the two-rank tensor-parallel sharding
  simulation (ch. 23), and the native MTP foundations (ch. 24) are
  built and tested on CPU only (`.venv/bin/python -m pytest -q` →
  221 passed). Framing follows Kiely, *Inference Engineering*
  (2026), sec. 2.2, sec. 2.4–2.5, sec. 4.2, sec. 5.2, and sec. 5.4;
  prose here is original.
- **Phase 1c (CPU integration): done; integration status — CPU suite 261 passed, 8 skipped (`.venv/bin/python -m pytest tests -q`, this window), Arm B still UNMEASURED, no win claimed.** Checkpoint assembly onto `HybridLM` validates on the real snapshot (851 tensors, 26,895,998,464 params), real layers 0/3 match HF (layer 3 bit-exact; layer 0 bf16 rel_rms 1.5e-3, fp32 1.2e-7), the TP execution simulation and static decode state (ch. 25) equal their dense/legacy paths, and `run_bench.py --tokenizer-dir` now builds exact-token prompts with Arm A/B defined in `benchmarks/arms/`.
- **GPU windows (roofline + bridge): done.** Inside user-approved
  windows the incumbent was stopped via `launch.sh --stop` and the
  roofline microbenchmarks were measured on both cards (bandwidth,
  BF16/FP8 GEMM, 2-rank NCCL all-reduce; see "Measured roofline"
  above and `benchmarks/roofline/SUMMARY.md`). The pipeline-by-layers
  bridge then ran the full 27B model across both GPUs (32/32 exact
  on all 3 probes; see Status above) and was smoke-rechecked at
  8 tokens / 1 prompt in the integration window. Arm B (vLLM + MTP)
  is still "not yet measured" and no win is claimed. The incumbent
  was left stopped at the end of each window; restore it with
  `/home/nick/llm/vllm/launch.sh qwen3.8-27b-fp8 --no-cc`.

## GPU-window protocol (run only when the user approves a pause)

The box normally serves production traffic on both GPUs. Do not stop,
restart, or load it outside an approved window. When the user approves:

1. Confirm the window and the exact incumbent config to restore after
   (fill in the launch flags Arm A leaves unrecorded; see
   `benchmarks/arms/arm-a-incumbent.json`).
2. Microbench first: run the candidate microbench
   (`benchmarks/microbench_hybrid_decode.py`, or
   `scripts/benchmark.py` for the tiny model) so kernel-level numbers
   exist before any serving load.
3. Arm A (incumbent as-is, `benchmarks/arms/arm-a-incumbent.json`):
   `vllm bench serve` (or the incumbent's own bench) against the live
   server, then
   `python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <real-snapshot> --output benchmarks/results-incumbent.json`.
4. Arm B (vLLM + MTP, `benchmarks/arms/arm-b-mtp.json`, UNMEASURED
   until run): enable MTP in vLLM with
   `--speculative-config {"method": "mtp", "num_speculative_tokens": K}`
   for each `K` in `[1, 2, 3]` as separate entries, restart once per
   entry inside the window, and rerun step 3's `run_bench.py` command
   per entry.
5. Ours: start the candidate server on a free port and run the same
   `run_bench.py` command against it.
6. Restore the original serving config with
   `/home/nick/llm/vllm/launch.sh qwen3.8-27b-fp8 --no-cc`, verify
   `/v1/models` answers, and record every output file with its
   command line. Outside the window, the only permitted live call
   is a single tiny smoke (`run_bench.py --smoke`, 16 tokens,
   greedy).
