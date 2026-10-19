# What counts as a win

This document defines the metrics, the incumbent reference numbers, and
the thresholds a candidate engine must clear. Nothing here is a new
measurement; live comparisons happen only inside an approved GPU window
(see the protocol at the end).

## Status: ours-bridge first full-model run (measured)

Our pipeline-by-layers bridge ran the FULL 27B model across both GPUs and generated tokens: 32/32 token ids exact vs the recorded vLLM probes on all 3 quality prompts, decode 1.83 tok/s (unoptimized reference-dequant bridge), peak ~15.9 GB/GPU. Raw record: `benchmarks/baseline/ours_pipeline_first_run.json`; write-up in `docs/22_hybrid_qwen38.md`. Correctness milestone met; the win bar below (throughput vs incumbent) is not yet attempted.

Integration smoke re-check (this window, incumbent stopped first via `launch.sh --stop`): `scripts/gpu_smoke_hybrid.py` → `RESULT: PASS` (fp32 prefill/decode max_abs ≤1.5e-07, static vs legacy 0.0, chunkwise 1.19e-07; peak 9,043,968 B on cuda:0); `scripts/run_qwen38_gpu.py --max-new-tokens 8` on the single `capital` probe → 8/8 generated ids exact vs the recorded vLLM prefix (`first_divergence: null`, finite logits, prefill 0.92 s, decode 1.82 tok/s, load 27.4 s, peak 15,885,218,304 B / 15,931,358,720 B). Smoke re-check only; no new throughput claim. Restore the incumbent with `/home/nick/llm/vllm/launch.sh qwen3.8-27b-fp8 --no-cc`.

## Candidate table (measured; bridge rows same 3 probes × 32 tok)

| Candidate | Decode tok/s | TPOT p50 | Token agreement |
|---|---|---|---|
| vLLM Arm A (greedy short c1) | ~49/stream (2745 tok/min) | 20.3 ms | reference |
| ours bridge v1 (reference dequant) | 1.83 | ~548 ms | 32/32 exact ×3 |
| ours bridge v2 (Triton fused, default) | 19.6–19.7 | ~50.7 ms | 32/32 exact ×3 |
| ours TP2 v1 (true TP2, Triton resident FP8) | 22.9–23.6 | 42.5–43.7 ms | 32/32 exact ×3 |
| ours TP2 v2 (v1 + fused RMSNorm) | 23.4–23.9 | 41.8–42.8 ms | 32/32 exact ×3 |
| ours TP2 v3 (static + CUDA graph, vocab lm_head) | 34.6 | 28.9 ms | 32/32 exact ×3 |
| ours TP2 v3 + FP8 tile retune (BLOCK_N=32) | 40.9 (capital ABBA) | 24.48 ms | 32/32 exact (capital) |
| ours TP2 + MTP mtp1-eager (speculative) | 25.7 eff (capital 128) | 38.68 ms eff | 128/128 exact vs graph-nonspec; 32/32 x3 |
| ours TP2 + MTP mtp1-graph (chunk+draft graphs) | 30.6 eff (capital 128) | 32.51 ms eff | 128/128 exact vs graph-nonspec; 32/32 x3 |
| ours TP2 + MTP mtp1-graph2 (merged graph, 1 sync/cycle) | 30.4 eff (capital 128) | 32.90 ms eff | 128/128 exact vs graph-nonspec; 32/32 x3 |
| ours TP2 graph + fused gated-delta (nonspec, best) | 42.25 | 23.67 ms | 32/32 exact x3 |
| ours TP2 graph + `--scale-cache on` (Harness² round 1, local runner) | 43.73 | 22.87 ms | 32/32 exact, ABBA ×5 vs base 23.73 ms |
| ours TP2 graph + scale-cache on (round 2 confirmed, graph default) | 43.76 | 22.850 ms | exact all shapes, ABBA vs 23.698 ms |
| ours TP2 graph + scale-cache + `--fp8-config gemv` (round 3, shipped default) | 44.90 | 22.270 ms | exact everywhere, ABBA vs 22.820 ms |
| ours TP2 graph + scale-cache + `--fp8-config gemv-bk512` (round 4, shipped default) | 45.22 | 22.112 ms | exact everywhere, ABBA vs 22.271 ms |
| ours TP2 graph + scale-cache + gemv-bk512 + `--gated-norm fused` (round 5, shipped default) | 45.73 | 21.867 ms | exact everywhere, ABBA vs 22.106 ms |
| vLLM Arm B (MTP K=1, 256608) | ~71/stream c1 short (4272-4291 tok/min) | n/a (chunk TPOT) | 32/32 exact x3; MEASURED-WITH-DEVIATION |
| Roofline floor (TP2 batch-1, bound only) | 52.2 ceiling | 19.17 ms floor | n/a — not a measurement |

#### Throughput ladder — tokens/min by concurrency (measured only)

Arm A/B are `run_bench.py` serving numbers (short 2048→512); ours batched is the local TP2 runner (6-token prompts, 64 greedy tokens, eager, ignore-EOS) — different harness/shape, so aggregate crossings are **not** WIN-threshold wins.

| Concurrency | Arm A (short) | Arm B (MTP K1, dev.) | Ours (measured) | Verdict |
|---|---|---|---|---|
| 1 | 2,745 (TPOT 20.3 ms) | 4,272–4,291 (~71 tok/s) | graphed B=1 2,535 (23.67 ms); eager B=1 1,367 | lose per-stream vs both |
| 2 | — | — | eager B=2 2,721 | no Arm bar |
| 4 | 8,530 | 12,113–12,136 | eager B=4 5,360 | lose vs both |
| 8 | — | — | eager B=8 10,746 | passes Arm A c4 numerically only |
| 16 | — | — | eager B=16 20,001 | passes Arm B c4 numerically only |

Prefill tok/s v2 (`ours_tp2_prefill_v2.json`, ours-vs-ours): 2048 chunk128 1,412.8 / chunk512 1,731.6 / chunk1024 1,596.0; 8192 chunk512 1,631.3; last-argmax equal. Prefill round 2 (`benchmarks/harness/prefill_round2.json`, ABBAAB): pure-prefill chunk512 vs chunk128 +22.1%/+21.9%/+20.0% tok/s at 512/2048/8192 (1,745.0/1,731.8/1,631.8 tok/s), argmax equal, peak +≤0.17 GiB/rank. Arm A effective prompt rate ~2.9–3.4k tok/s (from TTFT) — **we do not beat Arm A/B on prefill**.

Prefix TTFT v2 (`ours_prefix_ttft_v2.json`, 2048+256=2304, interval 256): cpu cold 1.926s / warm 0.252s; gpu cold 1.643s / warm 0.184s; n_reused=2048, argmax 12356 cold==warm. v1 was cold 10.66s / warm 1.31s. Prefill round 2 prefix bench (ABBAAB, `benchmarks/harness/prefill_round2.json`): shipped bench defaults are now `--chunk 512 --snapshot-interval 512` (v2 baseline via `--chunk 128 --snapshot-interval 256`); candidate vs base cold/warm −17.5%/−15.8% (512+256), −18.1%/−14.8% (2048+256, gpu storage), −15.0%/−12.4% (8192+512, cpu storage), warm n_reused == shared and argmax identical everywhere; base OOMs at 8192+512 on gpu storage in all 3 runs while the candidate completes (capability difference, not a timed win). Library defaults (`tp_prefix_cache` interval 256, `prefill_chunkwise_static` chunk 128) are unchanged — only the measured bench surface adopted 512/512. Arm A agent TTFT 1.81s @8704 is a different shape — **no TTFT win claimed**. Ours MTP graph2 (~32.9 ms eff.) stays slower than nonspec graphed (23.67 ms) and is not defaulted.

**Where we do/don't beat:** per-stream decode — lose to Arm A (best local 21.867 vs 20.3 ms) and Arm B (~71 tok/s); prefill rate — lose; prefix TTFT — no comparable-shape claim; aggregate tok/min — B=8/B=16 cross the c4 bars numerically on a different shape only. **No candidate is a win vs Arm A or Arm B under the thresholds below.**

Harness² round 1 detail (`benchmarks/harness/decode_round1.json`, summary in `benchmarks/harness/decode_round1_summary.md`): single-variable candidates on the graphed local runner (capital ×32); ABBA medians C0 base 23.726 ms vs C1 `--scale-cache on` 22.866 ms / 43.73 tok/s, 32/32 exact on every run, selected by `scripts/opt_harness.py import-metrics` (`benchmarks/harness/decode_round1_selection.json`). The gain is graph-path-specific and contradicts the earlier eager paired scale-cache result, so the shipped default stays base (`--scale-cache` opt-in) pending a confirming multi-probe round. 22.87 ms remains above Arm A 20.3 ms — **not a win**.

Harness² rounds 2–3 (integrated): round 2 (`benchmarks/harness/decode_round2.json`) confirmed scale-cache on all shapes (aggregate 22.850 vs 23.698 ms, -3.58%, exact everywhere), so `--scale-cache` now defaults to `on` for `--decode-mode graph` only (eager stays `off`). Round 3 (`benchmarks/harness/decode_round3.json`) promoted the M=1 GEMV kernel (aggregate 22.270 vs 22.820 ms, -2.41%, exact everywhere, no per-shape regression), so `--fp8-config gemv` became the shipped default (M>1 as `new`).

Harness² prefill round 1 + decode round 4 (integrated): prefill candidate `prefill-s2` (`num_stages=2`) was rejected — slower at every shape/chunk (-1.3% to -7.5%; chunk512 2048: 1703.4 vs 1728.3 tok/s, 8192: 1608.5 vs 1629.9 tok/s, last-token argmax equal; `benchmarks/harness/prefill_round1.json`) — and its code was removed, not kept as an opt-in. Decode candidate `gemv-bk512` (M=1 GEMV `BLOCK_K` 256→512) was promoted to the shipped default (`benchmarks/harness/decode_round4.json`): aggregate median 22.112 vs 22.271 ms (-0.71%, 45.22 vs 44.90 tok/s), per-shape -0.67% to -0.72%, exact everywhere (x32 32/32 vs bridge + Arm A; x128 128/128 vs base), so `--fp8-config gemv-bk512` is now the default and `gemv` remains the round-3 A/B baseline.

Harness² decode round 5 + prefill round 2 (integrated): decode candidate `--gated-norm fused` (one `F.rms_norm` per gated-delta layer, 48/token) was promoted (`benchmarks/harness/decode_round5.json`): aggregate median 21.867 vs 22.106 ms (-1.08%, 45.73 vs 45.24 tok/s), per-shape -1.07% to -1.10%, exact everywhere (x32 32/32 vs bridge + Arm A; x128 128/128 vs base), peak memory identical, so `--gated-norm fused` is now the runner default and `manual` remains the A/B baseline. Prefill candidate `prefix-chunk512` was promoted on the measured bench surface (`benchmarks/harness/prefill_round2.json`): `scripts/prefix_ttft_bench.py` defaults are now `--chunk 512 --snapshot-interval 512` (numbers above; library defaults unchanged). Local-runner only: 21.867 ms is still 1.57 ms above Arm A 20.3 ms and 2.70 ms above the 19.17 ms floor — **not a win vs Arm A or Arm B**, and local prefill/decode results are not serving-suite vLLM wins.


Bridge v2 detail: prefill 0.54 s (first, JIT) / 0.10–0.12 s, load 23.5 s, peak ~15.0 / ~15.1 GB/GPU. Re-check this window: 8-token capital probe 8/8 prefix exact at 18.98 tok/s, prefill 0.52 s. No candidate clears the win thresholds; Arm B retry MEASURED-WITH-DEVIATION (see below).

TP2 v1 detail: `scripts/run_qwen38_tp2.py` (torchrun TP2, resident-FP8 rank shards, replicated embed/head, 2 all-reduces/block/step), raw record `benchmarks/baseline/ours_tp2_first_run.json`: load 21.2 s/rank, peak 16.39 GiB/rank, prefill 0.61 s first (Triton JIT) then 0.14–0.17 s, TPOT 42.5–43.7 ms vs Arm A 20.3 ms and the 19.17 ms roofline floor — faster than bridge v2 but roughly 2.1× off Arm A, so **not a win**; the gap to the floor is unprofiled.

TP2 v2 detail: profile (`scripts/profile_qwen38_tp2.py` → `benchmarks/profile/SUMMARY_tp2.md`) found decode host-launch-bound (clean 44.4 ms/token vs 31.3 ms GPU busy; FP8 kernel 18.9 ms, collectives exactly 128 all-reduces = 1.28 ms, no redundant gather). Single shipped fix: fused RMSNorm (`--norm fused`, default; `--norm manual` A/B), raw record `benchmarks/baseline/ours_tp2_v2.json`: TPOT 41.8–42.8 ms (23.4–23.9 tok/s), ids identical to v1 and 32/32 exact ×3 — still ~2.2× off Arm A/floor, **not a win**.

TP2 v3 detail: static state (`TPStaticState`) + one captured `decode_token_static` graph replayed per token removes the ~13.1 ms host-launch gap (44.37 ms wall = 31.28 GPU busy + ~13.1 host; floor 19.17 ms, vLLM 20.3 ms). Graph record `benchmarks/baseline/ours_tp2_graph.json`: 34.6 tok/s, TPOT 28.9 ms, 32/32 exact ×3. Micro-opts (`benchmarks/baseline/ours_tp2_microopts_ab.json`, paired ABBA): vocab-parallel lm_head 43.73 vs 45.60 ms/token replicated — shipped as default; FP32 scale cache no consistent winner — ships opt-in, default off. Tile retune after graphs (`benchmarks/baseline/ours_tp2_tile_ab.json`, ABBA): BLOCK_N=32 at M=1, 28.88→24.48 ms/token (−15.2%), ids identical — shipped as `--fp8-config new` default. Best-mode 8-token smoke this session: 40.7 tok/s, TPOT 24.55 ms, 8/8 exact. Remaining gap: 24.48 − 19.17 = 5.31 ms above floor, 4.18 ms above vLLM 20.3 ms — **not a win**.

MTP speculative detail (2026-10-18, artifact `benchmarks/baseline/ours_tp2_mtp_matrix.json`): gamma=1 native-MTP speculation is exact everywhere (32/32 x3 probes vs bridge+Arm-A; capital 128/128 vs this run's graph-nonspec reference ids) with acceptance 0.88-0.94, but it is **slower** than graph-nonspec (40.5-40.8 tok/s, TPOT 24.5 ms): mtp1-eager effective TPOT 38.68 ms and mtp1-graph 32.51 ms on capital x128. CUDA-event profile per cycle (capital x128): draft 5.8 ms, verify chunk 52.8 ms eager / 34.6 ms graph, snapshot 0.46 ms, rollback copy ~0.03 ms, reject re-feed ~3.3 ms, host/capture overhead 12.7-18.8 ms. Expected tokens/cycle at gamma=1 is 1+alpha = 1.92; phase-only predicted effective TPOT for mtp1-graph is 22.96 ms/token, but measured is 32.51 ms -- the M=2 verify chunk costs 34.6 ms vs 24.5 ms for the M=1 graph step and host/capture overhead consumes the rest. Runner speculative default stays `off`; no speedup is claimed and this is **not a win**.

Waterfall vs Arm A / Arm B (measured, same-shape only where noted): Arm A TPOT 20.3 ms (greedy short c1, serving harness) is the reference; ours best nonspec fused-delta graph is 23.67 ms (42.25 tok/s, local runner x32, 32/32 exact x3) — 3.37 ms above Arm A, 4.50 ms above the 19.17 ms roofline floor, dominated by Triton FP8 14.57 ms/token (profile `benchmarks/profile/tp2_graph_profile.json`: elementwise residual 5.34 ms fused, dense GEMV 1.73 ms, NCCL 1.39 ms). Spec candidates are slower at capital x128: mtp1-graph 32.51 ms, mtp1-graph2 32.90 ms, mtp1-eager 38.68 ms — all exact, none beats nonspec. Arm B (MEASURED-WITH-DEVIATION, max-len 256608) has no per-token TPOT comparable to Arm A (chunk TPOT only; effective c1 short ~71 tok/s), so no TPOT waterfall vs Arm B is drawn. **No candidate is a win vs Arm A or Arm B.**

Raw v2 record: `benchmarks/baseline/ours_pipeline_fastfp8.json` (`--sampling-sync batched --grad-mode inference`; A/B toggles `--fp8-path reference --sampling-sync per-token --grad-mode no-grad` reproduce v1 at 1.82 tok/s, 8/8 prefix). v2 is ~10.7× v1, driven by removing the profiled dequant cost (`benchmarks/profile/SUMMARY.md`: ~243 ms/token/GPU dequant vs ~25 ms matmul). Batched-vs-per-token sampling sync alone was 19.57 vs 19.56 tok/s — no measurable difference once dequant is gone, so no speed is claimed for it. Other profile candidates were audited and skipped where evidence does not justify a change: no `BlockFP8Linear` is called twice with the same weight in one forward (each projection fires once per block), and the pipeline boundary transfer is one `[1, 1, 5120]` BF16 tensor per token (~10 KB), negligible next to the dequant traffic — neither was changed. `--grad-mode inference` (default) replaces `no_grad` with `inference_mode`; `--grad-mode no-grad` reproduces the old context. **Not a win vs Arm A:** Arm A greedy short c1 is TPOT p50 20.3 ms (~49 tok/s/stream, 2745 tok/min); v2 TPOT is ~50.7 ms (~19.7 tok/s, ~1180 tok/min). v2 clears neither threshold above; Arm B is MEASURED-WITH-DEVIATION with no comparable per-token TPOT (below).

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
  enabled. First attempt 2026-10-18 (exact Arm A flags + MTP K=1) FAILED_TO_START:
  KV cache 4.49 GiB needed > 4.4 GiB available at `max_model_len` 262144
  (vLLM estimated max 256608). **Retry 2026-10-18: MEASURED-WITH-DEVIATION** —
  exactly one deviation, `--max-model-len 256608` instead of 262144, everything
  else identical to Arm A + MTP K=1 (isolated `vllm-armb-qwen38-mtp`, 127.0.0.1:8003).
  Quality 3/3 exact vs Arm A; greedy c1 short 4272–4291 tok/min, agent-shared-prefix
  TTFT p50 3.33 s cold / 0.90 s warm; short c4 12113–12136 tok/min; spec acceptance
  3574/3617 (98.8%); failed=0. Raw: `benchmarks/baseline/arm_b_mtp_k1_256608.json`
  (+`_c4`), `arm_b_quality_probes.json`, `arm_b_metrics_*.txt`. Caveat: `run_bench`
  TPOT is per streamed chunk and MTP chunks carry multiple tokens, so Arm B chunk
  TPOT (~24.7 ms) is not comparable to Arm A per-token TPOT; no TPOT win/loss is
  claimed from it. Status in `benchmarks/arms/arm-b-mtp.json`: MEASURED-WITH-DEVIATION.

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
- **Phase 1c (CPU integration): done; integration status — CPU suite 261 passed, 8 skipped (`.venv/bin/python -m pytest tests -q`, this window), Arm B FAILED_TO_START (2026-10-18, exact-flags KV overflow; see Arm B section), no win claimed.** Checkpoint assembly onto `HybridLM` validates on the real snapshot (851 tensors, 26,895,998,464 params), real layers 0/3 match HF (layer 3 bit-exact; layer 0 bf16 rel_rms 1.5e-3, fp32 1.2e-7), the TP execution simulation and static decode state (ch. 25) equal their dense/legacy paths, and `run_bench.py --tokenizer-dir` now builds exact-token prompts with Arm A/B defined in `benchmarks/arms/`.
- **GPU windows (roofline + bridge): done.** Inside user-approved
  windows the incumbent was stopped via `launch.sh --stop` and the
  roofline microbenchmarks were measured on both cards (bandwidth,
  BF16/FP8 GEMM, 2-rank NCCL all-reduce; see "Measured roofline"
  above and `benchmarks/roofline/SUMMARY.md`). The pipeline-by-layers
  bridge then ran the full 27B model across both GPUs (32/32 exact
  on all 3 probes; see Status above) and was smoke-rechecked at
  8 tokens / 1 prompt in the integration window. Arm B (vLLM + MTP)
  was attempted 2026-10-18 and FAILED_TO_START (exact-flags KV overflow);
  no win is claimed. The incumbent
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
4. Arm B (vLLM + MTP, `benchmarks/arms/arm-b-mtp.json`, FAILED_TO_START 2026-10-18;
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
