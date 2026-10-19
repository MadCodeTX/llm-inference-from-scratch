# Prefill round 1: prefill-s2 vs shipped gemv — REJECT

Candidate: `--fp8-config prefill-s2` (PREFILL_S2_TUNED, num_stages=2, M>=64) vs shipped default `--fp8-config gemv`. Adopted baseline preserved (graph-scoped scale-cache on, M=1 FP8 GEMV, best exact local TPOT 22.270 ms — decode not re-measured, defaults untouched).

## Method
- Command (x6, ABBAAB): `.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/prefill_bench_tp2.py --lengths 512,2048,8192 --chunks 128,512,1024 --no-static --fp8-config {gemv|prefill-s2} --out /tmp/prefill_run<N>_<cfg>.json`; logs `/tmp/prefill_run<N>_<cfg>.log`, all rc=0.
- 3 measured runs per config; every run covers all lengths/chunks. GPU before/after: both RTX 4090 free (1 MiB, no processes). No probe remains.
- Git HEAD a5f1e7d; dirty candidate files: scripts/prefill_bench_tp2.py, scripts/run_qwen38_tp2.py, src/inference_from_scratch/fp8_triton.py, tests/test_fp8_tile_config.py, tests/test_fp8_triton_gpu.py. Not committed.
- TTFT proxy = model-only prefill seconds (B=1, no server/queue/sampling). Peak mem ~15.4 GiB/rank both arms.

## Medians (tok/s, s2 vs gemv)
| Shape | chunk512 | chunk128 | chunk1024 |
|---|---|---|---|
| 2048 | 1703 vs 1728 (-1.4%) | 1319 vs 1413 (-6.6%) | 1567 vs 1592 (-1.6%) |
| 8192 | 1609 vs 1630 (-1.3%) | 1264 vs 1358 (-6.9%) | 1495 vs 1518 (-1.5%) |
| 512 | 1716 vs 1742 (-1.5%) | 1325 vs 1432 (-7.5%) | 1716 vs 1743 (-1.5%) |

Prefill seconds at best chunk512: 2048: 1.202s vs 1.185s; 8192: 5.093s vs 5.026s.

## Exactness
Last-token argmax equal within every run and across configs at all lengths/chunks (no mismatch). Prior lane also showed kernel-level bitwise equality. Exactness passes; speed fails.

## Verdict: REJECT (do not adopt, do not keep as opt-in recommendation)
prefill-s2 is slower at every shape/chunk (-1.3% to -7.5%); num_stages=2 loses K-loop latency hiding. `opt_harness import-metrics` on derived metrics (/tmp/prefill_import_metrics.json) also selects gemv as sole frontier. Revert/leave unadopted per prior lane instruction.

vLLM note: Arm A TTFT (0.79s/5.69s/14.34s, different shapes) is effective-rate context only — no TTFT/serving win claimed from local prefill. scripts/prefix_ttft_bench.py not used: no --fp8-config flag, fixed chunk128, cannot express candidate.
