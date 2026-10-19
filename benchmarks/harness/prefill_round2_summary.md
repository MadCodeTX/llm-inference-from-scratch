# Prefill round 2: prefix-chunk512 vs shipped defaults — RECOMMEND (integration separate)

Candidate: `prefix-chunk512` = `--chunk 512 --snapshot-interval 512` on
`scripts/prefix_ttft_bench.py` vs shipped defaults (chunk 128,
snapshot-interval 256, `--fp8-config gemv-bk512`). Base preserved: HEAD
`f73525a9da79fdfe419435353c23eed8946b8da4`, graph decode + graph-scoped
scale-cache on, exact local TPOT 22.112 ms (decode not re-measured). This
round edited no source, changed no default, and committed nothing; candidate
stays opt-in until a separate integration step.

## Method

- Arms: A = shipped defaults (prefix runs pass no chunk/interval/fp8 flags);
  B = candidate flags only. Pure-prefill support on
  `scripts/prefill_bench_tp2.py`: A `--chunks 128` vs B `--chunks 512`,
  `--lengths 512,2048,8192 --no-static`, fp8 `gemv-bk512` default both.
- Order ABBAAB (A,B,B,A,A,B), 3 process runs per arm per shape; each prefix
  run is internally median-of-3 cold + median-of-3 warm. 30 torchrun
  processes: 27 rc=0, 3 base GPU-storage agent runs rc=1 (OOM, retained).
  Full commands, per-run values, and raw-log paths (`/tmp/pr2_*.log`,
  `/tmp/pr2_*.json`, `/tmp/pr2_*.mem`) are in `prefill_round2.json`.
- GPU gate: both RTX 4090 free before the first run (1 MiB, 0%, no compute
  processes); driver re-checked <500 MiB before every run and after the last
  (both 1 MiB, 0%). No external work or vLLM script was touched; no probe
  or bench process remains.
- TTFT proxy = model-only B=1 prefill seconds. Neither bench generates
  tokens, so exactness is last-token argmax equality (no decode-token
  exactness claimed). Any argmax or reused-token mismatch rejects.

## Prefix bench (GPU storage; agent shape on CPU storage, see below)

Medians of 3 process runs (each internally median-of-3):

| Shape | Metric | A base | B candidate | B vs A |
|---|---|---:|---:|---:|
| 512+256 | cold / warm | 0.5416 / 0.1821 s | 0.4469 / 0.1534 s | −17.5% / −15.8% |
| 2048+256 | cold / warm | 1.6437 / 0.1834 s | 1.3460 / 0.1562 s | −18.1% / −14.8% |
| 8192+512 (CPU storage) | cold / warm | 7.5113 / 0.5344 s | 6.3865 / 0.4680 s | −15.0% / −12.4% |

Warm `n_reused` equals the shared length in every run (512 / 2048 / 8192);
last-token argmax is identical across arms and cold/warm (12356 at
512/2048 shapes, 43221 at the agent shape). Exactness passes everywhere.

## Agent shape on GPU storage: base cannot run

Base A at 8192+512 with `--storage gpu` OOMs in all 3 runs
(`torch.OutOfMemoryError` on both ranks, device peak 24056 MiB; interval-256
snapshots hold ~34 GPU-resident states). Candidate B completes: cold
5.3927 s, warm 0.3237 s, `n_reused` 8192, argmax 43221/43221, device peak
20686 MiB. The like-for-like agent comparison above therefore uses CPU
storage for both arms; the GPU-storage rows are reported as a capability
difference, not a timed win.

## Pure prefill (`prefill_bench_tp2.py`, medians of 3)

| Prompt | A chunk128 s / tok/s | B chunk512 s / tok/s | tok/s Δ | Peak GiB/rank A→B | Argmax |
|---:|---|---|---:|---:|---|
| 512 | 0.3584 / 1428.7 | 0.2934 / 1745.0 | +22.1% | 15.19→15.36 | 30876 |
| 2048 | 1.4410 / 1421.3 | 1.1826 / 1731.8 | +21.9% | 15.24→15.41 | 2336 |
| 8192 | 6.0236 / 1360.0 | 5.0203 / 1631.8 | +20.0% | 15.58→15.75 | 310 |

Argmax is single-valued per length across all runs and arms. Peak memory
rises ≤0.17 GiB/rank (≤0.36 GiB device peak in prefix runs: 17394→16944 MiB
at 2048+256 actually falls; worst prefix rise is 16524→16672 MiB).

## Importer

`opt_harness import-metrics` on `/tmp/pr2_import_metrics.json` (cold
throughput/latency for the 3 prefix shapes) selects `cand512_512`. Limits:
the importer pools per-shape entries as separate candidates and sees only
imported cold metrics — warm times, pure-prefill rows, peak memory, and the
GPU-storage agent OOM are not represented. The paired ABBA medians above
remain the verdict basis.

## Verdict: RECOMMEND default change (separate integration step)

Candidate is exact (argmax + reuse), aggregate faster (median cold −17.5%,
warm −14.8% across prefix shapes; +20–22% pure-prefill tok/s), and has no
shape regression >1% in any measured cell. It also removes the GPU-storage
OOM at the agent shape. Known tradeoff, unchanged: snapshot interval 512
reuses less for shared prefixes that are not 512-aligned (worst case <512
extra recomputed tokens vs interval 256); all tested shared lengths are
512-aligned and reuse exactly. Recommendation only — defaults are unchanged
in this round and promotion needs its own integration change.

Ours-vs-ours local B=1 only: no serving TTFT or vLLM win is claimed; Arm A/B
figures are context unless measured in the same serving harness.
