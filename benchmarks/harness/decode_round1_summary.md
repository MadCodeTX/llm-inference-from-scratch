# Harness² round 1 — decode TPOT (run summary)

Source artifact: [decode_round1.json](decode_round1.json) (live GPU round at
git HEAD `c263db2`, local TP2 runner, capital probe ×32 greedy, graph decode).
Selection below was produced on CPU by the deterministic machinery, no GPU:

```bash
.venv/bin/python scripts/opt_harness.py import-metrics \
  benchmarks/harness/decode_round1.json --task-id decode-tpot
```

Raw selection output: [decode_round1_selection.json](decode_round1_selection.json).

| Candidate | Median TPOT | tok/s | Exact (bridge + recorded Arm A) |
|---|---|---|---|
| C0 base (shipped defaults) | 23.726 ms | 42.15 | 32/32, every run |
| C1 `--scale-cache on` | 22.866 ms | 43.73 | 32/32, every run |
| C2 `--lm-head replicated` | 24.987 ms | 40.02 | 32/32 (first probe) |
| C3 `--norm manual` | 24.312 ms | 41.13 | 32/32 (first probe) |
| C4 `--fp8-config old` | 28.183 ms | 35.48 | 32/32 (first probe) |
| C5 `--delta ref` | 24.530 ms | 40.77 | 32/32 (first probe) |

C0/C1 medians are ABBA repeats (5 runs each); C2–C5 are first probes only.

**Selected:** `C1_scale_cache_on` (−0.86 ms, −3.6% vs C0 median), the only
Pareto-frontier candidate under throughput ↑ then latency ↓.

**Not adopted as the shipped default.** The artifact itself records that
this contradicts the earlier eager-path paired result
(`ours_tp2_microopts_ab.json`: no consistent scale-cache winner), so the
win is graph-path-specific until a confirming multi-probe round runs.
Base (C0) remains the shipped default; `--scale-cache` stays opt-in.

**No vLLM win.** 22.866 ms is above the Arm A 20.3 ms bar, and the local
runner shape is not the Arm A serving harness in any case.
