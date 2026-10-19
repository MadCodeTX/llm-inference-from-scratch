# Harness² round 2 — decode TPOT (confirming round)

Source artifact: [decode_round2.json](decode_round2.json) (live GPU round at
git HEAD `71995f2`, local TP2 graph runner). Round 1
([decode_round1.json](decode_round1.json), HEAD `c263db2`) found
`--scale-cache on` faster on the graph path at capital ×32 only and left the
shipped default unchanged pending a multi-probe confirming round. The only
commit since round 1 is the harness-machinery commit itself; runner/model
source is unchanged.

**Question:** does C1 (`--scale-cache on`, single variable over shipped
graph defaults C0) hold up on all recorded probes and at ×128?

Design: ABBA order `C0,C1,C1,C0,C0,C1` per batch, 3 measured runs per
candidate per shape, 12 torchrun processes total, all return code 0, no
runs discarded. Batch A runs all three recorded probes (capital, water,
counting) ×32 greedy in one process; batch B runs capital ×128 greedy.
GPUs were idle before (1 MiB, no compute apps) and after; no probe process
remains. Raw logs/JSON under `/tmp/harness_r2/`.

| Shape | C0 median TPOT (min–max) | C1 median TPOT (min–max) | Δ C1 vs C0 | Exact |
|---|---|---|---|---|
| capital ×32 | 23.680 ms (23.680–23.725) | 22.844 ms (22.816–22.869) | −3.53% | 32/32 vs bridge + Arm A, every run |
| water ×32 | 23.683 ms (23.683–23.725) | 22.844 ms (22.829–22.874) | −3.54% | 32/32 vs bridge + Arm A, every run |
| counting ×32 | 23.679 ms (23.675–23.724) | 22.842 ms (22.825–22.870) | −3.53% | 32/32 vs bridge + Arm A, every run |
| capital ×128 | 23.720 ms (23.714–23.722) | 22.856 ms (22.855–22.859) | −3.64% | 32/32 prefix vs both refs; 128/128 vs C0 reference, every run |
| **Aggregate (pooled, n=12/candidate)** | **23.698 ms, 42.20 tok/s** | **22.850 ms, 43.76 tok/s** | **−3.58%** | exact everywhere |

Peak memory (runner-reported, per rank): C0 16,364,142,592 B and
C1 16,367,239,168 B at ×32 (~+3 MB for the scale cache); ×128 peaks are in
the artifact. Logits finite on every run.

Selection rule (pre-stated): C1 is confirmed only if exact on every run,
median-faster on the aggregate, and not more than 1% slower on any shape.
All three hold, so **the round-1 result is CONFIRMED** and the
recommendation is to **flip the shipped `--scale-cache` default to `on`
for the graph decode path**. This round does not change any default
itself (no source edited); the flip is a separate code change. The
eager-path contradiction from `ours_tp2_microopts_ab.json` is untouched:
this evidence is graph-path only, so any flip should be scoped or
re-measured for eager decode.

Deterministic selection on CPU reproduces the pick (capital ×32 is the
task-card shape the importer scores):

```bash
.venv/bin/python scripts/opt_harness.py import-metrics \
  benchmarks/harness/decode_round2.json --task-id decode-tpot
```

Frontier: `C1_scale_cache_on` (recorded in the artifact's
`import_metrics` block; raw stdout at `/tmp/harness_r2/import_stdout.json`).

**No vLLM win.** C1's aggregate median 22.850 ms is above the Arm A
20.3 ms bar, and the local-runner shape is not the Arm A serving harness
in any case; even a sub-20.3 ms local result would be labelled
local-runner only.
