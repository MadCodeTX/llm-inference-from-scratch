# Harness round 5 — decode TPOT (gated-norm elementwise/copy candidate)

Source artifact: [decode_round5.json](decode_round5.json) (live GPU round at
git HEAD `f73525a9da79fdfe419435353c23eed8946b8da4`, the shipped baseline;
the gated-norm candidate is uncommitted working-tree code — dirty files
listed in the artifact, untouched by this round — and this round edits no
source and commits nothing). Shipped baseline preserved as the A/B base:
graph decode, graph-scoped scale-cache on, `--fp8-config gemv-bk512`
default, exact local TPOT 22.112 ms (round 4). The profile lane then
proposed exactly one opt-in single-variable candidate from the
elementwise/copy bucket (4.487 ms/token, 3507 kernels/token in round 4):
`--gated-norm fused` (`RMSNormGated.use_fused_rmsnorm`, one `F.rms_norm`
kernel/layer replacing the pow/mean/rsqrt/cast chain, preserving the
intermediate dtype round-trip; isolated microprobe estimate ~0.31 ms/token).

**Question:** does candidate `R5C_candidate_gated_fused` (shipped defaults
plus `--gated-norm fused`, single variable) beat base `R5B_base_shipped`
(`--decode-mode graph --scale-cache on --fp8-config gemv-bk512
--gated-norm manual`, the current shipped defaults) exactly and faster?

Design: ABBA order `B,C,C,B,B,C` per batch, 3 measured runs per candidate
per shape, 12 torchrun processes total (24 probe-runs), all return code 0,
no runs discarded. Batch A runs all three recorded probes (capital, water,
counting) ×32 greedy in one process; batch B runs capital ×128 greedy.
GPUs were idle before (1 MiB, no compute apps) and after; no probe process
remains. Kernel smoke before the round:
`.venv-gpu/bin/python -m pytest tests/test_rmsnorm_gated_fused.py -q`
→ 6 passed. Raw logs/JSON under `/tmp/harness_r5/`.

| Shape | Base (shipped) median TPOT (min–max) | fused median TPOT (min–max) | Δ fused vs base | Exact |
|---|---|---|---|---|
| capital ×32 | 22.104 ms (22.102–22.107) | 21.867 ms (21.861–21.868) | −1.07% | 32/32 vs bridge + Arm A, every run |
| water ×32 | 22.109 ms (22.106–22.113) | 21.865 ms (21.861–21.873) | −1.10% | 32/32 vs bridge + Arm A, every run |
| counting ×32 | 22.103 ms (22.100–22.103) | 21.865 ms (21.856–21.903) | −1.08% | 32/32 vs bridge + Arm A, every run |
| capital ×128 | 22.154 ms (22.153–22.157) | 21.915 ms (21.906–21.916) | −1.08% | 32/32 prefix vs both refs; 128/128 vs base reference, every run |
| **Aggregate (pooled, n=12/candidate)** | **22.106 ms, 45.24 tok/s** | **21.867 ms, 45.73 tok/s** | **−1.08%** | exact everywhere |

Peak memory (runner-reported, per rank) is identical for base and fused:
16,367,239,168 B at ×32 and 16,346,798,592 B at ×128. Logits finite on
every run. The gain is uniform across prompts and lengths (−1.07% to
−1.10%), consistent with the profile lane's isolated estimate (~0.24 ms
saved vs ~0.31 ms upper estimate). Note the prior lane's caveat stands in
general — fused reduction order differs, so logits are tolerance-equal
rather than guaranteed bit-exact — but greedy tokens proved bit-identical
to bridge and recorded Arm A on every run of this round, including the
full 128-token capital continuation.

Selection rule (pre-stated, zero tolerance): any inexact candidate run
rejects the candidate regardless of speed; otherwise the candidate
becomes default only if exact everywhere, aggregate-median-faster, and
no shape median regresses by more than 1%. All conditions hold (exact
everywhere; −1.08% aggregate; worst shape −1.07%, i.e. no regression),
so the recommendation is **BECOME DEFAULT**: promote `--gated-norm fused`
to the default RMSNormGated path. This round itself changes no default
(no source edited); the promotion is a separate commit/default flip.

Deterministic selection on CPU reproduces the pick (capital ×32 is the
task-card shape the importer scores):

```bash
.venv/bin/python scripts/opt_harness.py import-metrics \
  benchmarks/harness/decode_round5.json --task-id decode-tpot
```

Frontier/chosen: `R5C_candidate_gated_fused` (recorded in the artifact's
`import_metrics` block; raw stdout at `/tmp/harness_r5/import_stdout.json`).

**No vLLM win.** The fused aggregate median 21.867 ms is above the Arm A
20.3 ms bar (and above the 19.17 ms roofline floor), so no beating-vLLM
claim is made. Even a sub-20.3 ms local result would be labelled
local-runner only, not a same-harness suite win. Arm B remains separate.
