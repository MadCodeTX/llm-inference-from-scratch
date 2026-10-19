# Harness round 4 — decode TPOT (GEMV BLOCK_K candidate)

Source artifact: [decode_round4.json](decode_round4.json) (live GPU round at
git HEAD `a5f1e7dad4b87529341e3f9c938a1899d2ba1a05`; the gemv-bk512 candidate
is uncommitted working-tree code — dirty files listed in the artifact,
including the pre-existing dirty prefill-s2 files, untouched — and this round
edits no source and commits nothing). Round 3
([decode_round3.json](decode_round3.json)) confirmed the M=1 GEMV kernel
(`--fp8-config gemv`, now the shipped default, 22.270 ms aggregate median).
The profile lane then proposed exactly one opt-in single-variable candidate:
`--fp8-config gemv-bk512` (`M1_GEMV_BK512_CONFIG={BLOCK_N:16, BLOCK_K:512,
num_warps:4, num_stages:3}`, GEMV only at M=1; M>1 exactly as `gemv`/`new`).

**Question:** does candidate `R4C_candidate_gemv_bk512` (shipped defaults
plus `--fp8-config gemv-bk512`, single variable) beat base `R4B_base_gemv`
(`--decode-mode graph --scale-cache on --fp8-config gemv`, the current
shipped defaults) exactly and faster?

Design: ABBA order `B,C,C,B,B,C` per batch, 3 measured runs per candidate
per shape, 12 torchrun processes total, all return code 0, no runs
discarded. Batch A runs all three recorded probes (capital, water,
counting) ×32 greedy in one process; batch B runs capital ×128 greedy.
GPUs were idle before (1 MiB, no compute apps) and after; no probe process
remains. Kernel smoke before the round:
`.venv-gpu/bin/python -m pytest tests/test_fp8_triton_gpu.py -k bk512 -q`
→ 5 passed. Raw logs/JSON under `/tmp/harness_r4/`.

| Shape | Base (gemv) median TPOT (min–max) | bk512 median TPOT (min–max) | Δ bk512 vs base | Exact |
|---|---|---|---|---|
| capital ×32 | 22.271 ms (22.267–22.272) | 22.111 ms (22.105–22.119) | −0.72% | 32/32 vs bridge + Arm A, every run |
| water ×32 | 22.271 ms (22.267–22.276) | 22.114 ms (22.111–22.115) | −0.71% | 32/32 vs bridge + Arm A, every run |
| counting ×32 | 22.268 ms (22.266–22.274) | 22.107 ms (22.107–22.109) | −0.72% | 32/32 vs bridge + Arm A, every run |
| capital ×128 | 22.309 ms (22.308–22.314) | 22.159 ms (22.158–22.162) | −0.67% | 32/32 prefix vs both refs; 128/128 vs base reference, every run |
| **Aggregate (pooled, n=12/candidate)** | **22.271 ms, 44.90 tok/s** | **22.112 ms, 45.22 tok/s** | **−0.71%** | exact everywhere |

Peak memory (runner-reported, per rank) is identical for base and bk512:
16,367,239,168 B at ×32 and 16,346,798,592 B at ×128. Logits finite on
every run. The gain is small but uniform across prompts and lengths
(−0.67% to −0.72%), consistent with the profile lane's real-model estimate
(FP8 GEMV 13.95 → 13.77 ms/token on the HBM-bound kernel).

Selection rule (pre-stated, zero tolerance): any inexact candidate run
rejects the candidate regardless of speed; otherwise the candidate
becomes default only if exact everywhere, aggregate-median-faster, and
no shape median regresses by more than 1%. All conditions hold (exact
everywhere; −0.71% aggregate; worst shape −0.67%, i.e. no regression),
so the recommendation is **BECOME DEFAULT**: promote `gemv-bk512` to the
default M=1 FP8 path. This round itself changes no default (no source
edited); the promotion is a separate commit/default flip.

Deterministic selection on CPU reproduces the pick (capital ×32 is the
task-card shape the importer scores):

```bash
.venv/bin/python scripts/opt_harness.py import-metrics \
  benchmarks/harness/decode_round4.json --task-id decode-tpot
```

Frontier/chosen: `R4C_candidate_gemv_bk512` (recorded in the artifact's
`import_metrics` block; raw stdout at `/tmp/harness_r4/import_stdout.json`).

**No vLLM win.** The bk512 aggregate median 22.112 ms is above the Arm A
20.3 ms bar (and above the 19.17 ms roofline floor), so no beating-vLLM
claim is made. Even a sub-20.3 ms local result would be labelled
local-runner only, not a same-harness suite win. Arm B remains separate.
