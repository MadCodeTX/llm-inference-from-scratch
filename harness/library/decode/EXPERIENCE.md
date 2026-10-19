# EXPERIENCE — decode

## D1 — Reference dequant is the bridge bottleneck

- **cue**: pipeline-bridge decode at ~1.8 tok/s with matmul looking cheap.
- **lesson**: profile before optimising — materialised dequant, not matmul,
  bound the reference bridge; do not tune GEMV shapes against a dequant wall.
- **evidence**: `benchmarks/profile/SUMMARY.md` + `benchmarks/profile/bridge_profile.json`:
  551.5 ms/token wall (1.81 tok/s); dequant 243 ms/GPU/token (~88% summed over
  both segments) vs FP8 `F.linear` matmul 25 ms/GPU/token; baseline artifact
  `benchmarks/baseline/ours_pipeline_first_run.json` at 1.83 tok/s, 32/32 exact ×3.
- **confidence**: high (profiler + CUDA-event timers agree; ids exact).
- **applicability limits**: describes the v1 reference-dequant bridge only;
  says nothing about the fused path's costs.

## D2 — Fused FP8 removes the dequant wall (device-path fix)

- **cue**: dequant (D1) dominating; a fused kernel can read FP8 once.
- **lesson**: switch the default FP8 path from reference dequant to the
  Triton fused dequant-GEMM; keep `--fp8-path reference` only as the A/B
  fallback that reproduces v1.
- **evidence**: `benchmarks/baseline/ours_pipeline_fastfp8.json`: bridge v2
  19.6–19.7 tok/s (TPOT ~50.7 ms), 32/32 exact ×3, load 23.5 s; A/B toggles
  reproduce v1 at 1.82 tok/s. `benchmarks/WIN.md` records ~10.7× v1→v2.
  Batched-vs-per-token sampling sync measured 19.57 vs 19.56 tok/s — no
  speed claimed for it.
- **confidence**: high (exact ×3; A/B reproducible).
- **applicability limits**: bridge (pipeline) numbers; TP2 reuses the same
  Triton path but its ranking must be re-profiled there (see D3).

## D3 — TP2 decode is host-launch-bound; CUDA graphs are the structural fix

- **cue**: TP2 clean wall well above profiler GPU-busy time.
- **lesson**: when host gap ≈13 ms/token, kernel micro-optimisation is
  absorbed; capture the static decode step in a CUDA graph instead.
  Prerequisite: static state with fixed addresses (ch 25).
- **evidence**: `benchmarks/profile/SUMMARY_tp2.md` +
  `benchmarks/profile/tp2_profile.json`: clean 44.37 ms/token vs GPU busy
  31.28 ms/token (~13.1 ms host gap); 128 all-reduces/token = 1.28 ms (no
  redundant collective). Paired A/B in the same profile: fused RMSNorm
  53.68→51.64 ms shipped as the single eager fix (TP2 v2 41.8–42.8 ms,
  `benchmarks/baseline/ours_tp2_v2.json`). Graph result:
  `benchmarks/baseline/ours_tp2_graph.json` 34.6 tok/s, TPOT 28.9 ms,
  32/32 exact ×3.
- **confidence**: high (paired in-process A/B; graph artifact exact ×3).
- **applicability limits**: graph capture needs the static-state contract;
  tile conclusions drawn pre-graph do not transfer (see D4).

## D4 — Retune FP8 tiles after graphs, not before

- **cue**: graphed decode where the Triton FP8 kernel is again top cost.
- **lesson**: tile rankings invert once host launch is removed — re-sweep
  BLOCK_N at M=1 under the graph (BLOCK_N=32 won) instead of trusting the
  eager-era tile choice. The same retune pre-graph measured ~0 wall gain.
- **evidence**: `benchmarks/baseline/ours_tp2_tile_ab.json` (ABBA):
  old 28.88 ms → new 24.48 ms/token (−15.2%), ids identical, verdict
  "ship new". Pre-graph paired row in `benchmarks/profile/SUMMARY_tp2.md`:
  BLOCK_N=32/2-warp tile 48.48→48.80 ms (~0, kernel gain absorbed).
- **confidence**: high (ABBA cross-process + in-profile paired row; exact).
- **applicability limits**: M=1 decode bucket on this Triton kernel/box;
  prefill M-buckets need their own sweep (prefill library).

## D5 — Fused gated-delta is the current best decode mode

- **cue**: graphed, tile-retuned decode with delta-rule elementwise cost left.
- **lesson**: fuse the gated-delta step (`--delta fused`, the default);
  it stacks on top of D3+D4 rather than replacing them.
- **evidence**: `benchmarks/baseline/ours_tp2_graph_v2.json`: TPOT
  23.67 ms, 42.25 tok/s, 32/32 exact ×3 (per-probe 23.668/23.673/23.668 ms);
  WIN.md ladder: 24.48 → 23.67 ms from this lever alone.
- **confidence**: high (exact ×3 across three probes).
- **applicability limits**: nonspec graphed decode at B=1; does not make
  speculative modes faster (speculative library) and does not close the
  gap to Arm A (20.3 ms) — no win claimed.

## D6 — Scale-cache helps on the graph path only (not yet default)

- **cue**: graphed decode re-measured with `--scale-cache on` as a
  single-variable candidate against shipped defaults.
- **lesson**: do not flip the shipped `--scale-cache` default on this
  evidence alone — the graph-path gain contradicts the eager-path paired
  result, so treat scale-cache as decode-mode-specific and require a
  confirming multi-probe round before changing the default.
- **evidence**: `benchmarks/harness/decode_round1.json` (Harness² round 1,
  ABBA 5 runs each): C1 `--scale-cache on` median TPOT 22.866 ms /
  43.73 tok/s vs C0 base 23.726 ms / 42.15 tok/s (−3.6%), 32/32 exact on
  every run; selection reproduced by
  `scripts/opt_harness.py import-metrics` →
  `benchmarks/harness/decode_round1_selection.json`. Prior eager paired
  result `benchmarks/baseline/ours_tp2_microopts_ab.json` found no
  consistent scale-cache winner.
- **confidence**: medium (exact and repeated on the graph path, but
  single prompt/length and contradicted across decode modes).
- **applicability limits**: graph decode, batch 1, capital ×32 only;
  no vLLM win (22.866 ms > Arm A 20.3 ms, different harness regardless).

## D7 — Scale-cache confirmed on graph path; GEMV is the M=1 default

- **cue**: rounds 2-3 confirming/promoting decode candidates.
- **lesson**: `--scale-cache` defaults to `on` only for `--decode-mode
  graph` (eager stays `off` unless explicit); `--fp8-config gemv`
  (M=1 GEMV kernel, M>1 as `new`) is the shipped FP8 default.
- **evidence**: round 2 `benchmarks/harness/decode_round2.json`:
  C1 22.850 vs C0 23.698 ms aggregate (-3.58%), exact on all shapes;
  round 3 `benchmarks/harness/decode_round3.json`: GEMV 22.270 vs
  base 22.820 ms (-2.41%), exact everywhere, worst shape -2.39%.
- **confidence**: high (ABBA, 12 runs/candidate/round, exact).
- **applicability limits**: local TP2 graph runner only; 22.270 ms
  still exceeds Arm A 20.3 ms — no vLLM/suite win claimed.
