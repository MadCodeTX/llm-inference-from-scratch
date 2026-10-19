# EXPERIENCE — batching

## B1 — Batched eager scales aggregate throughput; it does not fix per-seq latency

- **cue**: per-row static state (`position` [B], ch 28) makes B>1 decode
  possible and aggregate tok/min climbs with B.
- **lesson**: report aggregate and per-seq together. Eager step cost is
  ~flat (43.7→47.3 ms from B=1 to B=16), so aggregate scaling is real but
  per-seq speed (~22 tok/s) is unchanged and still loses per-stream; the
  next lever is a graphed batched step, not a bigger eager batch.
- **evidence**: `benchmarks/baseline/ours_tp2_batched_decode.json`
  (eager, 6-token prompts, 64 greedy tokens, ignore-EOS): aggregate
  tok/min B=1 1,366.7 / B=2 2,720.5 / B=4 5,360.2 / B=8 10,745.9 /
  B=16 20,000.6; per-seq 22.78 → 20.83 tok/s; row0 ids identical,
  `b1_row0_eq_single_run: true`. `TPEngineCore` continuous batching has an
  exact staggered-start proof and no throughput measurement.
- **confidence**: high (single measured sweep, row0 identity checked).
- **applicability limits**: local-runner shape only. B=8/B=16 crossing the
  Arm A c4 (8,530) / Arm B c4 (12,113–12,136) tok/min bars numerically is a
  cross-harness, cross-shape coincidence — explicitly not a win
  (`benchmarks/WIN.md`).

## B2 — Graphed batched decode is exact and faster at c1/c4/c8

- **cue**: a graphed batched step (`TPEngineCore` `decode_mode="graph"`)
  proposed against the eager engine at suite-short geometry.
- **lesson**: adopt as the engine default on CUDA (explicit `eager`
  preserved; explicit `graph` still fails closed off CUDA). The gain
  shrinks with batch (+67% / +15% / +5% aggregate), as expected for a
  host-launch-overhead lever.
- **evidence**: `benchmarks/harness/throughput_round1.json` (2048
  filler -> 512 greedy, ignore-EOS, adopted fused defaults): median
  aggregate tok/min graph vs eager c1 2,334 vs 1,397, c4 5,109 vs
  4,462, c8 7,298 vs 6,979; token-identical on all 14 runs;
  captures=1, replays=511 per graph run.
- **confidence**: high (ABBA, exactness-gated, verifier-recomputed).
- **applicability limits**: local engine harness, not `run_bench.py`
  serving; TTFT/TPOT not observable (step indices only); c16 not
  attempted; no vLLM win may be inferred.
