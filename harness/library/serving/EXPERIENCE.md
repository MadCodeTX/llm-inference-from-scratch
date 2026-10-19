# EXPERIENCE — serving

## V1 — Cross-shape / cross-harness numbers are context, never wins

- **cue**: a local-runner number (different prompt shape, batch, or
  harness) that numerically passes an Arm A / Arm B serving bar.
- **lesson**: the only win is `benchmarks/run_bench.py` on
  `benchmarks/suite.json`, same shapes, same box, quality gate passed
  (`benchmarks/WIN.md` thresholds). Everything else is labelled
  ours-vs-ours or context-only at the moment it is written down.
- **evidence**: `benchmarks/WIN.md` throughput ladder: ours eager B=8
  10,746 / B=16 20,001 tok/min vs Arm A c4 8,530 / Arm B c4 12,113–12,136
  tok/min — recorded there as "passes numerically only", verdict "no win
  claimed"; per-stream decode (23.67 vs 20.3 ms), prefill rate, and prefix
  TTFT all recorded as losses or non-comparable in the same file.
- **confidence**: high (this is the repo's own recorded verdict, applied
  consistently across every ladder row).
- **applicability limits**: none — this rule binds every domain's
  reporting, including future serving candidates.

## V2 — Serving surface ships before any serving win exists

- **cue**: an OpenAI-compatible server wrapping the TP engine is
  ready to be measured, but no same-harness suite run exists.
- **lesson**: ship the surface, state the blockers, claim nothing.
  Driver exceptions must fail the request (never hang the consumer),
  `ignore_eos` must reach the engine (a server-only flag cannot stop
  engine EOS), and TP shutdown must broadcast + join the driver
  before context teardown. All three are covered by committed
  stub-engine tests in `tests/test_openai_server.py`.
- **evidence**: `docs/30_serving_openai.md`; real-27B evidence is a
  tiny-prompt smoke only (TPOT p50 ~88 ms including graph capture
  and per-step `broadcast_object_list` overhead). Blockers: TP
  broadcast overhead, static-KV sizing (8x16384 defaults vs suite
  medium/long/c16 on 24 GiB/rank), greedy-only sampled arm.
- **confidence**: high for the defect fixes (probed + tested);
  none for serving speed (unmeasured).
- **applicability limits**: until a `run_bench.py` suite run passes
  the WIN.md quality gate, every serving-speed statement is a
  blocker statement, not a result.

## V3 — Poll only when idle; snapshot only live rows

- **cue**: serving TPOT ~2x the in-process engine step at the same
  geometry, and c16 collapses into HTTP 500s after the first batch.
- **lesson**: block the driver on the pending queue only when no
  job is active (a 50 ms poll quantum adds directly to every token);
  eager steps snapshot only live non-planned rows and reset
  stateless rows instead of cloning full state; graph capture
  warmup runs on the real state and its sizing guard fails closed
  to eager on any oracle/agreement/capture failure.
- **evidence**: `benchmarks/harness/serving_overhead_round1.json`
  ABBA: c1 TPOT p50 113.89 -> 63.98 ms, c4 1868.8 -> 3036.2
  tok/min, greedy hashes identical; 16x4096 c16 16/16 twice
  (~7452 tok/min) vs shipped 1/16 + 15x HTTP 500; 8x16384 still
  infeasible. Diagnosis: `serving_overhead_diagnosis.md`.
- **confidence**: high (ABBA + unit tests + fake-engine additive
  proof of the 50 ms quantum).
- **applicability limits**: remaining c1 gap is graphed B=8
  geometry (per-occupancy graphs unimplemented); numbers are
  short-shape `run_bench.py`, not a suite win vs vLLM.
