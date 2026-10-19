---
id: serving-head-to-head
domain: serving
status: open
current_best:
  result: "serving surface integrated (ch 30); serving-overhead round 1 ABBA: c1 TPOT 113.89 -> 63.98 ms, c4 3036.2 tok/min, 16x4096/c16 feasible 16/16 twice, 8x16384 still infeasible; serving-occupancy round 1 ABBA (8x8192, short): per-occupancy buckets adopted as serving default, c1 63.86 -> 25.77 ms, c4 46.95 ms, c8 flat, hashes exact, peak ~23.3-23.7 GiB/rank; no same-harness suite win yet"
  artifact: docs/30_serving_openai.md
  exactness: "fake/stub contract tests pass; real-27B evidence is a tiny-prompt smoke only, no suite quality-gate run"
arm_a:
  result: "short c1 2745 tok/min / TPOT 20.3 ms; c4 8530 tok/min; agent TTFT p50 1.81 s @8704 (c1)"
  artifact: benchmarks/baseline/arm_a_exact_greedy_c1_c4.json
  comparable: true
arm_b:
  result: "short c1 4272-4291 tok/min; c4 12113-12136 tok/min; acceptance 98.8%"
  artifact: benchmarks/baseline/arm_b_mtp_k1_256608.json
  status: MEASURED-WITH-DEVIATION (--max-model-len 256608)
  comparable: "tokens/min + TTFT; chunk TPOT not comparable"
objective_metrics:
  - {name: tokens_per_minute, direction: maximize, shape: "suite.json shapes, c1 and c4, run_bench.py"}
  - {name: tpot_p50_s, direction: minimize, shape: "same run_bench.py run (Arm B chunk TPOT excluded)"}
  - {name: ttft_p50_s, direction: minimize, shape: "agent-shared-prefix and cold long, same run"}
exactness_gate: "WIN.md quality gate: greedy short-shape ids token-identical to incumbent (or bounded+explained), failed == 0, completion_tokens == requested budget"
probe_command: ".venv/bin/python benchmarks/run_bench.py --base-url http://127.0.0.1:<candidate-port> --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <snapshot> --output <artifact>.json"
stop_condition: "WIN.md thresholds met (see below), or the candidate fails the quality gate twice"
candidate_dimensions:
  invariant: ["same suite.json, same harness, same box for every compared run", "WIN.md win thresholds applied literally"]
  domain_specific: ["candidate server wrapping of the TP2 engine (TPEngineCore continuous batching)", "concurrency 1-4 short/medium focus (threshold 1 scope)"]
win_thresholds:
  - ">=1.25x incumbent tokens_per_minute AND >=20% lower p50 TPOT (c1-c4, short/medium)"
  - ">=30% lower TTFT p50 on agent-shared-prefix"
  - ">=15% lower TTFT p50 on cold long"
---

# Task card: serving-head-to-head

This is the only card whose numbers may be called a win, and only via
`benchmarks/run_bench.py` on `benchmarks/suite.json` inside an approved
GPU window. Everything on the other cards is a local-runner contrast
until it is re-measured here. Thresholds are copied from
`benchmarks/WIN.md`; do not soften them on this card.
