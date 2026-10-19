---
id: throughput-batching
domain: batching
status: open
current_best:
  result: "graphed batched decode (engine default on CUDA): median aggregate tok/min c1 2334, c4 5109, c8 7298 vs eager 1397/4462/6979 at suite-short 2048->512 (local TPEngineCore harness); eager B sweep retained below as context"
  artifact: benchmarks/harness/throughput_round1.json
  exactness: "graph vs eager token-identical on all 14 runs c1/c4/c8; captures=1, replays=511 per graph run"
arm_a:
  result: "c4 short 8530 tok/min (run_bench.py serving harness)"
  artifact: benchmarks/baseline/arm_a_exact_greedy_c1_c4.json
  comparable: false
  note: "different harness/shape (local 6-token prompts, eager, ignore-EOS); aggregate crossings are not wins"
arm_b:
  result: "c4 short 12113-12136 tok/min (run_bench.py serving harness)"
  artifact: benchmarks/baseline/arm_b_mtp_k1_256608_c4.json
  status: MEASURED-WITH-DEVIATION (--max-model-len 256608)
  comparable: false
objective_metrics:
  - {name: aggregate_tok_per_min, direction: maximize, shape: "local batched runner, B in 1,2,4,8,16, 64 greedy tokens"}
  - {name: per_seq_tok_per_s, direction: maximize, shape: "same run; guards against aggregate-only wins"}
  - {name: step_ms, direction: minimize, shape: "decode-loop seconds / 63 steps"}
exactness_gate: "row0 generated ids identical at every batch size; staggered-start exactness via TPEngineCore proof for scheduling candidates"
probe_command: ".venv-gpu/bin/torchrun --nproc_per_node=2 scripts/batched_decode_bench.py --batches 1,2,4,8,16 --decode-tokens 64 --out <artifact>.json"
stop_condition: "met for c1/c4/c8 (graph exact and faster, adopted as CUDA default); reopen for c16 or a per-seq regression"
candidate_dimensions:
  invariant: ["row0-identity check at every B", "cross-harness non-claim discipline vs Arm A/B c4 bars"]
  domain_specific: ["CUDA-graphed batched decode (eager is the measured caveat: ~44 ms/step at every B)", "per-row position handling (ch 28: position is [B])", "continuous-batching scheduling in TPEngineCore (exactness shown; throughput unmeasured)"]
---

# Task card: throughput-batching

Aggregate throughput already scales near-linearly in eager mode because
the step cost barely moves (43.7 ms @B=1 → 47.3 ms @B=16). The open
problem is that eager per-seq latency (~22 tok/s) still loses to Arm A
per-stream, and only a graphed batched step can change that. Never report
the B=8/B=16 aggregate crossings of the Arm c4 bars as wins.
