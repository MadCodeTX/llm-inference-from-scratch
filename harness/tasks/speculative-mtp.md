---
id: speculative-mtp
domain: speculative
status: open
current_best:
  result: "exact everywhere, slower than nonspec: capital x128 effective TPOT eager 38.68 ms, graph 32.51 ms, graph2 32.90 ms vs nonspec 23.67-24.49 ms; acceptance 0.88-0.94"
  artifact: benchmarks/baseline/ours_tp2_mtp_matrix.json
  exactness: "32/32 x3 vs bridge+Arm A; capital 128/128 vs graph-nonspec reference"
arm_a:
  result: "TPOT p50 20.3 ms nonspec reference (the bar speculative must beat via nonspec first)"
  artifact: benchmarks/baseline/arm_a_exact_greedy_c1_c4.json
  comparable: true
arm_b:
  result: "vLLM+MTP K=1: 4272-4291 tok/min c1 short, spec acceptance 98.8% (3574/3617)"
  artifact: benchmarks/baseline/arm_b_mtp_k1_256608.json
  status: MEASURED-WITH-DEVIATION (--max-model-len 256608)
  comparable: "acceptance + tokens/min context; chunk TPOT not comparable"
objective_metrics:
  - {name: tpot_effective_ms, direction: minimize, shape: "capital x128, TP2 local runner, vs same-run graph-nonspec"}
  - {name: acceptance, direction: maximize, shape: "proposed/accepted over the speculative loop; secondary to TPOT"}
exactness_gate: "128/128 vs graph-nonspec reference ids on capital; 32/32 x3 vs bridge+Arm A; runner default stays --speculative off until the gate + TPOT both pass"
probe_command: ".venv-gpu/bin/torchrun --nproc_per_node=2 scripts/run_qwen38_tp2.py --decode-mode graph --speculative mtp1-graph --probe capital --out <artifact>.json"
stop_condition: "effective TPOT < 23.67 ms (current nonspec best) at 128/128 exact, or two successive k=1 steps adopt no candidate; acceptance gains without TPOT gains do not stop the card"
candidate_dimensions:
  invariant: ["effective-TPOT (not acceptance) as the adoption metric", "same-run nonspec reference for 128-token exactness"]
  domain_specific: ["verify-chunk cost at M=2 (34.6 ms graphed vs 24.5 ms M=1 step is the known blocker)", "merged single-graph cycle (graph2: 1 sync/cycle, measured 32.90 ms - kept, not default)", "draft cost (5.8 ms/cycle profile) and snapshot/rollback overhead"]
---

# Task card: speculative-mtp

High acceptance is not speed. At acceptance 0.92 the expected 1.92
tokens/cycle should give ~23 ms effective TPOT, but the measured M=2
verify chunk (34.6 ms) plus host/capture overhead lands at 32.51 ms.
Attack the verify chunk and the per-cycle overhead, not the draft's
acceptance — and keep `--speculative off` as the shipped default until
effective TPOT actually beats nonspec.
