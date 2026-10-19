---
id: decode-tpot
domain: decode
status: open
current_best:
  result: "TPOT 23.67 ms, 42.25 tok/s (graphed TP2 + fused gated-delta, fp8-config new)"
  artifact: benchmarks/baseline/ours_tp2_graph_v2.json
  exactness: "32/32 exact x3 vs bridge and Arm A"
arm_a:
  result: "TPOT p50 20.3 ms, 2745 tok/min (short 2048->512, c1, greedy)"
  artifact: benchmarks/baseline/arm_a_exact_greedy_c1_c4.json
  comparable: true
arm_b:
  result: "4272-4291 tok/min c1 short (~71 tok/s); chunk TPOT not comparable"
  artifact: benchmarks/baseline/arm_b_mtp_k1_256608.json
  status: MEASURED-WITH-DEVIATION (--max-model-len 256608)
  comparable: "tokens/min only; TPOT not comparable"
objective_metrics:
  - {name: tpot_ms, direction: minimize, shape: "capital probe x32, TP2 local runner"}
  - {name: decode_tok_per_s, direction: maximize, shape: "capital probe x32, TP2 local runner"}
exactness_gate: "generated ids identical to bridge + Arm A references in the same artifact (exact_match true, 32/32 x3); logits finite"
probe_command: ".venv-gpu/bin/torchrun --nproc_per_node=2 scripts/run_qwen38_tp2.py --decode-mode graph --delta fused --fp8-config new --probe capital --out <artifact>.json"
stop_condition: "TPOT <= 20.3 ms at 32/32 exact x3 on the local runner (Arm A parity), or two successive k=1 steps adopt no candidate"
candidate_dimensions:
  invariant: ["ABBA pairing discipline", "host-sync accounting (profile-first)", "exactness gate enforcement"]
  domain_specific: ["FP8 tile config (BLOCK_N/BLOCK_M per M bucket)", "fused elementwise residual path (6.32 ms/token elementwise in tp2_graph_profile.json)", "CUDA graph scope (single vs merged graphs)", "lm_head path (vocab-parallel shipped; revisit only with paired evidence)"]
---

# Task card: decode-tpot

Close the per-stream decode gap: best measured 23.67 ms vs Arm A 20.3 ms
(4.50 ms above the 19.17 ms roofline floor — bound only). Profile first:
in `benchmarks/profile/tp2_graph_profile.json` the FP8 kernel is
14.57 ms/token and elementwise/copy/reduce is 6.32 ms/token, so the lever
order is FP8 kernel → elementwise fusion → everything else. Any candidate
that only moves GPU kernel time while a host gap remains is suspect
(the pre-graph profile showed kernel savings absorbed by host pacing).
