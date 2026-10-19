---
id: decode-tpot
domain: decode
status: open
current_best:
  result: "TPOT 22.112 ms, 45.22 tok/s (graphed TP2 + fused gated-delta + scale-cache on (graph default) + fp8-config gemv-bk512 (default))"
  artifact: benchmarks/harness/decode_round4.json
  exactness: "exact everywhere: x32 32/32 vs bridge and Arm A every run; x128 prefix-32 + 128/128 within-round"
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
probe_command: ".venv-gpu/bin/torchrun --nproc_per_node=2 scripts/run_qwen38_tp2.py --decode-mode graph --delta fused --probe capital --out <artifact>.json"
stop_condition: "TPOT <= 20.3 ms at 32/32 exact x3 on the local runner (Arm A parity), or two successive k=1 steps adopt no candidate"
candidate_dimensions:
  invariant: ["ABBA pairing discipline", "host-sync accounting (profile-first)", "exactness gate enforcement"]
  domain_specific: ["FP8 tile config (BLOCK_N/BLOCK_M per M bucket)", "fused elementwise residual path (6.32 ms/token elementwise in tp2_graph_profile.json)", "CUDA graph scope (single vs merged graphs)", "lm_head path (vocab-parallel shipped; revisit only with paired evidence)"]
---

# Task card: decode-tpot

Close the per-stream decode gap: best measured 22.112 ms (round 4, local
runner) vs Arm A 20.3 ms (1.81 ms gap; 2.94 ms above the 19.17 ms roofline
floor — bound only). No suite win: local-runner shapes only. Profile first:
in `benchmarks/profile/tp2_graph_profile_round4.json` (adopted path) the
FP8 GEMV kernel is 13.95 ms/token (13.77 ms with the promoted BK512
config) and elementwise/copy/reduce is ~4.5 ms/token over ~3507 kernels,
so the next lever is elementwise fusion, not further GEMV tuning. Any
candidate that only moves GPU kernel time while a host gap remains is
suspect (the pre-graph profile showed kernel savings absorbed by host
pacing). Remaining gap to Arm A: 1.81 ms; Arm B is separate (no
comparable TPOT).
