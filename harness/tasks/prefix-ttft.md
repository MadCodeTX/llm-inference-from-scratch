---
id: prefix-ttft
domain: prefix
status: open
current_best:
  result: "GPU cold 1.64 s / warm 0.18 s (CPU cold 1.93 s / warm 0.25 s), 2048+256=2304, interval 256, n_reused=2048"
  artifact: benchmarks/baseline/ours_prefix_ttft_v2.json
  exactness: "last argmax cold == warm == 12356"
arm_a:
  result: "agent-shared-prefix TTFT p50 1.81 s @8704 tokens (c1)"
  artifact: benchmarks/baseline/arm_a_exact_greedy_c1_c4.json
  comparable: false
  note: "different shape (8704 vs 2304); no TTFT win claimed from v2 numbers"
arm_b:
  result: "agent-shared-prefix TTFT p50 3.33 s cold / 0.90 s warm"
  artifact: benchmarks/baseline/arm_b_mtp_k1_256608.json
  status: MEASURED-WITH-DEVIATION (--max-model-len 256608)
  comparable: false
objective_metrics:
  - {name: warm_ttft_s, direction: minimize, shape: "2048 shared + 256 unique = 2304, ours-vs-ours"}
  - {name: cold_ttft_s, direction: minimize, shape: "same shape, ours-vs-ours"}
  - {name: n_reused, direction: maximize, shape: "must stay 2048; a warm win with less reuse is not a win"}
exactness_gate: "last argmax cold == warm; n_reused == 2048 on warm reps"
probe_command: ".venv-gpu/bin/torchrun --nproc_per_node=2 scripts/prefix_ttft_bench.py --shared 2048 --unique 256 --snapshot-interval 256 --storage both --out <artifact>.json"
stop_condition: "warm TTFT improves >=20% at n_reused=2048 with argmax equality, or two successive k=1 steps adopt no candidate"
candidate_dimensions:
  invariant: ["cold==warm argmax equality", "cross-shape non-claim discipline vs Arm A/B TTFT"]
  domain_specific: ["snapshot_interval_tokens (256 current; must stay a multiple of block_size 16)", "snapshot storage cpu vs gpu", "block_size / hash-chain granularity (tp_prefix_cache.py)"]
---

# Task card: prefix-ttft

v1 (10.66 s cold / 1.31 s warm) to v2 (GPU 1.64 s / 0.18 s) came from the
snapshot-interval fix plus the prefill-v2 dispatch; treat further gains as
suspect until `n_reused` and argmax equality are re-verified — a faster
"warm" run that reuses less is a regression wearing a costume.
