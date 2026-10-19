---
id: prefill-ttft
domain: prefill
status: open
current_best:
  result: "prefix-chunk512 bench defaults (round 2): pure-prefill 1745.0/1731.8/1631.8 tok/s @512/2048/8192; prefix 2048+256 cold 1.3460s / warm 0.1562s (gpu storage)"
  artifact: benchmarks/harness/prefill_round2.json
  exactness: "last-argmax equal across chunk sizes (argmax_equal true)"
arm_a:
  result: "effective prompt rate ~2.9-3.4k tok/s (from TTFT); short c1 TTFT p50 0.79 s @2048"
  artifact: benchmarks/baseline/arm_a_exact_greedy_c1_c4.json
  comparable: "rate comparison is context only; TTFT shapes differ per harness"
arm_b:
  result: "short c1 TTFT p50 0.76-0.78 s"
  artifact: benchmarks/baseline/arm_b_mtp_k1_256608.json
  status: MEASURED-WITH-DEVIATION (--max-model-len 256608)
  comparable: "TTFT at same suite shape via run_bench.py only"
objective_metrics:
  - {name: prefill_tok_per_s, direction: maximize, shape: "2048 and 8192 tokens, chunkwise, TP2 local runner"}
  - {name: ttft_p50_s, direction: minimize, shape: "suite shapes via benchmarks/run_bench.py only"}
exactness_gate: "last-argmax equal across all probed chunk sizes; for serving claims, greedy ids exact on short shape + failed == 0"
probe_command: ".venv-gpu/bin/torchrun --nproc_per_node=2 scripts/prefill_bench_tp2.py --lengths 512,2048,8192 --chunks 128,512,1024 --out <artifact>.json"
stop_condition: "prefill rate >= Arm A effective prompt rate at matched length with argmax equality, or two successive k=1 steps adopt no candidate"
candidate_dimensions:
  invariant: ["argmax-equality check across chunk sizes", "ours-vs-ours labelling (never a vLLM TTFT claim from the local runner)"]
  domain_specific: ["chunk size per prompt length (128/512/1024 buckets)", "Triton M-bucket tile configs (TRITON_MAX_M=1024 in v2)", "reference-dequant fallback threshold (M>128 in v1; removed as a lever unless dispatch regresses)"]
---

# Task card: prefill-ttft

Chunkwise prefill already beat token-by-token static prefill by ~57x at
512 tokens (1,316 vs 23 tok/s, `benchmarks/baseline/ours_tp2_prefill_bench.json`);
the open problem is the remaining gap to Arm A's effective prompt rate.
Chunk256 through the reference FP8 path (635 tok/s @512) is the standing
warning: dispatch path matters more than chunk size alone. Prefill round 1
(`benchmarks/harness/prefill_round1.json`) rejected the `prefill-s2`
(`num_stages=2`) candidate — slower at every shape/chunk (-1.3% to -7.5%,
argmax equal) — and its code was removed; the shipped prefill family
(`num_stages=3`, chunk512 best) is unchanged. Prefill round 2
(`benchmarks/harness/prefill_round2.json`, ABBAAB) promoted
`prefix-chunk512` on the bench surface: `scripts/prefix_ttft_bench.py`
defaults are now `--chunk 512 --snapshot-interval 512` (pure-prefill
+20–22% tok/s; prefix cold/warm −12–18%; argmax + reuse exact; base
OOMs at 8192+512 gpu storage, candidate completes). Library defaults
(`prefill_chunkwise_static` chunk 128, `TPBlockPrefixCache` interval
256) are unchanged. Tradeoff: non-512-aligned shared prefixes may
recompute <512 extra tokens. No TTFT/serving win is claimed from the
local prefill runner.
