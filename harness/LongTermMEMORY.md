# Long-term memory (c3) — durable measured facts

Only facts with an artifact behind them belong here. Detail lives in
`harness/library/`; this file is the always-on summary.

- Model: `Qwen/Qwen3.8-27B-FP8`, 27B hybrid (48 gated-delta linear layers +
  16 full-attention layers), TP2 over 2× RTX 4090. Bars: Arm A TPOT p50
  20.3 ms (short c1), Arm B ≈71 tok/s c1 short (MEASURED-WITH-DEVIATION,
  `--max-model-len 256608`). Roofline floor 19.17 ms / 52.2 tok/s ceiling —
  a bound, not a result. (`benchmarks/WIN.md`)
- Best measured decode: graphed TP2 + fused gated-delta, TPOT 23.67 ms,
  42.25 tok/s, 32/32 exact ×3 — still above Arm A's 20.3 ms; **no win
  claimed anywhere yet**. (`benchmarks/baseline/ours_tp2_graph_v2.json`)
- Decode was host-launch-bound before CUDA graphs (clean 44.4 ms/token vs
  31.3 ms GPU busy); graphs + the post-graph FP8 tile retune
  (BLOCK_N=32 at M=1: 28.88 → 24.48 ms) are the two biggest decode levers
  so far. (`benchmarks/profile/SUMMARY_tp2.md`,
  `benchmarks/baseline/ours_tp2_tile_ab.json`)
- Native MTP (γ=1) is exact everywhere with acceptance 0.88–0.94 but slower
  than nonspec graphed decode (capital ×128 effective TPOT: eager 38.68 ms,
  graph 32.51 ms, merged graph2 32.90 ms vs 23.67 ms nonspec). The runner
  default stays `--speculative off`. (`benchmarks/baseline/ours_tp2_mtp_matrix.json`,
  `benchmarks/baseline/ours_tp2_mtp_graph2.json`)
- Prefill: chunkwise ≫ token-by-token (chunk128 1,316 tok/s @512 vs static
  23 tok/s); best chunk512 1,731.6 tok/s @2048 — still behind Arm A's
  ~2.9–3.4k prompt tok/s. (`benchmarks/baseline/ours_tp2_prefill_v2.json`)
- Prefix TTFT (2048+256=2304, ours-vs-ours): v2 snapshot interval 256 gives
  GPU cold 1.64 s / warm 0.18 s (n_reused=2048). Arm A's 1.81 s is at 8704
  tokens — a different shape, no TTFT win claimed.
  (`benchmarks/baseline/ours_prefix_ttft_v2.json`)
- Batched eager decode scales aggregate throughput (B=16: 20,001 tok/min)
  at ~flat per-step cost, but per-seq latency still loses to Arm A/B; the
  aggregate crossings of the c4 bars are cross-shape and are **not** wins.
  (`benchmarks/baseline/ours_tp2_batched_decode.json`)
