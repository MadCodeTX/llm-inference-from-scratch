# Bridge decode attribution — 551 ms/token, measured (2026-10-17, both RTX 4090 free)

Source: [profile script](/home/nick/muse/llm-inference-from-scratch/scripts/profile_qwen38_bridge.py) → [bridge_profile.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/profile/bridge_profile.json). Capital probe, prefill + 7 profiled decode tokens: **551.5 ms/token wall (1.81 tok/s)**, generated ids `[11751, 13, 198, 760, 6511, 314, 9564, 369]` — exact prefix match vs recorded vLLM. Profiler (CUDA+CPU, record_shapes) + CUDA-event timers via process-local monkey-patches (restored in `finally`; no engine change). Overlapping categories are marked memo and never summed.

## Ranked costs (per decode token)

| Rank | Cost | ms/token | Share of 551 ms | Notes |
|---|---|---|---|---|
| 1 | Block-FP8 dequant (`dequantize_block_fp8`) | 243 per GPU ×2 segments = 486 summed | ~88% summed; ~44% of latency per segment | fp32 cast + mul + bf16 copy per weight, every token |
| 2 | MLP module total (memo, subset of 1+3) | 386 summed | — | contains its own dequant+linear; not additive |
| 3 | Linear-attn (GatedDelta) total (memo) | 122 summed | — | runner calls `forward_prefill` even for T=1 |
| 4 | FP8 `F.linear` matmul | 25 per GPU ×2 = 50 summed | ~9% | M=1 GEMV, launch-bound |
| 5 | Full-attn total (memo) | 38 summed | — | 16 of 64 layers |
| 6 | Sampling host sync (`.item()` wall, memo) | 85 wall | overlaps queue drain | wait, not extra GPU work |

Profiler rollup corroborates: top CUDA ops are `aten::copy_` 1937 ms, `aten::mul` 1477 ms (dequant elementwise), `aten::mm` only 367 ms over the profiled window; aten names cannot isolate dequant, hence the event-timer split above.

## Pipeline-serial vs dequant vs launch overhead

- Serial split: GPU0 segment ≈ 243 dequant + 25 matmul ≈ 268 ms; GPU1 segment ≈ same. Latency = sum of segments ≈ 536 ms ≈ wall. Each GPU is idle ~49% of every token — the pipeline costs a full 2× serialization vs tensor-parallel, independent of kernels.
- Dequant overhead: micro-bench M=1 `BlockFP8Linear` (20 reps): 5120×17408 = 2.00 ms/call at 223 GB/s effective vs 0.48 ms at 920 GB/s roofline (**4.1× slow**); all four shapes 222–225 GB/s, 4.1× slow. Dequant materialises fp32→bf16 copies, multiplying weight traffic ~5×.
- Launch overhead: 16,394 `copy_` + 11,102 `mul` calls in 7 tokens (~3,900 small kernels/token); `aten::mm` averages ~0.1 ms over 3,479 calls — small-kernel launch, not FLOPs, bounds matmul.

## Predicted decode time at roofline

| Change | Predicted ms/token | tok/s |
|---|---|---|
| Today (measured) | 551 | 1.8 |
| Dequant fused (read FP8 once/GPU, 13.5 GB @ 920 GB/s ×2 serial segments) | ~35–50 | 20–28 |
| + TP2 instead of pipeline (task roofline floor) | 19.2 + overhead | ≤52 |

Next lever is fused FP8 matmul (no materialised dequant), then TP or CUDA graphs for launch overhead. Chrome trace: rerun with `--trace` (not committed).
