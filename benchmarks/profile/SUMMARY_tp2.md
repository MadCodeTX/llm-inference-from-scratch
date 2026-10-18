# TP2 decode profile — measured

Source: [profile script](/home/nick/muse/llm-inference-from-scratch/scripts/profile_qwen38_tp2.py) → [tp2_profile.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/profile/tp2_profile.json). Probe `capital`, 4 warmup + 16 clean + 16 instrumented decode tokens, both RTX 4090, TP2 resident FP8 (Triton). Generated ids cross-rank equal: True; vs pipeline bridge record: 32/32 exact (prefix compared over the recorded 32; profile generates 37).

Headline: clean wall **44.37 ms/token (22.54 tok/s)**; instrumented wall 89.07 ms/token (event recording perturbs it). Roofline TP2 floor 19.17 ms/token → clean is 2.31× the floor. Pipeline bridge (fastfp8) is 50.8 ms/token and the old dequant bridge profile was 551.5 ms/token; vLLM Arm A c1 TPOT is ~20.3 ms.

## Ranked costs (per decode token, rank 0 events; profiler for collectives)

| Rank | Cost | ms/token | Notes |
|---|---|---|---|
| 1 | Decode step total (event, rank 0) | 89.068 | whole model.step + argmax, event-timed |
| 2 | Gated-delta step (48 layers, memo) | 35.648 | includes its FP8 linears + 1 all-reduce per layer |
| 3 | Triton FP8 linears (subset of mixer+MLP) | 34.073 | subset of the mixer/MLP rows; never added to them |
| 4 | MLP (64 layers, memo) | 18.866 | includes its FP8 linears + 1 all-reduce per layer |
| 5 | Full attention (16 layers, memo) | 17.331 | includes its FP8 linears + 1 all-reduce per layer |
| 6 | Norms (block + final, memo) | 12.831 | QwenRMSNorm only; gated norm sits inside gated-delta |
| 7 | Collectives (NCCL, profiler rank 0) | 1.279 | 128 all-reduces/token counted on rank 0 |

Profiler (rank 0): GPU busy 31.28 ms/token, host gap (profiled wall − GPU busy) **57.79 ms/token**. Collective count 2048 over 16 tokens (128.0/token) on rank 0; FP8 linear calls 400.0/token.

## Reading

Overlapping categories are marked memo and never summed. The non-overlapping check is: decode-step event total vs clean wall vs GPU busy. Next lever is whichever of FP8 linears / attention / gated-delta / collectives / host gap dominates above; see the fix section appended after the A/B run.

## Diagnosis: host-launch-bound decode

The non-overlapping split is clean wall 44.37 ms/token vs profiler GPU busy 31.28 ms/token (kernels only, rank 0): a ~13.1 ms/token clean-run host gap, and the instrumented run's larger gap (57.8 ms) is event-recording overhead on top of it. Kernel-time ranking (rank 0, per token): Triton FP8 GEMM 18.89 ms (400 calls), lm_head GEMV 2.65 ms (1 call, shape [1,5120]x[5120,248320]), small elementwise/copy/reduce ops ~7 ms in aggregate, NCCL all-reduce 1.28 ms. Collectives are exactly the designed 128 all-reduces/token (10.0 us each, matching the fitted 10.2 us model) with no all-gather in decode — there is no redundant collective to remove. Because the host paces the stream, GPU-kernel savings alone do not move wall time (proven below), so the fix targets host launches.

## Candidate fixes, paired interleaved A/B (same process, alternating 16-token segments)

| Candidate | Paired wall (ms/token) | Delta | Verdict |
|---|---|---|---|
| Fused RMSNorm (F.rms_norm, all QwenRMSNorm) | 53.68 -> 51.64 | -2.04 | **shipped** |
| Vocab-parallel lm_head (shard + all-gather logits) | 49.25 -> 48.21 | -1.04 | measured, not shipped |
| Cached FP32 FP8 scale (skip per-call cast) | 48.73 -> 47.96 | -0.78 | measured, not shipped |
| Triton M<=8 tile BLOCK_N=32/2 warps | 48.48 -> 48.80 | ~0 (kernel 18.89 -> 15.09 absorbed) | rejected: no wall gain |

Absolute values drift with GPU clocks between processes; each row is paired within one process. Generated ids were identical under every candidate.

## Fix shipped: fused RMSNorm (single fix)

`QwenRMSNorm.use_fused_rmsnorm` (opt-in, default off; dense/pipeline paths unchanged) collapses the 9-op manual norm into one `F.rms_norm` kernel in FP32 with the same (1 + weight) gain. The TP2 runner enables it with `--norm fused` (default; `--norm manual` is the A/B baseline): 161 norm calls/token drop from ~29 us to ~12 us of launch-bound time each in microbench. Runner A/B from JSON artifacts (32-token probes, back-to-back processes):

| Run | Artifact | TPOT (ms) | tok/s | Exact vs bridge |
|---|---|---|---|---|
| TP2 v1 (manual norms) | benchmarks/baseline/ours_tp2_first_run.json | 43.72 / 42.46 / 42.82 | 22.87 / 23.55 / 23.35 | 32/32 x3 |
| Manual (fresh A/B baseline) | /tmp/ours_tp2_manual_ab.json (this session) | 45.42 / 44.37 / 44.24 | 22.02 / 22.54 / 22.61 | 32/32 x3 |
| Fused (TP2 v2) | benchmarks/baseline/ours_tp2_v2.json | 42.81 / 41.89 / 41.77 | 23.36 / 23.87 / 23.94 | 32/32 x3 |

Generated ids are identical across manual, fused and v1 on all 3 probes, also 32/32 vs vLLM Arm A with finite logits: ~2.5 ms/token (~5.6%) from the one fix. Remaining gap: ~42 ms/token is still 2.2x the 19.17 ms roofline floor; the FP8 kernel (~15-19 ms) plus ~30 ms GPU busy and the residual host path are next, with CUDA-graph capture the structural lever (deliberately not built here).
