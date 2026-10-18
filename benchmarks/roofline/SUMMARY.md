# Measured roofline — GPU window (2x RTX 4090, PCIe, no NVLink)

All numbers below were measured in this window with
`.venv-gpu/bin/python benchmarks/microbench_roofline.py ... --i-understand-gpu-window`
(torch 2.11.0+cu128, CUDA 12.8) after the incumbent was stopped via
`/home/nick/llm/vllm/launch.sh --stop`. Raw JSONL:
`bandwidth_gpu0.jsonl`, `bandwidth_gpu1.jsonl`, `gemm_gpu0.jsonl`,
`allreduce.jsonl` in this directory. Derived figures:
`roofline_derived.json`, computed by `benchmarks/roofline_compute.py`
from those files plus the checkpoint header manifest. Settings:
defaults throughout (bandwidth sizes 64/256/1024 MiB, warmup 2,
repeats 5, best-of-repeats CUDA-event timing; GEMM at the script's
27B shapes, M in {1,16,128,1024}; all-reduce 1/16/64/256 MiB via
`torchrun --nproc_per_node=2`, NCCL).

## Achieved HBM bandwidth (copy_, effective = 2x bytes / time)

| GPU | 64 MiB | 256 MiB | 1024 MiB |
|---|---|---|---|
| 0 | 910.6 GB/s | 919.8 GB/s | 919.5 GB/s |
| 1 | 914.4 GB/s | 920.0 GB/s | 919.0 GB/s |

Peak used below: **920.0 GB/s** (GPU1, 256 MiB).

## GEMM on GPU0 (TFLOP/s)

| Shape (M x K x N) | BF16 | FP8 `torch._scaled_mm` |
|---|---|---|
| mlp_gate_up_decode 1x5120x17408 | 0.99 | 1.61 |
| mlp_gate_up_small 16x5120x17408 | 13.02 | 23.84 |
| mlp_gate_up_prefill 128x5120x17408 | 102.21 | 165.05 |
| mlp_gate_up_large 1024x5120x17408 | 149.42 | 320.61 |
| mlp_down_prefill 128x17408x5120 | 94.42 | 162.72 |
| qkv_prefill 128x5120x12288 | 97.69 | 199.89 |
| lm_head_decode 1x5120x248320 | 0.94 | 1.82 |
| lm_head_prefill 128x5120x248320 | 106.21 | 192.29 |

FP8 note: `torch._scaled_mm` was available and ran at every shape
above (recorded in the JSONL, not assumed). The FP8 rows use random
FP8 operands with unit scales; the served model's lm_head is BF16,
so the FP8 lm_head rows price the kernel, not the served layout.
Decode-shaped GEMMs (M=1) are latency/bandwidth-bound, as expected.

## All-reduce, 2 ranks NCCL over PCIe

| Size | Time | Bus BW |
|---|---|---|
| 1 MiB | 178.3 us | 5.88 GB/s |
| 16 MiB | 2459.6 us | 6.82 GB/s |
| 64 MiB | 9861.1 us | 6.81 GB/s |
| 256 MiB | 39437.4 us | 6.81 GB/s |

Least-squares fit (seconds = latency + bytes/bus): fixed latency
**8.67 us** per collective, bus **6.81 GB/s**. A batch-1 decode step
issues 128 all-reduces of 20,480 B (BF16 [1,5120] partials, ch. 23),
so fitted collective latency is ~1.11 ms/token — small next to the
~19 ms weight-read floor below, but not free.

## Ridge points and batch-1 decode ceilings

Byte model (FP8-resident, language+MTP only; see
`benchmarks/roofline_compute.py`): FP8 payload 24,699,207,680 B
(407 tensors, manifest headers) + BF16 language 5,242,980,352 B +
scale_inv 3,015,040 B. TP2 per rank: 17,570,642,112 B/step
(FP8 shard 12,349,603,840 + BF16 replicated 5,195,784,192 + BF16
sharded 23,592,960 + scales 1,661,120). Single-device resident
total 29,945,203,072 B exceeds one 24 GiB card, so that column is
hypothetical. KV read at S=2048 (16 full layers, FP8 KV):
67,108,864 B/token, included below. FLOPs/token 52.61 GFLOP
(2*params on FP8 weights + BF16 lm_head, plus attention term).

| Quantity | Value |
|---|---|
| Ridge (BF16 peak 149.42 TFLOP/s) | 162.4 ops/byte |
| Ridge (FP8 peak 320.61 TFLOP/s) | 348.5 ops/byte |
| Decode intensity, TP2 per rank (S=2048) | 2.98 ops/byte — memory-bound |
| Decode ceiling, TP2 per rank | 52.2 tok/s (TPOT floor 19.17 ms) |
| Decode ceiling, single-device hypothetical | 30.7 tok/s (TPOT floor 32.62 ms) |

Context: the incumbent's measured TPOT p50 at c1 was 0.0203–0.0216 s
(`benchmarks/baseline/arm_a_exact_greedy_c1_c4.json`), i.e. within
~6–13% of the 19.17 ms TP2 weight-read floor at S=2048 — before
collective latency (~1.1 ms fitted) and kernel overheads. These are
roofline lower bounds, not candidate measurements; no win is claimed.
