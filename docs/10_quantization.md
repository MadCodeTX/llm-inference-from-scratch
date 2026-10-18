# 10 — Quantization: fewer bits per number

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 5.1. **Production question:** how few bits per weight can be stored before quality drops, and how much decode speed and memory does that buy on memory-bound hardware? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Quantization stores weights (and sometimes the KV cache) in fewer bits — fp32→fp16→int8→int4 — because decode is memory-bound, so halving the bytes read per token roughly halves time per token, at a controlled accuracy cost.

## Why inference needs this

Chapter 00 established that decode streams every weight from memory to produce one token. Token latency is therefore approximately `weight_bytes / memory_bandwidth`. You cannot easily raise bandwidth, but you can shrink the numerator: 27 billion parameters occupy 54 GB in fp16 and 13.5 GB in int4 (arithmetic below — those are round *assumed* parameter counts for formula practice, not a specific model's config). Quantization is the single highest-leverage decode optimization on bandwidth-starved hardware, which is why on-device (Glimmer-style) deployments are quantized essentially by definition.

The cost is error: a quantized weight `ŵ` differs from `w`, and every downstream activation inherits a perturbation. The craft is choosing granularity (per-tensor vs per-channel scales) so the error stays where the model tolerates it.

## The math/shapes

**Bytes.** `bytes_for_params(n, bits) = ceil(n * bits / 8)`. Sub-byte widths pack: an odd int4 count rounds up to a whole byte.

| Format | Bits | Bytes for 1e9 params | Bytes for 27e9 params (assumed count) |
|--------|------|----------------------|----------------------------------------|
| fp32 | 32 | 4.0 GB | 108 GB |
| fp16/bf16 | 16 | 2.0 GB | 54 GB |
| int8 | 8 | 1.0 GB | 27 GB |
| int4 | 4 | 0.5 GB | 13.5 GB |

**Per-channel int8 (the dissected scheme).** For a weight matrix `W [out, in]`, each **output row** gets its own scale:

```text
scale_i = max|W[i, :]| / 127
q[i, j] = round(W[i, j] / scale_i)        (int8, in [-127, 127])
ŵ[i, j] = q[i, j] * scale_i
```

Per-element error is at most `scale_i / 2` — half a quantization step — because rounding moves each value by at most half a step. Per-row scales matter: one outlier row must not coarsen the grid for every other row, as a single per-tensor scale would.

**Concrete example.** Row `w = [0.5, -1.0, 0.25]`. `max|w| = 1.0`, so `scale = 1/127 ≈ 0.007874`. Quantized: `q = [round(63.5), -127, round(31.75)] = [64, -127, 32]`. Dequantized: `[0.5039, -1.0, 0.2520]`. Errors: `[0.0039, 0, 0.0020]`, each ≤ `scale/2 ≈ 0.00394`. Note the largest element is *exact* by construction; the grid is pinned to it. An all-zero row is the edge case: `scale` would be 0, so it is pinned to 1.0 — `q` is all zeros and dequantizes back to exact zeros regardless.

**Error metrics.** `quantization_error(w, ŵ)` reports `max_abs` (worst element), `mean_abs`, and `rmse` (root-mean-square error — the typical perturbation scale an activation would feel).

**KV cache in fewer bits.** The chapter-05 formula with a smaller `bytes_per_elem`: `2 * L * B * H_kv * T * head_dim * bytes_per_elem`. KV int8 halves the cache versus fp16 with usually-small quality cost; the scales bookkeeping is the same per-channel idea along the token/head axes in production kernels.

## Walk the code

Everything is in [quantization.py](../src/inference_from_scratch/quantization.py).

- `quantize_int8_per_channel(w) -> (q, scales)` (`quantization.py:117`): computes row-wise `max_abs / 127` in float32, pins zero-row scales to 1.0 (`quantization.py:146`), and returns `q` int8 with the weight's shape plus `scales` fp32 of shape `[out_features]`. Guarantees per-element roundtrip error ≤ `scale_i / 2` (plus float rounding).
- `dequantize_int8(q, scales)` (`quantization.py:151`): returns `q.float() * scales[:, None]`, validating that `scales` has exactly one entry per row.
- `quantization_error(w, w2)` (`quantization.py:173`): returns `{"max_abs", "mean_abs", "rmse"}` as plain floats; identical inputs give all zeros.
- `bytes_for_params(n_params, bits)` (`quantization.py:199`): `(n * bits + 7) // 8`, raising `ValueError` on `bits <= 0`. `bytes_for_dtype(n, dtype)` (`quantization.py:214`) maps `fp32/fp16/bf16/int8/int4` through the `BITS_PER_PARAM` table.
- `kv_cache_bytes(config, seq_len, batch, bytes_per_elem)` (`quantization.py:226`): the chapter-05 formula read off a `ModelConfig` — `2 * n_layer * batch * n_kv_head * seq_len * head_dim * bytes_per_elem`.
- `worked_example()` (`quantization.py:261`): computes the table above live from the formulas using **explicitly assumed** round dims (27e9 params; L=64, d=5120, H=40, H_kv=8) — labelled in its docstring as *not* any real model's config. Real numbers come from `weights.load_hf_config` on a real `config.json`.

Tests: [test_quantization.py](../tests/test_quantization.py) checks the half-scale error bound, zero-tensor/zero-row exactness, scale shapes, bytes math including int4 packing, and KV-bytes values.

## Block FP8 anatomy (real Qwen3.8-27B-FP8 checkpoint)

The real FP8 checkpoint does not use per-channel int8. Each large Linear weight is stored as `float8_e4m3fn` (safetensors dtype `F8_E4M3`, 1 byte/param) plus a bf16 `weight_scale_inv` grid with one scale per 128×128 block: `scale_inv.shape == [ceil(R/128), ceil(C/128)]` for a weight `[R, C]`. Dequantization *multiplies* by `scale_inv` despite the name — verified on `layers.3.self_attn.k_proj.weight` `[1024, 5120]` with `weight_scale_inv` `[8, 40]`: every FP8 block saturates at absmax 448, the dequantized tensor is finite with absmax 0.210 and std 0.0159, and each block's dequantized absmax matches `scale_inv * 448` to 0.3% (multiplying is correct; dividing would give absmax ~3.4e6).

| Real tensor (layer 0/3) | FP8 weight shape | `weight_scale_inv` shape |
|---|---|---|
| `linear_attn.in_proj_qkv.weight` | `[10240, 5120]` | `[80, 40]` |
| `linear_attn.in_proj_z.weight` | `[6144, 5120]` | `[48, 40]` |
| `linear_attn.out_proj.weight` | `[5120, 6144]` | `[40, 48]` |
| `mlp.gate_proj.weight` / `up_proj.weight` | `[17408, 5120]` | `[136, 40]` |
| `mlp.down_proj.weight` | `[5120, 17408]` | `[40, 136]` |
| `self_attn.k_proj.weight` (layer 3) | `[1024, 5120]` | `[8, 40]` |

Code: `dequantize_block_fp8(w_fp8, scale_inv, block=(128,128), out_dtype=bf16)` (reshape-multiply, edges padded then cropped), `quantize_block_fp8(w, block)` (per-block `amax/448` stored as `scale_inv`), and `fp8_block_matmul(x, w_fp8, scale_inv)` — a CPU reference (`x @ dequant.T`). Tests: [test_fp8_block.py](../tests/test_fp8_block.py). Small tensors (norms, `A_log`, `dt_bias`, conv1d) stay bf16 in the checkpoint; only the large projections are FP8.

## Fast path: Triton fused dequant matmul (measured)

**`torch._scaled_mm` cannot express the checkpoint layout exactly** (probe: [probe_scaled_mm.py](../scripts/probe_scaled_mm.py), torch 2.11.0+cu128, RTX 4090 SM 8.9). Its 128×128 BlockWise mode requires float32 scales in a near-inner-dim-major, 16-byte-aligned blocked layout; plain, float32-cast, transposed, and padded variants of the checkpoint's contiguous BF16 `[ceil(N/128), ceil(K/128)]` grid all raise `Invalid scaling configuration` at 128- and 512-sized shapes. It would also require FP8 activations with matching block scales — dynamically quantising BF16 activations is lossy, so results would differ from the BF16-activation reference by activation FP8 rounding, not just weight rounding. No `_scaled_mm` path is implemented.

The fast path is therefore [fp8_triton.py](../src/inference_from_scratch/fp8_triton.py): each Triton program streams FP8 tiles (`BLOCK_K=128` = one scale column, `BLOCK_N` divides 128 = one scale row), applies the block scale in-kernel, converts to BF16 (mirroring reference rounding), and accumulates in FP32 via `tl.dot`. Activations stay BF16. Autotune is off; `choose_config` picks fixed tiles by M. Enable via `BlockFP8Linear(..., use_triton=True)`, `from_linear(..., use_triton=True)`, or `INFERENCE_FP8_TRITON=1`. Dispatch uses Triton only for CUDA inputs with flattened `1 <= M <= 128` (`TRITON_MAX_M`, chosen from the bench below: Triton won at every measured M); any import/launch failure is logged with `falling back to reference dequant path` and recorded on `_triton_fallback_reason` — never silent.

Measured with [fp8_linear_bench.py](../benchmarks/fp8_linear_bench.py) (`.venv-gpu`, RTX 4090, 20 reps after 5 warmup; effective GB/s = FP8 weight + scale + activation bytes / time, so repeated-rep L2 reuse can push small shapes above DRAM roofline; reference GB/s on the same byte count understates its real traffic, which also materialises FP32→BF16 dequant copies):

| Shape (N×K) | M | Reference | Triton | Speedup | Max rel diff |
|---|---|---|---|---|---|
| gate_up 17408×5120 | 1 | 2.000 ms / 44.6 GB/s | 0.132 ms / 674 GB/s | 15.1× | 1.9e-3 |
| gate_up 17408×5120 | 4 | 2.049 ms / 43.6 GB/s | 0.133 ms / 672 GB/s | 15.4× | 1.8e-3 |
| gate_up 17408×5120 | 16 | 2.051 ms / 43.8 GB/s | 0.135 ms / 664 GB/s | 15.2× | 3.3e-3 |
| gate_up 17408×5120 | 128 | 2.077 ms / 45.7 GB/s | 0.258 ms / 368 GB/s | 8.0× | 7.0e-3 |
| down 5120×17408 | 1 | 2.004 ms / 44.5 GB/s | 0.136 ms / 657 GB/s | 14.8× | 2.6e-3 |
| down 5120×17408 | 128 | 2.079 ms / 45.6 GB/s | 0.346 ms / 274 GB/s | 6.0× | 8.0e-3 |
| qkv 10240×5120 | 1 | 1.162 ms / 45.1 GB/s | 0.043 ms / 1216 GB/s | 26.9× | 2.4e-3 |
| qkv 10240×5120 | 128 | 1.212 ms / 46.5 GB/s | 0.155 ms / 363 GB/s | 7.8× | 3.6e-3 |

Correctness: Triton vs dequant reference rel diff ≤ 8e-3 across all benched shapes (BF16 dot-order rounding). GPU tests: [test_fp8_triton_gpu.py](../tests/test_fp8_triton_gpu.py) (skip-gated); CPU dispatch/fallback tests: [test_fp8_triton_dispatch.py](../tests/test_fp8_triton_dispatch.py).

## Tile retune after CUDA graphs (measured)

Once CUDA graphs removed the host-launch gap (TP2 graph TPOT 28.89 ms vs eager 42 ms; docs/25), kernel tiles matter again. [fp8_tile_sweep.py](../benchmarks/fp8_tile_sweep.py) sweeps `BLOCK_N {32,64,128} x num_warps {4,8} x BLOCK_M {16,32}` at M=1 on the stored orientations, CUDA-event median of interleaved rounds, raw rows in [fp8_tile_sweep.json](../benchmarks/profile/fp8_tile_sweep.json). Legacy default is `BLOCK_M=16, BLOCK_N=128, num_warps=4`; the winner on every primary shape (and, in the repeat run, both TP2 shard shapes `8704x5120` / `5120x8704`) was `BLOCK_M=16, BLOCK_N=32, num_warps=4`:

| Shape N×K (M=1) | Legacy ms / GB/s | Tuned BN32 ms / GB/s | Kernel gain | Rel diff |
|---|---|---|---|---|
| 5120×17408 | 0.1351 / 660 | 0.1050 / 849 | 22.3% | 2.6e-3 |
| 17408×5120 | 0.1308 / 682 | 0.1032 / 864 | 21.1% | 1.9e-3 |
| 5120×12288 | 0.0992 / 634 | 0.0487 / 1294 | 51.0% | 2.9e-3 |
| 5120×10240 | 0.0803 / 654 | 0.0664 / 790 | 17.3% | 3.5e-3 |

GB/s above DRAM peak on the smaller shapes is L2 reuse across bench reps; the within-shape config comparison is the decision basis, and the two large (89 MB, L2-exceeding) shapes still gain 21–22%. Selection is fixed, not autotuned (graph safety): `choose_config` in [fp8_triton.py](../src/inference_from_scratch/fp8_triton.py) looks up `TUNED_CONFIGS` keyed by `(M bucket, N, K)` — bucket 1 is M=1 alone — applies the measured M=1 family config (`BLOCK_N=32`) to unlisted M=1 shapes, and falls back to the legacy heuristic for all other buckets. `--fp8-config old|new` on the TP2 runner (or `INFERENCE_FP8_TILE_CONFIG`) toggles it. CPU tests: [test_fp8_tile_config.py](../tests/test_fp8_tile_config.py).

End-to-end A/B on real TP2 (graph decode, capital, 32 tokens, ABBA cross-process, [ours_tp2_tile_ab.json](../benchmarks/baseline/ours_tp2_tile_ab.json)): old TPOT 28.87 / 28.89 ms (34.63 tok/s), new TPOT 24.48 / 24.48 ms (40.85 tok/s) — **−4.40 ms (−15.2%)**, ids exact 32/32 in all four runs. Verdict: **ship new defaults**, on the end-to-end improvement (not merely the kernel-only gain).

## How the big systems do it

**Qwen-family deployments** are commonly served in bf16, or weight-quantized (AWQ/GPTQ-style int4 with per-group scales and activation-aware calibration — refinements of the per-channel grid above, using small groups of columns per scale and a calibration set to protect salient weights). W4A16 ("4-bit weights, 16-bit activations") is the standard on-device recipe: dequantize weights to fp16 on the fly during the GEMV, pay ~1/4 the weight traffic, keep activations precise.

**vLLM-style serving** supports AWQ/GPTQ/FP8 checkpoints and quantized KV caches, with fused dequantize-into-GEMM kernels so the smaller storage also becomes smaller *traffic*, not just smaller files. The serving win is exactly chapter 00's formula: fewer bytes/token ⇒ lower TPOT and room for more concurrent sequences' KV caches.

**llama.cpp / GGUF (Glimmer-style)** is the purest form: GGUF defines a family of block-quantized formats (Q4/Q5/Q8 with per-block scales, and importance-matrix variants) tuned for CPU SIMD dequantization. A Glimmer-class deployment picks the most aggressive format whose `quantization_error`-style degradation stays acceptable on eval prompts — the same tradeoff, measured the same way, at phone scale.

## Worked: pricing a deployment before downloading it

You are given a checkpoint directory. Before loading anything:

1. `cfg = load_hf_config("ckpt/config.json")`; build `DecoderLM(cfg)` and read `model.num_params()` — call it `P`. (Or sum the safetensors header shapes from ch 09 without instantiating.)
2. Weight bytes at each precision via `bytes_for_dtype(P, ...)`: fp32 `4P`, bf16 `2P`, int8 `P`, int4 `P/2` (rounded up). This is the download size and the decode traffic per token (one full weight read per token, batched or not — batching amortizes it across sequences, ch 11).
3. KV budget: pick target `batch` and `seq_len`, then `kv_cache_bytes(cfg, seq_len, batch, bytes_per_elem=2)` for fp16 KV (halve for int8 KV). Add activations/runtime headroom (~10–20%).
4. Feasibility is now arithmetic: `weight_bytes + kv_bytes + headroom ≤ device memory`, and TPOT ≈ `weight_bytes / effective_bandwidth` at batch 1, improving with batch until compute-bound.

Every quantity came from the checkpoint's own config or from `P` — at no point did anyone need to "know" the model's dimensions from memory. If step 1's file is missing, the honest deliverable is this recipe plus the formulas, not invented numbers.

### What quantization does not shrink

Activations stay in working precision (W4A16 dequantizes *into* fp16 math), attention scores are computed in fp, and the KV cache is a separate decision from weights. A fully int4 deployment can still OOM on KV at long context — the two budgets in step 2/3 are independent levers, and ch 11's block allocator exists to manage the second one dynamically.

## Common pitfalls

- **Per-tensor scales on real weights.** One scale for an entire `[4096, 4096]` matrix lets its single largest element coarsen 16M values. Per-channel (this repo) or per-group (production AWQ/GPTQ) scales are not optional garnish.
- **Quantizing the embedding/norm gains blindly.** Norm gains and (tied) embeddings are unusually sensitive; production recipes often keep them at higher precision. Measure `quantization_error` *and* downstream generation quality, never the former alone.
- **Rounding modes.** `round()` here is round-half-to-even via `torch.round`; a truncating implementation doubles worst-case error to a full `scale` step and invalidates the `scale/2` bound the tests assert.

## Break it / exercises

1. Quantize a row containing one huge outlier (`[0.01, 0.02, 100.0]`) per-channel, then quantize a 2-row matrix where only one row has the outlier, per-*tensor* (single scale). Compare `rmse` on the small values: why does per-channel win?
2. Zero out one row of a random matrix, quantize, dequantize, and assert that row returns *exactly*. Which line makes that true?
3. Use `kv_cache_bytes` on the tiny config (`L=2, H_kv=2, head_dim=16`) at `T=8192, B=8` for fp16 vs int8 KV. How many extra sequences fit in a fixed 1 GiB cache budget under int8?

## Check yourself

<details>
<summary>1. Why does quantization speed up decode far more than prefill?</summary>

Decode is memory-bound: its cost tracks bytes read, and quantization cuts weight bytes 2-4x. Prefill is compute-bound (weights are reused across T rows), so the same byte saving matters proportionally less — though it still helps.
</details>

<details>
<summary>2. Per-element int8 error is bounded by scale_i/2. What sets scale_i, and why per-row?</summary>

`scale_i = max|row i| / 127`: the largest magnitude in the row pins the grid so it exactly fills the int8 range. Per-row scales stop one outlier row from coarsening every other row's grid, as a global scale would.
</details>

<details>
<summary>3. An int4 checkpoint for an assumed 27e9-param model is 13.5 GB. What else must fit in memory before it can serve?</summary>

The KV cache (chapter 05/10 formula — potentially GiBs at long context and batch), activations, and runtime overhead. Weight bytes are the floor, not the total; that is why KV quantization and paging exist.
</details>

## Summary in 30 seconds

This chapter's core is per-channel int8, byte arithmetic, KV pricing. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 11 (serving), which budgets these bytes across concurrent requests. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_quantization.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
