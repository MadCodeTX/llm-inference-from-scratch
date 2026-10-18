# 10 — Quantization: fewer bits per number

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
