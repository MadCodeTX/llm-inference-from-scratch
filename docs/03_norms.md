# 03 — Normalization: RMSNorm, LayerNorm, and the residual stream

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2.2. **Production question:** how does a deep stack keep activation scale controlled so low-precision inference stays numerically safe? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Normalization rescales each token's `[D]` residual-stream row to a controlled magnitude before a sublayer reads it, and the modern variant (RMSNorm) does it with one statistic and no mean subtraction instead of two.

## Why inference needs this

Stack `L` blocks, each adding attention and MLP outputs back into the residual stream, and activations drift: magnitudes grow, some rows blow up, softmaxes saturate. Norms keep every sublayer's *input* well-scaled regardless of depth, which is also a numerical-precision issue at inference time — fp16 has a narrow range, and an unnormalized stream overflows it.

The layout matters more than the flavour. This repo is **pre-norm**:

```text
x = x + attn(norm(x))
x = x + mlp(norm(x))
```

The residual stream itself is never normalized inside the stack; it is an uninterrupted identity highway from embedding to the final norm before the head. Post-norm (normalize after each add) puts a norm on that highway in every block, trains less stably at depth, and is not what Llama/Qwen ship.

## The math/shapes

Both norms act per-row over the last dim: input `[..., D]` → output `[..., D]`, with a learned gain `[D]`.

**LayerNorm** (GPT-2):

```text
mu  = mean(x)
var = mean((x - mu)^2)
y   = (x - mu) / sqrt(var + eps) * weight + bias
```

**RMSNorm** (Llama/Qwen):

```text
rms(x) = sqrt(mean(x^2) + eps)
y      = x / rms(x) * weight
```

No mean subtraction, no bias. One statistic, one affine parameter.

**Concrete example.** Take the row `x = [3, 4]` (pretend `D = 2`), `weight = [1, 1]`, `eps ≈ 0`.

- RMSNorm: `mean(x^2) = (9 + 16)/2 = 12.5`, `rms = 3.5355`. Output: `[3/3.5355, 4/3.5355] = [0.8485, 1.1314]`. The row's direction is unchanged; only its scale is set so the root-mean-square is 1. Note the output mean is `0.99`, not 0 — RMSNorm does not center.
- LayerNorm: `mu = 3.5`, `var = 0.25`, `sqrt(var) = 0.5`. Output: `[(3-3.5)/0.5, (4-3.5)/0.5] = [-1, 1]`. Centered *and* scaled.

A constant row like `[2, 2]` exposes the philosophical difference: RMSNorm returns `[1, 1]` (a pure rescale — constant stays constant), while LayerNorm returns `[0, 0]` (centering annihilates it).

## Walk the code

Both norms are hand-written in [layers.py](../src/inference_from_scratch/layers.py); `F.rms_norm` / `nn.LayerNorm` appear only as test oracles.

- `RMSNorm(d, eps)` (`layers.py:32`): holds a single `weight` parameter of ones (`layers.py:60`) — gain only, no bias. `forward` (`layers.py:62`) upcasts to float32, computes `ms = xf.pow(2).mean(dim=-1, keepdim=True)`, returns `xf * rsqrt(ms + eps) * weight`, cast back to the input dtype. The float32 reduction is deliberate: `mean(x^2)` in fp16 loses precision fast, and production kernels upcast the same way.
- `LayerNorm(d, eps)` (`layers.py:74`): `weight` and `bias` (`layers.py:97`), computes `mu`, centered `xc`, biased variance (divide by `D`, not `D-1`), then `(x - mu)/sqrt(var + eps) * weight + bias` (`layers.py:100`).
- Wiring: `_make_norm(config)` in [model.py](../src/inference_from_scratch/model.py) (`model.py:63`) picks the class from `config.norm_type`, and each `TransformerBlock` builds `attn_norm` and `mlp_norm` from it (`model.py:102`). The block applies them pre-norm style: `x = x + self.attn(self.attn_norm(x), ...)` then `x = x + self.mlp(self.mlp_norm(x))` (`model.py:127`).
- Count: norms run `2 * n_layer + 1` times per token (two per block plus `final_norm`, `model.py:202`).

Tests in [test_layers.py](../tests/test_layers.py) compare both implementations against PyTorch reference norms and check the centering difference between them.

## How the big systems do it

**Qwen-family dense models** use RMSNorm everywhere, with the epsilon read from `config.json`'s `rms_norm_eps` field (mapped to `ModelConfig.rms_eps` by `load_hf_config`, chapter 09). The gain vectors are part of the checkpoint (`input_layernorm.weight`, `post_attention_layernorm.weight`, final `norm.weight`) and map onto this repo's `attn_norm` / `mlp_norm` / `final_norm` names in chapter 09.

**vLLM-style serving** fuses RMSNorm with the residual add (`x + sublayer(norm(x))`) into a single kernel, so the `[B*T, D]` stream is read and written once instead of three times. At decode time, when everything is memory-bound, that fusion is worth more than the arithmetic saved.

**llama.cpp / GGUF** implements RMSNorm in its CPU/GPU kernels with the same float32 accumulation trick. On-device, the norm is also where quantization scales often get folded in — another reason to keep the formula in plain sight, as here, rather than behind a library call.

Why did RMSNorm win? Empirically (Zhang & Sennrich, 2019) re-centering is unnecessary for pre-norm Transformers, so LayerNorm's mean subtraction and bias are pure cost: one fewer reduction pass per norm, `2L+1` times per token.

## Worked: one residual block, numerically

Track a single token row with `D = 2` through one pre-norm block, using identity sublayers to isolate the norm/residual arithmetic. Start: `x = [3, 4]`.

1. `h = RMSNorm(x)` with unit gain: `[0.8485, 1.1314]` (this chapter's example).
2. Pretend `attn(h) = h` (identity). Residual add: `x' = x + h = [3.8485, 5.1314]`. Note `x'` is **not** renormalized — the stream grew, by design.
3. `h2 = mlp_norm(x')`: RMS of `x'` is `sqrt((14.81 + 26.33)/2) ≈ 4.536`, so `h2 ≈ [0.8485, 1.1314]` again — the norm strips the accumulated scale right back off before the MLP reads it.
4. `x'' = x' + mlp(h2)`. The stream carries the sum of every sublayer output so far; each sublayer only ever *reads* a unit-RMS view.

That loop is the pre-norm contract: the stream is an accumulator (an "identity highway"), and norms are read-time lenses, not stream mutations. Post-norm would normalize `x'` itself at step 2, putting a squashing nonlinearity on the highway that every later block must fight through — the depth-instability Llama/Qwen avoided by going pre-norm.

### The eps, honestly

`eps` sits *inside* the square root: `sqrt(mean(x²) + eps)`. Its job is singular: a zero row (possible for padding or a dead residual early in training) must not divide by zero. At `rms_eps = 1e-6` it is invisible for any row with RMS ≫ 1e-3, and merely softens the blow-up for tiny rows. Setting it large (exercise 2) rescales *small* rows toward zero output while barely touching large ones — a silent, input-dependent damping.

## Common pitfalls

- **Normalizing over the wrong axis.** Norms reduce over the last dim (`D`) only, per token row. A reduction over `T` would couple tokens, break decode (where `T=1` makes statistics degenerate), and invalidate cached positions.
- **Forgetting the gain.** The learned `weight [D]` is part of the checkpoint (`attn_norm.weight` etc., ch 09). A norm with gain forced to 1 loads "successfully" and generates subtly wrong text.
- **Mean-centering the stream.** Porting LayerNorm intuition ("outputs have mean 0") to RMSNorm leads to wrong unit tests and wrong fused kernels. RMSNorm rows have RMS 1, mean arbitrary.

## Break it / exercises

1. Feed `RMSNorm` the row `[2, 2, 2, 2]` and `LayerNorm` the same row. Predict both outputs before running, then explain the difference in one sentence.
2. Set `eps` to `1.0` in a tiny model's `rms_eps` and generate greedily. What breaks qualitatively, and why does a *large* eps damage small activations more than large ones?
3. Remove the float32 upcast in `RMSNorm.forward` (compute in the input dtype) and run the layer tests in float16 on a large-magnitude input. Find an input where the outputs visibly diverge.

## Check yourself

<details>
<summary>1. In pre-norm, what is normalized: the residual stream, or a copy of it?</summary>

A copy. `attn_norm(x)` reads a normalized view; the result of attention is added back to the *unnormalized* `x`. The stream itself is normalized only once, by `final_norm` before the head.
</details>

<details>
<summary>2. RMSNorm output rows have RMS 1 (before the gain). Do they have mean 0?</summary>

No. There is no mean subtraction, so a row like `[3, 4]` maps to `[0.8485, 1.1314]` with mean ≈ 0.99. Only LayerNorm centers.
</details>

<details>
<summary>3. Why compute <code>mean(x^2)</code> in float32 when the model runs in fp16?</summary>

Squaring and averaging in fp16 overflows and loses precision quickly; production kernels upcast the reduction internally for the same reason. The multiply back is done in the working dtype.
</details>

## Summary in 30 seconds

This chapter's core is one-statistic RMSNorm and the pre-norm residual highway. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 04 (attention), the first consumer of normalized rows. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_layers.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
