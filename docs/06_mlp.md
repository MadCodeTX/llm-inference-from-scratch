# 06 — The MLP: where the parameters live

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.1.2 and 2.2.2, with a pointer to MoE at sec. 2.2.4. **Production question:** where do most parameters and prefill FLOPs sit, and what would change if only a few expert MLPs ran per token? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Each token independently passes through a widening-then-narrowing MLP (`d_model → d_ff → d_model`), and the modern SwiGLU variant spends its parameter budget on a *gated* product of two projections instead of one plain projection.

## Why inference needs this

Attention moves information *between* tokens; the MLP is where each token's representation is actually transformed — "thinking" versus "talking". It holds the majority of parameters in a typical decoder (two of every three weight matrices per block, three for SwiGLU), so it dominates both the model's memory footprint and prefill FLOPs.

"Independently" is the inference-relevant word: there is no cross-token math here. The MLP is identical in prefill and decode except for how many rows (`B * T`) are pushed through it — `T` rows at prefill, `B` rows (one per sequence) at decode. No cache, no mask, no position bookkeeping.

## The math/shapes

**GELU MLP** (GPT-2 style), two bias-free matrices:

```text
hidden = gelu(x @ W_up)        [B, T, D] -> [B, T, F]
y      = hidden @ W_down       [B, T, F] -> [B, T, D]
```

Parameters: `2 * D * F`.

**SwiGLU** (Llama/Qwen style), three bias-free matrices:

```text
hidden = silu(x @ W_gate) * (x @ W_up)     [B, T, F] elementwise product
y      = hidden @ W_down                   [B, T, D]
```

where `silu(z) = z * sigmoid(z)`. Parameters: `3 * D * F`.

**Concrete example.** Let `D = 4, F = 8`. GELU MLP: `2 * 4 * 8 = 64` parameters. SwiGLU: `3 * 4 * 8 = 96`. To compare fairly at equal budget, SwiGLU shrinks `F`: with `F = 5` (and change 64→60 params) it roughly matches — this is why real SwiGLU models use `d_ff ≈ (2/3)` of what a two-matrix MLP would use at the same parameter count, often rounded to a multiple of 256. In `config.json` terms, `intermediate_size` already reflects that shrinkage; you read it, you don't derive it.

**Why a gate helps.** The `up` branch proposes features; `silu(gate)` decides, per feature and per token, how much of each proposal survives (values near 0 suppress, large values pass). That data-dependent product is a strictly richer function than a fixed nonlinearity on one projection, and it consistently wins at equal parameter count (Shazeer, 2020) — worth one extra matrix.

## Walk the code

Both variants are in [mlp.py](../src/inference_from_scratch/mlp.py), bias-free throughout (Llama/Qwen style; biases are a negligible win at this scale and inference here is deterministic — no dropout either).

- `SwiGLU(d_model, d_ff)` (`mlp.py:26`): `gate_proj`, `up_proj` (`[D → F]`), `down_proj` (`[F → D]`) (`mlp.py:50`). `forward` (`mlp.py:56`) computes `hidden = F.silu(gate) * up` elementwise (`mlp.py:61`) and returns `down_proj(hidden)` (`mlp.py:63`). Note the order: SiLU applies to the **gate** branch only; the up branch enters the product raw.
- `GeluMLP(d_model, d_ff)` (`mlp.py:66`): `up_proj`, `down_proj` (`mlp.py:85`). `forward` (`mlp.py:89`) applies `F.gelu` — the exact erf form, not the tanh approximation — between the two projections.
- Selection: `_make_mlp(config)` in [model.py](../src/inference_from_scratch/model.py) (`model.py:70`) picks by `config.mlp_type` (`"swiglu"` | `"gelu"`), and each `TransformerBlock` calls it as `x = x + self.mlp(self.mlp_norm(x))` (`model.py:133`).
- State-dict names (chapter 09 maps checkpoints onto these): SwiGLU exposes `mlp.{gate,up,down}_proj.weight`; GELU exposes `mlp.{up,down}_proj.weight` (`model.py:96`).

Tests: [test_mlp.py](../tests/test_mlp.py) checks output shapes, that SwiGLU with a zeroed gate outputs zero (the gate really gates), and GELU against a manual erf computation.

## How the big systems do it

**Qwen-family dense models** are SwiGLU throughout; `intermediate_size` in `config.json` is the `d_ff` to read (via `load_hf_config`, chapter 09). Some large variants replace the dense MLP with mixture-of-experts (MoE): many `d_ff`-sized experts, a router picking a few per token. The per-token math for an activated expert is exactly `SwiGLU.forward` above — MoE changes *which* MLP runs, not what an MLP is. This repo stays dense deliberately: MoE routing is a serving problem (chapter 11/12 territory), not an MLP problem.

**vLLM-style serving** fuses gate+up into one kernel with the SiLU-multiply as its epilogue, so the `[B*T, F]` hidden tensor is written once instead of twice. During decode the MLP is pure weight-streaming (one row per sequence), so fused, quantized GEMV kernels matter more than FLOPs.

**llama.cpp / GGUF** runs the same three-matrix gated MLP with quantized weights (chapter 10): on-device, `gate`/`up`/`down` are typically stored at 4-bit-ish precision and dequantized on the fly, because reading 3*D*F full-precision weights per token would blow the memory-bandwidth budget that decode lives on.

## Worked: the gate, numerically

One token, `D = 2, F = 2`, toy weights chosen so the arithmetic is checkable. Let `x = [1, 2]`, `W_gate = I`, `W_up = [[1, 0], [0, 2]]` (rows are output features), `W_down = I`:

- `gate = x @ W_gate^T = [1, 2]`; `silu([1, 2]) = [0.731, 1.762]` (since `silu(z) = z·sigmoid(z)`: `1×0.731`, `2×0.881`).
- `up = x @ W_up^T = [1, 4]`.
- `hidden = silu(gate) * up = [0.731, 7.048]`. Feature 1 was *amplified* (gate > 1 after SiLU × a large up value); had the gate pre-activation been very negative, `silu → 0` and the feature would vanish regardless of `up`.
- `y = hidden @ W_down^T = [0.731, 7.048]`.

Compare GELU on the same `up`: `gelu([1, 4]) ≈ [0.841, 4.000]` — a fixed squashing of a single projection, with no second opinion about which features deserve to survive. That second opinion, computed per token from the token itself, is the entire SwiGLU idea.

### Parameter share, at real-ish proportions

In a SwiGLU block with width `D` and `F ≈ 8D/3` (a common ratio before rounding), MLP parameters are `3·D·F = 8D²`, versus attention's ~`4D²` (Q/O at `D²` each, K/V narrower under GQA). The MLP is roughly two-thirds of the block — which is why MLP weight quantization (ch 10) and MoE sparsity are where large-model efficiency work concentrates, and why `intermediate_size` is the config field to read first when sizing a checkpoint.

## Common pitfalls

- **SiLU on the wrong branch.** `silu(up) * gate` has identical shapes and plausible outputs. It is a different function from the checkpoint's, and only weight-loading shape checks plus golden-output comparisons (ch 09) catch it — unit tests over shapes cannot.
- **Dropping the elementwise product.** `down(silu(gate) + up)` "looks" equivalent in a skim and destroys the gating semantics entirely. The multiply *is* the chapter.
- **Bias nostalgia.** GPT-2 MLPs have biases; Llama/Qwen (and this repo) do not. Adding bias parameters creates checkpoint keys with no source tensors — `validate_state_dict` (ch 09) reports them as missing, correctly.

## Break it / exercises

1. Zero the `gate_proj` weight in a `SwiGLU` and confirm the output is exactly zero for any input. Then zero `up_proj` instead — same result, different reason. Explain both.
2. Swap the SiLU onto the `up` branch (`hidden = gate * F.silu(up)`) and run the model tests. Shapes still pass — so what *would* catch this bug when loading a real checkpoint in chapter 09?
3. Compute parameter counts for `D=64`: GELU with `F=128` vs SwiGLU with `F=85`. Which is closer to a tie, and why do real configs round `F` to a multiple of 256 anyway? (Hint: hardware tile sizes, chapter 12.)

## Check yourself

<details>
<summary>1. Why is the MLP untouched by the KV cache and causal mask?</summary>

It has no cross-token operations: each row `[D]` is transformed independently. Masking and caching exist solely for attention's token-mixing; the MLP neither knows nor cares which step it is in.
</details>

<details>
<summary>2. SwiGLU has 3 matrices to GELU's 2, yet models prefer it "at equal parameter count". How is the count equalized?</summary>

By shrinking `d_ff` to roughly 2/3 of the two-matrix width. The gated product extracts more quality per parameter, so the smaller-but-gated MLP wins.
</details>

<details>
<summary>3. During decode with B=8, how many MLP rows are computed per step, and during prefill of a 200-token prompt?</summary>

Decode: 8 rows (one per sequence, `B * 1`). Prefill: `8 * 200 = 1600` rows in one batched pass. Same weights, same code path — only the row count changes.
</details>

## Summary in 30 seconds

This chapter's core is SwiGLU gating and the parameter majority. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 07 (full model), which wires MLP and attention into blocks. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_mlp.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
