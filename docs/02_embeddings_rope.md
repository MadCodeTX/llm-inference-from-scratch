# 02 — Embeddings and RoPE: giving tokens identity and position

## The one-sentence idea

Token embeddings turn ids into vectors, and RoPE (rotary position embedding) injects position by rotating each query/key vector by an angle proportional to its absolute position — so attention scores end up depending on relative distance.

## Why inference needs this

Attention (chapter 04) is permutation-invariant: shuffle the input tokens and the outputs shuffle identically. A language model therefore needs position information injected somewhere. The two classic options are:

- **Learned positions** (GPT-2 style): add row `t` of a `[T_max, D]` table to token `t`'s embedding.
- **RoPE** (Llama/Qwen style): leave the embeddings alone; rotate queries and keys inside attention.

RoPE has a decisive inference advantage: the rotation for position `t` depends only on `t`, never on neighbours. A key rotated during prefill is already correct when decode queries attend to it later — no cache invalidation, ever. That property is what makes the KV cache (chapter 05) trivially compatible with position.

## The math/shapes

**Embedding.** `tok_embedding` is `[V, D]`. Lookup is row selection: ids `[B, T]` → vectors `[B, T, D]`. When `tie_embeddings=True`, the same `[V, D]` matrix is reused as the LM head (transposed in effect), saving `V*D` parameters.

**RoPE.** Split each head vector of width `D_h` into `D_h/2` pairs. Pair `i` at position `t` is rotated by angle:

```text
angle(i, t) = t * theta^(-2i / D_h)
```

Low pairs (`i=0`) spin fast — angle grows by 1 radian per token — giving local detail. High pairs barely move, carrying long-range signal.

**Concrete example.** Let `D_h = 4`, `theta = 10000`. The inverse frequencies are `theta^(-2i/4)` for `i = 0, 1`: `[1.0, 0.01]`. At position `t = 2` the two pair-angles are `[2.0, 0.02]` radians, with cosines `[-0.4161, 0.9998]`. Pair 0 has rotated a lot (position matters locally); pair 1 is nearly unrotated. At `t = 0` every angle is 0, so RoPE is the identity — position 0 is always unrotated.

This repo uses the **rotate-half** layout: the vector is split into two halves `x = [x1, x2]`, and rotation is `x * cos + rotate_half(x) * sin` where `rotate_half(x) = [-x2, x1]`. Pairs are `(x_i, x_{i + D_h/2})` (half-split, GPT-NeoX convention), not adjacent elements. The cos/sin tables are therefore built by duplicating the half-size angle table: `emb = cat([freqs, freqs])`, shape `[T_max, D_h]`.

Key algebraic facts (all tested): RoPE preserves norms (`||rope(x)|| == ||x||`, rotation is orthogonal), and `dot(rope(q, m), rope(k, n))` depends only on `m - n` — absolute rotations produce relative attention.

## Walk the code

Embeddings live in [model.py](../src/inference_from_scratch/model.py); RoPE lives in [layers.py](../src/inference_from_scratch/layers.py).

- `DecoderLM.__init__` (`model.py:158`) creates `tok_embedding = nn.Embedding(V, D)` (`model.py:164`). For `pos_type="learned"` it also creates `pos_embedding = nn.Embedding(max_seq_len, D)` (`model.py:170`); for `"rope"` that attribute is `None` — there is no position table at all.
- RoPE tables are precomputed once in `__init__` via `precompute_rope_cache(max_seq_len, head_dim, theta=rope_theta)` (`model.py:183`) and registered as **non-persistent buffers** (`model.py:189`), so they never pollute a checkpoint file — they are pure functions of the config.
- `precompute_rope_cache` (`layers.py:113`) returns `(cos, sin)`, each `[max_seq_len, head_dim]`, computed in float32, and raises `ValueError` on odd `head_dim`.
- `apply_rope(x, cos, sin, start_pos=0)` (`layers.py:164`) takes `x` of shape `[B, H, T, D_h]`, slices rows `start_pos : start_pos+T` from the tables, and returns `x * c + _rotate_half(x) * s`. Prefill passes `start_pos=0`; each decode step passes its absolute position.
- In `DecoderLM.forward` (`model.py:234`), the learned variant adds `pos_embedding` rows `start_pos .. start_pos+T-1` (`model.py:262`) and passes `cos = sin = None` to attention; the RoPE variant hands the tables down for attention to slice.
- Attention applies RoPE to queries and keys **only** (`attention.py:196`): values carry no position. Keys are rotated once, before being stored in the KV cache, and never re-rotated.

Tests: [test_layers.py](../tests/test_layers.py) checks position-0 identity, norm preservation, and the relative-distance property; [test_model_generate.py](../tests/test_model_generate.py) checks the learned variant gives identical logits token-by-token vs full-sequence.

## How the big systems do it

**Qwen-family dense models** are RoPE models. The two numbers that matter — `head_dim` and `rope_theta` — come from the checkpoint's `config.json` (`hidden_size / num_attention_heads`, and the `rope_theta` field, which long-context variants set much larger than the classic 10000 to slow the high-pair rotation). Read them with `load_hf_config` (chapter 09); never copy values from a blog post. Long-context variants also ship `rope_scaling` settings (e.g. YaRN-style scaling) that stretch angles beyond the training length; the interface in this chapter — precomputed cos/sin tables — is unchanged, only their values differ.

**vLLM-style serving** fuses the RoPE rotation into the QKV projection kernel epilogue and caches cos/sin tables on-device. The maths is element-for-element the rotate-half formula above; the win is avoiding a separate pass over `[B, H, T, D_h]` tensors in memory.

**llama.cpp / GGUF** precomputes the same tables (or computes angles on the fly per position) and applies RoPE inside its attention kernels, including on CPU SIMD paths. On-device models lean on RoPE's cache-friendliness just as much: a phone decoding token 500 cannot afford to re-rotate 499 cached keys.

## A worked rotation

Take one pair `(a, b) = (1, 0)` at position `t` in plane `i=0` (angle increment 1 radian/token), so the angle is exactly `t` radians. Rotation gives `(cos t, sin t)`:

| `t` | rotated pair | reading |
|-----|--------------|---------|
| 0 | `(1, 0)` | identity — position 0 never moves |
| 1 | `(0.540, 0.841)` | rotated 1 rad |
| 2 | `(-0.416, 0.909)` | rotated 2 rad |

In the rotate-half layout with `D_h = 4`, the vector `[a, b, c, d]` pairs as `(a, c)` and `(b, d)`: halves, not neighbours. `apply_rope` computes `x * cos + [-x2, x1] * sin` where `x1`/`x2` are the two halves — verify by hand that row `t=2` of the `precompute_rope_cache` tables (cosines `[-0.4161, 0.9998]` from this chapter's example) reproduces the pair-0 rotation above and barely touches pair 1.

### Why relative distance falls out

`rope(q, m)` and `rope(k, n)` are both rotations; their dot product contains terms in `cos((m - n) * freq)` because rotation matrices compose by subtracting angles: `R(m)^T R(n) = R(n - m)`. Absolute rotations in, relative geometry out — no relative-position table is ever stored, and nothing about positions `m, n` individually survives except their difference. This is also why RoPE extrapolates gracefully-ish beyond trained lengths in the low-frequency planes (they have barely rotated) and degrades first in the high-frequency ones — the motivation for the `rope_scaling` settings real long-context checkpoints ship.

## Common pitfalls

- **Rotating values.** Only Q and K carry position. Rotating V corrupts retrieved content; `attention.py` applies `apply_rope` to `q` and `k` only, and a stray third call is a silent quality bug, not a crash.
- **Relative instead of absolute `start_pos` at decode.** Passing 0 on every decode step rotates every new token as if it were position 0. Prefill still looks right; generations quietly lose all position sense beyond the prompt.
- **Interleaved vs half-split pairing.** GPT-J-style interleaved pairing `(x_0, x_1), (x_2, x_3)...` and this repo's half-split `(x_i, x_{i+D_h/2})` are different conventions. Loading weights is unaffected (RoPE has no parameters), but mixing conventions between Q rotation and the cached K rotation destroys scores without any shape error.

## Break it / exercises

1. Call `apply_rope` with `start_pos=5` on a `T=1` tensor, then with the equivalent full-sequence call, and verify the rotated vectors match. This is the prefill/decode consistency the KV cache relies on.
2. Set `theta` very small (e.g. `1.0`) in `precompute_rope_cache` and inspect the cos table. What happens to position information, and which test would catch the damage?
3. Switch a tiny `ModelConfig` from `pos_type="rope"` to `"learned"` and compare `num_params()`. Where did the extra `max_seq_len * d_model` parameters come from?

## Check yourself

<details>
<summary>1. Why are values (V) never rotated by RoPE?</summary>

Position should affect *where a query looks* (the Q·K score), not *what is retrieved*. Rotating V would corrupt the content that gets mixed into the output; Q/K rotation alone makes scores relative-position aware.
</details>

<details>
<summary>2. A cached key was rotated at prefill for position 7. At decode step 20, must it be re-rotated?</summary>

No. Its rotation depends only on its own absolute position (7), which never changes. The *query* at step 20 is rotated for position 20, and the dot product sees the difference, 13.
</details>

<details>
<summary>3. What shape are the cos/sin tables, and why is <code>head_dim</code> (not <code>d_model</code>) their width?</summary>

`[max_seq_len, head_dim]`. RoPE rotates each attention head's vector independently, and heads have width `head_dim = d_model // n_head`; the rotation planes live inside a head.
</details>

## Summary in 30 seconds

This chapter's core is position as rotation, cache-compatible by construction. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 03 (norms) prepares the vectors attention rotates. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_layers.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
