# 04 — Attention: scores, causal masks, and grouped queries

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2.3 and 2.5. **Production question:** how do tokens exchange information without seeing the future, and how does head sharing shrink the cache that decode must stream? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Attention lets each token build its output as a softmax-weighted average of earlier tokens' value vectors, where the weights come from scaled query·key dot products — and grouped-query attention (GQA) shares each key/value head among several query heads to shrink the KV cache.

## Why inference needs this

The MLP (chapter 06) processes each token in isolation. Attention is the *only* place tokens exchange information, so it defines both model quality (what context a prediction can use) and the central inference data structure: keys and values, once computed, are reused at every later step — the KV cache of chapter 05.

Causality is the constraint that makes generation possible: token `t` may attend to positions `≤ t` only. Without the mask, "predict the next token" would cheat by reading it.

## The math/shapes

Per head, with queries `Q [T_q, D_h]`, keys `K [T_k, D_h]`, values `V [T_k, D_h]`:

```text
scores  = Q @ K^T / sqrt(D_h)      [T_q, T_k]
scores  = mask_future(scores)      (causal: -inf above the diagonal)
weights = softmax(scores, dim=-1)  rows sum to 1
out     = weights @ V              [T_q, D_h]
```

The `1/sqrt(D_h)` scale keeps dot products from growing with dimension, which would push softmax into its saturated tails (one weight ≈ 1, gradients and resolution gone).

**Concrete example.** One head, `D_h = 2`. A query `q = [1, 0]`; two keys `k1 = [1, 0]`, `k2 = [0, 1]`; values `v1 = [10, 0]`, `v2 = [0, 20]`. Raw scores: `q·k1 = 1`, `q·k2 = 0`. Scaled by `1/sqrt(2) ≈ 0.707`: `[0.707, 0]`. Softmax: `[0.67, 0.33]`. Output: `0.67*[10,0] + 0.33*[0,20] = [6.7, 6.6]`. The query "asked" mostly for whatever `k1` advertises, and got mostly `v1`.

**Causal mask, prefill vs decode.** Prefill (`T_q = T_k = 4`): row `i` may see keys `0..i` — a lower-triangular mask. Decode (`T_q = 1`, `T_k = 9`, the new query sits at absolute position 8): it may see **all 9 keys**. The unifying rule used in this repo: query row `i` sits at absolute position `i + (T_k - T_q)` and may attend keys `j ≤` that position. One formula covers both regimes, including chunked prefill (`T_q > 1, T_k > T_q`).

**GQA shapes.** With `H` query heads and `H_kv` KV heads, `n_rep = H / H_kv`. Only `H_kv` heads of K/V are ever computed or cached; at attention time each KV head is repeated `n_rep` times to meet the queries. `H_kv = H` is classic multi-head attention (MHA); `H_kv = 1` is multi-query attention (MQA). Cache size scales with `H_kv`, so GQA with `n_rep = 4` quarters the KV cache at a small quality cost.

## Walk the code

Everything is in [attention.py](../src/inference_from_scratch/attention.py), built from matmul + softmax only — `F.scaled_dot_product_attention` is never called here (chapter 12 dissects what it would replace).

- `manual_scaled_dot_product_attention(q, k, v, is_causal=False)` (`attention.py:33`): takes `[B, H, T_q, D_h]` queries and `[B, H, T_k, ·]` keys/values. It computes `scores = q @ k^T / sqrt(d_h)` (`attention.py:71`), builds the causal mask from absolute positions `q_pos = arange(T_q) + (T_k - T_q)` and `masked_fill`s future entries with `-inf` (exact zeros after softmax, not approximate ones) (`attention.py:77`), then softmaxes (`attention.py:86`) and returns `weights @ v` (`attention.py:89`).
- `repeat_kv(x, n_rep)` (`attention.py:92`): `[B, H_kv, T, D_h] → [B, H_kv*n_rep, T, D_h]` with **grouped** ordering — KV head `g`'s copies are consecutive, serving query heads `g*n_rep .. (g+1)*n_rep - 1`. Interleaving instead would pair every query with the wrong KV head and silently corrupt GQA. `n_rep == 1` returns the input unchanged.
- `CausalSelfAttention(config)` (`attention.py:119`): bias-free projections sized `q_proj: [D → H*D_h]`, `k_proj`/`v_proj: [D → H_kv*D_h]`, `o_proj: [H*D_h → D]` (`attention.py:152`). `forward(x, cos, sin, kv_cache, layer_idx, start_pos)` (`attention.py:159`) projects and reshapes to `[B, H, T, D_h]` (`attention.py:189`), applies RoPE to Q/K only (`attention.py:198`), appends K/V to the cache if given (`attention.py:205`, dispatching on `StaticKVCache` vs concat cache), expands KV heads with `repeat_kv` (`attention.py:216`), runs the manual attention with `is_causal=True` (`attention.py:224`), then merges heads and applies `o_proj` (`attention.py:228`).

Tests: [test_attention.py](../tests/test_attention.py) compares the manual op against `F.scaled_dot_product_attention` as an oracle, checks masked positions get exactly zero weight, decode-shape causality, and `repeat_kv` ordering.

## How the big systems do it

**Qwen-family dense models** use GQA: `num_key_value_heads < num_attention_heads` in `config.json` (read the real values from the checkpoint — chapter 09). Some Qwen variants add biases to the Q/K/V projections; this repo's projections are bias-free Llama-style, and chapter 09's weight validator reports such biases as unexpected rather than dropping them silently.

**vLLM-style serving** never materializes the `[T_q, T_k]` score matrix: FlashAttention-style kernels tile it through SRAM (chapter 12), and PagedAttention reads keys/values from scattered cache blocks (chapter 11). The grouped KV-head layout is exploited directly — kernels index the shared KV head instead of physically repeating it as `repeat_kv` does. The maths per element is identical to the manual version; that is precisely why the manual version is the reference.

**llama.cpp / GGUF** implements the same scaled dot-product with causal masking in hand-tuned CPU/GPU kernels, with KV heads shared GQA-style to fit the cache in device memory. On-device, avoiding the materialized score matrix matters even at modest context lengths because memory, not FLOPs, is the scarce resource.

## Worked: the mask, cell by cell

Prefill with `T_q = T_k = 3`. Query absolute positions are `[0, 1, 2]`; the boolean `future` matrix (`k_pos > q_pos`) is:

```text
        k0   k1   k2
q0  [  keep MASK MASK ]   token 0 sees only itself
q1  [  keep keep MASK ]   token 1 sees 0..1
q2  [  keep keep keep ]   token 2 sees all
```

Masked cells get `-inf` before softmax, so row `q0`'s softmax is a point mass on `k0` (weight exactly 1.0 — there is nothing else), and its output is exactly `v0`. That is the correct semantics for the first token's representation during teacher-forced prefill: it is the state from which token 1 will be predicted.

Now the decode twin: one new query, `T_q = 1`, `T_k = 4` (three cached + itself). `q_pos = [0 + (4 - 1)] = [3]` and `k_pos = [0, 1, 2, 3]`; no key position exceeds the query position, so the row is `[keep keep keep keep]`. One formula, both regimes; the chunked-prefill case (`T_q = 2, T_k = 5`, `q_pos = [3, 4]`) masks only the genuinely-future triangle.

### Where the FLOPs and bytes go

Per head, prefill computes a `[T, T]` score matrix: at `T = 512` that is 262,144 scores, each a `D_h`-multiply dot product — and, in this manual implementation, a materialized tensor that chapter 12's FlashAttention analysis avoids writing. Decode computes a `[1, T]` row: trivial arithmetic, but it must *read* `[T, D_h]` of K and the same of V. Attention at decode is a streaming reduction over the cache; its cost table is chapter 05's byte formula, not a FLOP count.

## Common pitfalls

- **Masking with a large negative instead of `-inf`.** Leaks ~`exp(-1e9 + max)` — underflows to 0 in practice, but the *exact*-zero guarantee (and exact prefill/decode agreement) comes from `-inf`; keep the stronger invariant.
- **Applying RoPE after caching.** Keys must be rotated *before* storage (ch 02). Rotate-after-read would need the position of every cached key at every step — the cache would no longer store finished products.
- **Wrong `repeat_kv` order.** Grouped `[kv0, kv0, kv1, kv1]` matches how query heads are laid out; interleaved expansion is a shape-valid, semantics-destroying bug (exercise 2).

## Break it / exercises

1. In `manual_scaled_dot_product_attention`, remove the `/ sqrt(d_h)` factor and set `D_h = 64` with random Q/K. Inspect the softmax rows: how peaked do they get, and why does that hurt?
2. Change `repeat_kv` to interleave copies (`[kv0, kv1, kv0, kv1, ...]`) and run the GQA tests. Which tests fail, and why is the failure silent (no shape error) in real code?
3. Pass `is_causal=False` during a prefill forward and generate greedily. Construct a prompt where the output visibly "cheats" — i.e. depends on a token that should have been in its future.

## Check yourself

<details>
<summary>1. A decode query is at absolute position 8 with 9 cached keys. How many keys does the causal mask hide?</summary>

None. Query row 0 sits at position `0 + (9 - 1) = 8` and may attend all keys `j ≤ 8`, which is all of them. The mask only bites when a query has genuine future keys, as in prefill.
</details>

<details>
<summary>2. Why <code>-inf</code> in the mask instead of a large negative number like -1e9?</summary>

`exp(-inf) = 0` exactly, so masked tokens contribute nothing to the softmax denominator or the weighted sum. A finite penalty leaks a small but nonzero weight, which breaks exact prefill/decode agreement.
</details>

<details>
<summary>3. GQA with H=32, H_kv=8: what is n_rep, and what does the KV cache save versus MHA?</summary>

`n_rep = 32 // 8 = 4`. Only 8 KV heads are stored instead of 32, so the KV cache is 4x smaller; Q computation and the score maths are unchanged.
</details>

## Summary in 30 seconds

This chapter's core is scaled dot-product, absolute-position causal masking, GQA. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 05 (KV cache), which stores this chapter's K/V. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_attention.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
