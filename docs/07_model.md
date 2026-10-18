# 07 — The full model: wiring blocks into a language model

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2.1. **Production question:** how do embeddings, blocks, and a head assemble into one decoder whose every dimension comes from the checkpoint's `config.json` rather than from code? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

A decoder-only language model is just embeddings, a stack of identical pre-norm blocks (attention + MLP, each wrapped in a residual add), a final norm, and a head that scores every vocabulary token as the next one.

## Why inference needs this

Every earlier chapter built one component. This one is the assembly — and the assembly choices (pre-norm residuals, tied embeddings, one shared KV-cache object threaded through all layers) are exactly the choices that make the generation loop in chapter 08 possible. There is no encoder and no cross-attention: the only sequence mixing is causal self-attention, which is what makes the KV cache valid and the prefill/decode split clean.

## The math/shapes

The forward pass, with `B` batch, `T` tokens, `D = d_model`, `V = vocab_size`, `L = n_layer`:

```text
idx [B, T] --tok_embedding--> x [B, T, D]
           (+ pos_embedding rows, learned variant only)
for each of L blocks:
    x = x + attn(attn_norm(x))     [B, T, D] -> [B, T, D]
    x = x + mlp(mlp_norm(x))       [B, T, D] -> [B, T, D]
x = final_norm(x)
logits = lm_head(x)                [B, T, V]
```

Position `t`'s logits predict token `t+1`. Generation reads only the last position (chapter 00).

**Parameter count, concretely.** Tiny config: `V=256, D=32, L=2, H=4, H_kv=2, F=64`, RoPE, SwiGLU, tied embeddings.

- Embedding (tied, counted once): `256 * 32 = 8192`
- Per block: attention `q: 32*32 + k: 32*16 + v: 32*16 + o: 32*32 = 3072`; MLP `3 * 32 * 64 = 6144`; norms `2 * 32 = 64`. Block total: `9280`
- Two blocks: `18560`; final norm: `32`
- **Total: `8192 + 18560 + 32 = 26,784`** — which is what `num_params()` should report (verify in the exercises). Attention is a minority; the MLP and embedding dominate, as in real models.

**Tied embeddings.** When `tie_embeddings=True`, `lm_head.weight is tok_embedding.weight`: one `[V, D]` matrix maps ids→vectors on the way in and vectors→id-scores on the way out, saving `V*D` parameters. Small models (GPT-2, small Qwen sizes) do this; large ones usually untie.

## Walk the code

The model is [model.py](../src/inference_from_scratch/model.py).

- `_make_norm(config)` (`model.py:63`) and `_make_mlp(config)` (`model.py:70`) are the single construction points for the `norm_type` / `mlp_type` variants — blocks never branch on flavour themselves.
- `TransformerBlock(config)` (`model.py:77`): builds `attn_norm`, `attn` (`CausalSelfAttention`), `mlp_norm`, `mlp` (`model.py:102`). `forward(x, cos, sin, kv_cache, layer_idx, start_pos)` (`model.py:107`) is the two pre-norm residual adds (`model.py:127`): attention first (tokens talk), then MLP (tokens think), forwarding `layer_idx`/`start_pos` verbatim to attention.
- `DecoderLM(config)` (`model.py:137`): `tok_embedding` (`model.py:164`); `pos_embedding` only for the learned variant (`model.py:170`); RoPE `cos`/`sin` `[T_max, head_dim]` as non-persistent buffers (`model.py:189`); `layers = nn.ModuleList(TransformerBlock ...)` (`model.py:196`); `final_norm` (`model.py:202`); bias-free `lm_head` (`model.py:205`), tied by assignment `self.lm_head.weight = self.tok_embedding.weight` (`model.py:209`) when configured. `_init_weights` uses std-0.02 normals so untrained logits start near-uniform (loss ≈ ln V).
- `DecoderLM.forward(idx, kv_caches=None, start_pos=0)` (`model.py:234`) returns `(logits, kv_caches)`. Crucially, `kv_caches` is **one** cache object for the whole model: it is threaded through every block with that block's `layer_idx` (`model.py:274`) and returned as the same object, so `generate.py` can feed it back step after step. The learned variant adds `pos_embedding` rows `start_pos .. start_pos+T-1` (`model.py:265`) — a decode step must pass its absolute position, the same bookkeeping RoPE needs.
- `num_params()` (`model.py:230`) sums `p.numel()` over `parameters()`, which deduplicates the tied weight automatically.

Tests: [test_model_generate.py](../tests/test_model_generate.py) covers logits shapes for both position variants, cache identity (same object returned), tied vs untied weights, `num_params`, and the layernorm/gelu variant stack.

## How the big systems do it

**Qwen-family dense models** are this exact graph at scale: the checkpoint's `config.json` (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`) maps field-by-field onto `ModelConfig` via `load_hf_config` (chapter 09), and the same `DecoderLM` class instantiates it. Differences to expect in real checkpoints: sliding-window attention on some layers, QKV biases on some variants, and MoE MLPs on the largest — all reported honestly by chapter 09's validator rather than silently forced into this shape.

**vLLM-style serving** instantiates the same block stack but shards it: tensor parallelism splits each `[D, D]`-ish matrix across GPUs with an all-reduce per block. The residual-stream math is unchanged; chapter 12 discusses why the communication, not the math, becomes the limit.

**llama.cpp / GGUF (Glimmer-style)** builds the identical graph from a single self-contained file, with weights quantized in place. On-device, `num_params()` translates directly into a storage/bandwidth budget via chapter 10's `bytes_for_params` — the model either fits the phone's RAM alongside its KV cache or it does not ship.

## Worked: a two-token forward, shape by shape

Tiny config as in the parameter example (`V=256, D=32, L=2, H=4, H_kv=2, F=64`, RoPE). Input `idx = [[104, 105]]`:

| Stage | Tensor | Shape |
|-------|--------|-------|
| ids | `idx` | `[1, 2]` |
| embed | `tok_embedding(idx)` | `[1, 2, 32]` |
| block 0 attn | Q `[1,4,2,8]`, K/V `[1,2,2,8]` → out | `[1, 2, 32]` |
| block 0 MLP | hidden `[1, 2, 64]` → out | `[1, 2, 32]` |
| block 1 | (repeat; head_dim 8 throughout) | `[1, 2, 32]` |
| final norm | `final_norm(x)` | `[1, 2, 32]` |
| head | `lm_head(x)` | `[1, 2, 256]` |

Two rows of logits come out; generation keeps row 1 (`logits[:, -1]`, shape `[1, 256]`) and samples the next token. With a KV cache attached, the same call also leaves K/V for positions 0–1 in both layers' cache slots, so the next call needs only the new token's `[1, 1]` ids at `start_pos=2`.

### Reading a state dict like a map

`DecoderLM.state_dict()` keys are the architecture written out: `tok_embedding.weight`, then per layer `layers.{i}.attn_norm.weight`, `layers.{i}.attn.{q,k,v,o}_proj.weight`, `layers.{i}.mlp_norm.weight`, `layers.{i}.mlp.{gate,up,down}_proj.weight`, then `final_norm.weight`, `lm_head.weight`. Chapter 09's entire mapping table is derived by diffing this key set against a HF checkpoint's — you can regenerate it yourself with `print(list(model.state_dict()))` instead of trusting the book.

## Common pitfalls

- **Post-norm temptation.** Moving the norms after the residual adds "to match an old diagram" changes the architecture; checkpoints are trained pre-norm and will load shape-perfectly into a post-norm graph while producing garbage.
- **One cache per block.** `forward` takes a single `kv_caches` object and indexes it by `layer_idx`. Building `L` separate caches and passing the wrong one to a block passes shapes and scrambles layers.
- **Forgetting `final_norm`.** The head expects normalized input. Skipping it (or normalizing twice) loads fine and degrades output quality in ways that look like "the model is just weak".

## Break it / exercises

1. Build the tiny config from the parameter example above and assert `num_params() == 26784`. If your count differs, find which term you mis-derived (untie the embeddings and watch it grow by exactly `V*D`).
2. Set `tie_embeddings=False` and compare the state dict keys before/after. Why does the tied model's state dict still *list* `lm_head.weight` even though it is the same Parameter?
3. Pass `start_pos=0` on a decode step (T=1, cache non-empty) instead of the absolute position, with the learned variant. Which test catches this, and what goes wrong geometrically for the RoPE variant?

## Check yourself

<details>
<summary>1. Why does <code>forward</code> return the cache object it was given, unchanged in identity?</summary>

So the caller (`generate`) can keep feeding the same object back each decode step while blocks mutate it in place via `layer_idx`. Returning it makes the dataflow explicit and lets stubs/models that ignore the cache still satisfy the contract.
</details>

<details>
<summary>2. The residual stream is normalized how many times inside the block stack?</summary>

Zero. Each block normalizes a *copy* for its sublayers (`attn_norm`, `mlp_norm`); the stream itself is normalized once, by `final_norm`, after the last block.
</details>

<details>
<summary>3. What changes in the graph when <code>pos_type</code> flips from <code>rope</code> to <code>learned</code>?</summary>

A `pos_embedding [T_max, D]` table appears and its rows are added to the token embeddings; `cos`/`sin` become `None` and attention skips RoPE. Position moves from inside attention to the input, and `max_seq_len * d_model` parameters appear.
</details>

## Summary in 30 seconds

This chapter's core is embed, L pre-norm blocks, final norm, (tied) head. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 08 (sampling/generation), which drives this model token by token. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_model_generate.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
