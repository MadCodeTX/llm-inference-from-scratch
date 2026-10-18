# 00 — Overview: inference is two different workloads

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), Ch 0 and sec. 1.2, 1.4. **Production question:** between a request arriving and tokens streaming back, where does time go, and how should that latency be reported? **Online/offline:** online serving lives or dies by TTFT and decode TPS at p95/p99; offline batch work trades those percentiles for aggregate TPS. See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Inference is one parallel **prefill** pass over the prompt followed by many serial **decode** steps that each emit one token, and almost every later optimization exists because those two phases stress the hardware differently.

## Why inference needs this

Training processes a known sequence in parallel and updates weights. Inference has no labels and no weight updates: it must produce token `t+1`, feed it back in, and repeat. That feedback loop is why inference is dominated by memory traffic, not arithmetic.

Two quantities follow immediately:

- **Time to first token (TTFT)** — how long prefill takes. Users feel this as startup delay.
- **Time per output token (TPOT)** — how long each decode step takes, in seconds per token. Users feel this as streaming speed.
- **Tokens per second (TPS)** — the rate form of the same quantities. Per request, decode TPS ≈ `1 / TPOT`; system TPS counts all tokens finished per second across the batch, which is the throughput batching raises. Always say which one you mean, and report TTFT and TPOT as percentiles (p50/p95/p99), not only means: a mean hides the slow tail users remember.

## The math/shapes

A decoder-only model maps token ids `[B, T]` to logits `[B, T, V]`:

```text
ids [B, T] --embed--> x [B, T, D] --L blocks--> x [B, T, D] --head--> logits [B, T, V]
```

`B` is batch, `T` sequence length, `D = d_model`, `V = vocab_size`. Only `logits[:, -1]` (shape `[B, V]`) is needed to choose the next token during generation.

**Concrete tiny example.** Take `B=1, T=4, D=32, V=256` (a byte model). The embedding lookup turns `[1, 4]` ids into `[1, 4, 32]` floats. The head turns that into `[1, 4, 256]` logits. Four distributions come out, but generation samples only from position 3, the last one.

**Why prefill is compute-bound and decode is memory-bound.** In prefill, a weight matrix `[D, D]` multiplies `T=256` token rows at once, so each weight loaded from memory is reused 256 times. In decode, the same matrix multiplies **one** row: every weight is loaded to do roughly one multiply-add. Arithmetic intensity collapses to ~1 flop/byte, so the GPU/ CPU sits waiting on memory. The KV-cache chapter (05) shows the same effect for cached keys and values.

Rule of thumb:

| Phase | Tokens per forward | Dominant cost | Repo entry point |
|-------|--------------------|---------------|------------------|
| Prefill | `T_prompt` (often >1) | matmul FLOPs | `generate.py` `_generate_cached`, first forward |
| Decode | 1 | weight + KV reads | `generate.py` `_generate_cached`, loop |

> **Reading a real model's scale — do not memorize numbers.** Open the checkpoint's own `config.json` and read `hidden_size` (= `D`), `num_hidden_layers` (= `L`), `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, and `vocab_size`. Every cost formula in this repo (parameter bytes in chapter 10, KV bytes in chapters 05/10) takes those fields as inputs. This book never hard-codes a large model's dimensions.

## Walk the code

Start with the shape contract in [config.py](../src/inference_from_scratch/config.py):

- `ModelConfig` (`config.py:28`) bundles `vocab_size, d_model, n_layer, n_head, n_kv_head, d_ff, max_seq_len` plus `rope_theta=10000.0`, `rms_eps=1e-6`, `tie_embeddings=True`, and the three variant switches `norm_type`, `mlp_type`, `pos_type`.
- `ModelConfig.__post_init__` (`config.py:76`) fails fast if `d_model % n_head != 0` or `n_head % n_kv_head != 0`, so a bad shape dies at construction instead of deep inside attention.
- `ModelConfig.head_dim` (`config.py:99`) is a property: `d_model // n_head`. RoPE and the `1/sqrt(head_dim)` attention scale are sized by this, never by `d_model`.

The prefill/decode split itself lives in [generate.py](../src/inference_from_scratch/generate.py): `generate()` (`generate.py:91`) dispatches to `_generate_cached` (one prefill forward at `start_pos=0`, then `T==1` decode forwards) or `_generate_no_cache` (the wasteful reference path). Chapter 08 walks that loop line by line.

## How the big systems do it

**Qwen-family dense models** are decoder-only stacks of the same shape: embedding, `L` pre-norm blocks with grouped-query attention and SwiGLU MLPs, final norm, LM head. Nothing about prefill/decode changes at scale — only `D`, `L`, and `V` grow, which makes decode *more* memory-bound, not less.

**vLLM-style serving** keeps this exact per-request loop but runs thousands of requests concurrently: continuous batching (chapter 11) admits and retires requests every step so the weight reads of one decode step are amortized over many sequences.

**llama.cpp / GGUF (Glimmer-style on-device)** runs the same two phases on a phone or laptop CPU/GPU with quantized weights (chapter 10). Quantization is a direct attack on the decode bottleneck: fewer weight bytes read per token means faster TPOT on bandwidth-starved hardware.

## A worked end-to-end trace

Follow one tiny request through the whole system. Prompt `"hi"`, byte tokens, so `idx = [[104, 105]]` with shape `[1, 2]`:

1. **Encode** (ch 01): `ByteTokenizer.encode("hi") = [104, 105]`. No training, no vocabulary file.
2. **Prefill** (ch 08): `DecoderLM.forward(idx, kv_caches, start_pos=0)` embeds both tokens (`[1, 2, D]`), runs both blocks, caches K/V for positions 0 and 1 in every layer, and emits logits `[1, 2, V]`. Only `logits[:, -1]` — position 1's prediction — is sampled.
3. **Decode step 1**: the sampled token (say id 33) is fed alone as `[1, 1]` with `start_pos=2`. Its K/V are appended at position 2; its query attends to all 3 cached positions. One new token is sampled.
4. **Decode steps 2..n**: identical, with `start_pos` incrementing 3, 4, 5... Each step costs one full read of the weights and cache (ch 05) for one token.
5. **Decode the answer**: collected ids are byte-decoded back to text. If sampling stopped at EOS, that token is present in the id list and the caller strips it.

Notice what never happens: positions 0 and 1 are never re-embedded, their K/V never recomputed, and no forward ever sees a future token. Those three invariants are the whole design; every chapter protects one of them.

### Cost accounting for the trace

Suppose the tiny model has `P` parameters in fp16 and the prompt is `T_p` tokens with `n` tokens generated:

- Prefill FLOPs scale like `T_p * P` (every token uses every weight, in parallel).
- Decode FLOPs also scale like `n * P`, but spread over `n` serial steps that each *read* `2P` weight bytes.
- Total latency ≈ TTFT(`T_p`) + `n` × TPOT. For long outputs, TPOT dominates: generation length, not prompt length, sets the bill.

That last line is the most useful intuition in this book. It explains why serving systems optimize decode (batching, quantization, speculation — ch 10–12) far more aggressively than prefill.

## Common pitfalls

- **Confusing `T` across phases.** In prefill `T` is the prompt length; in decode `T=1` always, and the *cache* holds the growing length. Shapes like `[B, H, T, D_h]` mean different `T`s in the two phases (ch 04–05).
- **Reading all logits.** `logits[:, -1]` is the only row generation consumes. Averaging or summing logits over positions is a category error.
- **Optimizing FLOPs when bandwidth binds.** A "faster" kernel that reads the same bytes is not faster at decode. Count bytes first (ch 10's `bytes_for_params`, ch 05's KV formula).

## Break it / exercises

1. Set `max_new_tokens=0` in `generate()` and confirm it returns the prompt unchanged (both paths). Which line handles it on the cached path?
2. In `ModelConfig`, try `d_model=30, n_head=4`. Read the `ValueError` from `__post_init__` and explain why failing here beats failing at a `reshape`.
3. Time prefill for prompt lengths 16, 64, 256 with `scripts/benchmark.py`. Plot ms vs length: is prefill time linear in `T` at these sizes? Why might it look sub-linear on a tiny CPU model?

## Check yourself

<details>
<summary>1. Why does generation only read <code>logits[:, -1]</code>?</summary>

Because position `t`'s logits predict token `t+1`. Only the last position has no known successor yet, so only its distribution is sampled. Earlier positions' predictions are already "answered" by the prompt itself.
</details>

<details>
<summary>2. Prefill processes 256 tokens in one forward; decode processes 1. Why is decode not 256x faster per step?</summary>

Because decode still reads every weight and the whole KV cache from memory. Its cost is bandwidth, not token count — that is the memory-bound regime described above.
</details>

<details>
<summary>3. Where do a real checkpoint's layer count and hidden size come from in this repo?</summary>

From its own `config.json`, translated by `load_hf_config` in `weights.py` (`num_hidden_layers -> n_layer`, `hidden_size -> d_model`). Never from numbers memorized in docs.
</details>

## Summary in 30 seconds

This chapter's core is inference's two-phase shape. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 01 (tokenizer) turns text into the ids this chapter's trace consumes. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_model_generate.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
