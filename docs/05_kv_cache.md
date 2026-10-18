# 05 — The KV cache: never recompute yesterday's keys

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.5 and 5.3. **Production question:** why store past keys and values instead of recomputing them, and when does that memory become the binding constraint? **Online relevance:** cache bytes set how many concurrent online requests fit; offline jobs feel it as max batch size. See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Because each token's keys and values depend only on that token (and its position), decode can compute them once, store them, and reuse them at every later step — trading memory for an entire recomputation.

## Why inference needs this

At decode step `t`, attention needs the keys and values of all `t` previous tokens. Keys/values are pure functions of their own token (RoPE rotation included — chapter 02), so recomputing them each step turns decode into O(T²) work per token. The cache stores them once instead: prefill writes `T` entries in one shot; each decode step appends exactly one entry per layer.

The trade is memory, and it is not small. Every cached token costs, across all layers:

```text
bytes/token = 2 (K and V) * n_layer * n_kv_head * head_dim * bytes_per_elem
```

For long contexts the cache dwarfs the weights — which is why GQA (chapter 04), paged allocation (chapter 11), and KV quantization (chapter 10) all exist.

The cache is also *why decode is memory-bound*: a decode step does O(T) arithmetic per head but must **read** the entire `[H_kv, T, D_h]` K and V blocks from memory to do it — roughly 1 flop per byte read. Prefill reuses data in big matmuls; decode streams the cache (plus all weights) to produce a single token.

## The math/shapes

Per layer, K and V are each `[B, H_kv, T, D_h]`. Time is the append axis (dim 2): new entries go to the right, and attention reads the filled prefix `[..., :T_filled, :]`.

**Concrete example.** Tiny config: `n_layer=2, B=1, H_kv=2, head_dim=16`, fp16 (`bytes_per_elem=2`). Per token: `2 * 2 * 2 * 16 * 2 = 256` bytes. At `T=128`: `128 * 256 = 32768` bytes = 32 KiB. Trivial here — now scale the *formula*, not the intuition: with assumed illustrative dims `L=64, H_kv=8, head_dim=128`, fp16, batch 1, `T=4096`, the same formula gives `2*64*8*4096*128*2 = 1,073,741,824` bytes = 1 GiB. (Those dims are round assumed numbers for arithmetic practice, **not** any real model's config — for a real checkpoint, read `num_hidden_layers`, `num_key_value_heads`, `hidden_size`, `num_attention_heads` from its `config.json` and compute `head_dim = hidden_size / num_attention_heads`.)

**Two layouts, same contract:**

| | `KVCache` (concat) | `StaticKVCache` (preallocated) |
|--|--------------------|-------------------------------|
| Storage | one growing tensor per layer | `[n_layer, B, H_kv, T_max, D_h]` buffer, allocated once |
| Append cost | `torch.cat` copies all previous entries → O(T²) total copying | slice-write in place → O(1) |
| Memory | exactly what's filled | worst case `T_max` always reserved |
| Returns | the full concatenated tensors | views of the filled prefix only |

## Walk the code

Both classes live in [kv_cache.py](../src/inference_from_scratch/kv_cache.py) and are consumed by `CausalSelfAttention.forward` in [attention.py](../src/inference_from_scratch/attention.py).

- `KVCache(n_layer)` (`kv_cache.py:49`): `keys`/`values` are lists of `None`, one slot per layer (`kv_cache.py:65`). `update(layer_idx, k, v)` (`kv_cache.py:68`) stores the first chunk directly, otherwise `torch.cat(..., dim=2)` appends (`kv_cache.py:81`), and returns the full `(k_all, v_all)` for attention to consume. `seq_len` (`kv_cache.py:85`) reports the filled length from the first non-empty layer; `clear()` (`kv_cache.py:94`) drops all tensors.
- `StaticKVCache(n_layer, batch_size, n_kv_head, max_seq_len, head_dim, ...)` (`kv_cache.py:102`): allocates zeroed `k_cache`/`v_cache` of shape `[n_layer, B, H_kv, T_max, D_h]` (`kv_cache.py:134`). `update(layer_idx, k, v, start_pos)` (`kv_cache.py:139`) slice-writes at `start_pos`, raises `ValueError` on overflow past `max_seq_len` rather than truncating (`kv_cache.py:151`), and returns views `[..., :start_pos+T_new, :]` — the unfilled tail is never exposed as attention input. `clear()` zeroes the buffers but keeps them (that is the point of preallocation).
- Dispatch: attention checks `isinstance(kv_cache, StaticKVCache)` to pass `start_pos`, else calls the concat `update` (`attention.py:206`). Either way it reads back *all* cached positions plus the new ones.
- Bookkeeping lives with the caller: `generate.py` passes `start_pos = out.shape[1] - 1` on each decode step (chapter 08), which is both the RoPE position and, for the static cache, the write offset.

The load-bearing test property: **cached and uncached greedy generation must produce identical token ids** ([test_model_generate.py](../tests/test_model_generate.py)). Any divergence is a cache, mask, or position bug — greedy has no randomness to blame. [test_kv_cache.py](../tests/test_kv_cache.py) covers append semantics, `seq_len`, `clear`, static-prefix views, and overflow.

## How the big systems do it

**Qwen-family dense models** don't change the algorithm; GQA is their cache strategy. Fewer KV heads (`num_key_value_heads` in `config.json`) linearly shrink every term in the bytes/token formula. Long-context variants feel cache pressure first, which is why KV-head counts dropped across model generations while query heads stayed high.

**vLLM-style serving** abolishes the contiguous per-request tensor entirely. PagedAttention carves the cache into fixed-size blocks scattered across GPU memory, addressed per-request by a block table (dissected in chapter 11 via `BlockAllocator`). No O(T²) copying, no worst-case preallocation per request, and freed blocks are reused non-contiguously — the two teaching caches here are the two naive endpoints (copy everything vs reserve everything) that paging interpolates between.

**llama.cpp / GGUF (Glimmer-style)** keeps a contiguous cache like `StaticKVCache` but quantizes it (e.g. int8/int4 KV) and sizes it to the device. On a phone, the cache competes with the OS for RAM, so `max_seq_len` is a hard product decision, not a config curiosity — the overflow `ValueError` in `StaticKVCache.update` is the honest version of that limit.

## Worked: filling the cache, step by step

Tiny model, `L = 2` layers, prompt of 3 tokens, then 2 decode steps. Watch layer 0's K tensor (`H_kv = 1`, `D_h` suppressed) in the concat `KVCache`:

| Step | Call | K shape after | Positions held |
|------|------|---------------|----------------|
| Prefill | `update(0, k[1,1,3,·], v)` | `[1, 1, 3, D_h]` | 0, 1, 2 |
| Decode 1 | `update(0, k[1,1,1,·], v)` | `[1, 1, 4, D_h]` | 0..3 |
| Decode 2 | `update(0, k[1,1,1,·], v)` | `[1, 1, 5, D_h]` | 0..4 |

`seq_len` reads 3, 4, 5. Layer 1 keeps its own pair of lists in the same `KVCache` object — recall `DecoderLM.forward` threads **one** cache through all blocks, indexed by `layer_idx` (ch 07). The `StaticKVCache` performs the same writes into columns `start_pos:start_pos+T` of a preallocated `[L, B, H_kv, T_max, D_h]` buffer and returns only the filled prefix as a view — identical values, different storage economics (O(1) append, worst-case reservation).

Total stored after the trace, by the chapter formula (`H_kv = 2` here, `D_h = 16`, fp16): `2 (K,V) * 2 (layers) * 1 (batch) * 2 (H_kv) * 5 (positions) * 16 (D_h) * 2 (bytes)` = 1280 bytes ≈ 1.25 KiB. Small here; the formula is the lesson (ch 10 prices it at serving scale).

## Common pitfalls

- **Caching logits or hidden states.** Only K and V are reused by future attention. Caching anything else wastes the scarcest resource (memory bandwidth at decode) on data never re-read.
- **Sharing one cache across requests.** Cache entries are position-indexed per sequence. Two requests interleaved into one `KVCache` corrupt each other's positions silently — serving isolates caches per request (block tables, ch 11).
- **Reading the unfilled tail.** `StaticKVCache` returns filled-prefix views precisely so uninitialized columns cannot leak into a softmax. Bypassing the view (indexing `k_cache` directly) reintroduces that bug class.

## Break it / exercises

1. In `CausalSelfAttention.forward`, append K/V to the cache *before* applying RoPE. Which equivalence test fails, and why does the failure only appear at decode steps (not prefill)?
2. Benchmark decode with `KVCache` vs `StaticKVCache` at `T=512` using `scripts/benchmark.py`-style timing. Where does the concat cache's O(T²) copying show up?
3. Using `quantization.kv_cache_bytes`, compute the cache for batch 8 at `T=8192` with the tiny config above, then with `H_kv` halved. Which single config field buys the most?

## Check yourself

<details>
<summary>1. Why can keys be cached at all — what property of K/V makes recomputation wasteful rather than necessary?</summary>

Each token's K and V depend only on that token and its absolute position (RoPE rotation included), never on later tokens. Once computed they are final, so storing them loses nothing.
</details>

<details>
<summary>2. <code>StaticKVCache.update</code> returns views ending at <code>start_pos + T_new</code>. Why not return the whole <code>T_max</code> buffer?</summary>

The tail beyond the filled prefix is uninitialized capacity. Letting attention read it would mix garbage keys into every softmax. The filled-prefix view makes the valid region unrepresentable to get wrong.
</details>

<details>
<summary>3. Decode does O(T) math per head but is called memory-bound. Why?</summary>

It must read the full `[H_kv, T, D_h]` K and V blocks (plus all weights) from memory to perform that O(T) math — about a flop per byte. Bandwidth, not arithmetic, sets TPOT.
</details>

## Summary in 30 seconds

This chapter's core is compute-once/store-forever K/V and its byte formula. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 06 (MLP), the per-token half of each block. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_kv_cache.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
