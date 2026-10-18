# 12 — Optimizations: read-only dissection of the fast paths

## The one-sentence idea

Every production speedup — FlashAttention/SDPA, `torch.compile`, CUDA graphs, speculative decoding, continuous batching — replaces exactly one dissected component in this repo while preserving its semantics, and this chapter is a guided tour of those seams (no code changes: pointers only).

## Why inference needs this

The preceding chapters optimize for *legibility*: materialized score matrices, Python-loop decode, concatenated caches. Production optimizes for bandwidth and kernel-launch overhead. The risk in jumping straight to the fast paths is not knowing what they compute. Having built the slow versions, you can now read each optimization as "same math, different data movement" — and, critically, verify any fast path against the slow reference already in the repo (that is what the tests' oracles are for).

**Ground rule for this chapter:** each section names the repo file/function the optimization *would* replace. Nothing here edits them; the manual paths stay as references. When you adopt a fast path in your own work, keep the slow one and assert agreement, exactly as [test_attention.py](../tests/test_attention.py) already does.

## The math/shapes

**IO-awareness (why FlashAttention wins).** Naive attention materializes `scores [B, H, T, T]` in HBM (GPU main memory): at `T=4096`, fp16, one head, that is `4096² * 2 = 33.5 MB` written and re-read per head per layer — pure traffic. FlashAttention tiles Q/K/V through SRAM (on-chip, ~100x faster), maintains a *running* softmax (online rescaling by the running max/sum), and never stores the `[T, T]` matrix. Same result, O(T) extra memory instead of O(T²), and far fewer HBM round-trips. FLOPs are identical (even slightly higher due to recomputation); the win is entirely data movement — chapter 00's moral again.

**Speculative decoding, as acceptance arithmetic.** A small draft model proposes `k` tokens; the target model scores all `k` in one parallel forward (a mini-prefill). Each proposal is accepted with probability related to how well the draft matches the target; on rejection, sampling resumes from the corrected distribution. Expected tokens per target forward is `> 1` whenever the draft is decent, and the *output distribution is provably the target's* — speculation changes speed, never semantics. It attacks decode's serialism: one memory-bound weight-read yields several tokens.

**CUDA graphs / `torch.compile`, as overhead removal.** A decode step launches dozens of small kernels; at `T=1` sizes, launch overhead rivals math time. CUDA graphs capture the whole step once and replay it with new data pointers; `torch.compile` fuses elementwise chains (norms, SiLU-multiplies) into single kernels. Neither changes results (up to fusion numerics); both shrink the non-math fraction that dominates tiny-batch decode.

## Walk the code

Seams, file by file. Each entry: what production does, and the exact repo component it replaces.

- **SDPA / FlashAttention →** `manual_scaled_dot_product_attention` in [attention.py](../src/inference_from_scratch/attention.py) (`attention.py:33`), called from `CausalSelfAttention.forward` (`attention.py:224`). `F.scaled_dot_product_attention(q, k, v, is_causal=True)` computes the identical scaled-mask-softmax-matmul (and dispatches to FlashAttention kernels on suitable hardware). The repo already treats it as the test oracle; swapping it in production means deleting the materialized `scores` tensor and gaining GQA support natively (no `repeat_kv` materialization — kernels index shared KV heads). Keep the manual op for chunked/nonstandard masks where explicit `-inf` control matters.
- **RoPE/norm fusion →** `apply_rope` ([layers.py](../src/inference_from_scratch/layers.py), `layers.py:164`), `RMSNorm.forward` (`layers.py:62`), and the residual adds in `TransformerBlock.forward` ([model.py](../src/inference_from_scratch/model.py), `model.py:127`). Serving kernels fuse norm+residual and RoPE-into-QKV so each `[B*T, D]` tensor crosses memory once. `torch.compile` on `DecoderLM` gets part of this automatically.
- **`torch.compile` / CUDA graphs →** the decode loop `_generate_cached` in [generate.py](../src/inference_from_scratch/generate.py) (`generate.py:140`) and `DecoderLM.forward` (`model.py:234`). Compile the model once; the prefill (varying `T`) and decode (`T=1`, static shapes) paths compile separately — decode's fixed shapes are the graph-friendly case. Watch for graph breaks at the Python-level cache dispatch (`isinstance(kv_cache, StaticKVCache)`, `attention.py:206`): static-shape, preallocated buffers graph-capture cleanly; the concat `KVCache` (new tensor every step) does not — one more reason production preallocates or pages.
- **Speculative decoding →** the sampler seam `sample_next` in [sampling.py](../src/inference_from_scratch/sampling.py) (`sampling.py:124`) plus a second, smaller `DecoderLM` as drafter. `generate()`'s loop would verify `k` drafted tokens per iteration instead of sampling one. The greedy special case is the teaching version: accept the longest drafted prefix matching the target's argmax, resample the first mismatch.
- **Continuous batching (recap) →** already dissected in [serving.py](../src/inference_from_scratch/serving.py): `ContinuousBatcher.step` (`serving.py:192`) is the iteration-level scheduler vLLM runs; production adds chunked prefill, preemption, and prefix sharing around the same admit/step/retire skeleton (chapter 11).
- **PagedAttention (recap) →** `BlockAllocator` (`serving.py:256`) is the allocator half; the kernel half (gathering K/V by block table inside attention) replaces the contiguous cache reads implied by `KVCache`/`StaticKVCache` in [kv_cache.py](../src/inference_from_scratch/kv_cache.py). Same semantics, scattered storage.

## How the big systems do it

**Qwen-family deployments** combine all of the above: FlashAttention kernels, GQA-native KV indexing, AWQ/GPTQ or FP8 weights (chapter 10), continuous batching with paged KV (chapter 11), and optionally speculative decoding with a smaller sibling model as drafter. Read the knobs off the deployment's serving config the same way you read model dims off `config.json` — never assume a technique is on because it exists.

**vLLM-style serving** is the reference integration: its block manager, scheduler, and fused sampler are production versions of `serving.py`'s three classes, and its attention backends (FlashAttention / FlashInfer) are `attention.py:33`'s fast counterparts. The dissection value flows both ways — vLLM's source is readable *because* you have built the naive versions.

**llama.cpp / GGUF (Glimmer-style)** shows the CPU/on-device flavour: no CUDA graphs, but persistent thread pools, SIMD-fused dequantize+GEMV, quantized KV, and speculative decoding with a tiny draft model. The ordering of wins differs by hardware; the seams in this repo do not.

## Decision guide: which optimization, when

| Symptom | Likely binding constraint | First lever (repo seam) |
|---------|--------------------------|-------------------------|
| Slow prefill on long prompts | score-matrix traffic, matmul efficiency | SDPA/FlashAttention (`attention.py:33`) |
| High TPOT, batch 1, small model | kernel-launch + Python overhead | `torch.compile`, CUDA graphs (`generate.py:140` loop) |
| High TPOT, large model, any batch | weight bytes per token | quantization (ch 10; `bytes_for_params`) |
| High TPOT, draft sibling available | serial decode itself | speculative decoding (`sampling.py:124` seam) |
| Low GPU utilization, many users | scheduling gaps, idle slots | continuous batching (`serving.py:192`) |
| OOM at long context / high concurrency | contiguous KV reservations | paged blocks (`serving.py:256`, ch 11) |
| Cache append dominates profile | O(T²) copying | static/paged cache (`kv_cache.py`) |

Measure before choosing: `scripts/benchmark.py` already separates prefill (ms, tokens/s) from decode (cache vs no-cache tokens/s), and `serving.measure_prefill_decode` splits TTFT from TPOT on a real model. An optimization aimed at the wrong row of this table is the most common serving mistake — e.g. FlashAttention will not fix a launch-bound batch-1 deployment, and quantization will not fix scheduler idle slots.

### The verification discipline

Each fast path gets adopted the same way: keep the dissected reference, run both on identical inputs, assert agreement (exactly for integer/greedy paths, within tolerance for fused numerics). This repo models the pattern — SDPA is the *oracle* in `test_attention.py`, the no-cache loop is the oracle for cached generation, and `validate_state_dict` is the pre-flight oracle for weight loading. Optimizations that cannot be checked against a slow truth eventually become unverifiable folklore.

## Common pitfalls

- **Benchmarking the oracle away.** Deleting the manual attention/cache "because production uses the fast path" removes the only independent check on the fast path. Disk is cheap; silent numeric drift is not.
- **Compiling the variable-shape path.** `torch.compile` and CUDA graphs reward static shapes; the `T=1` decode step qualifies, arbitrary-`T` prefill largely does not. Compile them as separate paths or expect recompilation stalls.
- **Expecting speculation to help a bad draft.** Acceptance rate is everything: a draft that mismatches early wastes its verification forward. Draft/target must share a tokenizer (ch 01) — a mismatch there makes acceptance near zero and the "optimization" a slowdown.

## Break it / exercises

1. Write a throwaway script (not a repo change) timing `manual_scaled_dot_product_attention` vs `F.scaled_dot_product_attention` at `T = 64, 512, 2048` on the largest device available. Where does the materialized `[T, T]` matrix start to dominate? Compare against the 33.5 MB/head figure derived above.
2. Draft a greedy speculative-decoding loop on paper against `generate.py`: with draft length `k=4` and a perfect draft, how many target forwards produce 20 tokens? With a draft that is wrong at position 2 every time?
3. List every Python-level branch on the decode path (`generate.py` → `model.py` → `attention.py`) that would break a CUDA-graph capture. For each, name the chapter-05/11 mechanism that removes it.

## Check yourself

<details>
<summary>1. FlashAttention does slightly <em>more</em> FLOPs than naive attention, yet is faster. Why?</summary>

It recomputes some softmax statistics during tiling instead of storing/re-reading the `[T, T]` score matrix in HBM. The workload is bandwidth-bound, so trading a few FLOPs for a large traffic reduction wins.
</details>

<details>
<summary>2. Why does speculative decoding not change the output distribution?</summary>

Proposals are accepted/rejected by a correction rule derived from the target model's own probabilities; rejected positions are resampled from the adjusted target distribution. The draft only affects *how many* tokens each target forward finalizes, never their law.
</details>

<details>
<summary>3. Which repo cache graphs cleanly and which does not, and why?</summary>

`StaticKVCache` (fixed buffers, in-place slice writes) captures cleanly; the concat `KVCache` allocates a new, differently-shaped tensor every step, which breaks static-shape graph capture. Same reason production systems preallocate or page.
</details>

## Summary in 30 seconds

This chapter's core is same semantics, less data movement, verified against slow references. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to the glossary and chapter 00, to re-read the whole arc with production eyes. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_attention.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.

Resist benchmark maximalism. The right question for any fast path is never "is it faster" but "is it faster *on my phase, shape, and hardware, while still agreeing with the reference*". A 2x attention kernel that only engages at T≥1024 does nothing for a chat workload of short prompts, and a speculative setup whose draft disagrees is slower than plain decode. Measure the workload you have.
