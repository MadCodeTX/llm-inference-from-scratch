# 11 — Serving: continuous batching and paged KV memory

## The one-sentence idea

A serving system runs one shared decode loop for many requests at once — admitting and retiring requests at *every step* (continuous batching) and allocating their KV caches in fixed-size, possibly scattered blocks (paged attention) — instead of running each request to completion in isolation.

## Why inference needs this

Chapter 08's `generate()` is one request's loop. A service has hundreds of concurrent requests with different prompt lengths and output lengths. Two naive strategies both waste the machine:

- **One-at-a-time**: decode is memory-bound (chapter 00), so a lone request leaves the hardware idle waiting on weight reads that could have served 50 requests at once. Batching amortizes each weight read over the whole batch.
- **Static batching** (admit a fixed group, run until *all* finish): a request that needs 5 tokens waits for its batch-mate that needs 500. The batch's latency is its longest member's, and slots sit finished-but-occupied.

Continuous (iteration-level) batching fixes the second: the scheduler steps the whole running set once, retires whoever finished, and admits waiting requests into freed slots — every iteration. Utilization stays high and short requests never queue behind long ones inside a batch.

The memory side needs the matching fix. If each request reserves a contiguous `max_seq_len` cache (chapter 05's `StaticKVCache`), memory is exhausted by reservations, not usage. **Paged attention** carves the KV cache into fixed blocks allocated on demand and addressed by a per-request block table — virtual memory, applied to KV. Internal fragmentation is at most one partially-filled block per sequence.

## The math/shapes

**Latency vs throughput metrics** (all defined over a request's life):

- **TTFT** (time to first token): prefill time for the prompt. Compute-bound; grows with prompt length.
- **TPOT** (time per output token): mean decode-step time after the first token.
- **Throughput**: total tokens (or requests) completed per second across the system. Batching raises throughput while barely moving per-request TPOT — until compute saturates.

**Block math, concretely.** `block_size = 16` tokens, `num_blocks = 8` (128 token-slots). A request needing 40 tokens needs `ceil(40/16) = 3` blocks (48 slots, 8 wasted). Three such requests use 9 blocks — but we only have 8, so the third must wait even though `3*40 = 120 ≤ 128` raw slots would fit. That gap *is* fragmentation, now bounded and visible. When a request finishes, its blocks return to the free pool and can serve a new request **non-contiguously** — block ids `[0, 5, 2]` are fine, because the block table, not physical adjacency, defines order.

**Scheduler step, as a state machine.** Each `step()`: (1) admit waiting requests into free running slots, including any added mid-flight; (2) advance every running request by one token; (3) retire finished requests to the finished list, freeing their slots for the *next* step's admission.

## Walk the code

Everything is in [serving.py](../src/inference_from_scratch/serving.py) — a model-independent dissection (the sampler is injected, so the scheduler can be tested without a model).

- `Request` (`serving.py:68`): dataclass with `prompt_ids`, `max_new_tokens`, `generated` (accumulates one id per step), and `eos_id`. `finished` (`serving.py:94`) is a **derived property**, never stored: true once `max_new_tokens` tokens exist or the last generated token is `eos_id`. `total_len` (`serving.py:105`) is prompt + generated.
- `ContinuousBatcher` (`serving.py:130`): `waiting`, `running`, `finished` lists plus `step_count`. `add(request)` (`serving.py:165`) enqueues to waiting. `step(sampler=None)` (`serving.py:192`) increments the counter, calls `_admit()` (FIFO, capped by `max_batch_size`), advances each running request with the sampler — which may be a 1-arg `(request)` or 2-arg `(request, step_count)` callable, or the default rule `last_token + 1` (`serving.py:117`) — appends newly-finished requests to `finished`, and returns how many requests are still running. A `max_new_tokens=0` request finishes on the first step that admits it. `run(sampler)` (`serving.py:227`) steps until no work remains, with a `max_steps` runaway guard.
- `BlockAllocator(num_blocks, block_size)` (`serving.py:256`): `blocks_for_tokens(n) = ceil(n / block_size)`; `allocate(n_tokens)` (`serving.py:312`) returns a list of free block ids, or `None` **atomically** when blocks run out — it never hands out a partial allocation. `free(ids)` (`serving.py:330`) returns blocks to the pool and raises `ValueError` on double-free. `free_blocks` / `allocated_blocks` / `wasted_slots(n)` expose the fragmentation accounting.
- `compute_tpot(decode_seconds, n_decode_tokens)` and `compute_throughput(n_tokens, elapsed)` (`serving.py:359`): the metric definitions as functions. `measure_prefill_decode(...)` (`serving.py:407`) times a real `DecoderLM` prefill/decode loop with `time.perf_counter`, separating TTFT from TPOT, and its produced tokens match `generate()` greedy exactly.

Tests: [test_serving.py](../tests/test_serving.py) covers finish order, mid-flight admission, batch-size caps, allocator refusal/free/fragmentation, and measured-helper token equality.

## How the big systems do it

**vLLM-style serving** is this chapter industrialized: a scheduler loop admits/retires requests per iteration; PagedAttention block tables (managed much like `BlockAllocator`, plus reference counting for shared prefixes — many requests share a system prompt, so its blocks are allocated once and read by all) address a block-scattered KV cache; sampling (chapter 08's knobs, per request) runs fused on-GPU. Prefix caching, chunked prefill (splitting a long prompt across iterations so decode never stalls), and preemption (swapping blocks out under memory pressure) are refinements of the same two ideas: schedule at iteration granularity, allocate KV in blocks.

**Qwen-family models on bigbox** sit behind exactly such a stack (see the control-center notes: a dashboard endpoint fronting the loaded model). The model's job ends at logits; TTFT/TPOT targets, max batch, and block counts are serving-layer decisions made with this chapter's formulas and the checkpoint's real `config.json` dims.

**llama.cpp / GGUF (Glimmer-style)** serves few concurrent requests (often one), so continuous batching matters less than the memory discipline: a fixed KV budget, block-ish reuse across conversation turns, and speculative decoding (chapter 12) to cut TPOT. The lesson transfers: even a single-user device is a scheduler with a memory allocator attached.

## Worked: three requests through the batcher

`ContinuousBatcher(max_batch_size=2)`. Requests: A (needs 3 tokens), B (needs 1), C (needs 2). Sampler: any rule; EOS never emitted.

| Step | Admit | Advance | Retire | Running after |
|------|-------|---------|--------|---------------|
| 1 | A, B | A→1, B→1 | B (done) | A |
| 2 | C (into B's slot) | A→2, C→1 | — | A, C |
| 3 | — | A→3, C→2 | A, C | — |

Finish order: B, then A and C together at step 3 — `finished == [B, A, C]`. C arrived "late" (added before step 1 but admissible only at step 2) and still overlapped A's decode; under static batching, C would wait for a fresh batch and B's slot would idle through steps 2–3. Multiply by thousands of requests and that idle-slot effect *is* the throughput difference between serving generations.

On the memory side, give each request blocks as it grows: with `block_size=16`, A (prompt 10 + 3 generated = 13 tokens) needs 1 block the whole time; a 40-token request needs 3. `BlockAllocator.allocate` returning `None` is the scheduler's signal to leave a request waiting — backpressure, not an error.

## Common pitfalls

- **Treating `finished` as state to set.** It is derived from `generated` vs `max_new_tokens`/`eos_id` (`serving.py:94`). Manually flagging requests finished desynchronizes the scheduler from the token lists it also uses for lengths and block accounting.
- **Freeing blocks at admission time.** Blocks are freed when a request *retires*, then reallocated (possibly non-contiguously) to another. Free-early corrupts live sequences; double-free raises `ValueError` by design — let it.
- **Optimizing TTFT and TPOT with the same knob.** Bigger batches raise throughput and can worsen TPOT under contention; longer chunked-prefill slices protect TPOT at TTFT's expense. There is no single "latency" to minimize — report both, always.

## Break it / exercises

1. Create a `ContinuousBatcher(max_batch_size=2)`, add three requests needing 1, 5, and 1 tokens, and step manually. Record which step each finishes on and when the third is admitted. What would static batching have cost request 1?
2. With `BlockAllocator(4, 16)`, allocate for 40, 40, then 40 tokens. Show the third returns `None` despite `120 ≤ 256` total slots... wait, recompute: is it slots or blocks that ran out? Then free the first and retry — note the reused ids need not be contiguous.
3. Make a sampler that returns `eos_id` immediately for one request but never for another. Run `run()` and explain why the never-finishing request alone determines total steps — and which guard eventually saves you.

## Check yourself

<details>
<summary>1. Why is <code>Request.finished</code> a derived property rather than a stored boolean?</summary>

It is a pure function of `generated`, `max_new_tokens`, and `eos_id`. Storing it would create a second source of truth that can disagree with the token list; deriving it makes inconsistency unrepresentable.
</details>

<details>
<summary>2. Continuous batching admits a new request mid-flight. What must be true of its KV cache for that to be safe?</summary>

Its cache blocks are independent of the running requests' blocks (block-table addressing, not a shared contiguous tensor), and its prefill can be interleaved without disturbing others' positions — each request's `start_pos` bookkeeping is its own.
</details>

<details>
<summary>3. <code>allocate</code> returns <code>None</code> instead of a partial block list. Why is atomicity important?</summary>

A partial allocation would leave a request holding blocks it cannot use while starving others — leaked memory with no progress. Refusing atomically lets the scheduler simply retry next step after some request frees blocks.
</details>

## Summary in 30 seconds

This chapter's core is iteration-level scheduling and block-allocated KV. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 12 (optimizations), which replaces each dissected seam with its fast path. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_serving.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.

If you operate a deployment, the two dashboards that matter follow directly from this chapter: TTFT percentiles (prefill health, prompt-length driven) and TPOT percentiles (decode health, batch/contention driven), plus KV-block occupancy as the saturation early-warning. Throughput alone hides every pathology that matters to a waiting user.
