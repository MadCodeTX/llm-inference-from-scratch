# 14 — Techniques: reuse work, draft ahead, and measure the cost

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 5.1.3, 5.2, 5.3.1, 5.4 and 5.5. **Production question:** which repeated or serial work can be removed without changing what the target model would have said, and how is any quality change that remains actually measured? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Prefix caching and greedy speculative decoding both delete work while preserving the target model's output exactly — one reuses keys and values for a shared opening, the other lets a cheap draft run ahead and keeps only what the target verifies — while quantization deliberately trades a small, measured output change for fewer bytes.

## Why inference needs this

Chapter 13 showed decode paying a full weight and cache read for one token, and prefill paying again for openings that many requests share verbatim. Three different levers follow. Reuse the shared opening instead of recomputing it. Let several tokens be settled per target forward instead of one. Or shrink the bytes each step must move, and accept a numeric change only after measuring it on outputs. The first two are exactness-preserving optimisations; the third is a trade, and confusing those categories is how silent quality regressions ship.

## The math/shapes

**Prefix reuse (sec. 5.3.1).** A stored prefix of length `P` for a prompt of length `T` leaves only `T - P` positions to forward. The saved work is the prefill arithmetic and KV writes for `P` tokens; time-to-first-token falls roughly in proportion for the prompt-processing part, while decode is untouched. Reuse is valid because a token's keys and values depend only on that token, its position, and earlier tokens (chapters 02, 04, 05) — never on what follows.

**Speculative verification (sec. 5.2).** With draft length `gamma` and per-token agreement probability summarised crudely as independent with rate `a`, the expected number of draft tokens kept per target forward is `(1 - a^(gamma+1)) / (1 - a) - 1` in the usual geometric sketch, plus the one target-chosen token that always follows the kept run. The sketch only guides choosing `gamma`: a longer run helps when agreement is high and wastes draft work when the first proposal is usually wrong. The exact greedy rule built here needs no probability model at all.

**Quality measurement (sec. 5.1.3).** For reference and candidate distributions `p` and `q` at each position: `KL(p || q) = sum_v p_v * (log p_v - log q_v)`, averaged over positions (`mean_kl`, in nats, 0 when identical); `top1_agreement` is the fraction of positions whose argmax agrees; `logit_rmse` is the root-mean-square raw-logit difference. Weight error alone (chapter 10) does not predict these: small weight perturbations can be amplified or absorbed by later layers.

## Walk the code

### Prefix caching — built

- [prefix_cache.py](../src/inference_from_scratch/prefix_cache.py): `PrefixKVCache(max_entries)` keys entries by the exact tuple of token ids. `put(tokens, kv_cache)` stores a clone of the cache, truncated to exactly the key length, and evicts the least recently used entry past the bound. `lookup(tokens) -> (matched_len, kv_cache_clone)` returns the longest *stored* key that opens `tokens`, again as a fresh clone, and records counters.
- Counters are `hits`, `misses` and `tokens_saved` (a hit adds its matched length), available as attributes and as the `stats` dict — `cache.stats["hits"]` or `cache.stats()`. `tokens_saved` counts skipped prefill positions, not seconds.
- `prefill_with_prefix(model, idx, prefix_cache) -> (logits_last, kv_cache, n_reused)` looks the prompt up, forwards only `idx[:, n_reused:]` with `start_pos=n_reused`, and returns the `[1, V]` logits that predict the next token plus the completed cache. It deliberately does not store anything: the caller picks reuse boundaries with `put`, typically once after prefilling a shared system opening. If the entire prompt is already stored, the helper trims the clone by one position and re-forwards the last token, because a KV cache holds keys/values but not the logits that token produced.

Tests: [test_prefix_cache.py](../tests/test_prefix_cache.py) pins the longest-prefix rule, clone isolation in both directions, reuse *only* at `put` boundaries (a divergent continuation of an unstored opening finds nothing), LRU eviction, and — the load-bearing check — suffix prefills whose last logits, greedy tokens and per-layer KV tensors match an independent full prefill.

### Speculative decoding — draft-target built

- [speculation.py](../src/inference_from_scratch/speculation.py): `speculative_generate(target, draft, idx, max_new_tokens, gamma=4) -> (tokens, stats)` (alias `greedy_speculative_generate`). Each round the draft greedily proposes `gamma` tokens, then *one* target forward over prompt-plus-proposals verifies them. The longest agreeing run is kept, followed by one target-chosen token: the correction at the first disagreement, or a bonus token when every proposal agrees. `stats` reports `accepted` (kept proposals), `drafted` (proposals made) and `target_forwards` (verification passes).
- Both models run cache-free on the full sequence. That wastes constant-factor work and is stated in the module rather than hidden: with this repo's concatenating cache, rejected draft tokens would sit in the cache needing rollback, and the teaching point — one verification forward settling several tokens — is clearer without that bookkeeping. Production keeps caches and truncates them to the accepted length.

Tests: [test_speculation.py](../tests/test_speculation.py) holds the invariant two ways. With the draft equal to the target, every proposal is accepted and `target_forwards` is strictly less than the token count. With a different tiny model as draft, the output still equals target-only greedy (`generate(..., temperature=0)`) token for token.

### Quantization quality — built (extends chapter 10)

- [quantization.py](../src/inference_from_scratch/quantization.py): `quantize_model_weights_(model)` fake-quantizes every Linear weight in place through the chapter-10 int8 round-trip, writing dequantized floats back into the same Parameter objects (tied weights stay tied). `measure_quality_impact(reference_model, candidate_model, idx) -> {mean_kl, top1_agreement, logit_rmse}` runs both models on the same ids and compares output distributions, not weights.

Tests: the additions in [test_quantization.py](../tests/test_quantization.py) check a model against itself (all three metrics at their identity values) and that an int8 fake-quantized copy sits far closer to its reference — in KL and logit RMSE — than a re-randomized model of the same shape. On the tiny seeded setup used there, the quantized copy's mean KL is orders of magnitude below the re-randomized model's; treat that as a sanity direction, not a promise about a real checkpoint, whose number must be measured on its own eval prompts.

## The wider families (read-only)

### Speculation beyond draft-target (sec. 5.2)

- **Draft-target (built above)** uses a separate smaller model sharing the vocabulary and tokenizer. Its cost is running and serving a second model; its acceptance rate is set by how well that model mimics the target.
- **Medusa-style heads** attach extra prediction heads to the target itself, proposing several future tokens from one target state. There is no second model to serve, but the heads must be trained and their proposals are less independent than a real draft run.
- **EAGLE-style drafting** drafts in the target's feature space rather than token space, reusing target representations to propose more accurately; again the proposer is trained with the target and verification remains the target's own forward.
- **n-gram speculation** proposes continuations copied from the prompt or a corpus (for example repeated spans in code or retrieval text). It costs no model at all and works exactly when the text literally repeats.

All four share the same verification contract built here: proposals are cheap guesses, the target disposes, and the emitted text stays the target's.

### Caching beyond one node (sec. 5.3)

- **KV placement.** Cached keys/values can live in device memory, host memory, or be recomputed; each tier trades capacity against the cost of bringing entries back. The block allocator in chapter 11 is the device-tier bookkeeping in miniature.
- **Cache-aware routing.** A shared prefix only helps if requests opening with it reach the machine holding its entries. Routers that hash the opening tokens can steer such requests together; a load-only router silently turns a warm cache cold by scattering them.
- **Long context.** As context grows, KV bytes (chapter 10's formula) dominate and prefix hits save proportionally more prefill — but capacity pressure rises too, which is why eviction policy and block-level sharing matter more there than the exact-prefix map built here.
- **Production prefix sharing** generalises this chapter's map into a tree over shared openings with the KV chopped into fixed blocks (compare `BlockAllocator` in [serving.py](../src/inference_from_scratch/serving.py)): sharing can stop at internal, block-aligned points rather than only where a caller once said `put`.

### Parallelism and disaggregation (sec. 5.4–5.5, read-only)

- **Tensor parallelism** splits individual matrices across devices and pays collective communication inside every block; **expert parallelism** instead places different MoE experts on different devices and routes tokens to them. Chapter 06's MoE pointer and the glossary carry the vocabulary; this repo's single-device model builds neither.
- **Disaggregation** runs prefill and decode on separate pools — each scaled for its own bound from chapter 13 — and transfers KV between them. The seam is already visible in [serving.py](../src/inference_from_scratch/serving.py): its scheduler admits, steps and retires requests (chapter 11), and `measure_prefill_decode` already reports the two phases separately. A disaggregated deployment inserts a KV handoff exactly at that phase boundary; nothing in the per-request maths changes.

## How the big systems do it

Serving engines combine all three levers at once: block-shared prefix caches with routing that respects them, speculative verification (often with a small sibling checkpoint as draft), and quantized weights whose quality was signed off on eval traffic before rollout. The ordering matters operationally: exactness-preserving reuse and speculation are switched on by measurement of speed alone, while a quantized checkpoint is a new model artefact that re-enters evaluation like any other model change.

## Common pitfalls

- **Mutating a stored cache.** KV objects are append targets. Storing or returning the live object (instead of clones) lets one request's decode rewrite another's prefix; the tests probe both directions.
- **Counting a hit you cannot use.** A prefix match only saves work if the caller also gets positions right: suffix forwards must start at `start_pos=matched_len`, or RoPE and learned positions silently shift.
- **Judging quantization by weight error.** Per-element weight RMSE is a mechanism metric. Ship/no-ship belongs to output metrics — KL, top-1 agreement, and task evals — on prompts resembling production.
- **Drafting with a mismatched tokenizer.** Proposal ids the target would never emit make acceptance collapse; speculation then costs draft forwards and gains nothing.
- **Expecting speculation to fix a memory-bound step by itself.** It amortises target reads over more tokens per forward, but each verification still reads the target weights; quantization and batching attack that term differently.

## Break it / exercises

1. Store a prefix, look it up, append to the returned cache, and look it up again. Which clone (on `put`, on `lookup`, or both) is load-bearing for the second lookup's correctness?
2. Set `gamma=1` and `gamma=8` with the draft equal to the target and predict `target_forwards` for 16 tokens before running. Then use a re-seeded draft and explain the direction `accepted` moves.
3. Fake-quantize only the MLP Linears of a tiny copy (skip attention) and compare `measure_quality_impact` against quantizing every Linear. What does the comparison suggest about where int8 error hurts this model?
4. A router sends identical system-prompt requests to two replicas alternately. Using the counters here, explain why measured `tokens_saved` halves even though the prompt never changed.

## Check yourself

<details>
<summary>1. Why can a suffix forward reuse a stored prefix without recomputing its keys and values?</summary>

A token's keys and values depend only on that token, its absolute position, and earlier tokens. Provided the suffix is forwarded with `start_pos` equal to the prefix length, its attention reads the stored entries exactly as a full prefill would have produced them.
</details>

<details>
<summary>2. A draft proposal disagrees with the target at the second of four proposed tokens. How many tokens does that verification forward finalise, and which are they?</summary>

Two: the first (agreeing) proposal, plus the target's own token at the disagreement position, which replaces the wrong proposal. The remaining proposals are discarded.
</details>

<details>
<summary>3. A quantized copy shows tiny weight RMSE but large mean KL on eval prompts. Which number governs the rollout decision, and why?</summary>

The KL (with top-1 agreement and task evals). Deployment cares about output distributions; weight error is only a proxy and can be amplified or absorbed by the layers between weights and logits.
</details>

## Summary in 30 seconds

Reuse exact prefixes at `put` boundaries, verify draft runs with one target forward, and measure any deliberate quality trade on outputs. If you remember one invariant, make it this: prefix caching and speculation must reproduce the unoptimised result exactly, while quantization must quote `{mean_kl, top1_agreement, logit_rmse}` rather than assert it is close enough.

## Where to go next

Continue to ch 15 (hardware), which prices the devices these techniques run on, or revisit ch 13 to recompute which bound each lever actually moves.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test files now: `.venv/bin/python -m pytest tests/test_prefix_cache.py tests/test_speculation.py tests/test_quantization.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
