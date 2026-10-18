# 08 — Sampling and generation: from logits to a token stream

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2 (autoregressive mechanics). **Production question:** how does a distribution over the vocabulary become a token stream, and which per-request knobs control it without touching the model? **Online relevance:** this loop sets decode TPS, the rate users watch stream in. See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Generation is a loop — take the last position's logits, turn them into one next token by a *sampling strategy* (greedy, temperature, top-k, top-p), append it, repeat — run once as a parallel prefill and then as single-token decode steps.

## Why inference needs this

The model outputs a distribution, not an answer. *How* you pick from that distribution is a separate, model-independent policy: greedy is deterministic but repetitive; sampling trades determinism for diversity. Serving systems expose exactly these knobs per request, applied to identical logits — which is why sampling lives in its own module of pure functions here.

Separating sampling from the model also gives inference its sharpest correctness test: under greedy decoding (no randomness anywhere), the cached and uncached generation paths must emit **identical** tokens. That equivalence audits the KV cache, the causal mask, and all position bookkeeping at once.

## The math/shapes

Logits `[B, V]` → next ids `[B, 1]`. The pipeline (order matters):

1. `temperature == 0` → greedy `argmax`. (`temperature < 0` is a `ValueError`.)
2. Divide logits by `temperature`.
3. Top-k filter, then top-p filter (each optional): removed logits become `-inf`.
4. Softmax → probabilities; draw one id per row (`torch.multinomial`, optionally seeded).

**Temperature, concretely.** Logits `[2, 1, 0]`:

| T | Scaled logits | Softmax | Effect |
|---|---------------|---------|--------|
| 0.5 | `[4, 2, 0]` | `[0.867, 0.117, 0.016]` | sharper — the leader dominates |
| 1.0 | `[2, 1, 0]` | `[0.665, 0.245, 0.090]` | the model's own distribution |
| 2.0 | `[1, 0.5, 0]` | `[0.506, 0.307, 0.186]` | flatter — tail tokens gain |

Division happens *before* softmax, so temperature truly rescales the distribution rather than tweaking it afterwards.

**Top-k and top-p.** Top-k keeps the `k` highest logits (`k=1` ≡ greedy; `k ≥ V` is a no-op). Top-p (nucleus) keeps the *smallest* set of tokens whose cumulative probability reaches `p`, always including the argmax token. Top-p adapts: a peaked distribution keeps few tokens, a flat one keeps many — the property fixed-k lacks. Filters never renormalize; they set logits to `-inf` so softmax assigns exactly 0, keeping filter/sampler composition clean.

**The generation loop, shape-wise.** Prompt `[B, T_p]`. Prefill: one forward `[B, T_p] → logits [B, T_p, V]`, sample from `logits[:, -1]`, append → `[B, T_p+1]`. Decode × n: forward `[B, 1]` at `start_pos = current_len - 1`, sample, append. Stop when every row has emitted `eos_id` (the EOS token stays in the output) or `max_new_tokens` is reached.

## Walk the code

Sampling: [sampling.py](../src/inference_from_scratch/sampling.py). Generation: [generate.py](../src/inference_from_scratch/generate.py).

- `greedy(logits)` (`sampling.py:36`): `argmax(dim=-1, keepdim=True)` → `[B, 1]`. The rank-2 result always concatenates cleanly onto `[B, T]` ids.
- `apply_temperature(logits, T)` (`sampling.py:48`): returns `logits / T`; raises `ValueError` for `T <= 0` — the "0 means greedy" policy lives only in `sample_next`, keeping one validation point.
- `top_k_filter(logits, k)` (`sampling.py:68`): thresholds at the k-th largest value (`torch.topk(...).values[..., -1]`) and `where`s the rest to `-inf`. Returns a new tensor; the caller's logits are never mutated.
- `top_p_filter(logits, p)` (`sampling.py:88`): sorts descending, softmaxes, cumulative-sums, and removes tokens *after* the cumulative mass crosses `p` — implemented by shifting the removal mask right by one so the crossing token survives (`sampling.py:115`), then scattering back to vocab order. Validates `p ∈ (0, 1]`.
- `sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None)` (`sampling.py:124`): the composed pipeline above; a bare `[V]` input is treated as `B=1` and still returns `[1, 1]`. `generator` makes seeded runs exactly reproducible.
- `generate(model, idx, max_new_tokens, ..., eos_id=None, use_cache=True, generator=None)` (`generate.py:91`): validates inputs, forces `model.eval()` + `no_grad`, and dispatches. `_generate_cached` (`generate.py:140`) creates one `KVCache(n_layer=...)`, runs the **PREFILL** block (whole prompt, `start_pos=0`), then the **DECODE** loop feeding only the just-sampled token with `start_pos = out.shape[1] - 1` (`generate.py:176`). `_generate_no_cache` (`generate.py:190`) re-runs the entire growing sequence every step (`start_pos=0`, no cache) — deliberately wasteful, existing purely as ground truth. EOS handling tracks a per-row `finished` mask and stops when all rows finished (`generate.py:183`).

Tests: [test_sampling.py](../tests/test_sampling.py) (temperature extremes, top-k=1 ≡ greedy, top-p keeps the crossing token, seeded reproducibility) and [test_model_generate.py](../tests/test_model_generate.py) (cache ≡ no-cache under greedy across MHA/GQA/MQA, EOS stop, `max_new_tokens` 0/1/5).

## How the big systems do it

**Qwen-family serving stacks** expose this exact knob set per request — `temperature`, `top_p`, `top_k`, `max_tokens`, stop/EOS ids — plus penalties (repetition/frequency/presence) that adjust logits before filtering. The model is untouched; only the sampler configuration travels with the request.

**vLLM-style serving** fuses sampling on-GPU over thousands of concurrent rows, each with its own temperature and filters, and layers speculative-decoding rejection sampling on top (chapter 12). Crucially, finished rows retire individually instead of waiting for the whole batch — the `finished`-mask idea in `generate.py`, promoted to a scheduler (chapter 11). Production serving deliberately does *not* seed per request; reproducibility there comes from logging, not RNG control.

**llama.cpp / GGUF (Glimmer-style)** implements the same sampler chain on-device (its `--temp`, `--top-k`, `--top-p` flags are `sample_next`'s parameters), plus grammar-constrained sampling that masks logits to a formal language — a vivid reminder that "filter logits, then sample" composes with almost any policy.

## Worked: one greedy generation, step by step

Prompt ids `[104, 105]` (`"hi"`, bytes), `max_new_tokens=3`, `temperature=0`, cached path:

| Step | Input `[1, T]` | `start_pos` | What is sampled |
|------|---------------|-------------|-----------------|
| Prefill | `[104, 105]` | 0 | `argmax(logits[:, -1])` → say 33 |
| Decode 1 | `[33]` | 2 | next argmax → say 119 |
| Decode 2 | `[119]` | 3 | next argmax → say 33 |

Output: `[104, 105, 33, 119, 33]`, shape `[1, 5]`. The no-cache path reaches the same list by re-running `[104, 105]`, then `[104, 105, 33]`, then `[104, 105, 33, 119]` in full — same last-position argmax each time. That equality, across MHA/GQA/MQA and both position variants, is exactly what `test_model_generate.py` asserts; it is the strongest cheap audit of ch 02/04/05 bookkeeping combined.

With `temperature=1.0` instead, each step draws from the (optionally top-k/top-p filtered) softmax, and two runs differ unless they share a seeded `generator`. Note EOS semantics: if step 2 had produced `eos_id`, generation still returns it in the output and stops only once *every* batch row has finished — callers strip the EOS themselves.

## Common pitfalls

- **Sampling from the wrong position.** `logits[:, -1]` — not `logits[:, 0]`, not a mean over positions. An off-by-one here generates fluent nonsense that no sampler tuning will fix.
- **Filtering before temperature.** The pipeline order (temperature → top-k → top-p → softmax) is deliberate: top-p's "cumulative probability ≥ p" must be judged on the *tempered* distribution. Reordering changes which tokens survive.
- **Mutating the model's logits.** Filters return new tensors. Writing `-inf` in place would poison the caller's logits for any second sampler or logging pass.

## Break it / exercises

1. Sample from logits `[2, 1, 0]` 10,000 times at T=0.5, 1.0, 2.0 with a seeded generator. Compare empirical frequencies to the table above.
2. Make `_generate_cached` pass `start_pos=0` on decode steps and run the greedy cache≡no-cache test. Explain why prefill still passes but decode diverges.
3. Set `top_p=0.01` on a peaked distribution and on a near-uniform one. Count surviving tokens in each case (`(filtered > -inf).sum()`) and state the adaptive property in your own words.

## Check yourself

<details>
<summary>1. Why must cache and no-cache agree exactly under greedy, but not under T=1 sampling?</summary>

Greedy has no randomness: both paths compute the same logits and take the same argmax, so any divergence is a real bug. Sampling draws random ids; two paths can legitimately differ unless they share a seeded generator and consume it identically.
</details>

<details>
<summary>2. Why do filters write -inf instead of zeroing probabilities?</summary>

So filtering composes *before* softmax in logit space: temperature → top-k → top-p → one softmax. Zeroing probabilities would require renormalizing between filters and breaks that pipeline; `-inf` softmaxes to exactly 0.
</details>

<details>
<summary>3. In <code>_generate_cached</code>, why is decode's <code>start_pos = out.shape[1] - 1</code> and not the prompt length?</summary>

`start_pos` is the absolute position of the token being fed — always the newest token, which sits at the end of the sequence built so far. It grows by one each step; the prompt length is only correct for the first decode step.
</details>

## Summary in 30 seconds

This chapter's core is prefill-then-decode loop and the temperature/top-k/top-p pipeline. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 09 (weights), which loads real parameters into the model driven here. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_sampling.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
