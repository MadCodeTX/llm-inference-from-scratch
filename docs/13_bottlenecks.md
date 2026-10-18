# 13 — Bottlenecks: will arithmetic or data movement set the pace?

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.4 and 2.5, with measurement vocabulary from sec. 1.4. **Production question:** for this model shape, on this device, which ceiling — peak arithmetic or peak memory traffic — binds each phase, and by roughly how much? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Divide the arithmetic a phase performs by the bytes it must move; if that ratio sits below the device's balance point the phase waits on memory, and if it sits above, it waits on arithmetic — a prediction you can make from a `config.json` and a spec sheet before running anything.

## Why inference needs this

Prefill and decode run identical weights and produce wildly different speeds per token. The reason is reuse. In prefill each loaded weight serves many token rows at once, so arithmetic piles up per byte fetched. In decode each step loads the full weight set (plus the stored keys and values) to advance a single row, so traffic piles up per operation. Guessing which regime you are in leads to the classic wasted fix: a faster arithmetic kernel bolted onto a phase that was waiting on memory, or extra bandwidth chased for a phase already limited by arithmetic.

This chapter turns the qualitative labels from chapter 00 — compute-bound and memory-bound — into arithmetic you can check.

## The math/shapes

**Arithmetic intensity.** For any phase or kernel:

```text
intensity = flops / bytes_moved          [ops per byte]
```

**Ridge point.** A device advertises two peaks: arithmetic (`peak_flops`, operations per second) and traffic (`memory_bandwidth_bytes_s`, bytes per second). They meet at:

```text
ridge = peak_flops / memory_bandwidth_bytes_s     [ops per byte]
bound = compute  if intensity >= ridge  else  memory
time_per_token >= max(flops / peak_flops,  bytes / bandwidth)
```

The last line is the roofline floor: no implementation beats both ceilings at once, and real ones reach only a fraction of each.

**FLOPs per token — the 2×params rule plus attention.** A weight matrix touched once per token costs one multiply and one add per element, so matmuls cost `2 * P_matmul` per token. Which parameters count, and what is assumed:

- Counted: Q/K/V/O projections, MLP matrices, and the LM head, per layer where applicable. The head counts even when tied, because the head multiplication still runs.
- Not counted as matmul: the embedding lookup (a row gather), norms, rotary rotations, softmax, sampling. These are elementwise or small at the scales of interest — an assumption, stated plainly. On this repo's tiny CPU model their host overhead can dominate *measured* time; the estimator predicts a floor, not that measurement (see *Walk the code* on end-to-end versus on-device time).
- Attention's own score/value work is added: each of the score product and the weighted value sum costs `2 * d_model * S` per layer per token at context `S`. Decode, where one query meets `S` stored keys, adds `4 * L * d_model * S`. Prefill averages over the causal triangle — position `i` sees `i + 1` keys, mean `(S + 1) / 2` — and adds `2 * L * d_model * (S + 1)`.

**Bytes per token — the honest asymmetry.**

```text
weight_total = total_params * weight_bytes        (2 for fp16/bf16)

decode:   bytes/token = weight_total / batch  +  2*L*H_kv*S*head_dim*kv_bytes
prefill:  bytes/token = weight_total / (batch * S)  +  2*L*H_kv*head_dim*kv_bytes
```

Decode re-reads every weight each step; batching shares that single read across the batch's tokens, while each sequence's key/value read stays its own. Prefill reads the weights once for the whole prompt, so the per-token share divides by prompt length, and its KV term is the one-time *write* of each token's own keys/values. Activation traffic is treated as staying on-chip — so every byte count here is a floor and every intensity an optimistic estimate.

## Worked: decode versus prefill on the tiny config

The benchmark configuration used across this repo: `vocab_size=64, d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128`, tied embeddings, fp16 storage, context `S = 128`, batch 1. Every number below is produced live by [bottlenecks.py](../src/inference_from_scratch/bottlenecks.py) (the tests recompute them):

| Quantity | Prefill | Decode |
|---|---|---|
| Matmul params `P_matmul` | 77,824 | 77,824 |
| Matmul FLOPs/token (`2P`) | 155,648 | 155,648 |
| Attention FLOPs/token | 33,024 | 65,536 |
| Total FLOPs/token | 188,672 | 221,184 |
| Weight bytes/token | 1,221 | 156,288 |
| KV bytes/token | 256 (write) | 32,768 (read) |
| Total bytes/token | 1,477 | 189,056 |
| Intensity (ops/byte) | **127.7** | **1.17** |
| Roofline floor/token (illustrative device) | 1.89 ns | 189 ns |

The illustrative device is [make_example_spec](../src/inference_from_scratch/bottlenecks.py) — round invented numbers (`100 TFLOP/s`, `1 TB/s`, ridge **100 ops/byte**) chosen so the arithmetic is checkable by hand, explicitly *not* a real product. Against that ridge: prefill at 127.7 lands just on the compute side; decode at 1.17 sits two orders of magnitude on the memory side. The FLOPs are nearly identical — the verdict flips purely on bytes. Batching helps decode (intensity at batch 8 is ≈4.2, at 32 ≈5.9) but cannot rescue it here: as the weight read is shared away, the per-sequence KV read remains, capping intensity near `221,184 / 32,768 ≈ 6.7`. That ceiling is why shrinking the cache — fewer KV heads, fewer KV bytes — attacks decode in a way batching alone cannot.

## Walk the code

All of this chapter lives in two small modules, each mirrored by a test file.

- [bottlenecks.py](../src/inference_from_scratch/bottlenecks.py): `HardwareSpec(name, peak_flops, memory_bandwidth_bytes_s, memory_bytes)` holds spec-sheet peaks, with `None` allowed where a value is genuinely unknown. `ridge_point(spec)` divides the peaks and *refuses* unknown ones rather than guessing. `count_params(config)` reproduces `DecoderLM.num_params()` analytically — the tests assert exact agreement across tied/untied, SwiGLU/GELU and position variants, so the estimator can price a `config.json` without building the model. `estimate_flops_per_token(config, seq_len, phase)` and `estimate_bytes_per_token(config, seq_len, phase, weight_bytes=2, kv_bytes=2, batch=1)` implement the formulae above with their assumptions in the docstrings. `arithmetic_intensity`, `classify(...) -> {intensity, ridge, bound, flops, bytes, ...}` and `roofline_time_lower_bound(...)` (accepting either raw flops/bytes or a config form) finish the chain.
- No fabricated device table exists, by design. `make_example_spec()` is labelled illustrative in its name and docstring. `try_read_local_gpu()` shells out to `nvidia-smi --query-gpu=name,memory.total` when present and returns only name and capacity, leaving both peaks `None` with instructions to fill them from the vendor sheet for the exact card and precision — tensor-core half-precision peaks and desktop clocks are per-product facts this repo will not invent.
- [metrics.py](../src/inference_from_scratch/metrics.py): `percentile(values, p)` uses linear interpolation between closest ranks over `p` in `[0, 100]` and validates empty/out-of-range input loudly; `summarize(values)` returns `{n, mean, p50, p90, p99, min, max}`. `measure_generate(model, idx, max_new_tokens, **kw)` re-runs the prefill-then-decode loop under `time.perf_counter` brackets — without changing `generate()`'s interface — and returns `{ttft_s, decode_s, tps, tokens, ...}`. Its docstring draws the sec. 1.4 distinction explicitly: these are end-to-end wall times including Python and sampling overhead, not on-device kernel times from a profiler, and on tiny CPU models the host share dominates by design honesty.
- [scripts/benchmark.py](../scripts/benchmark.py) now prints p50/p90/p99 for TTFT and TPS over repeated `measure_generate` runs, alongside the older prefill table and cache-versus-no-cache decode comparison.

Tests: [test_bottlenecks.py](../tests/test_bottlenecks.py) pins decode intensity far below prefill at batch 1, intensity growth with batch, a bound flip across a synthetic ridge, the roofline as the maximum of its two terms, and parameter-count agreement with the real model. [test_metrics.py](../tests/test_metrics.py) pins known percentile values, summary shape, and a tiny-model measurement with positive TTFT and TPS.

## Applying it to a real Qwen config and a real GPU sheet

The recipe transfers verbatim — only the inputs stop being tiny:

1. Translate the checkpoint's own `config.json` with `load_hf_config` (chapter 09): `hidden_size → d_model`, `num_hidden_layers → n_layer`, `num_attention_heads → n_head`, `num_key_value_heads → n_kv_head`, `intermediate_size → d_ff`, `vocab_size → vocab_size`. Never substitute remembered dimensions.
2. Pick the context `S` you actually serve and the precision you actually store (`weight_bytes`, `kv_bytes` from chapter 10's table — halving weight bytes halves decode's dominant traffic term directly).
3. From the device vendor's sheet for the exact SKU, take tensor-core peak FLOPs *at that precision* and peak memory bytes/s; build `HardwareSpec` with those two numbers plus capacity. If only name/capacity are discoverable locally, `try_read_local_gpu()` gives exactly those and leaves the peaks for this step.
4. Call `classify(config, spec, S, "prefill")` and `classify(config, spec, S, "decode", batch=B)` for the batch you plan to run. Read `bound` for the verdict and `roofline_time_lower_bound` for the floor; check `weight bytes + kv_cache_bytes(...) ≤ memory_bytes` for fit before believing either.
5. Only then benchmark (`measure_generate`, `scripts/benchmark.py`) and compare measured TPS against the floor. A large gap on the predicted side means implementation overhead; a measurement *beating* the floor means one of the stated assumptions was violated — find which before trusting either number.

## How the big systems do it

**Serving engines generally** run this same calculation per kernel with a profiler: achieved operations and bytes replace the spec-sheet peaks, and the ridge comparison decides whether a kernel gets arithmetic work (fusion that raises reuse) or traffic work (compression, fewer cache bytes, better locality). The phase split survives intact — prefill kernels are tuned toward the compute ceiling, decode kernels toward the bandwidth ceiling, and batching is valued precisely because it slides decode's intensity rightward along the roofline.

**On-device deployments** face the same chart with a far lower bandwidth ceiling and a modest arithmetic one, which is why aggressive weight compression is near-mandatory there: it cuts the byte term that binds decode, at a quality cost that must be measured separately (chapter 10), while prefill remains comparatively comfortable.

## Common pitfalls

- **Pricing decode with prefill intuition.** Similar FLOPs, opposite bounds. Always estimate both phases.
- **Trusting a borrowed device table.** Peaks vary by SKU, precision and clock policy. An intensity verdict is only as real as its two spec-sheet inputs.
- **Comparing wall time to kernel time.** End-to-end TPS (host overhead included) and profiler TPS answer different questions; mixing them manufactures phantom wins and losses.
- **Reporting a mean where a tail matters.** A p50 TTFT can look healthy while p99 violates the product's promise. Percentiles are the reporting unit, not decoration.

## Break it / exercises

1. Double `n_kv_head` on the tiny config (GQA toward MHA) and recompute decode intensity at `S = 512`. Which term moved, and why does prefill barely notice?
2. At what batch does decode on the illustrative device cross the ridge for `S = 16`? Derive it from the formulae before calling `classify`.
3. Price the same tiny model in int8 weights (`weight_bytes=1`) and int4 (`weight_bytes=1` is wrong — work out why sub-byte storage needs the byte accounting of chapter 10 rather than a fractional argument here). What breaks in this module's simple interface, and what is the honest fix?
4. Run `scripts/benchmark.py` and compare measured decode TPS with the roofline floor. Name the assumption most responsible for the gap on this hardware.

## Check yourself

<details>
<summary>1. Two phases have nearly equal FLOPs per token, yet one is compute-bound and the other memory-bound. What actually differs?</summary>

The bytes moved per token. Prefill amortizes one weight read over the whole prompt; decode pays a full weight read (plus the stored keys/values) per token. Intensity — flops divided by bytes, not flops alone — decides the bound.
</details>

<details>
<summary>2. Why does batching stop helping decode at long context?</summary>

Batching shares only the weight read. Each sequence's KV read is private and grows with context length, so once weight traffic is amortized away the KV term caps intensity regardless of batch.
</details>

<details>
<summary>3. A colleague reports a decode time below your roofline floor. What do you conclude?</summary>

One of the inputs or assumptions is wrong — miscounted parameters, a borrowed peak for the wrong precision, bytes silently staying on-chip, or a wall/kernel time mix-up. The floor is a consistency check; beating it indicts the bookkeeping, not physics.
</details>

## Summary in 30 seconds

Intensity against ridge, per phase, from the model's own config and the device's own sheet. If you remember one formula, make it `bound = compute iff flops/bytes ≥ peak_flops/bandwidth`; if you remember one code location, make it `classify` in [bottlenecks.py](../src/inference_from_scratch/bottlenecks.py). Everything else — the worked tiny numbers, the batching ceiling, the Qwen recipe — is that comparison with honest inputs.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 14 (prefix caching and speculation), which attacks the memory-bound decode this chapter diagnoses from two different directions. If a concept from an earlier chapter felt shaky here, revisit it now: this curriculum is cumulative, and the byte formulae of chapters 05 and 10 are load-bearing throughout.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test files now: `.venv/bin/python -m pytest tests/test_bottlenecks.py tests/test_metrics.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.
