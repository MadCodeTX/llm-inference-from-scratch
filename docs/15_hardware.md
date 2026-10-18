# 15 — Hardware: three budgets, not one speed number

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), Ch 3, including sec. 3.5 on local inference. **Production question:** given a device's arithmetic rate, memory bandwidth, and memory capacity, what model, context, and batch can it actually serve? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

A GPU is three separate budgets — operations per second, bytes per second, and bytes of capacity — and inference fails differently when each one runs out, so read all three off the device before choosing precision, context, or concurrency.

## Why inference needs this

Chapter 13 reduced a phase to a ratio: arithmetic intensity against a ridge point. That ratio needs two device numbers, and serving needs a third. Capacity decides whether weights plus KV cache plus activations fit at all. Bandwidth decides how fast decode can stream those weights and entries back. Arithmetic rate decides how fast prefill can chew a prompt once the data arrives. A card can be generous in one budget and tight in another; quoting a single headline figure hides which wall a workload will hit.

This chapter is read-only about products. It teaches how to interrogate whatever device is in front of you, using this repo's own helpers, and how the same three budgets explain both a single-GPU home server and an on-device deployment.

## The math/shapes

**The three budgets.** For a chosen precision:

```text
fit:      weight_bytes + kv_cache_bytes(...) + activation/workspace  <=  memory_bytes
prefill:  time  >=  flops / peak_flops          (usually the binding term)
decode:   time/token  >=  bytes_moved / memory_bandwidth_bytes_s   (usually the binding term)
ridge     =  peak_flops / memory_bandwidth_bytes_s        [ops/byte, ch 13]
```

Capacity uses [kv_cache_bytes](../src/inference_from_scratch/quantization.py) from chapter 10:

```text
kv_cache_bytes = 2 * L * B * H_kv * S * D_h * bytes_per_elem
```

On this repo's tiny config (`L=2, H_kv=2, D_h=16`) that is 128 bytes per token per sequence in fp16 — 131,072 bytes at `S=512, B=1`, and 8,388,608 bytes at `S=4096, B=8`. The formula, not the tiny result, is the point: on a real checkpoint read `L, H_kv, D_h` from its own `config.json` (chapter 09) and the same line prices any context and batch. Halving `bytes_per_elem` (fp16 to int8 KV) or halving `H_kv` (stronger grouping, chapter 04) halves the capacity and the decode traffic at once.

**Generations, read-only.** Successive GPU generations mostly move the three budgets by different amounts: arithmetic (especially low-precision tensor arithmetic) tends to grow fastest, bandwidth next, capacity slowest and in steps. The practical consequence is that a model that was bandwidth-bound stays bandwidth-bound but faster, while a capacity wall may not move at all between generations at the same memory tier. Never carry a ridge point across generations — recompute it from the exact card's sheet at the exact precision, as chapter 13 insists.

**Multi-GPU and MIG, read-only.** Splitting one model across devices (tensor or expert parallelism, chapter 14) pools capacity and can pool bandwidth, at the price of collective traffic between devices on every block. Several independent replicas instead pool throughput but not capacity per request. Hardware partitioning (MIG on supported datacenter cards) carves one device into isolated slices with fixed shares of all three budgets, trading peak single-request speed for predictable multi-tenant isolation. This repo builds none of these; the vocabulary matters because engine comparisons in chapter 16 assume it.

## What is actually in this machine

Measured here, not remembered:

```text
$ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NVIDIA GeForce RTX 4090, 24564 MiB
```

So this host has one RTX 4090 with 24,564 MiB (25,757,220,864 bytes) of device memory. [try_read_local_gpu](../src/inference_from_scratch/bottlenecks.py) reports exactly that name and capacity and deliberately leaves `peak_flops` and `memory_bandwidth_bytes_s` as `None`: peaks depend on precision and clock policy and must come from the vendor sheet for this exact card, not from memory. Note also that this repo's `.venv` carries a CPU build of PyTorch (`torch.cuda.is_available()` is `False` here), so the card is present on the host but is not the device these chapters benchmark on. Both facts are stated because conflating "present", "usable by this environment", and "measured" is how phantom GPU results get published.

## Local inference (sec. 3.5): two honest cases

**Desktop / home-server, single GPU (bigbox-style).** One card serves one model to a handful of clients. The serving shape is chapter 11's: continuous batching at modest concurrency, paged KV, and a capacity check that weights plus the planned KV working set fit with headroom for activations and fragmentation. The binding questions are capacity first (does the quantized checkpoint plus KV at the target context fit?), bandwidth second (what decode TPS follows?), arithmetic last (prefill is rarely the complaint at this scale). No product figures are needed: run the chapter-13 recipe with the real `config.json` and the card's own sheet.

**On-device, GGUF-style (Glimmer-style).** A phone or laptop-class deployment differs in degree, not kind. Memory is unified — one pool shared with the operating system and every other app — so the capacity budget is whatever the system leaves, and it can shrink under pressure. Bandwidth is far below a desktop card's, which pushes decode's ridge comparison further into memory-bound territory and makes weight compression close to mandatory: the int4/int8 byte savings of chapter 10 buy both fit and speed at once, at a quality cost that must be measured (chapter 14) rather than assumed. Context is usually capped hard, because `kv_cache_bytes` grows linearly in `S` against that small shared pool. Thermal and power limits act as a fourth, time-varying budget with no datacenter analogue: sustained TPS, not burst TPS, is the honest figure. The GGUF single-file format (chapters 09–10) exists for exactly this case — config, tokenizer, and block-quantized tensors in one artefact that a CPU/GPU runtime can stream.

In both cases the discipline is identical: no invented specifications. If a number did not come from `nvidia-smi`, a `config.json`, a vendor sheet for the exact SKU, or a measurement you ran, it does not go in the capacity plan.

## Walk the code

- [bottlenecks.py](../src/inference_from_scratch/bottlenecks.py): `HardwareSpec(name, peak_flops, memory_bandwidth_bytes_s, memory_bytes)` is the three budgets as a record, with `None` for honestly-unknown peaks. `ridge_point(spec)` refuses unknown peaks instead of guessing. `make_example_spec()` is labelled illustrative round numbers, not a product. `try_read_local_gpu()` shells to `nvidia-smi --query-gpu=name,memory.total` and returns only what that tool actually reports.
- [quantization.py](../src/inference_from_scratch/quantization.py): `kv_cache_bytes(config, seq_len, batch, bytes_per_elem)` prices capacity; `bytes_for_params` / `bytes_for_dtype` price weights at each precision.
- [metrics.py](../src/inference_from_scratch/metrics.py) and [scripts/benchmark.py](../scripts/benchmark.py): the measurement side — end-to-end wall time on whatever device actually ran, never a spec-sheet promise.

Tests: [test_bottlenecks.py](../tests/test_bottlenecks.py) pins the ridge refusal and intensity arithmetic; no test asserts a real product's numbers, by design.

## How the big systems do it

Fleet operators qualify each new device generation with the same three-budget worksheet, then let the serving engine (chapter 16) place prefill and decode pools accordingly — sometimes on different device types, since the phases stress different budgets (disaggregation, chapter 14). On-device runtimes instead spend their effort on compression and on streaming weights from storage, because their bandwidth and capacity budgets leave no other lever. The worksheet does not change; only which budget binds first changes.

## Common pitfalls

- **Planning from a borrowed spec table.** Peaks vary by SKU, precision, and clock policy. An intensity verdict is only as real as its inputs (chapter 13).
- **Checking FLOPs and forgetting capacity.** A model that does not fit with its KV working set has no TPS at all. Fit first, speed second.
- **Reporting burst speed on a thermally-limited device.** Sustained rates are the deployable number on local hardware.
- **Assuming "GPU present" means "GPU used".** Verify from inside the environment (`torch.cuda.is_available()`, device placement) — this repo's own venv is the counterexample.

## Break it / exercises

1. Using `kv_cache_bytes`, find the largest `B` at `S=4096` in fp16 whose KV alone stays under 1 GiB on the tiny config, then repeat in int8. Which lever — heads, bytes, or context — would you pull first on a real config, and why?
2. Run `try_read_local_gpu()` and explain, from its source, why it will not give you a ridge point. What two numbers must you fetch, and for what precision?
3. A colleague sizes a deployment using a previous generation's bandwidth with the new card's FLOPs. Which phase's prediction breaks, and in which direction?

## Check yourself

<details>
<summary>1. A checkpoint fits easily but decode is slow, and a larger batch barely helps. Which budget is binding?</summary>

Bandwidth. Decode streams weights plus KV per token (chapter 13); once batching has amortised the weight read, the per-sequence KV read remains, and only fewer bytes — compression or fewer KV heads — moves it.
</details>

<details>
<summary>2. Why does unified memory change the capacity calculation on-device?</summary>

The pool is shared with the OS and other apps and can shrink under pressure, so the safe KV-plus-weights budget is the memory the system reliably leaves available, not a nameplate total.
</details>

<details>
<summary>3. This machine reports an RTX 4090. Can this repo's test suite quote its GPU TPS?</summary>

No. The venv uses a CPU PyTorch build, so the card is present on the host but unused by these tests. Any GPU figure would need a CUDA-enabled environment and an actual run.
</details>

## Summary in 30 seconds

Capacity decides fit, bandwidth decides decode, arithmetic decides prefill. If you remember one formula, make it `kv_cache_bytes = 2·L·B·H_kv·S·D_h·bytes`; if you remember one code location, make it `HardwareSpec` in [bottlenecks.py](../src/inference_from_scratch/bottlenecks.py). Every device claim in this chapter traces to `nvidia-smi`, a `config.json`, a vendor sheet, or a stated `None`.

## Where to go next

Continue to ch 16 (software and engines), which stacks runtimes on top of these budgets. Run `.venv/bin/python -m pytest tests/test_bottlenecks.py -q` before moving on.
