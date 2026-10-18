# 16 — Software and engines: where this repo sits in the stack

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), Ch 4, including sec. 4.2.4 on reference implementations and sec. 4.5 on benchmarking and profiling. **Production question:** which layer owns which optimisation, and how does a hand-built teaching engine compare honestly with the systems that serve real traffic? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Inference software is a stack — fused kernels at the bottom, a tensor framework above them, file formats and reference implementations defining correctness, and serving engines at the top — and this repo deliberately occupies the reference layer so the layers above it can be judged.

## Why inference needs this

When a deployment is slow, the fix lives in exactly one layer: a kernel that moves too many bytes, a framework dispatch that launches too many kernels, weights stored in a format that forces conversion on load, or a scheduler that leaves the device idle between requests. Working only at the top layer makes every problem look like a scheduling problem. Having built the bottom mechanisms by hand (attention, KV cache, sampling), the reader can place each production system by which layers it replaces and which it leaves alone.

## The stack, bottom to top

**CUDA kernels and fusion (sec. 4.1).** The device executes kernels; every tensor round-trip to main memory costs bandwidth (chapter 15). Fusion merges elementwise chains — norm, residual add, rotary rotation, gated activation — so intermediate tensors cross memory once instead of per operation. FlashAttention (chapter 12) is the flagship example: identical arithmetic, radically less traffic. This repo's seams are named in chapter 12: manual attention in [attention.py](../src/inference_from_scratch/attention.py), norms and RoPE in [layers.py](../src/inference_from_scratch/layers.py), the decode loop in [generate.py](../src/inference_from_scratch/generate.py).

**PyTorch as the substrate.** This repo runs on PyTorch tensors and `nn.Linear`, which is also what most engines embed or export from. The framework gives portable operators and autograd-free inference dispatch; engines then bypass its per-operator launches on hot paths with fused kernels, compiled graphs, or captured CUDA graphs. Knowing the framework path matters because it is the fallback every engine is compared against — and, on this repo's tiny CPU models, the honest measured baseline.

**File formats (sec. 4.2.2).** Chapter 09 built safetensors by hand: a length-prefixed JSON header plus raw tensor bytes, no executable content. That format is the interchange layer — checkpoints ship as `config.json` plus safetensors shards. GGUF (chapters 09–10) is the on-device counterpart: one self-describing file bundling config, tokenizer, and usually block-quantized tensors for runtimes that stream from storage. An engine's loader is only ever translating one such container into device tensors; nothing about the model changes.

**Reference implementations (sec. 4.2.4) — this repo's stance.** A reference implementation is the slow, obviously-correct version against which fast paths are verified. That is exactly the role this repo assigns itself, following the stance the book describes for framework reference code: Hugging Face `transformers` model code plays that role in the ecosystem — readable, framework-native, the definition a kernel or engine port is checked against — and chapters 09 and 12 adopt the same discipline locally. SDPA is the test oracle for manual attention; the no-cache loop is the oracle for cached generation; `validate_state_dict` is the pre-flight oracle for weight loading. The rule for adopting any fast path: keep the reference, assert agreement, then optimise.

**Engines and Dynamo (sec. 4.3–4.4).** Above the kernels sit serving engines that own scheduling, KV memory, and request lifecycle (chapter 11). Dynamo-style systems sit one level higher still, coordinating many engine instances — including splitting prefill and decode pools (disaggregation, chapter 14). This repo builds neither a fleet nor a router; chapter 17 covers what that absence costs and when it starts to matter.

## Engines compared

Capability labels are about design centre, stated plainly. "Ours" means this repo as it stands today.

| System | Batching | Paging | Prefix cache | Speculation | Parallelism | Disaggregation |
|--------|----------|--------|--------------|-------------|-------------|----------------|
| This repo | Toy: iteration-level scheduler in [serving.py](../src/inference_from_scratch/serving.py), model-independent sampler | Toy: `BlockAllocator` accounting only; attention still reads contiguous tensors | Built, exact-prefix: [prefix_cache.py](../src/inference_from_scratch/prefix_cache.py) (ch 14) | Built, greedy draft-target: [speculation.py](../src/inference_from_scratch/speculation.py) (ch 14) | Absent: single-device only | Absent: phase split measured, never separated |
| vLLM | Continuous batching, chunked prefill | PagedAttention: block tables inside the attention kernel | Block-shared prefix caching | Draft-model and n-gram options | Tensor/pipeline parallelism | Supported via disaggregated serving modes |
| SGLang | Continuous batching with a radix-structured KV tree | Paged allocation over the same tree | Radix-tree prefix reuse as a core design point | Speculative decoding support | Tensor parallelism | Supported in newer deployments |
| TensorRT-LLM | In-flight (continuous) batching | Paged KV in compiled engines | KV reuse options | Speculative decoding support | Tensor/pipeline parallelism, heavily compiled per model | Primarily single-deployment; scaling via replicas |
| Dynamo | Orchestrates engines rather than batching itself | Inherits the engines it coordinates | Cache-aware routing across workers | Inherits engine support | Fleet-level placement | Its central use case: separate prefill/decode pools |

Read the first row as the chapter's honesty contract: the scheduler and allocator are dissections (the sampler is injected; the allocator never touches a kernel), prefix caching and speculation are real but exact-form teaching versions, and parallelism and disaggregation are absent by design. Nothing in this repo should be benchmarked against an engine's throughput; it should be used to understand what the engine's numbers are made of.

## Benchmarking versus profiling (sec. 4.5)

These are different instruments answering different questions, and chapter 13 already separated their outputs:

- **Benchmarking** measures the whole path end-to-end: TTFT, TPOT/TPS, percentiles. [metrics.py](../src/inference_from_scratch/metrics.py) `measure_generate` brackets the real generation loop with wall-clock timers, host overhead included, and [scripts/benchmark.py](../scripts/benchmark.py) reports p50/p90/p99 over repeated runs. A benchmark answers "what does a caller experience?" and is the only honest source for a latency claim.
- **Profiling** decomposes a run per kernel: achieved FLOPs/s, bytes/s, launch gaps. A profiler answers "which layer is responsible?" and its on-device times will always look better than wall time for the same work, because host gaps are excluded. Chapter 13's roofline floor is the bridge: compare measured wall time against the floor to size the overhead, then profile to locate it.

The failure mode to avoid is quoting profiler TPS beside benchmark TPS as if they were one quantity (sec. 1.4's end-to-end versus on-device distinction, carried through this repo since chapter 00).

## Walk the code

- Formats and loading: [weights.py](../src/inference_from_scratch/weights.py) — hand-parsed safetensors headers, HF `config.json` translation, name mapping, validation (ch 09).
- Reference discipline: [test_attention.py](../tests/test_attention.py) (SDPA as oracle), [generate.py](../src/inference_from_scratch/generate.py) (cached loop checked against no-cache), chapter 12's seam list.
- Engine-in-miniature: [serving.py](../src/inference_from_scratch/serving.py) — `ContinuousBatcher`, `BlockAllocator`, `measure_prefill_decode` (ch 11).
- Measurement: [metrics.py](../src/inference_from_scratch/metrics.py), [bottlenecks.py](../src/inference_from_scratch/bottlenecks.py), [scripts/benchmark.py](../scripts/benchmark.py).

## How the big systems do it

Production stacks assemble all layers at once: a checkpoint in safetensors is compiled or loaded by an engine whose fused kernels run under a continuous-batching scheduler, whose fleet behaviour may be coordinated across prefill and decode pools, and whose rollout of any change is gated by benchmarking against the reference model's outputs (quality measurement, chapter 14). Dynamo's contribution is coordination — routing and scaling those pools — not a new kernel. When a vendor number surprises you, ask which layer produced it; the stack tells you which questions to ask next.

## Common pitfalls

- **Benchmarking against the teaching engine.** This repo optimises for legibility. Its TPS measures overhead visibility, not competitiveness.
- **Deleting the reference after adopting a fast path.** Then numeric drift has no oracle (chapter 12).
- **Confusing a format with an engine.** Safetensors and GGUF describe bytes on disk; they imply nothing about scheduling quality.
- **Treating orchestration as acceleration.** Coordination layers place work; kernels and schedulers do it. Both matter, differently.

## Break it / exercises

1. Pick one row of the engine table and trace it to this repo: for vLLM's paging, name the two repo components (allocator, cache layout) whose fusion into a kernel is the actual production step.
2. Run `scripts/benchmark.py`, then explain in one paragraph why its TPS cannot be compared with a profiler trace of the same run. Which of the two would you quote to a user, and which to a kernel engineer?
3. Suppose a fused kernel disagrees with manual attention only at long context. Using chapter 12's discipline, write the exact assertion you would add before adopting it.

## Check yourself

<details>
<summary>1. An engine upgrade doubles throughput with no kernel changes. Which layers could plausibly be responsible?</summary>

Scheduling and memory layers: better batching admission, paged KV reducing fragmentation stalls, or prefix reuse cutting prefill. Kernels unchanged means per-forward cost is unchanged, so the win came from doing fewer or better-packed forwards.
</details>

<details>
<summary>2. What makes this repo a reference implementation rather than a slow engine?</summary>

Its outputs are the oracle: fast paths (SDPA, cached generation, loaded checkpoints) are asserted against its manual paths in tests. Engines are judged for speed; references are kept for truth.
</details>

<details>
<summary>3. Why will a profiler always report a friendlier TPS than measure_generate for identical work?</summary>

The profiler sums on-device kernel spans and excludes host dispatch, sampling, and Python gaps that wall-clock measurement includes. Different quantities, both useful, never interchangeable.
</details>

## Summary in 30 seconds

Kernels move bytes, frameworks dispatch, formats describe, references define correctness, engines schedule. If you remember one contrast, make it benchmark (wall time, user-visible) versus profile (kernel time, layer-locating); if you remember one code location, make it the oracle pattern in [test_attention.py](../tests/test_attention.py).

## Where to go next

Continue to ch 17 (production), which asks what wraps around an engine once real callers arrive. Run `.venv/bin/python -m pytest -q` before moving on.
