# Roadmap — full curriculum

Grouped by the categories of Kiely, *Inference Engineering* (2026):
Prerequisites, Models, Hardware, Software, Techniques, Modalities,
Production. Framing follows Kiely, *Inference Engineering* (2026);
see [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md) for the full cross-reference.
All chapter text and code here are original.

Status words are used strictly. **Built**: code plus tests exist.
**Read-only**: dissection only, no integration by design. **Planned**:
mapped here and in the textbook map; no chapter file or code yet.

## Prerequisites (book Ch 0–1)

Why run your own inference at all, what kind of application is asking,
and how latency and throughput get measured before any model runs.

### 00 — Inference vs training; prefill/decode — Built

- **Book framing:** Ch 0; sec. 1.2 (about your app), sec. 1.4
  (measuring latency/throughput).
- **Goal:** Prefill (parallel, compute-bound, `T>1`) versus decode
  (serial, `T=1`, memory-bound); TTFT, TPOT, TPS, and percentiles as
  the measurement language for everything after.
- **Builds on:** —
- **Qwen-class / local relevance:** A large Qwen-class decode step on
  bigbox reads all weights per token; an on-device Glimmer-style
  deployment pays the same cost from a far smaller bandwidth budget.

## Models (book Ch 2)

The Tiny-to-Qwen ladder lives here. One codebase climbs it: only
`ModelConfig` and the weights change.

1. **Tiny** — 2-layer, `d_model<=64` decoder, random weights, byte
   tokens; every tensor shape printable in a test.
2. **Small** — same code, character/BPE tokens, longer contexts;
   cache versus no-cache greedy equivalence.
3. **Real weights** — a Hugging Face `config.json` plus safetensors
   checkpoint (Llama/Qwen family) loaded into the same `DecoderLM`.
4. **Qwen-class / Glimmer** — a large model's real `config.json` read
   from disk at load time; its numbers are never hard-coded here.

### 01 — Tokenizer / BPE — Built

- **Book framing:** sec. 2.2, treated lightly in the book; this repo
  goes deeper (see the textbook map's gap note).
- **Goal:** Byte, char, and hand-trained BPE tokenizers; roundtrip
  losslessness; vocab size versus sequence length.
- **Builds on:** 00

### 02 — Embeddings + RoPE — Built

- **Book framing:** sec. 2.2.1–2.2.2.
- **Goal:** Token embeddings, tied embeddings, rotate-half RoPE,
  precomputed cos/sin tables; `theta` and `head_dim` from config.
- **Builds on:** 01

### 03 — Norms — Built

- **Book framing:** sec. 2.2.2.
- **Goal:** RMSNorm versus LayerNorm; pre-norm residual stream; eps
  and float32-reduction stability.
- **Builds on:** 02

### 04 — Attention: MHA / GQA — Built

- **Book framing:** sec. 2.2.3, 2.5.
- **Goal:** Hand-built scaled dot-product attention, causal masks for
  prefill and decode, grouped-query attention via `repeat_kv`.
- **Builds on:** 02, 03

### 05 — KV cache — Built

- **Book framing:** sec. 2.5, 5.3.
- **Goal:** Concat versus preallocated cache; `start_pos` bookkeeping;
  cache/no-cache greedy equivalence; KV bytes formula.
- **Builds on:** 04

### 06 — MLP: SwiGLU — Built

- **Book framing:** sec. 2.1.2, 2.2.2; MoE pointer at sec. 2.2.4.
- **Goal:** Gated SwiGLU versus GELU MLP; `intermediate_size`/`d_ff`;
  where the parameter majority lives.
- **Builds on:** 03

### 07 — Full model — Built

- **Book framing:** sec. 2.2.1, `config.json`-driven.
- **Goal:** `TransformerBlock` plus `DecoderLM`, tied embeddings,
  `num_params()`, learned-position variant.
- **Builds on:** 04, 05, 06

### 08 — Sampling + generation — Built

- **Book framing:** sec. 2.2, autoregressive mechanics.
- **Goal:** Greedy, temperature, top-k, top-p; the explicit
  prefill-then-decode loop; EOS; seeded generators.
- **Builds on:** 07

### 19 — Real weights: Qwen2.5-0.5B proved against HF — Built

- **Book framing:** sec. 2.2.1 (`config.json` as source of truth), sec. 4.2.2
  (model file formats), sec. 4.2.4 (reference implementations).
- **Goal:** Download Qwen2.5-0.5B-Instruct, load it into `DecoderLM` with
  zero validation errors (QKV bias gap found and fixed), and prove greedy
  equivalence with the HF reference on identical raw ids (logits, per-step
  top-1, exact generated ids) on CPU fp32; CPU TTFT/TPS + roofline.
- **Builds on:** 07, 08, 09, 13

### 20 — GGUF formats: safetensors vs GGUF vs pickle — Built

- **Book framing:** sec. 4.2.2 (model file formats), sec. 3.5
  (local inference).
- **Goal:** Dissect the GGUF container by hand (`struct` only): magic/
  version, typed metadata incl. arrays, tensor infos, alignment and
  data-start arithmetic; compare safetensors vs GGUF vs pickle and
  state what is not built yet (tensor dequantization, next build).
- **Builds on:** 09, 10, 19

### 09 — Safetensors + HF config + Qwen weight mapping — Built

- **Book framing:** sec. 4.2.2 (model file formats), 4.2.4 (reference
  implementations). Filed under Models here because it is what lets
  the ladder reach real weights.
- **Goal:** Hand-parsed safetensors headers, HF `config.json` field
  mapping, HF-to-`DecoderLM` tensor renaming and validation.
- **Builds on:** 07, 08

### 13 — Bottlenecks: ops:byte and arithmetic intensity — Built

- **Book framing:** sec. 2.4.
- **Goal:** Compute the ops:byte ratio for prefill and decode, place
  each phase against a device's ridge point, and predict
  compute-bound versus memory-bound before benchmarking.
- **Builds on:** 00, 05, 10

## Hardware (book Ch 3)

### 15 — Hardware: datacenter GPU and local inference — Written

- **Book framing:** Ch 3 (GPU compute and memory, generations,
  multi-GPU, local desktop/mobile inference).
- **Goal:** Read a device's compute, bandwidth, and capacity numbers
  the way ch 09 reads a `config.json`; case studies: the bigbox GPU
  serving Qwen-class models, and local/Glimmer-style on-device runs.
- **Builds on:** 13

## Software (book Ch 4)

### 16 — Software and engines — Written

- **Book framing:** Ch 4 (CUDA kernels and fusion, PyTorch, file
  formats, reference implementations, engines, Dynamo, profiling).
- **Goal:** Place this repo's hand-built engine beside vLLM, SGLang,
  TensorRT-LLM, and Dynamo: same seams as ch 12, compared as systems.
- **Builds on:** 09, 11, 12

### 12 — Optimizations — Read-only

- **Book framing:** sec. 4.1 (kernels/fusion); Ch 5 overview.
- **Goal:** SDPA/FlashAttention versus manual attention,
  `torch.compile`, speculative decoding concepts; measure, do not
  assume. The manual paths stay as references.
- **Builds on:** 04, 11

## Techniques (book Ch 5)

### 10 — Quantization — Built

- **Book framing:** sec. 5.1.
- **Goal:** Number formats, per-channel INT8 quantize/dequantize with
  error metrics, GGUF concepts, KV-cache bytes, quality impact.
- **Builds on:** 09

### 14 — Prefix caching + speculative decoding — Built

- **Book framing:** sec. 5.3.1 (prefix caching / KV reuse), sec. 5.2
  (draft-target, Medusa, EAGLE, n-gram speculation), sec. 5.1.3
  (measuring quality impact), sec. 5.4–5.5 (parallelism and
  disaggregation, read-only).
- **Goal:** Reuse KV for shared prefixes; draft tokens with a small
  model and verify with the target, outputs unchanged; measure any
  deliberate quality trade on model outputs.
- **Builds on:** 05, 08, 10, 12

### 11 — Serving: batching / PagedAttention — Built

- **Book framing:** sec. 4.3 (engines), sec. 7.2 (concurrency and
  batch sizing). Filed here as a technique; its production half is
  ch 17.
- **Goal:** Continuous (iteration-level) batching, block allocator as
  a PagedAttention dissection, TTFT/TPS tradeoffs under concurrency.
- **Builds on:** 05, 08

## Modalities (book Ch 6)

### 18 — Modalities beyond text — Written, read-only map

- **Book framing:** Ch 6 (vision-language models, embeddings, ASR,
  TTS, image and video generation).
- **Goal:** Map what changes for VLM and embedding workloads against
  the text pipeline built here, with a pointer to a Qwen-VL style
  deployment. No non-text code planned.
- **Builds on:** 07, 11

## Production (book Ch 7)

### 17 — Production serving — Written

- **Book framing:** Ch 7 (containers, autoscaling, cold starts,
  routing and queueing, scale to zero, capacity, testing, cost,
  observability, client streaming).
- **Goal:** Audit this repo's toy serving layer against that list:
  what it already does (iteration-level scheduling, block accounting,
  TTFT/TPS metrics) and what it deliberately does not (autoscaling,
  cold-start management, multi-request client streaming).
- **Builds on:** 11, 15, 16
