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

### 21 — Linear attention: the gated delta rule — Built

- **Book framing:** Repo extension beyond the book's survey; the
  mechanism is read from the HF `Qwen3_5GatedDeltaNet` reference and
  the real checkpoint's `text_config` (see docs/21).
- **Goal:** Fixed-size recurrent state `S` with a gated, delta-rule
  rank-1 write per token; causal depthwise conv, L2-normalised Q/K,
  gated output norm; prefill/decode/snapshot equivalence on CPU.
- **Builds on:** 04, 05, 10

### 22 — Hybrid Qwen3.8: gated full attention — Built

- **Book framing:** Repo extension beyond the book's survey; shapes
  and the Q+gate split are read from the real checkpoint's shards
  and the HF `Qwen3_5Attention` reference (see docs/22).
- **Goal:** Dispatch 48 linear layers and 16 full layers behind one
  mixed snapshot/restore; fused Q+gate projection, per-head Q/K
  norms, partial RoPE, GQA, sigmoid output gate; `HybridConfig`
  parses the real `text_config` (hidden 5120, 64 layers, vocab 248320).
- **Builds on:** 04, 05, 09, 10, 21

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

### 25 — Static decode: preallocated state — Built (CPU)

- **Book framing:** sec. 4.1 (CUDA kernels/fusion), sec. 4.5
  (performance benchmarking).
- **Goal:** Allocate all decode state once at fixed shapes/addresses
  (`StaticHybridState`, shared static KV cache, in-place gated-delta
  step) so a `T=1` step never reallocates state; prove static logits
  equal the legacy path and `data_ptr` is stable across steps/reset.
  The static path has since been smoke-tested on CUDA
  (`scripts/gpu_smoke_hybrid.py`: static vs legacy `max_abs` 0.0);
  CUDA-graph capture has since been built for TP decode (`TPStaticState` + captured `decode_token_static`, tiny TP2 graph check PASS, real graph 34.6 tok/s; see ch 23).
- **Builds on:** 05, 21, 22

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

### 23 — Tensor parallelism — Built (CPU simulation + distributed tiny + measured real TP2)

- **Book framing:** sec. 5.4 (tensor parallelism), with sec. 2.2
  (config/mechanics), sec. 2.4–2.5 (bottlenecks/attention),
  sec. 4.2 (formats/reference implementations), and sec. 5.2
  (speculation context).
- **Goal:** Partition every hybrid tensor family across two ranks —
  per-head `[q|gate]` pairs and segmented gated-delta QKV kept whole —
  with exact shard/merge roundtrips and dense-equivalent matmuls on
  CPU; price the 128 all-reduces per forward and the per-rank bytes.
  The 2-rank NCCL all-reduce cost was since measured on GPU
  (6.81 GB/s bus, 8.67 us fitted latency; `benchmarks/roofline/`).
  True TP2 has since run the full 27B model (`tp_distributed.py`,
  `tp_checkpoint.py`, `scripts/run_qwen38_tp2.py`): v1 22.9–23.6 tok/s,
  v2 (fused RMSNorm) 23.4–23.9 tok/s, both 32/32 exact ×3 vs the
  pipeline bridge (retained as oracle) and vLLM Arm A; profile shows
  host-launch-bound decode (128 all-reduces/token = 1.28 ms). This phase: TP static state + CUDA-graph decode 34.6 tok/s (TPOT 28.9 ms, 32/32 exact ×3), vocab-parallel lm_head shipped, scale cache opt-in/off, FP8 BLOCK_N=32 tile retune 24.48 ms/token, fused gated-delta 23.67 ms/token (42.25 tok/s, exact x3) — still above Arm A 20.3 ms and the 19.17 ms floor, no win claimed. See `docs/23_tensor_parallel.md`.
- **Builds on:** 21, 22

### 24 — Native MTP head: foundations — Built (CPU)

- **Book framing:** sec. 5.2 (speculation), with sec. 2.2
  (config/mechanics), sec. 2.4–2.5 (bottlenecks/attention),
  sec. 4.2 (formats/reference implementations), and sec. 5.4
  (tensor-parallel context).
- **Goal:** Build the checkpoint-derived MTP head (22 header-verified
  tensors, one full-attention block reusing `HybridBlock`) plus shared
  acceptance accounting with no fabricated numbers; record the
  shape-silent joins as inferred and design Phase 3 rollback
  (truncate KV, snapshot recurrent state) before wiring generation.
  Arm B retry 2026-10-18 **MEASURED-WITH-DEVIATION** (--max-model-len 256608 only; acceptance 98.8%, failed=0; no win claimed).
- **Status (2026-10-18, measured):** wiring sweep (48 candidates,
  teacher-forced) resolved=false — best acceptance 0.8721 with tied
  rows, so no winner was written in as default. The TP2 speculative
  loop (eager + graph-captured) is exact (32/32 x3, capital 128/128)
  with acceptance 0.88–0.94 but slower than graph-nonspec
  (32.51 vs 24.49 ms effective TPOT, capital x128); not a win.
  Arm B **MEASURED-WITH-DEVIATION** 2026-10-18 (--max-model-len 256608 only). TP chunkwise prefill chunk128 1316 tok/s @512 / 1306 tok/s @2048 (ours-vs-ours); TP prefix cache scaled-agent cold 10.66s / warm 1.31s median (8.14x ours-vs-ours, no win vs vLLM claimed). Also: mtp1-graph2 merged graph 32.90 ms eff capital x128 (exact, not faster); best is fused nonspec 23.67 ms (42.25 tok/s).
- **Builds on:** 14, 22

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
