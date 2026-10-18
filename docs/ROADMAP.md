# Roadmap — full curriculum

Each chapter: goal, builds-on, and Qwen-27B / Glimmer relevance. Status: scaffold (not started) for all chapters.

## 00 — Inference vs training; prefill/decode

- **Goal:** Why inference is memory-bandwidth bound; prefill (parallel, compute-bound, T>1) vs decode (serial, T=1, KV-bound). The two phases explain almost every later optimization.
- **Builds on:** —
- **Qwen-27B / Glimmer relevance:** Running a 27B-class Qwen decode step reads all weights per token; Glimmer serving cost is decode-dominated.

## 01 — Tokenizer / BPE

- **Goal:** Byte (0-255 UTF-8), char, and hand-trained BPE tokenizers; encode/decode roundtrips; vocab size vs sequence length trade-off.
- **Builds on:** 00
- **Qwen-27B / Glimmer relevance:** Real Qwen uses large-vocab BPE; byte fallback explains unknown-token robustness.

## 02 — Embeddings + RoPE

- **Goal:** Token embeddings, tied embeddings, rotary position embeddings (rotate-half), precomputed cos/sin cache.
- **Builds on:** 01
- **Qwen-27B / Glimmer relevance:** Qwen/Glimmer models are RoPE models; theta and head_dim come from config.

## 03 — Norms

- **Goal:** RMSNorm vs LayerNorm; pre-norm residual stream; eps and dtype stability.
- **Builds on:** 02
- **Qwen-27B / Glimmer relevance:** Qwen uses RMSNorm (`rms_norm_eps` in its config).

## 04 — Attention: MHA / GQA

- **Goal:** Manual scaled dot-product attention from matmul+softmax, causal masking for prefill and decode, grouped-query attention via `repeat_kv`.
- **Builds on:** 02, 03
- **Qwen-27B / Glimmer relevance:** Qwen-27B-class models use GQA (`num_key_value_heads` < heads) to shrink KV cache.

## 05 — KV cache

- **Goal:** Concat cache vs preallocated static cache; `start_pos` bookkeeping; cache/no-cache greedy equivalence; KV bytes math.
- **Builds on:** 04
- **Qwen-27B / Glimmer relevance:** Glimmer decode latency and max batch are set by KV-cache size (ch. 10 formula).

## 06 — MLP: SwiGLU

- **Goal:** Gated MLPs (gate/up/down) vs GELU MLP; `intermediate_size` / `d_ff`.
- **Builds on:** 03
- **Qwen-27B / Glimmer relevance:** Qwen MLPs are SwiGLU; most parameters live here.

## 07 — Full model

- **Goal:** `TransformerBlock` + `DecoderLM`, tied embeddings, `num_params()`, learned-position variant.
- **Builds on:** 04, 05, 06
- **Qwen-27B / Glimmer relevance:** The same `DecoderLM` instantiates tiny test models and, via ch. 09, real Qwen weights.

## 08 — Sampling + generation

- **Goal:** Greedy, temperature, top-k, top-p; explicit prefill-then-decode loop; EOS; seeded generators.
- **Builds on:** 07
- **Qwen-27B / Glimmer relevance:** Sampling is what a serving stack (Glimmer) exposes as request parameters.

## 09 — safetensors + HF config + Qwen weight mapping

- **Goal:** Parse the safetensors header by hand (8-byte LE length + JSON), map HF `config.json` fields to `ModelConfig`, rename HF state-dict keys to `DecoderLM` names.
- **Builds on:** 07, 08
- **Qwen-27B / Glimmer relevance:** Directly loads Qwen checkpoints; large-model config numbers are read from the real `config.json`, never fabricated.

## 10 — Quantization

- **Goal:** Dtypes and bytes/param, INT8 per-channel quantize/dequantize + error metrics, GGUF concepts, KV-cache bytes.
- **Builds on:** 09
- **Qwen-27B / Glimmer relevance:** W4A16-style deployments (Glimmer) trade weight bytes for bandwidth; INT8 here is the dissected analogue.

## 11 — Serving: batching / PagedAttention

- **Goal:** Request lifecycle, continuous (iteration-level) batching, block allocator as a PagedAttention dissection, throughput vs latency.
- **Builds on:** 05, 08
- **Qwen-27B / Glimmer relevance:** This is the vLLM/dashboard layer that serves Qwen-27B on bigbox/Glimmer.

## 12 — Optimizations

- **Goal:** SDPA/FlashAttention vs the manual attention, `torch.compile`, speculative decoding concepts; measure, don't assume.
- **Builds on:** 04, 11
- **Qwen-27B / Glimmer relevance:** Explains why production stacks beat this code and by how much; manual path stays the reference.

