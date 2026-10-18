# Glossary

One-line definitions, each pointing at the chapter that dissects it. Shapes use `B` batch, `T` sequence length, `D = d_model`, `V = vocab_size`, `L = n_layer`, `H` query heads, `H_kv` KV heads, `D_h = head_dim`.

**Prefill** — The single parallel forward pass that ingests the whole prompt (`T > 1`), fills the KV cache, and produces the first sampled token; compute-bound. Ch 00, 08.

**Decode** — The serial phase after prefill: one forward per generated token (`T = 1`), reading all weights and the full KV cache each step; memory-bandwidth-bound. Ch 00, 08.

**KV cache** — Stored keys and values for all previous tokens, per layer `[B, H_kv, T, D_h]`, so decode never recomputes them. Size: `2 * L * B * H_kv * T * D_h * bytes_per_elem`. Ch 05, 10.

**GQA (grouped-query attention)** — Attention where `H_kv < H`: each KV head is shared by `n_rep = H / H_kv` query heads, shrinking the KV cache by `n_rep`. `H_kv = H` is MHA; `H_kv = 1` is MQA. Ch 04.

**RoPE (rotary position embedding)** — Position encoding that rotates each Q/K head-vector pair by an angle proportional to its absolute position (`theta^(-2i/D_h)` per pair), making attention scores depend on relative distance. Ch 02.

**Logits** — The raw, unnormalized `[B, V]` scores the LM head emits for the next token (and `[B, T, V]` over a sequence); softmax turns a row into probabilities. Ch 07, 08.

**TTFT (time to first token)** — Wall-clock latency from request arrival to the first generated token; dominated by prefill. Ch 00, 11.

**TPOT (time per output token)** — Mean wall-clock time per decode token after the first; the streaming speed users perceive. Ch 00, 11.

**Quantization** — Storing weights (or KV) in fewer bits (fp16, int8, int4, ...) with per-channel/block scales, trading a bounded numeric error for proportionally fewer bytes read per decode token. Ch 10.

**GGUF** — llama.cpp's single-file model format: config, tokenizer (vocab + merges), and (usually block-quantized) tensors in one self-describing container. The on-device counterpart of a HF checkpoint directory. Ch 09, 10.

**Safetensors** — A tensor checkpoint format: 8-byte little-endian header length, a JSON header (`dtype`, `shape`, `data_offsets` per tensor), then a raw data buffer. No executable code. Ch 09.

**Paged attention** — vLLM's KV-cache scheme: the cache is carved into fixed-size blocks scattered in memory and addressed per-request by a block table, eliminating contiguous per-request reservations and bounding fragmentation to one partial block. Ch 11.

**TPS (tokens per second)** — Output rate. Per request, decode TPS ≈ `1 / TPOT` (TPOT in seconds per token); system TPS aggregates tokens finished per second across all concurrent requests. Ch 00, 11.

**Percentile (p50/p95/p99)** — A latency quantile: p95 TTFT is the value 95% of requests beat. Production latency is reported in percentiles because means hide the slow tail. Ch 00, 11.

**Compute-bound / memory-bound** — Which resource sets the pace: arithmetic throughput (compute-bound, typical of prefill) or bytes moved per second (memory-bound, typical of decode). Decided quantitatively by arithmetic intensity against the ridge point. Ch 00, 13.

**Ops:byte ratio** — Operations performed per byte moved from memory; the numerator-side view of arithmetic intensity for a specific kernel or phase. Low values (decode, ~1) signal memory-bound work. Ch 13.

**Arithmetic intensity** — Total arithmetic operations divided by total bytes transferred for a workload. Compare it with the device ridge point to predict the binding constraint before benchmarking. Ch 13.

**Ridge point** — In plain terms: the arithmetic intensity at which a device's compute ceiling and bandwidth ceiling meet. Work left of it is memory-bound; work right of it is compute-bound. Found by dividing peak FLOPs/s by peak bytes/s. Ch 13.

**Online inference** — Requests served as they arrive, judged by TTFT and per-request TPS at high percentiles. Opposite pole: offline inference. Ch 00, 17.

**Offline inference** — Batch jobs run for aggregate system TPS and cost per token, with loose per-request latency targets. Ch 00, 17.

**Prefix caching** — Reusing cached KV for a shared prompt prefix (for example a common system prompt) across requests instead of re-prefilling it; cuts TTFT on hits. Ch 05, 14.

**Speculative decoding** — A small draft model proposes several tokens; the target model verifies them in one parallel pass, accepting a prefix and resampling at the first mismatch, so outputs match the target's own distribution. Variants include draft-target, Medusa, EAGLE, and n-gram speculation. Ch 12, 14.

**Disaggregation** — Running prefill and decode on separate worker pools, each scaled and optimised for its own phase (compute-bound versus memory-bound), with KV transferred between them. Ch 17.

**Cold start** — Delay before a fresh replica can serve: loading weights, allocating caches, warming kernels. Paid on scale-out and after scale to zero. Ch 17.

**Scale to zero** — Letting idle capacity shrink to no running replicas, trading a cold start on the next request for zero idle cost. Ch 17.

**Tensor parallelism (TP)** — Splitting individual weight matrices across devices, with collective communication per block, mainly to cut per-request latency and fit large layers. Ch 16–17.

**Expert parallelism (EP)** — Placing different MoE experts on different devices and routing tokens to them, mainly to raise system TPS for sparse models. Ch 16–17.

**MoE (mixture of experts)** — An MLP replacement holding many expert sub-networks; a router sends each token to a few, so most parameters stay inactive per token. This repo builds the dense MLP an expert uses. Ch 06, 18.

**Container / NIM** — Only as packaging context for ch 17: a container freezes model-serving dependencies into a deployable unit, and a NIM is NVIDIA's pre-packaged, optimised serving container of that kind. No container work exists in this repo yet. Ch 17.
