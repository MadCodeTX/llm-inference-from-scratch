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
