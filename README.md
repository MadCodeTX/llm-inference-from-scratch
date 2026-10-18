# llm-inference-from-scratch

Educational, dissected LLM inference in Python + PyTorch. Every mechanism is
built by hand first (matmul/softmax attention, hand-parsed safetensors headers,
a toy paged KV allocator), then compared with the optimized library path.
No training framework, no `generate()` black box: the point is to see the wires.

> **Attribution.** Framing, categorisation, and terminology follow
> Philip Kiely, *Inference Engineering*, Baseten Books, January 2026,
> ISBN 9798994359730. All text, code, examples, and tests in this repo
> are original. See [docs/TEXTBOOK_MAP.md](docs/TEXTBOOK_MAP.md) for
> the chapter-by-chapter cross-reference.

## The Tiny-to-Qwen ladder

1. **Tiny** — a 2-layer, `d_model<=64` decoder trained on nothing; random weights,
   byte tokens. Small enough to print every tensor shape in a test.
2. **Small** — same code, character/BPE tokens, longer contexts; sampling and
   KV-cache equivalence (cache vs no-cache must match token-for-token greedy).
3. **Real weights** — load a Hugging Face `config.json` + safetensors checkpoint
   (Llama/Qwen architecture family) into the same `DecoderLM` via an explicit
   weight-name mapping. Start with the smallest available Qwen checkpoint.
4. **Qwen-class / Glimmer** — read the real `config.json` of a large model
   (e.g. a 27B-class Qwen deployment) *from disk at load time*; never hard-code
   its numbers here. Reason about its KV-cache bytes, quantization, and serving
   cost with chapters 10-11 even when it cannot run on this machine.

One codebase climbs the whole ladder: only `ModelConfig` and the weights change.

## Chapter index

| Ch | Doc | Code | Topic |
|----|-----|------|-------|
| 00 | [docs/00_overview.md](docs/00_overview.md) | `config.py` | Inference vs training; prefill / decode |
| 01 | [docs/01_tokenizer.md](docs/01_tokenizer.md) | `tokenizer.py` | Tokenizers: byte, char, BPE |
| 02 | [docs/02_embeddings_rope.md](docs/02_embeddings_rope.md) | `layers.py`, `model.py` | Embeddings + RoPE |
| 03 | [docs/03_norms.md](docs/03_norms.md) | `layers.py` | RMSNorm / LayerNorm |
| 04 | [docs/04_attention.md](docs/04_attention.md) | `attention.py` | Attention: MHA / GQA, causal masks |
| 05 | [docs/05_kv_cache.md](docs/05_kv_cache.md) | `kv_cache.py` | KV cache (concat + preallocated) |
| 06 | [docs/06_mlp.md](docs/06_mlp.md) | `mlp.py` | MLP: SwiGLU / GELU |
| 07 | [docs/07_model.md](docs/07_model.md) | `model.py` | Full decoder model |
| 08 | [docs/08_sampling_generation.md](docs/08_sampling_generation.md) | `sampling.py`, `generate.py` | Sampling + generation loop |
| 09 | [docs/09_weights.md](docs/09_weights.md) | `weights.py` | safetensors + HF config + Qwen weight mapping |
| 10 | [docs/10_quantization.md](docs/10_quantization.md) | `quantization.py` | Quantization: dtypes, INT8, GGUF concepts |
| 11 | [docs/11_serving.md](docs/11_serving.md) | `serving.py` | Serving: continuous batching, PagedAttention |
| 12 | [docs/12_optimizations.md](docs/12_optimizations.md) | — (read-only dissection) | Optimizations: SDPA/FlashAttention, torch.compile, speculative decoding |
| 13 | [docs/13_bottlenecks.md](docs/13_bottlenecks.md) | `bottlenecks.py`, `metrics.py` | Bottlenecks: ops:byte ratio, arithmetic intensity, ridge point |
| 14 | [docs/14_techniques.md](docs/14_techniques.md) | `prefix_cache.py`, `speculation.py`, `quantization.py` (quality) | Techniques: prefix caching, speculative decoding, quality measurement |
| 15 | [docs/15_hardware.md](docs/15_hardware.md) | `bottlenecks.py`, `quantization.py` (`kv_cache_bytes`) | Hardware: GPU compute/bandwidth/capacity budgets, local inference (sec. 3.5) |
| 16 | [docs/16_software_engines.md](docs/16_software_engines.md) | `weights.py`, `serving.py`, `metrics.py` (read-only comparison) | Software/engines: stack layers, this engine vs vLLM/SGLang/TensorRT-LLM/Dynamo |
| 17 | [docs/17_production.md](docs/17_production.md) | `serving.py`, `metrics.py` | Production: cold starts, autoscaling, cost, observability, client streaming |
| 18 | [docs/18_modalities.md](docs/18_modalities.md) | — (read-only map) | Modalities: VLM, embeddings, ASR/TTS, image/video (read-only) |

Textbook map: [docs/TEXTBOOK_MAP.md](docs/TEXTBOOK_MAP.md).
Glossary: [docs/GLOSSARY.md](docs/GLOSSARY.md). Full curriculum: [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```bash
cd llm-inference-from-scratch
uv venv && uv pip install -e ".[hf]" --group dev   # if uv is available
# fallback (no uv):
python3 -m venv .venv && .venv/bin/pip install -e ".[hf,dev]"
```

On a CPU-only box, install CPU torch first if the default wheel is too large:
`.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Test

```bash
.venv/bin/python -m pytest        # or: uv run pytest
```

Tests use tiny dims (`d_model<=64`, `n_layer<=2`) and run fast on CPU.

## Demo

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/demo_generate.py   # tiny char LM: train, generate, cache==no-cache check
.venv/bin/python scripts/benchmark.py       # prefill vs decode throughput
```

## Status

| Area | Status |
|------|--------|
| Scaffold (layout, contract, roadmap) | done |
| Code chapters 00-11 (`config` … `serving`) | implemented; tests passing (see below) |
| Chapter 12 (optimizations) | read-only dissection in `docs/12_optimizations.md`; no code changes by design |
| Teaching docs 00-12 + glossary + textbook map/roadmap | written; every internal link verified to exist |
| Chapter 13 | implemented (`bottlenecks.py`, `metrics.py`, `docs/13_bottlenecks.md`); tests passing |
| Chapter 14 | implemented (`prefix_cache.py`, `speculation.py`, quantization quality metrics, `docs/14_techniques.md`); tests passing |
| Chapters 15-18 | written (`docs/15_hardware.md`–`docs/18_modalities.md`); 15-17 map to existing code, 18 is a read-only map with no code by design |
| Qwen weight loading | machinery implemented + tested on hand-built checkpoints (`weights.py`: hand-parsed safetensors headers, `load_hf_config`, `map_hf_state_dict`, `validate_state_dict`); not yet run against a downloaded real checkpoint on this machine — config numbers are always read from the real `config.json` at load time |

Verified: `.venv/bin/python -m pytest -q` → **123 passed** (tiny CPU dims, including ch 13–14). Independently re-checked: greedy cache vs no-cache tokens identical (MHA/GQA/MQA + learned-pos variants), HF-rename roundtrip logits identical (`max_abs_diff 0.0`), `scripts/demo_generate.py` prints a sample and `scripts/benchmark.py` prints tokens/s plus p50/p90/p99 TTFT and TPS (ch 13). Fresh spot-checks for this update: prefix-cache reuse over a shared 4-token prefix matches full-prefill logits (`max_abs_diff` ~4e-08), speculative greedy output equals target-only greedy token-for-token, and `scripts/demo_generate.py` still runs. Honest gaps: no real Qwen checkpoint has been downloaded/run here; chapter 12 remains a read-only dissection (FlashAttention, torch.compile) while speculative decoding is built separately in chapter 14 — the manual implementations remain the reference. On this tiny CPU model the no-cache path can time faster than the cached path (overhead dominates); the cache's win grows with sequence length and model size.
