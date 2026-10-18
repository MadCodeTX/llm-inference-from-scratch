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
   weight-name mapping. **Done** (ch 19): Qwen2.5-0.5B-Instruct runs in
   `DecoderLM` and matches the HF reference token-for-token.
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
| 19 | [docs/19_real_weights_qwen.md](docs/19_real_weights_qwen.md) | `scripts/load_qwen_small.py`, `weights.py` | Real weights: Qwen2.5-0.5B proved id-exact vs HF reference |
| 20 | [docs/20_gguf_formats.md](docs/20_gguf_formats.md) | `gguf.py` | GGUF formats: safetensors vs GGUF vs pickle; GGUF header by hand |
| 21 | [docs/21_linear_attention.md](docs/21_linear_attention.md) | `linear_attention.py` | Linear attention: gated delta rule, fixed-size recurrent state |
| 22 | [docs/22_hybrid_qwen38.md](docs/22_hybrid_qwen38.md) | `hybrid.py`, `hybrid_checkpoint.py`, `linear_attention.py` | Hybrid Qwen3.8: gated full attention in a linear-attention stack |
| 23 | [docs/23_tensor_parallel.md](docs/23_tensor_parallel.md) | `tensor_parallel.py` | Tensor parallelism: two-rank sharding simulation for the hybrid stack |
| 24 | [docs/24_mtp.md](docs/24_mtp.md) | `mtp.py`, `speculation.py` | Native MTP head: checkpoint-derived draft layer and acceptance accounting |

## Scaling program (beat vLLM on bigbox)

The endgame is a from-scratch engine that beats the incumbent vLLM
serving `Qwen/Qwen3.8-27B-FP8` on this box. What counts as a win —
metrics, incumbent reference numbers, thresholds, quality gate, and
the GPU-window protocol — is defined in
[benchmarks/WIN.md](benchmarks/WIN.md). No win is claimed until a
candidate is measured with `benchmarks/run_bench.py` on
`benchmarks/suite.json` inside an approved GPU window; the vLLM + MTP
arm is not yet measured.

| Phase | Status |
|-------|--------|
| Phase 0: benchmark suite, WIN criteria, incumbent baseline import | done (harness + dry-run plan; aggregate numbers only, no prompt text) |
| FP8 block-quant path: `F8_E4M3`/`F8_E5M2` parser + block dequant reference (CPU) | done (GPU `torch._scaled_mm` fast path is later work) |
| Hybrid foundations: gated delta rule (ch 21) + gated full attention (ch 22, CPU) | done (tiny-dims CPU parity vs HF; no GPU serving yet) |
| Phase 1a (CPU): streaming hybrid checkpoint loader (`hybrid_checkpoint.py`, headers-only manifest, lazy FP8 dequant) | done (real snapshot validated: 1606 entries, 1273 expected/present, `ok: True`) |
| Phase 1a (CPU): chunkwise gated-delta prefill (`method="chunkwise"`, sequential stays default) | done (CPU parity to fp32 tolerance; `benchmarks/microbench_linear_attn.py` reports measured medians) |
| Phase 1a (CPU): hybrid runtime state + generation (typed `HybridState`, explicit mRoPE positions, `generate_hybrid`) | done (tiny-dims CPU tests; greedy ids agree across prefill methods) |
| Phase 1b-prep (CPU): TP2 sharding simulation (ch 23) + MTP foundations (ch 24) | done on CPU (exact shard/merge roundtrips; checkpoint-derived MTP spec; no fabricated acceptance numbers). GPU execution, real collectives, and any win claim still pending an approved GPU window — the live vLLM service was not touched |

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
| Qwen weight loading | done (ch 19): real Qwen2.5-0.5B-Instruct checkpoint downloaded to git-ignored `.hf-cache/`, validated with zero missing/unexpected/shape errors, and proved against the HF reference on CPU fp32 (logits max-abs-diff 2.1e-05, greedy top-1 20/20, generated ids exactly equal). Config numbers always read from the real `config.json` at load time |
| GGUF formats | done (ch 20): `gguf.py` parses GGUF v2/v3 headers by hand (magic/version, typed metadata incl. arrays, tensor infos, alignment/data-start arithmetic); tests synthesise tiny GGUF files with a test-side writer. Tensor-data dequantization is the explicit next build |
| Scaling Phase 0 | done: `benchmarks/` suite + WIN criteria + incumbent baseline import (aggregate numbers only); `run_bench.py --dry-run` plans 48 batches / 336 requests with no network; microbench prints a plan only without `--i-understand-gpu-window` |
| FP8 block-quant | done (CPU): `F8_E4M3`/`F8_E5M2` safetensors parsing in `weights.py`; block-FP8 quantize/dequantize/matmul reference in `quantization.py` (ch 10) |
| Hybrid Qwen3.8 (ch 21–22) | done (CPU): gated delta rule in `linear_attention.py`; gated full attention + hybrid stack in `hybrid.py`; tiny-dims parity vs HF references |
| Hybrid checkpoint loader + runtime (ch 22) | done (CPU): `hybrid_checkpoint.py` streaming loader; typed `HybridState`, explicit positions, `generate_hybrid` in `hybrid.py` |
| Tensor parallelism + MTP (ch 23–24) | done (CPU simulation/foundations): `tensor_parallel.py` shard/merge proofs; `mtp.py` checkpoint-derived head + shared acceptance accounting. No GPU run yet; GPU window still pending |

Verified: `.venv/bin/python -m pytest -q` → **221 passed** (tiny CPU dims, including ch 13–14, 19–24, Phase-0 harness, FP8 block tests). Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2 (config/mechanics), sec. 2.4–2.5 (bottlenecks/attention), sec. 4.2 (formats/reference implementations), sec. 5.2 (speculation), and sec. 5.4 (tensor parallelism); all prose here is original. Real-model rung independently rerun on CPU fp32: `scripts/load_qwen_small.py` on Qwen2.5-0.5B-Instruct gives validate `missing=[] unexpected=[] shape_mismatch=[]` (291 tensors, 494,032,768 params), first-forward logits max-abs-diff `2.098e-05`, per-step top-1 `20/20`, generated ids exactly equal `True`, text `'The capital of France is Paris. It is the largest city in Europe and the third largest city in the world. It is'` (TTFT `0.039s`, TPS `27.97`). Independently re-checked: greedy cache vs no-cache tokens identical (MHA/GQA/MQA + learned-pos variants), HF-rename roundtrip logits identical (`max_abs_diff 0.0`), `scripts/demo_generate.py` prints a sample and `scripts/benchmark.py` prints tokens/s plus p50/p90/p99 TTFT and TPS (ch 13). Fresh spot-checks for this update: prefix-cache reuse over a shared 4-token prefix matches full-prefill logits (`max_abs_diff` ~4e-08), speculative greedy output equals target-only greedy token-for-token, and `scripts/demo_generate.py` still runs. Honest gaps: the real-checkpoint proof (ch 19) is one short prompt on CPU fp32 — a CUDA torch build, longer contexts, and bf16 comparison are future work; chapter 12 remains a read-only dissection (FlashAttention, torch.compile) while speculative decoding is built separately in chapter 14 — the manual implementations remain the reference. On this tiny CPU model the no-cache path can time faster than the cached path (overhead dominates); the cache's win grows with sequence length and model size.
