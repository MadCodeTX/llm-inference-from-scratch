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
| 23 | [docs/23_tensor_parallel.md](docs/23_tensor_parallel.md) | `tensor_parallel.py`, `tp_distributed.py`, `tp_checkpoint.py` | Tensor parallelism: sharding, distributed tiny model, FP8 rank loader, and measured real TP2 run |
| 24 | [docs/24_mtp.md](docs/24_mtp.md) | `mtp.py`, `speculation.py` | Native MTP head: checkpoint-derived draft layer and acceptance accounting |
| 25 | [docs/25_static_decode.md](docs/25_static_decode.md) | `hybrid.py`, `linear_attention.py` | Static decode: preallocated state for CUDA-graph-friendly stepping |
| 29 | [docs/29_harness2_method.md](docs/29_harness2_method.md) | `opt_harness/`, [harness/README.md](harness/README.md) | Harness² method: Propose–Probe–Compose optimisation loop, exactness-gated selection |
| 30 | [docs/30_serving_openai.md](docs/30_serving_openai.md) | `openai_server.py`, `scripts/serve_openai.py`, `scripts/serve_qwen38_tp2.py` | OpenAI-compatible serving surface for same-harness `run_bench.py` measurement; no vLLM win claimed yet |

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
| Phase 1c (CPU): checkpoint assembly onto `HybridLM` + real-layer parity (ch 22) | done on CPU (real snapshot meta-validation: 851 model tensors, 26,895,998,464 params, no missing/unexpected; layer 3 bit-exact, layer 0 whole-block rel_rms 1.5e-3 in bf16 collapsing to 1.2e-7 in fp32 — dtype noise, no fix needed) |
| Phase 1c (CPU): TP execution simulation + static decode state (ch 23 runtime, ch 25) | done on CPU (simulated two-rank blocks equal dense; static decode logits equal legacy, `data_ptr` stable; TP prefill and CUDA-graph capture not built) |
| Phase 1c (CPU): exact-token benchmark suite + arms (ch 25 bench prep) | done on CPU (exact token counts with `--tokenizer-dir`, Arm A incumbent / Arm B vLLM+MTP `k=1,2,3` defined; Arm B retry 2026-10-18 MEASURED-WITH-DEVIATION (only --max-model-len 256608; greedy c1 short 4272-4291 tok/min, spec acceptance 98.8%, failed=0; chunk TPOT not comparable, no win claimed; see benchmarks/WIN.md)) |
| Resident-FP8 execution (`fp8_linear.py`, ch 10/22) | done: `BlockFP8Linear` (FP8 bits + BF16 `weight_scale_inv` buffers, dequantize-then-`F.linear` reference forward), `replace_linears_with_fp8`, and `load_into_model(fp8_mode="resident")`; `torch._scaled_mm` probed on this box and kept off (its 128x128 blockwise mode needs a blocked, 16B-aligned float-scale layout plus FP8 activations, not the checkpoint's plain BF16 block grid). GPU proof `scripts/gpu_proof_fp8_resident.py` (real layer-3 `q_proj` [12288, 5120] on `cuda:0`, RTX 4090, seed 3): CUDA vs CPU-dequant `max_abs=0.015625`, peak allocated 700,669,440 B; resident bytes 62,922,240 vs BF16 125,829,120 |
| Arm A live baseline (exact-token) | done (measured subset): `benchmarks/run_bench.py` greedy + sampled at concurrency 1/4, all four shapes, 2 repeats/cell, `failed == 0` (16 batches each; concurrency 16 not run, so not the full suite); quality probes in `benchmarks/baseline/arm_a_quality_probes.json` (output ids `retokenized_local`; API returned no server ids). Medians in [benchmarks/WIN.md](benchmarks/WIN.md). Arm B (vLLM + MTP K=1) **MEASURED-WITH-DEVIATION** 2026-10-18 (--max-model-len 256608 only deviation; first exact-flags attempt FAILED_TO_START); no win claimed |
| Measured roofline (2× RTX 4090) | done: copy bandwidth peaks 919.8 / 920.0 GB/s (GPU0/GPU1), BF16 GEMM 149.42 TFLOP/s, FP8 `torch._scaled_mm` 320.61 TFLOP/s, 2-rank NCCL bus ~6.81 GB/s; batch-1 decode ceiling 52.2 tok/s at S=2048 (TPOT floor 19.17 ms). Raw JSONL + `benchmarks/roofline/SUMMARY.md`; bounds only, no win claimed |
| GPU tiny hybrid smoke | done: `scripts/gpu_smoke_hybrid.py` → `RESULT: PASS` (fp32 prefill/decode, static-vs-legacy 0.0, chunkwise, CPU-generator sampling; peak 9,043,968 B on `cuda:0`); skip-gated tests in `tests/test_gpu_smoke.py` |
| Two-GPU pipeline bridge (full 27B) | done (measured): v1 reference-dequant 1.83 tok/s; v2 Triton fused FP8 (default) 19.6–19.7 tok/s, both 32/32 greedy ids exact vs recorded vLLM probes on all 3 prompts; v2 load 23.5 s, peak ~15.0/15.1 GB/GPU; 8-token capital re-check 8/8 prefix exact at 18.98 tok/s. Still below Arm A (~49 tok/s/stream, TPOT 20.3 ms) and the 52.2 tok/s roofline ceiling — no win claimed. Retained as the correctness oracle for TP2 |
| True TP2 (full 27B, measured) | done (measured): `scripts/run_qwen38_tp2.py` v1 22.9–23.6 tok/s (TPOT 42.5–43.7 ms), v2 with fused RMSNorm 23.4–23.9 tok/s (TPOT 41.8–42.8 ms), both 32/32 exact ×3 vs pipeline bridge and vLLM Arm A; load ~21.2 s/rank, peak 16.39 GiB/rank. Profile: host-launch-bound (clean 44.4 ms/token vs 31.3 ms GPU busy; 128 all-reduces/token = 1.28 ms). Still ~2.1× off Arm A (TPOT ~20.3 ms) and the 19.17 ms/token roofline floor — no win claimed. This phase (measured): TP static state + CUDA-graph decode 34.6 tok/s (TPOT 28.9 ms, 32/32 exact ×3, `benchmarks/baseline/ours_tp2_graph.json`); vocab-parallel lm_head shipped (paired 43.73 vs 45.60 ms/token replicated), FP32 scale cache ships opt-in/default off (no consistent winner); FP8 tile retune after graphs (M=1 BLOCK_N=32) 28.88→24.48 ms/token ABBA (`benchmarks/baseline/ours_tp2_tile_ab.json`); fused gated-delta (`--delta fused`, default) 24.48→23.67 ms/token, 42.25 tok/s, 32/32 exact x3 (`benchmarks/baseline/ours_tp2_graph_v2.json`) — best measured mode. Still above vLLM Arm A 20.3 ms and the 19.17 ms floor — no win claimed |
| TP chunkwise prefill (measured) | done: `prefill_chunkwise_static` chunk128 1316 tok/s @512, 1306 tok/s @2048 vs static 23 tok/s @512 (`benchmarks/baseline/ours_tp2_prefill_bench.json`, ours-vs-ours, no TTFT-vs-vLLM claim); chunk256 635 tok/s (reference FP8 M>128) |
| TP prefix cache + TTFT (measured) | done: scaled agent 2048+256=2304 cold 10.66s / warm 1.31s median, 8.14x ours-vs-ours, warm n_reused=2048 (`benchmarks/baseline/ours_prefix_ttft.json`; Arm A 1.81s@8704 context only, no win claim) |
| Serving surface + graphed batched decode (integrated) | done: OpenAI-compatible server (`openai_server.py`, ch 30) with `BatchedEngineBackend` over `TPEngineCore` (driver-error, `ignore_eos`, and TP-shutdown defects fixed with committed stub-engine tests); graphed batched decode is now the engine default on CUDA — exact and faster at c1/c4/c8 on the local engine harness (`benchmarks/harness/throughput_round1.json`: 2,334 / 5,109 / 7,298 vs eager 1,397 / 4,462 / 6,979 tok/min, token-identical). Serving readiness round 1 (27B TP2): default sizing is now the proven 8×8192 (8×16384 still infeasible, HTTP 500 OOM), compact lockstep stays opt-in (no ABBA benefit), sampled arm implemented eager-only. Serving-overhead round 1: c1 TPOT 113.89 → 63.98 ms, c4 3036.2 tok/min, 16×4096/c16 feasible (16/16 twice). Local harness only: **no vLLM win claimed** until a same-harness `run_bench.py` suite run exists |
| Batched TP decode + prefill v2 + prefix v2 + engine (measured) | done: per-row `TPStaticState` + `generate_static_batched` (eager B=1..16: 1,367 / 2,721 / 5,360 / 10,746 / 20,001 tok/min, row0 ids identical; `benchmarks/baseline/ours_tp2_batched_decode.json`); prefill v2 chunk512 1,732 tok/s @2048 (`ours_tp2_prefill_v2.json`); prefix v2 interval-256 cold 1.64s gpu / warm 0.18s (`ours_prefix_ttft_v2.json`); `TPEngineCore` continuous batching (exact staggered proof, no throughput claim). Ladder in [benchmarks/WIN.md](benchmarks/WIN.md); per-stream still behind Arm A/B — no win claimed |

### Throughput ladder (measured only; different harnesses — see WIN.md)

| Concurrency / B | Arm A tok/min | Arm B tok/min | Ours tok/min |
|---|---|---|---|
| c1 / B=1 graphed | 2,745 (short) | 4,272–4,291 | 2,535 (23.67 ms) |
| c1 / B=1 eager batched | — | — | 1,367 |
| B=2 eager | — | — | 2,721 |
| c4 / B=4 | 8,530 (short) | 12,113–12,136 | 5,360 eager |
| B=8 eager | — | — | 10,746 |
| B=16 eager | — | — | 20,001 |

Prefill v2: 1,413 (chunk128) / 1,732 (chunk512) tok/s @2048 vs Arm A ~2.9–3.4k prompt tok/s — behind. Prefix TTFT v2 @2304: cold 1.93s cpu / 1.64s gpu, warm 0.25s / 0.18s vs v1 10.66s / 1.31s; Arm A 1.81s @8704 is a different shape. Aggregate B=8/B=16 numerically pass the Arm A/B c4 bars but on a different local-runner shape, so **no win is claimed**; per-seq latency still loses everywhere.

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
| Tensor parallelism + MTP (ch 23–24) | TP done (measured): `tensor_parallel.py` shard/merge proofs, `tp_distributed.py` distributed tiny model, `tp_checkpoint.py` FP8 rank loader, real TP2 23.4–23.9 tok/s (v2), graph decode 34.6 tok/s, tile-retuned 40.85 tok/s (TPOT 24.48 ms) at 32/32 exact (see Scaling program); pipeline bridge retained as oracle. MTP (measured, exact, not faster): wiring sweep over 48 candidates found no single winner (best teacher-forced acceptance 0.8721, tied rows; `benchmarks/mtp/wiring_sweep.json`, resolved=false); TP2 speculative loop (`--speculative mtp1` / `mtp1-graph`) is exact everywhere (32/32 x3 vs bridge+Arm-A, capital 128/128) with acceptance 0.88–0.94, but effective TPOT on capital x128 is 38.68 ms (eager) / 32.51 ms (graph) / 32.90 ms (mtp1-graph2 merged, 1 sync/cycle, `benchmarks/baseline/ours_tp2_mtp_graph2.json`) vs 23.67 ms fused graph-nonspec — no speedup claimed, runner default stays `off`. Host diagnosis (`benchmarks/profile/SUMMARY_spec.md`): ~35 ms/cycle is GPU draft+verify surfaced as sync waits; genuine host overhead ~8 ms/cycle. Arm B **MEASURED-WITH-DEVIATION** 2026-10-18 (`benchmarks/arms/arm-b-mtp.json`, --max-model-len 256608 only; acceptance 98.8%) |
| Checkpoint assembly + real-layer parity (ch 22) | done (CPU): `HybridLM` assembly validated on the real snapshot (meta only); layers 0 and 3 proved vs HF via `scripts/parity_real_layer.py` (gated tests skip by default) |
| TP execution + static decode (ch 23, 25) | done (CPU + CUDA smoke): simulated two-rank runtime in `tensor_parallel.py`; `StaticHybridState`/`decode_token_static` in `hybrid.py` + `step_inplace` in `linear_attention.py`; CUDA static-vs-legacy `max_abs` 0.0 in `scripts/gpu_smoke_hybrid.py`. No CUDA graph captured yet |
| Exact-token benchmark suite + arms | done (CPU harness + measured Arm A subset): `benchmarks/run_bench.py --tokenizer-dir` builds exact-token prompts; `benchmarks/arms/` defines Arm A/B. Arm A greedy + sampled at c1/c4 measured (`failed == 0`); Arm B **MEASURED-WITH-DEVIATION** (K=1, 256608). No win claimed |
| GPU execution (tiny + full-model bridge + TP2) | done (measured): tiny hybrid `RESULT: PASS` on CUDA; full 27B pipeline bridge 32/32 exact on all 3 probes; true TP2 v2 23.4–23.9 tok/s, graph 34.6 tok/s at 32/32 exact ×3. See Scaling program rows above and [benchmarks/WIN.md](benchmarks/WIN.md) |
| Harness² optimisation loop (ch 29) | done: `opt_harness/` deterministic Propose–Probe–Compose machinery (ledger `R=4k/P=3k`, exactness-gated Pareto selection, experience library) + [harness/](harness/README.md) task cards/library. Round 1 (live GPU, `benchmarks/harness/decode_round1.json`): `--scale-cache on` selected locally at median TPOT 22.866 ms / 43.73 tok/s vs base 23.726 ms / 42.15 tok/s (ABBA ×5, 32/32 exact every run), Rounds 2–3 confirmed scale-cache (graph-scoped default) and promoted the M=1 GEMV kernel to the FP8 default at 22.270 ms aggregate (local runner). Prefill round 1 rejected `prefill-s2` (slower everywhere, code removed); decode round 4 promoted `gemv-bk512` to the FP8 default at 22.112 ms aggregate / 45.22 tok/s; decode round 5 promoted `--gated-norm fused` at 21.867 ms / 45.73 tok/s and prefill round 2 promoted `prefix-chunk512` bench defaults (chunk512 pure-prefill +20–22% tok/s, prefix cold/warm −12–18%, exact everywhere, local runner); no vLLM win (Arm A 20.3 ms). Summaries: `benchmarks/harness/decode_round4_summary.md`, `benchmarks/harness/prefill_round1_summary.md` |

Verified: `.venv/bin/python -m pytest -q` → **419 passed, 25 skipped** (tiny CPU dims, including ch 13–14, 19–25, 29, Phase-0 harness, FP8 block tests; the 3 skips are the gated real-layer parity tests). Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2 (config/mechanics), sec. 2.4–2.5 (bottlenecks/attention), sec. 4.2 (formats/reference implementations), sec. 5.2 (speculation), and sec. 5.4 (tensor parallelism); all prose here is original. Real-model rung independently rerun on CPU fp32: `scripts/load_qwen_small.py` on Qwen2.5-0.5B-Instruct gives validate `missing=[] unexpected=[] shape_mismatch=[]` (291 tensors, 494,032,768 params), first-forward logits max-abs-diff `2.098e-05`, per-step top-1 `20/20`, generated ids exactly equal `True`, text `'The capital of France is Paris. It is the largest city in Europe and the third largest city in the world. It is'` (TTFT `0.039s`, TPS `27.97`). Independently re-checked: greedy cache vs no-cache tokens identical (MHA/GQA/MQA + learned-pos variants), HF-rename roundtrip logits identical (`max_abs_diff 0.0`), `scripts/demo_generate.py` prints a sample and `scripts/benchmark.py` prints tokens/s plus p50/p90/p99 TTFT and TPS (ch 13). Fresh spot-checks for this update: prefix-cache reuse over a shared 4-token prefix matches full-prefill logits (`max_abs_diff` ~4e-08), speculative greedy output equals target-only greedy token-for-token, and `scripts/demo_generate.py` still runs. Honest gaps: the real-checkpoint proof (ch 19) is one short prompt on CPU fp32 — a CUDA torch build, longer contexts, and bf16 comparison are future work; chapter 12 remains a read-only dissection (FlashAttention, torch.compile) while speculative decoding is built separately in chapter 14 — the manual implementations remain the reference. On this tiny CPU model the no-cache path can time faster than the cached path (overhead dominates); the cache's win grows with sequence length and model size.
