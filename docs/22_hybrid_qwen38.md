# 22 — Hybrid Qwen3.8: gated full attention in a linear-attention stack

> **Source of truth** — the real checkpoint on disk: `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/` (`config.json`, `model.safetensors.index.json`, per-layer shard headers read with our own [weights.py](../src/inference_from_scratch/weights.py) header parser — names/shapes only, weights never loaded); and `.venv` transformers 5.20.1, `transformers/models/qwen3_5/modeling_qwen3_5.py` (`Qwen3_5Attention`, `Qwen3_5DecoderLayer`, `Qwen3_5TextRotaryEmbedding`, `Qwen3_5RMSNorm`) + `configuration_qwen3_5.py` (`Qwen3_5TextConfig`). Our code: [hybrid.py](../src/inference_from_scratch/hybrid.py) (`HybridConfig`, `GatedFullAttention`, `HybridBlock`, `TinyHybridLM`, `list_hybrid_tensor_map`); tests: [test_hybrid.py](../tests/test_hybrid.py). Linear layers are chapter 21 ([linear_attention.py](../src/inference_from_scratch/linear_attention.py)).

## The one-sentence idea

The 27B model's 64 text layers are a hybrid — 48 gated-delta-rule linear layers carry a fixed-size recurrent state, and every 4th layer (16 total) is full softmax attention, but a gated, Q/K-normalised, partially-rotated variant that the plain decoder of chapter 07 cannot express.

## Architecture map (from the real config + tensors)

Top level: `architectures = ["Qwen3_5ForConditionalGeneration"]`, `model_type = "qwen3_5"`. Everything below is `text_config` unless noted.

| Property | Real value | Where verified |
|---|---|---|
| Hidden size | 5120 | `text_config.hidden_size`; every `[…, 5120]` tensor |
| Layers | 64 = 48 linear + 16 full, full at indices 3, 7, …, 63 | `text_config.layer_types` (`full_attention_interval = 4`) |
| Vocab | 248320, `tie_word_embeddings = false` | `embed_tokens.weight` / `lm_head.weight`, both `[248320, 5120]` in `outside.safetensors` |
| MLP | dense SwiGLU, intermediate 17408 | `mlp.{gate,up}_proj [17408, 5120]`, `down_proj [5120, 17408]` per layer |
| Full-attn heads | 24 Q / 4 KV (GQA ×6), `head_dim = 256` | `text_config`; `k/v_proj [1024, 5120]` |
| Full-attn widths | `q_proj [12288, 5120]`, `o_proj [5120, 6144]` | `layers-3.safetensors` header (first full layer) |
| RoPE | `rope_theta = 10000000`, `partial_rotary_factor = 0.25` → 64 of 256 dims rotate | `text_config.rope_parameters` |
| mrope | interleaved, sections `[11, 11, 10]` (over 32 rotary pairs) | `rope_parameters.mrope_section` |
| Linear-attn heads | 16 key / 48 value heads, per-head dim 128/128, conv kernel 4 | `text_config.linear_*`; `layers-0.safetensors` |
| Linear widths | `in_proj_qkv [10240, 5120]`, `in_proj_z [6144, 5120]`, `in_proj_a/b [48, 5120]`, `conv1d [10240, 1, 4]`, `out_proj [5120, 6144]` | `layers-0.safetensors` header |
| Norms | RMSNorm everywhere, `rms_norm_eps = 1e-6`, zero-centred gain `1 + w` | `Qwen3_5RMSNorm`; stored `q_norm.weight` mean ≈ 0.23, not ≈ 1 |
| Context | `max_position_embeddings = 262144` | `text_config` |
| Quantisation | FP8 e4m3, block `[128, 128]`, dynamic activations | `quantization_config`; e.g. `q_proj.weight_scale_inv [96, 40]` = (12288/128, 5120/128) |
| Tensors / size | 1606 tensors in the index (1250 language-model, 333 visual, 22 MTP, 1 `lm_head`); 30,866,866,928 safetensors bytes (≈ 30.87 GB) | `model.safetensors.index.json`; file sizes |
| Per-layer shards | one file per layer: 20 tensors (linear) / 18 (full), plus `outside.safetensors` (336) and `mtp.safetensors` (22) | shard headers |

Full-attention layer anatomy (layer 3, the template for `GatedFullAttention`): `input_layernorm`, `self_attn.{q,k,v,o}_proj` (+ each FP8 weight's `weight_scale_inv`), `self_attn.{q,k}_norm [256]`, `post_attention_layernorm`, `mlp.{gate,up,down}_proj` (+ scales). No biases anywhere (`attention_bias = false`).

## Gated full attention, step by step

1. **Fused Q+gate projection.** `q_proj` emits `2 · n_head · head_dim` per token. HF reshapes to `[B, T, n_head, 2·head_dim]` and chunks the last dim: *within each head*, the first `head_dim` is the query, the second is the gate. Splitting the flat vector in half instead pairs every head with the wrong gate — the split order is read from `Qwen3_5Attention.forward`, not guessed.
2. **Per-head Q/K norms.** `q_norm`/`k_norm` are RMSNorms over `head_dim` only (not the residual width), zero-centred (`1 + weight`), applied before RoPE.
3. **Partial RoPE.** Only the first `rotary_dim = 256 · 0.25 = 64` dims rotate, with frequencies built over 64 dims (`theta^(-2i/64)`); the remaining 192 dims pass through bit-identical (asserted in tests). For text, all mrope axes share one position, so interleaved mrope recomposition is the identity and 1-D partial RoPE is exact; distinct H/W positions arrive only with vision tokens.
4. **GQA attention.** 4 KV heads are expanded ×6 to meet 24 Q heads; our path is selectable — the repo's manual SDPA (reference) or `F.scaled_dot_product_attention` — and the two agree to 1e-6 in tests. KV-cache support reuses chapter 05's `KVCache`/`StaticKVCache`.
5. **Output gate.** The merged heads are multiplied by `sigmoid(gate)` before `o_proj` (transformers line 804; the config labels this `output_gate_type = "swish"` — `hybrid.py` defaults to the reference `sigmoid` and offers `gate_activation="silu"` explicitly).
6. **Parity.** With weights copied from HF `Qwen3_5Attention` (strict state-dict load — names and shapes already match), our output matches HF's to max-abs-diff 0.0 in fp32 once HF is given its production causal mask.

## What the old `DecoderLM` lacked

| Capability | `DecoderLM` (ch 07) | Hybrid stack |
|---|---|---|
| Layer dispatch | one block type × L | per-layer `linear_attention` / `full_attention` from `layer_types` |
| `head_dim` | forced `d_model // n_head` (5120/24 is not even an integer) | independent field (256) |
| Attention state | KV cache only | KV for full layers + recurrent `(S, conv_buffer)` for linear layers, one snapshot/restore |
| Q/K normalisation | none | per-head zero-centred RMSNorm |
| Output gate | none | fused q+gate projection, sigmoid gate before `o_proj` |
| RoPE | full-width, rotate-half | partial (prefix-only rotation, tail identity), mrope-aware config |
| Norm gains | ones-initialised `RMSNorm` (`y · w`) | zero-centred (`y · (1 + w)`) — checkpoint weights would be misread by the old norm |
| Weight format | dense, optionally tied head | FP8 block scales (`weight_scale_inv` per [128, 128] block), untied head, MTP + vision names in the checkpoint |

`ModelConfig` is deliberately untouched — old tests keep passing — and `HybridConfig` lives beside it in `hybrid.py`, accepting both HF (`hidden_size`, …) and repo (`d_model`, …) spellings.

## Mixed state: prefill, decode, snapshot

`TinyHybridLM` (embed → `HybridBlock`s → final norm → untied `lm_head`) runs prefill and decode through one path: `logits, state = model(idx)`, then `logits, state = model(next_ids, state)` with any `T ≥ 1` (`prefill`/`decode_step` are the named forms). `HybridState.layer_states[i]` holds `(k_all, v_all)` for full layers and `(S, conv_buffer)` for linear layers; `seq_len` is the next absolute position. `snapshot_state`/`restore_state` deep-copy both kinds, so a restored continuation is bit-identical (tested with `torch.equal`), and prefill-vs-token-step logits agree to ~1e-7 at tiny dims.

## Runtime and state: `HybridState`, explicit positions, `generate_hybrid`

The stack above is stateless maths; the runtime that makes it usable is three pieces, all in [hybrid.py](../src/inference_from_scratch/hybrid.py) (Kiely Inference Engineering sec. 2.2 for the config/mechanics framing, sec. 2.4–2.5 for why the state shapes below matter, sec. 5.4 for how this state would shard under tensor parallelism).

**Typed state.** `HybridState` stores one tagged entry per layer: `("kv", KVCache)` for full layers — a single-layer chapter-05 cache holding that layer's keys and values — and `("linear", (S, conv_buffer))` for gated-delta layers, with `("linear", None)` as the lazy zero state before the first token. `seq_len` counts positions already processed, so the next token's absolute position is always known. `clone()` / `snapshot()` / `restore()` deep-copy every tensor on both sides of the tag, which makes branching a first-class operation: snapshot once after a shared prompt, restore per branch, and each suffix generation proceeds without seeing the other branches' writes. The older untagged view survives as the `layer_states` property (`(k, v)` / `(S, conv)` pairs, `None` when empty), so pre-existing callers and tests keep working unchanged.

**Explicit positions.** `TinyHybridLM` and `GatedFullAttention` accept `position_ids` as `[T]` or `[3, T]`. The `[T]` form is text: the single axis is repeated to the temporal/height/width mRoPE axes, the interleaved recomposition (H owning pair slots 1, 4, 7, …, W owning 2, 5, 8, …, bounded by `mrope_section`) then writes each slot with the value already there, and the result is bit-identical to the default `start_pos` path — asserted with `torch.equal` in the tests. A vision-shaped `[3, T]` whose H/W axes differ changes exactly the pair slots those axes own and nothing else (also asserted, slot by slot, over the rotary pairs). Positions are absolute and are never offset by `start_pos`; the gated-delta layers receive no positions at all, since their recurrence carries no position embedding. This is the seam the deferred vision tower will use, and no vision tensor is executed here.

**Generation.** `generate_hybrid(model, idx, max_new_tokens, ...)` mirrors chapter 08's loop over the hybrid state: one prefill, then one decode step per token, sampling every step through `sampling.sample_next` (temperature / top-k / top-p / seeded generator, `temperature=0` for greedy, EOS included in the output and stopping only once every row has emitted it). Its one hybrid-specific knob is `prefill_method`: `"sequential"` runs the reference per-token scan for the prompt, `"chunkwise"` runs the chapter-21 chunked gated-delta form. The two agree only to fp32 tolerance in logits, but greedy token ids on the tiny models are asserted identical across both, and a snapshot-branching test asserts that two suffix generations restored from one prompt snapshot each reproduce a fresh from-scratch generation of the same suffix.

One audit note the runtime depends on: the checkpoint config labels the full-attention output gate `output_gate_type = "swish"`, but the reference implementation applies `attn_output * sigmoid(gate)` unconditionally (transformers `modeling_qwen3_5.py`, the `Qwen3_5Attention.forward` gate line). `silu(g) = g · sigmoid(g)` is a different function, so in this codebase `"swish"` resolves to `"sigmoid"` wherever it arrives — config or explicit argument — and `gate_activation="silu"` survives only as an explicitly non-HF alternative. The HF-parity test, which loads `Qwen3_5Attention` weights strictly and compares outputs, pins the sigmoid behaviour.

## MTP head anatomy (`mtp.safetensors`, 22 tensors)

Multi-token prediction: one extra full-attention decoder layer that predicts the *next-next* token from the current hidden state fused with the next token's embedding.

| Tensor | Shape | Role |
|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | [5120] | norm on the next-token embedding branch |
| `mtp.pre_fc_norm_hidden.weight` | [5120] | norm on the hidden-state branch |
| `mtp.fc.weight` | [5120, 10240] | projects the concatenated (hidden ‖ embedding) 10240 → 5120 |
| `mtp.layers.0.input_layernorm.weight` | [5120] | block pre-norm |
| `mtp.layers.0.self_attn.q_proj.weight` (+ `weight_scale_inv [96, 40]`) | [12288, 5120] | same gated-full-attention shapes as layer 3 |
| `mtp.layers.0.self_attn.k_proj / v_proj.weight` (+ scales [8, 40]) | [1024, 5120] | GQA KV projections |
| `mtp.layers.0.self_attn.o_proj.weight` (+ scale [40, 48]) | [5120, 6144] | output projection |
| `mtp.layers.0.self_attn.q_norm / k_norm.weight` | [256] | per-head norms |
| `mtp.layers.0.post_attention_layernorm.weight` | [5120] | MLP pre-norm |
| `mtp.layers.0.mlp.{gate,up}_proj.weight` (+ scales [136, 40]) | [17408, 5120] | SwiGLU up/gate |
| `mtp.layers.0.mlp.down_proj.weight` (+ scale [40, 136]) | [5120, 17408] | SwiGLU down |
| `mtp.norm.weight` | [5120] | final norm before the shared `lm_head` |

`text_config.mtp_num_hidden_layers = 1`, `mtp_use_dedicated_embeddings = false` (the MTP layer reuses the main embedding). Our `list_hybrid_tensor_map()` maps `mtp.*` names onto themselves — the anatomy is specified and name-mapped, but no MTP forward is implemented yet; speculative-decoding use (ch 14) is the natural consumer.

## Vision tower: deferred

`outside.safetensors` carries the vision tower (333 tensors: 27 blocks at hidden 1152, patch embedding, merger to 5120; `vision_config.depth = 27`, `patch_size = 16`). It is excluded from `list_hybrid_tensor_map()` and from `TinyHybridLM`: this chapter is text-only inference. The seam is known — vision tokens would enter as embeddings among text token ids (`image_token_id = 248056`) and activate the mrope H/W position axes noted above — but no vision tensor is read, mapped, or executed here.

## Check yourself

<details>
<summary>1. Why can't <code>ModelConfig</code> describe this checkpoint, even approximately?</summary>

Its `head_dim` is derived as `d_model // n_head`; here 5120/24 is not an integer and the real head_dim is an independent 256 (q width 6144 ≠ hidden 5120). It also has no layer-type dispatch, no recurrent state, no Q/K norms, no output gate, and full-width RoPE only.
</details>

<details>
<summary>2. A loader reads checkpoint <code>q_norm.weight</code> (mean ≈ 0.23) into a ones-style RMSNorm. What breaks?</summary>

Every gain is ~0.77 too small: the checkpoint stores zero-centred weights whose effective gain is `1 + w`. Shapes match, nothing crashes, and all Q/K magnitudes silently shrink.
</details>

<details>
<summary>3. Which layers grow state with sequence length, and by how much per token?</summary>

Only the 16 full layers: each appends 4 KV heads × 256 dims × 2 (K and V) values per token. The 48 linear layers keep a fixed `48 × 128 × 128` recurrent matrix plus a 3-vector conv buffer, independent of position.
</details>

## Streaming checkpoint loader (`hybrid_checkpoint.py`)

The loader that turns the on-disk checkpoint into tensors without ever holding the model (Kiely sec. 2.2 config/mechanics, sec. 4.2 formats/reference implementations; sharding context sec. 5.4, MTP/speculation context sec. 5.2). Three stages, strictly lazy:

1. **Manifest (headers only).** `discover_shards(dir)` resolves the 66 shard files from `model.safetensors.index.json` (falling back to a glob); `build_manifest(dir)` parses each shard with our own `weights.read_safetensors_header` — no tensor bytes are read — and emits entries `{checkpoint_name, target_name, shard, dtype, shape, bytes, kind}` with `kind` in `fp8_weight | scale_inv | bf16 | other`. Target names reuse `hybrid.list_hybrid_tensor_map`'s mapping; vision tensors map to `None` and stay deferred.
2. **Validation (formulas, not fixtures).** `expected_language_tensors(config)` derives every expected name and shape from `HybridConfig` arithmetic: embedding `[248320, 5120]`, per-layer widths by `layer_types` (e.g. full `q_proj [12288, 5120]`, linear `in_proj_qkv [10240, 5120]`), final norm, untied `lm_head`, and the MTP block, with each FP8 weight's `weight_scale_inv` at `[ceil(R/128), ceil(C/128)]`. `validate_manifest(manifest, config)` then reports `{missing, unexpected, shape_mismatch, ok, counts, total_bytes_by_dtype}`; vision tensors are out of scope on both sides.
3. **Lazy load (one tensor at a time).** `load_tensor(dir, checkpoint_name, dequantize=True)` opens only the owning shard with `safetensors.safe_open`, pairs an FP8 weight with its `*_weight_scale_inv` automatically, and returns BF16 via `quantization.dequantize_block_fp8`. Loading a scale tensor directly, an unknown name, or an FP8 weight with a missing scale raises a clear error. `iter_layer_tensors(dir, layer_idx)` is a generator over that primitive: only the requested layer materialises, scales are consumed by pairing and not yielded. `scripts/inspect_qwen38_checkpoint.py [--layer N]` prints the validation summary header-only; `--layer N` additionally loads only that layer's small norm tensors (never a full layer) to prove the lazy path.

Verified on the real snapshot (`scripts/inspect_qwen38_checkpoint.py`, CPU, headers + small norms only):

```
shards: 66
manifest entries: 1606
kinds: {'bf16': 792, 'fp8_weight': 407, 'scale_inv': 407}
total_bytes_by_dtype: {'BF16': 6167455584, 'F8_E4M3': 24699207680}
counts: {'expected': 1273, 'present': 1273, 'manifest_entries': 1606, 'missing': 0, 'unexpected': 0, 'shape_mismatch': 0}
ok: True
```

Total safetensors payload is 30,866,663,264 bytes (≈ 30.87 GB). The 1273 expected tensors are the language model + MTP + `lm_head`; the remaining 333 manifest entries are the deferred vision tower. Synthetic multi-shard fixtures and lazy-dequant parity are covered by `tests/test_hybrid_checkpoint.py`.

## Full-model assembly: mechanically provable without allocating 27B

`TinyHybridLM`'s `state_dict` naming already matches the checkpoint target names from `list_hybrid_tensor_map` (`embed_tokens.weight`, `layers.{i}.*` for both layer types, final `norm.weight`, `lm_head.weight`), so no renaming layer was needed; `HybridLM` in [hybrid.py](../src/inference_from_scratch/hybrid.py) is the general (any-size) name for the same stack, with `TinyHybridLM` preserved. Three functions in [hybrid_checkpoint.py](../src/inference_from_scratch/hybrid_checkpoint.py) assemble the full model on top of the manifest/`load_tensor` primitives above:

1. **`iter_checkpoint_state(snapshot_dir, config, include_mtp=False, dequantize=True)`** — a generator yielding `(target_name, tensor)` one tensor at a time in deterministic model order: `embed_tokens.weight`, per-layer tensors in `state_dict` order (linear: `dt_bias`/`A_log` before the `in_proj_*`/`conv1d`/`norm`/`out_proj` children; full: `q/k/v/o_proj`, `q/k_norm`; then the SwiGLU MLP), final `norm.weight`, then `lm_head.weight` (skipped when embeddings are tied). FP8 `weight_scale_inv` companions are consumed by dequantisation and never yielded; vision tensors are never yielded; MTP (`mtp.*`) is yielded only with `include_mtp=True`, appended after `lm_head` — excluded by default because the LM has no MTP parameters.
2. **`validate_state_against_model(model, snapshot_dir)`** — compares the header manifest against a (typically `meta`-device) model's `state_dict()` shapes. It returns exact `missing` (model order), `unexpected` (sorted; vision and, unless `include_mtp=True`, MTP excluded), `shape_mismatch` and `dtype_mismatch` lists, plus `ok`, `counts` and `parameter_count` computed from shapes. BF16-target policy: every target loads as BF16 (FP8 dequantises to BF16, BF16 loads as stored) and `copy_` casts into the parameter dtype, so the meta model's float32 dtype is not a mismatch and float storage (BF16/F32/F16 or FP8) is acceptable; non-float storage is a `dtype_mismatch`.
3. **`load_into_model(model, snapshot_dir, strict=True)`** — streaming load: it pre-validates with (2) *before* copying anything, so a strict failure raises `RuntimeError` with the model unmutated, then copies one tensor at a time via `iter_checkpoint_state` + `copy_` under `no_grad`. `strict=False` skips missing/mismatched/unexpected tensors and loads the rest. On the real 27B model this entry point is reserved for future GPU code; it is exercised end-to-end only on tiny synthetic fixtures.

Synthetic proof ([test_hybrid_assembly.py](../tests/test_hybrid_assembly.py)): a multi-shard checkpoint is built from a tiny `HybridLM`'s own `state_dict` transformed through the checkpoint naming, with one tensor artificially FP8-block-quantized. Meta-validation is `ok`; streaming iteration reproduces the model's `state_dict` order with no `scale_inv` yielded; a streaming load into a differently-seeded BF16 model reproduces logits with `torch.equal`, the quantized tensor loads exactly equal to its dequantized reference (tolerance only on that intentionally quantized path's logits), and a dropped `norm.weight` is reported as `missing == ["norm.weight"]` with the strict load raising before mutating any tensor.

Verified on the real snapshot (`scripts/validate_hybrid_assembly.py`, CPU, headers + meta shapes only, no tensor data loaded):

```
manifest entries: 1606
model state_dict tensors: 851
parameter_count: 26895998464
missing: []
unexpected: []
shape_mismatch: []
dtype_mismatch: []
counts: {'expected': 851, 'present': 851, 'manifest_entries': 1606, 'missing': 0, 'unexpected': 0, 'shape_mismatch': 0, 'dtype_mismatch': 0}
ok: True
```

The 851 model tensors are the language model without MTP (the 15 non-scale MTP tensors are out of scope unless `include_mtp=True`; the remaining manifest entries are 407 `scale_inv` companions and 333 deferred vision tensors). Meta-device construction of the real-config model needed no device-agnostic fixes: it already works, so the count above is computed from real shapes, not a formula.

## Real-layer parity: one linear + one full layer, measured

Until this section every real-checkpoint claim above was headers, shapes and names. [parity_real_layer.py](../scripts/parity_real_layer.py) closes the numeric loop for exactly one layer of each kind, on CPU, against transformers 5.20.1 in `.venv` (the reference implementation; Kiely sec. 4.2.4 stance):

```
.venv/bin/python scripts/parity_real_layer.py --layer 0 --tokens 4 --seed 0
.venv/bin/python scripts/parity_real_layer.py --layer 3 --tokens 4 --seed 0
.venv/bin/python scripts/parity_real_layer.py --layer 0 --tokens 4 --seed 0 --fp32
```

**Method.** The script builds the single HF `Qwen3_5DecoderLayer` for the real `text_config` at that index (layer 0 is `linear_attention`, layer 3 the first `full_attention`) and our `HybridBlock` at the same index. It loads, via `hybrid_checkpoint.iter_layer_tensors`, dequantized FP8→BF16 plus stored-BF16 tensors for *only that layer* (14 tensors for layer 0, 11 for layer 3; `weight_scale_inv` companions are consumed by dequantisation, never loaded as weights). Checkpoint target names `layers.{N}.<rest>` map explicitly to `<rest>`, the `state_dict` key in *both* modules; the script asserts `missing_hf=[] missing_ours=[] extra=[]` and exact key-set equality before `load_state_dict(strict=True)` into both. Both layers then receive the identical seeded hidden state `[1, 4, 5120]` (`randn * 0.5`, seed 0, bf16 — the checkpoint's dtype policy) plus the HF-signature position inputs: `Qwen3_5TextRotaryEmbedding` cos/sin from positions `[0..3]` for the full layer (with a production causal mask, as in the tiny-dims test) and no cache/mask for the linear layer (fresh zero recurrent state; HF evaluates it chunkwise, ours sequentially — the two production/reference forms of the same recurrence). Per-submodule diffs are isolated on a *common* input (each norm/mixer/MLP is fed the HF-side tensor, so errors are located, not cascaded), then the whole block is compared end-to-end. Tensors are deleted and `gc` run before exit; only one layer is ever materialised.

**Tolerance expectations.** bf16 outputs will not agree to 1e-6: bf16 carries ~3 decimal digits, so judge by relative error against the output RMS (`rel_rms = diff_rms / ref_rms`), not by absolute diff alone. The `--fp32` control reruns both sides with the same dequantized weights and inputs cast to fp32: a maths error (wrong q/gate split, ones- vs zero-centred norm gain, wrong RoPE width) would stay O(1) there, while dtype noise collapses to ~1e-6.

**Measured** (seed 0, T=4, this machine, CPU):

| Run | Submodule | max_abs | rel_rms |
|---|---|---|---|
| layer 0 (linear), bf16 | input_layernorm | 0 | 0 |
| layer 0 (linear), bf16 | mixer (gated delta) | 0.015625 | 0.00520 |
| layer 0 (linear), bf16 | mlp (common input) | 0 | 0 |
| layer 0 (linear), bf16 | whole block | 0.03125 | 0.00151 |
| layer 3 (full), bf16 | every submodule + whole block | 0 | 0 |
| layer 0 (linear), fp32 | mixer | 2.88e-06 | 6.73e-07 |
| layer 0 (linear), fp32 | whole block | 7.63e-06 | 1.21e-07 |
| layer 3 (full), fp32 | whole block | 0 | 0 |

Whole-block output RMS is 1.173 (layer 0) / 1.284 (layer 3) in bf16, output norms 167.8 / 183.8 over `[1, 4, 5120]`. Layer 3 agrees bit-exactly in both dtypes — both sides run the same bf16/fp32 matmuls, norms and eager attention arithmetic. Layer 0's only nonzero diffs are in the gated-delta mixer: `max_abs = 0.015625` is one bf16 ULP at that magnitude, and the fp32 control collapses the whole-block relative error from 1.5e-3 to 1.2e-7, which is the chunkwise-vs-sequential re-association signature, not a maths error. No engine fix was needed; the bisect (norms and MLP exact, mixer diff vanishing in fp32) is what rules out a root cause in our code.

The durable test is [test_real_layer_parity.py](../tests/test_real_layer_parity.py), env-gated (`RUN_REAL_LAYER_PARITY=1`, three subprocess runs of the script: both layers in bf16 plus the layer-0 fp32 control, asserting whole-block `rel_rms < 0.02` in bf16 and `< 1e-5` in fp32). It skips by default, and a skip is not counted as green evidence — the script runs above are the evidence. Measured gated run: `3 passed in 14.26s`.

## Full-model pipeline run across both GPUs: measured

[run_qwen38_gpu.py](../scripts/run_qwen38_gpu.py) runs the FULL real text model end-to-end (pipeline-by-layers bridge; true TP collectives are the next phase). It builds `HybridLM` from the real `HybridConfig` on meta, converts the 400 checkpoint FP8 linears to resident `BlockFP8Linear`, places embed + layers 0–31 on cuda:0 and layers 32–63 + final norm + `lm_head` on cuda:1 (`--split` CLI arg; `pipeline_device_plan` + `DeviceMappedHybridRunner` in [hybrid.py](../src/inference_from_scratch/hybrid.py)), streams tensors via `hybrid_checkpoint.load_into_model(fp8_mode="resident")` (never a full CPU state dict; `loaded_count=1251`, validation ok), and greedy-generates 32 tokens for the three [arm_a_quality_probes.json](../benchmarks/baseline/arm_a_quality_probes.json) prompts. Raw record: [ours_pipeline_first_run.json](../benchmarks/baseline/ours_pipeline_first_run.json).

Measured (2× RTX 4090, torch 2.11.0+cu128 in `.venv-gpu`, BF16 activations, resident-FP8 reference-dequant forward, single stream):

| Prompt | Ours vs vLLM ids | First divergence | Prefill (s) | Decode (tok/s) |
|---|---|---|---|---|
| capital | 32/32 exact | none | 0.93 | 1.83 |
| water | 32/32 exact | none | 0.71 | 1.83 |
| counting | 32/32 exact | none | 0.78 | 1.83 |

Load time 25.6 s (streamed resident load). Peak memory 15,887,028,736 B (cuda:0) / 15,934,093,824 B (cuda:1). Generated text is identical to the recorded vLLM text on all three probes. Decode is slow by design: every `BlockFP8Linear` dequantises its full weight matrix per forward, so each token costs a full 27B dequant+matmul pass — 1.83 tok/s is the unoptimized-bridge number, not a target. A `--log-rms` flag records per-layer hidden RMS on the first prefill for divergence bisect (not needed here: logits were finite and exact throughout).

## Summary in 30 seconds

Parse the real `text_config` into `HybridConfig`; dispatch each block by `layer_types`; run linear layers as gated-delta recurrences (ch 21) and full layers as gated attention — fused per-head Q+gate split, per-head zero-centred Q/K norms, 64-of-256 partial RoPE, GQA over a reused KVCache, sigmoid output gate — behind one mixed snapshot/restore, with checkpoint names mapped from shard headers and the MTP/vision towers specified but deliberately not executed.
