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

## Summary in 30 seconds

Parse the real `text_config` into `HybridConfig`; dispatch each block by `layer_types`; run linear layers as gated-delta recurrences (ch 21) and full layers as gated attention — fused per-head Q+gate split, per-head zero-centred Q/K norms, 64-of-256 partial RoPE, GQA over a reused KVCache, sigmoid output gate — behind one mixed snapshot/restore, with checkpoint names mapped from shard headers and the MTP/vision towers specified but deliberately not executed.
