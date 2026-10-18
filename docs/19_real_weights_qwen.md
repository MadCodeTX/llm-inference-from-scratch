# 19 — Real weights: Qwen2.5-0.5B-Instruct in our engine, proved against HF

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 2.2.1 (`config.json` as the architecture source of truth), sec. 4.2.2 (model file formats), and sec. 4.2.4 (HF transformers as the reference implementation to learn from and verify against, not the production path). **Production question:** does the hand-built engine in chapters 00–09 actually run a real checkpoint, token-for-token identically to the reference? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Point the existing `DecoderLM` at a downloaded snapshot's own `config.json` and `model.safetensors`, cast to fp32 on CPU, and the proof of correctness is not "it generates plausible text" but three exact comparisons against the HF reference on identical raw token ids: first-forward logits, per-step greedy top-1, and the full 20-token generated id sequence.

## Why inference needs this

Every earlier chapter was verified on tiny random models, where a wrong RoPE pairing, a missing bias, or a transposed projection can still produce *shaped* outputs and passing shape tests. A real checkpoint is the first oracle that punishes those bugs semantically: its weights encode one specific function, and any architectural deviation shows up as diverging tokens within a few steps. This chapter is the rung of the Tiny-to-Qwen ladder where "the machinery is tested on hand-built checkpoints" becomes "the engine runs the real weights, proved."

## The checkpoint, read from disk

Model: `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face Hub (public), downloaded with `snapshot_download` into the git-ignored `.hf-cache/` (never committed; `.gitignore` gained `.hf-cache/` before any download). Everything below was read from the snapshot's own files at run time.

**Annotated `config.json`** (snapshot `config.json`, 659 bytes; values verbatim):

| HF field | Real value | `ModelConfig` field | Meaning |
|---|---|---|---|
| `architectures` | `["Qwen2ForCausalLM"]` | — (family detection) | decoder-only causal LM; also the Qwen2 bias evidence |
| `hidden_size` | 896 | `d_model` | residual-stream width `D` |
| `num_hidden_layers` | 24 | `n_layer` | block count `L` |
| `num_attention_heads` | 14 | `n_head` | query heads `H` |
| `num_key_value_heads` | 2 | `n_kv_head` | KV heads (GQA: `n_rep = 14 // 2 = 7`) |
| `intermediate_size` | 4864 | `d_ff` | SwiGLU hidden width |
| `vocab_size` | 151936 | `vocab_size` | token id range; embedding is `[151936, 896]` |
| `max_position_embeddings` | 32768 | `max_seq_len` | RoPE/cache position bound |
| `rope_theta` | 1000000.0 | `rope_theta` | long-context RoPE base |
| `rms_norm_eps` | 1e-06 | `rms_eps` | RMSNorm epsilon |
| `tie_word_embeddings` | true | `tie_embeddings` | head shares the embedding matrix |
| `torch_dtype` | `bfloat16` | — (see dtype below) | storage dtype of the checkpoint |
| `hidden_act` | `silu` | `mlp_type="swiglu"` | gated MLP activation |
| `use_sliding_window` | false | — (not implemented) | see below |
| `sliding_window` / `max_window_layers` | 32768 / 21 | — | window equals max context; moot while disabled |

Derived, never stored: `head_dim = 896 // 14 = 64`. There is **no `attention_bias` key** in the file — that absence is the bias story below.

**On-disk layout** (sec. 4.2.2): one `model.safetensors` (988,097,824 bytes), 290 tensors, 494,032,768 parameters, every tensor BF16 — read with this repo's own hand parser (`read_safetensors_header`), not the `safetensors` library. `lm_head.weight` is absent (tied); the mapper aliases it to the embedding tensor, so 291 tensors load into `DecoderLM` (494,032,768 unique params).

**Sliding window, honestly:** not implemented. The real config disables it (`use_sliding_window: false`), and its window (32768) equals `max_position_embeddings`, so even enabled it could not truncate any sequence this engine is verified on. Implementing it would be dead code for short-sequence equivalence.

## The bias gap, and the fix

First contact exposed exactly one architectural gap. The checkpoint carries 72 bias tensors — per layer `self_attn.q_proj.bias [896]`, `k_proj.bias [128]`, `v_proj.bias [128]` (128 = `2 KV heads × 64 head_dim`) — and the original bias-free projections had no slot for them: validation reported them as *unexpected*, and dropping them would have silently changed every attention output.

Fix (minimal, in the engine — not a tolerance): `ModelConfig.attention_bias` (default `False`, so Llama-style configs are unchanged); `CausalSelfAttention` builds Q/K/V with `bias=attention_bias` while `o_proj` stays bias-free (the checkpoint has no `o_proj` bias either); `weights.load_hf_config` infers the flag via `_infer_attention_bias` — an explicit `attention_bias` key wins, otherwise the Qwen2 family (`model_type`/`architectures`) implies `True` and Llama `False`. The safetensors header remains ground truth: with the wrong flag, `validate_state_dict` flags the disagreement instead of hiding it. After the fix, the real checkpoint validates with **zero missing, zero unexpected, zero shape mismatches**.

## Dtype decisions

- Checkpoint storage is BF16; both engines here run **fp32 on CPU**. The `.venv` torch build is CPU-only (`torch 2.14.1+cpu`, `torch.cuda.is_available() == False`) even though an RTX 4090 is present — deliberate: fp32 CPU arithmetic is the most reproducible common ground between two implementations, so a mismatch can only mean an architecture bug, not kernel rounding. Installing a CUDA torch build is future work, not a requirement for correctness proof.
- Cast happens once, at load: every mapped tensor `.to(torch.float32)` before the strict `load_state_dict`. RMSNorm already upcasts internally (ch 03); RoPE tables are fp32 by construction (ch 02).
- The HF reference is loaded with `dtype=torch.float32` for the same reason. Comparing our fp32 against HF bf16 would have priced dtype noise into every number below.

## Comparison method and numbers

Script: [load_qwen_small.py](../scripts/load_qwen_small.py) (`--model-dir <snapshot>`). One fixed plain-completion prompt, encoded by `AutoTokenizer` from the same snapshot and fed as **raw ids to both sides — no chat template anywhere** (a chat-template asymmetry would compare two different prompts, not two engines):

```text
"The capital of France is" -> ids [785, 6722, 315, 9625, 374]
```

Three comparisons, all on CPU fp32:

| Check | Method | Result |
|---|---|---|
| (c) First-forward logits | one forward over the prompt, both engines, max-abs-diff over `[1, 5, 151936]` | **2.098e-05** |
| (a) Per-step top-1 | 20 hand-rolled greedy steps per side (argmax, teacher-forced on own picks) | **20/20 agree** |
| (b) Generated ids | our `generate()` (greedy, KV cache) vs HF greedy ids | **exactly equal** |

The 2.1e-05 logits gap is accumulated fp32 reduction-order drift across 24 layers (per-layer hidden-state diffs stay at 1e-06–1e-05 scale), three orders below the <1e-3 "acceptable drift" bar — and with ids exactly equal, it is drift, not a bug. Debugging path, had it diverged: compare embedding output, block-0 attention output, block-0 output, and final norm against HF `output_hidden_states` in a throwaway `/tmp` script, layer by layer, until the first diverging tensor names the guilty submodule. That probe was run during development: embeddings match exactly (0.0) and blocks match to ~1e-06.

**Sample generation** (real decoded output, both engines identical):

> `The capital of France is Paris. It is the largest city in Europe and the third largest city in the world. It is`

**A trap worth naming:** the checkpoint's `generation_config.json` ships sampling defaults (`do_sample`, temperature/top-k/top-p, and a repetition penalty). Calling HF `generate(do_sample=False)` still inherits the repetition penalty and *diverges from step 4* — while hand-rolled greedy on both sides agrees exactly, and HF's own full-forward argmax on our sequence agrees with us at every position. The lesson generalises: a reference comparison must control the sampler explicitly, or it benchmarks two decoding rules, not two engines. Our cached and no-cache paths also agree token-for-token on this model, re-confirming ch 05/08 at real scale.

## CPU timing and roofline (real dims, this machine)

`metrics.measure_generate` on the real model (5-token short run, CPU fp32, single run — indicative, not a benchmark): **TTFT 0.043 s, decode TPS ≈ 28.2, end-to-end TPS ≈ 27.1**. TTFT here is one 5-token prefill plus first-token sampling, end-to-end wall time including Python overhead (sec. 1.4 distinction, ch 13).

`bottlenecks.classify` with the real config dims against `make_example_spec()` — **explicitly illustrative round numbers (100 TFLOP/s, 1 TB/s, ridge 100 ops/byte), not any real GPU** — priced in fp32 bytes (weight/kv 4 bytes, matching how the model actually ran), `S = 25`:

| Phase | FLOPs/token | Bytes/token | Intensity | Bound (illustrative) |
|---|---|---|---|---|
| Prefill | 9.890e08 | 7.907e07 | 12.51 | memory |
| Decode | 9.901e08 | 1.977e09 | 0.50 | memory |

Decode's byte term is the whole fp32 weight set (~1.98 GB) re-read per token; that, not arithmetic, is why CPU decode sits at tens of tokens per second, and why the roofline — even with invented device peaks — already tells you quantisation and batching (ch 10/11) are the levers, not faster matmuls.

## What changes at 27B scale

Nothing structural — the same `config.json` fields drive every formula; only the inputs grow (read them from that model's real `config.json` at load time; this repo never hard-codes a large model's dims):

- **Weights:** bytes `= params × bytes_per_param`. Halving precision (bf16→int8→int4, ch 10) divides decode's dominant traffic term by the same factor.
- **KV cache:** bytes `= 2 × L × H_kv × head_dim × S × kv_bytes` per sequence. GQA (`H_kv << H`) and short KV dtypes are what keep long-context decode feasible; the cache grows linearly in `S` while weights do not.
- **Decode intensity:** `≈ 2·P_matmul / (P·weight_bytes + KV_read)` stays ~1 op/byte at batch 1 regardless of scale — bigger models are *more* memory-bound per token, and batching (sharing the weight read) plus KV compression are the only terms that move it.
- **Prefill intensity** grows with `S` (weight reads amortise over the prompt), so prefill crosses toward compute-bound while decode never does.
- **Loading:** sharded safetensors + `model.safetensors.index.json`, mmap/lazy reads (ch 09) — the three translations are unchanged; only *when bytes move* changes.

## How the big systems do it

Production loaders (vLLM, TensorRT-LLM) perform these same three translations, then add tensor parallelism, fused kernels, and paged KV management (ch 11/16). HF `transformers` here plays exactly the sec. 4.2.4 role: the readable reference implementation you verify a hand-built engine against before trusting it — not the serving path. The verification habit transfers verbatim: fixed raw prompt, fp32 common ground, per-layer hidden-state diffing to localise the first divergence.

## Common pitfalls

- **Trusting plausible text.** Wrong-but-shaped engines generate fluent continuations from real weights for a few tokens. Only id-exact comparison against a reference is proof.
- **Inherited sampler settings.** A checkpoint's `generation_config.json` can silently change the reference's decoding rules (above). Pin every sampling knob explicitly.
- **Mixing dtypes in a comparison.** fp32-vs-bf16 comparisons measure rounding, not architecture. Cast both sides to one dtype first.
- **Dropping "unexpected" tensors.** Unexpected keys are a finding (here: the QKV biases). Dropping them to make a strict load pass manufactures a subtly wrong model.

## Break it / exercises

1. Set `attention_bias=False` in the loaded config and re-validate. Which report list fills, and why is that the *good* failure mode?
2. Feed the prompt through the model's chat template on one side only. Where does the comparison first diverge, and why is that not an engine bug?
3. Re-run the comparison in bf16 on both sides. How does the logits max-abs-diff move, and do the ids survive?

## Check yourself

<details>
<summary>1. The real config has no <code>attention_bias</code> key, yet our model loads QKV biases. Where did the True come from, and what catches a wrong guess?</summary>

`load_hf_config` infers it from the Qwen2 family (`model_type`/`architectures`) when the key is absent; `validate_state_dict` against the safetensors header catches any disagreement as missing/unexpected keys.
</details>

<details>
<summary>2. Why is "it generated a fluent sentence" not evidence the load is correct?</summary>

Real weights make even architecturally wrong engines locally fluent. Proof is exact: logits diff at fp32-drift scale, per-step top-1 agreement, and id-exact generations against the reference on identical raw ids.
</details>

<details>
<summary>3. HF <code>generate(do_sample=False)</code> diverged from our greedy output. Who was wrong?</summary>

Neither engine: HF inherited a repetition penalty from the checkpoint's `generation_config.json`. Hand-rolled greedy on both sides agrees 20/20, and HF full-forward argmax confirms our sequence at every position.
</details>

## Summary in 30 seconds

Real `config.json` in, validated safetensors in, fp32 CPU forward — and three exact comparisons (logits 2.1e-05, top-1 20/20, ids identical) against the HF reference turn "loads a real checkpoint" into "proved equivalent on this prompt." The one real bug found on the way was architectural (missing QKV biases), fixed in the engine; the one false alarm was a sampler default, fixed in the comparison.

## Where to go next

Back to ch 10 (quantization) with real stakes: decode here is weight-traffic-bound, so every byte-per-param saved is decode speed. A CUDA torch build and longer-context runs are the honest next rungs; the comparison script already generalises (`--prompt`, `--max-new-tokens`).

### Reader checkpoints

- Read any field of the real `config.json` and name the `ModelConfig` field and tensor shape it controls.
- Explain why both sides run fp32 on CPU for the proof, and what a CUDA build would and would not change about the verdict.
- Reproduce the comparison: `.venv/bin/python scripts/load_qwen_small.py --model-dir <snapshot>` and confirm zero validation errors, 20/20 top-1, and identical ids.
- State the KV-bytes and decode-bytes formulas and what each term does as `S` and batch grow.
