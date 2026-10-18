# 09 — Weights: safetensors by hand, HF config, and the Qwen name mapping

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 4.2.2 and 4.2.4. **Production question:** how do checkpoint bytes and a `config.json` become a runnable model, and how are loading mistakes caught before generation? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Turning a checkpoint on disk into a runnable model is three explicit translations — bytes→header, HF config→`ModelConfig`, HF tensor names→`DecoderLM` names — and each one is done by hand here so nothing about the format stays magic.

## Why inference needs this

A checkpoint is not a model. Before a single token can be generated from real weights, a loader must answer: what tensors exist, what shape/dtype is each, where do its bytes live, what architecture do they belong to, and which slot in *our* model does each fill? Production loaders hide those questions behind `from_pretrained`. Dissecting them is what makes "load Qwen weights into our `DecoderLM`" a renaming exercise instead of a leap of faith — and what makes failures (missing keys, shape mismatches, stray biases) diagnosable instead of silent.

## The math/shapes

**The safetensors container.** No pickle, no executable code — just length-prefixed JSON plus raw bytes:

```text
| 8 bytes        | N bytes      | rest of file     |
| uint64 LE = N  | UTF-8 JSON   | raw tensor bytes |
| header length  | header       | "data buffer"    |
```

The header maps each tensor name to `{"dtype": "F32", "shape": [V, D], "data_offsets": [begin, end)}`, plus one optional `"__metadata__"` string→string entry. Offsets are relative to the **start of the data buffer** (file offset `8 + N`), `end` exclusive. Tensors are contiguous, row-major, little-endian.

**Concrete example.** A file whose first 8 bytes are `40 00 00 00 00 00 00 00` declares `N = 0x40 = 64`. Bytes 8..72 are the JSON header; the data buffer starts at file offset 72. An entry `"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}` occupies file bytes 72..96 — check: `2*3` elements × 4 bytes = 24 = `end - begin`. That arithmetic, `numel(shape) * dtype_bytes == end - begin`, is the integrity check a correct reader can always run. (Sharded checkpoints split tensors *across* files, never within one entry; real files may leave offset gaps, so honour the offsets exactly.)

**HF config → our config.** The field mapping (`hidden_size`→`d_model`, etc.) is tabulated in the walkthrough below. `head_dim` is never a config field: it is derived, `hidden_size / num_attention_heads`.

**Never fabricate a large model's numbers.** Every dimension of a real checkpoint — layers, heads, KV heads, `d_ff`, vocab — is read from *its* `config.json` at load time by `load_hf_config`. If a checkpoint is absent, the correct move is to state which fields you *would* read and give the formulas (chapters 05/10), not to quote dimensions from memory.

## Walk the code

All of chapter 09 is [weights.py](../src/inference_from_scratch/weights.py). No `safetensors` library is used for parsing.

- `read_safetensors_header(path) -> dict` (`weights.py:209`): reads exactly 8 bytes, unpacks the little-endian uint64 (`struct.Struct("<Q")`, `weights.py:129`), reads that many JSON bytes, and validates every entry via `_validate_header` (`weights.py:168`) — dtype must be in `_DTYPE_TO_TORCH` (`weights.py:133`), shapes must be int lists, offsets must be consistent. `safetensors_table` / `format_safetensors_table` (`weights.py:246`) render the header as an inspectable table.
- `LazySafetensorsFile` (`weights.py:300`): parses only the header up front; `load(name)` (`weights.py:348`) seeks to `8 + N + begin` and reads one tensor's byte range, converting via `_tensor_from_bytes` (`weights.py:286`, BF16 handled through its uint16 bit pattern since NumPy has no BF16 dtype). `load_safetensors(path)` (`weights.py:378`) is the eager counterpart.
- `load_hf_config(path_or_dict)` (`weights.py:404`): accepts a dict, a `config.json` path, or its directory. Mapping: `vocab_size→vocab_size`, `hidden_size→d_model`, `num_hidden_layers→n_layer`, `num_attention_heads→n_head`, `num_key_value_heads→n_kv_head`, `intermediate_size→d_ff`, `max_position_embeddings→max_seq_len`, `rope_theta→rope_theta`, `rms_norm_eps→rms_eps`, `tie_word_embeddings→tie_embeddings`. Defaults follow HF: missing `num_key_value_heads` ⇒ `n_head` (plain MHA), `rope_theta` ⇒ 10000 (a nested `rope_scaling.rope_theta` is honoured), `rms_norm_eps` ⇒ 1e-6, tie ⇒ False, `max_position_embeddings` ⇒ 32768. The five shape-defining fields are **required** — a `ValueError` names every missing one rather than inventing a number. Flavour is fixed to Llama/Qwen: `rmsnorm` + `swiglu` + `rope`.
- `map_hf_state_dict(sd, tie_embeddings=False)` (`weights.py:551`): pure renaming, values are the same objects (no copy, no transpose — HF and this repo both store Linear weights `[out, in]`). Key rows: `model.embed_tokens.weight→tok_embedding.weight`; `model.layers.{i}.input_layernorm→layers.{i}.attn_norm`; `...self_attn.{q,k,v,o}_proj→layers.{i}.attn.{q,k,v,o}_proj`; `...post_attention_layernorm→layers.{i}.mlp_norm`; `...mlp.{gate,up,down}_proj→layers.{i}.mlp....`; `model.norm.weight→final_norm.weight`; `lm_head.weight→lm_head.weight`. Two honest caveats are documented in the module: Qwen2-style QKV **biases** rename along but have no home in our bias-free model (validation reports them as unexpected), and tied checkpoints usually omit `lm_head.weight` — with `tie_embeddings=True` the mapper aliases it to the embedding tensor.
- `validate_state_dict(mapped, model)` (`weights.py:590`): returns `{missing, unexpected, shape_mismatch, ok}` sorted deterministically, tie-aware (either tied name present covers both), *before* any `load_state_dict` is attempted.

Tests: [test_weights.py](../tests/test_weights.py) hand-builds safetensors files byte-by-byte, checks header/table/roundtrip, the config defaults and missing-field errors, and pins the rename table against a live `DecoderLM.state_dict()`.

## How the big systems do it

**Qwen-family checkpoints** ship as `config.json` + one or more `model-*.safetensors` shards with a `model.safetensors.index.json` mapping tensor names to shard files. Everything above still applies per-shard; production loaders add mmap zero-copy reads (pages faulted in on first touch), dtype casting on load, and device placement. To inspect a real one: read its `config.json` fields with `load_hf_config`, list its header with `read_safetensors_header`, rename with `map_hf_state_dict`, and pre-flight with `validate_state_dict` — in that order, before copying a single weight.

**vLLM-style serving** loads through the same translations, then shards tensors across GPUs (tensor parallelism) and optionally quantizes on the fly. Its loader is this chapter plus distribution.

**llama.cpp / GGUF** converts HF checkpoints *offline* into a single `.gguf` file with its own header (tensor table + tokenizer + config in one container — compare the safetensors layout above) and often pre-quantized weights. The conversion script performs exactly our three translations; GGUF just freezes their result.

## Worked: loading a checkpoint, in order

The safe sequence for any real HF checkpoint directory `ckpt/` (do not skip steps):

1. `cfg = load_hf_config("ckpt/config.json")` — every dimension now comes from disk. Print it: `hidden_size→d_model`, `num_hidden_layers→n_layer`, heads, KV heads, `intermediate_size→d_ff`, `vocab_size`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`. If a required field is missing you get a `ValueError` naming it — fix the path, not the code.
2. `model = DecoderLM(cfg)` — an untrained shell with exactly the right shapes.
3. `hdr = read_safetensors_header("ckpt/model.safetensors")` (per shard if sharded) — eyeball `format_safetensors_table(hdr)`: names, dtypes (expect F32/BF16), shapes consistent with `cfg`.
4. `sd = load_safetensors(...)` or lazy-load per tensor; `mapped = map_hf_state_dict(sd, tie_embeddings=cfg.tie_embeddings)`.
5. `report = validate_state_dict(mapped, model)` — require `report["ok"]`. Missing keys mean the mapping or config is wrong; unexpected keys usually mean checkpoint biases our bias-free model cannot host (a *finding* to report, not to silently drop); shape mismatches mean the config did not come from this checkpoint.
6. Only now: `model.load_state_dict(mapped, strict=True)`, then a greedy generation smoke test against a reference implementation if one is available.

Skipping step 5 is how "it loaded" becomes "it generates garbage": `load_state_dict(strict=False)` plus a rename bug is indistinguishable from a working load until text comes out wrong.

> **FP8 note:** Real FP8 checkpoints (e.g. Qwen3.8-27B-FP8) use safetensors dtypes `F8_E4M3`/`F8_E5M2` (1 byte/element). `weights.py` maps them to `torch.float8_e4m3fn`/`torch.float8_e5m2`, so `read_safetensors_header` validates their offset spans (`numel * 1` byte) and `LazySafetensorsFile.load` materialises them via a uint8 bit-reinterpretation (NumPy has no float8 dtype). Each FP8 weight is paired with a bf16 `…weight_scale_inv` block-scale tensor — see ch 10's block-FP8 section.

## Common pitfalls

- **Guessing dimensions.** Any hard-coded layer/hidden count — for any model, however famous — rots the moment a checkpoint variant differs. The config file is the only source (step 1).
- **Assuming one file.** Sharded checkpoints need the index (`model.safetensors.index.json`) to know which shard holds which tensor; each shard is still the same 8-byte-header format.
- **Copying during rename.** `map_hf_state_dict` deliberately returns the same tensor objects. A "helpful" `.clone()`/transpose pass doubles memory for a 50 GB checkpoint and can silently corrupt layout.

## Break it / exercises

1. Hand-craft a minimal safetensors file in Python (`struct.pack("<Q", len(header)) + header_json + raw_bytes`) with one `[2, 3]` F32 tensor, then read it back with `read_safetensors_header` and `LazySafetensorsFile.load`. Verify the offset arithmetic by hand.
2. Corrupt one copy: change a `data_offsets` end so `end - begin != numel * dtype_bytes`. Where does validation catch it — and if you bypass that, what silently goes wrong?
3. Call `load_hf_config` on a dict missing `intermediate_size`. Read the `ValueError`. Why is raising better than defaulting `d_ff` to, say, `4 * hidden_size`?

## Check yourself

<details>
<summary>1. A tensor entry says shape [4, 8], dtype F16, offsets [100, 164]. Consistent?</summary>

`4*8 = 32` elements × 2 bytes = 64 = `164 - 100`. Yes, consistent.
</details>

<details>
<summary>2. Why does <code>map_hf_state_dict</code> never transpose anything?</summary>

Because HF Llama/Qwen and this repo both store Linear weights as `[out, in]` and embeddings as `[V, D]`. The architectures are the same shape; only the *names* differ. A transpose would be a bug, not a conversion.
</details>

<details>
<summary>3. A tied Qwen checkpoint has no <code>lm_head.weight</code>. What does strict loading into a tied <code>DecoderLM</code> need?</summary>

`map_hf_state_dict(sd, tie_embeddings=True)`, which aliases `lm_head.weight` to the embedding tensor — mirroring the shared Parameter in the tied model. Without the flag, `validate_state_dict` still judges the omission tie-aware and does not call it missing.
</details>

## Summary in 30 seconds

This chapter's core is safetensors bytes, HF config translation, name mapping, validation. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 10 (quantization), which shrinks the tensors loaded here. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_weights.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.

Keep the header table from a real inspection next to you for the rest of the book: it is the ground truth for parameter counting (ch 10's `P`), for shape validation against `ModelConfig`, and for spotting surprises like QKV biases or an unexpected rotary buffer. Ten minutes with `format_safetensors_table` on a real checkpoint teaches more than any diagram of the format, and it costs nothing but reads.
