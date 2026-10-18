# 20 — Model file formats: safetensors vs GGUF vs pickle, dissected in GGUF bytes

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 4.2.2 (model file formats), sec. 4.2.4 (HF transformers as the reference implementation to learn from and verify against, not the production path), and sec. 3.5 (local inference). **Production question:** why does an on-device runtime ship one self-contained `.gguf` file instead of a `config.json` plus safetensors shards — and what exactly is inside that file? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

GGUF packs architecture hyperparameters, tokenizer, and (usually block-quantized) tensors into one little-endian binary container with an explicit header a runtime can parse and `mmap` in milliseconds — and this chapter parses that header by hand, byte by byte, the way chapter 09 parsed safetensors.

## Why inference needs this

Chapter 09 turned a checkpoint into a runnable model in three translations: bytes→header, `config.json`→`ModelConfig`, HF names→our names. A phone or laptop runtime cannot assume that sidecar ecosystem exists. It gets one downloaded file and must answer every loader question from that file alone: what architecture, what tokenizer, what quantization, where is each tensor? GGUF is the answer the llama.cpp ecosystem converged on, and understanding its header is understanding what an on-device (Glimmer-style) deployment actually ships.

No real `.gguf` file is downloaded in this chapter. The parser is proved on tiny synthetic GGUF files built byte-by-byte in the tests (the same discipline as ch 09's hand-built safetensors). The real Qwen2.5-0.5B numbers quoted in ch 19 come from that snapshot's own `config.json`; a future conversion of that checkpoint to GGUF would carry those same fields as GGUF metadata — read from the file, never copied from memory.

## The math/shapes

**The GGUF container (version 3).** All integers little-endian. A *string* is a `uint64` length plus UTF-8 bytes, no terminator:

```text
| 4 B "GGUF" | 4 B version=3 | 8 B tensor_count | 8 B metadata_kv_count |
| metadata entries (kv_count of them) ...                                |
| tensor infos (tensor_count of them) ...                                |
| zero padding to alignment | raw tensor data ...                        |
```

Each metadata entry is `key: string`, `value_type: uint32`, then the value. The twelve value types are `UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64` (codes 0–12). An `ARRAY` is `uint32 elem_type + uint64 length + elements`; elements are any scalar or string type (nested arrays are not allowed). Tokenizer vocabularies ride in exactly this mechanism, e.g. a string-array metadata entry holding every token.

Each tensor info is `name: string`, `n_dims: uint32`, `dims: uint64 × n_dims`, `type: uint32` (a ggml type code), `offset: uint64`.

**Alignment arithmetic — the formula this chapter exists to pin:**

```text
header_end  = file offset just after the last tensor info
alignment   = metadata["general.alignment"]   (uint32; default 32 if absent)
data_start  = ceil(header_end / alignment) * alignment
absolute_offset(tensor) = data_start + tensor.offset
```

**Concrete example.** Suppose the tensor infos end at file offset 100 and `general.alignment` is 32. Then `data_start = ceil(100/32)*32 = 128`: bytes 100–127 are zero padding. A tensor whose info declares `offset = 48` starts at file byte 176. Reading at `100 + 48 = 148` instead silently reads padding and the previous tensor's tail — which is why the parser returns `header_end`, `alignment`, and `data_start` explicitly instead of hiding them.

**ggml dimension order.** `dims` are stored fastest-varying-first, the reverse of the PyTorch/safetensors convention. A logical `[V, D]` embedding stored by a converter appears on disk with `dims = [D, V]`. The parser returns the on-disk order verbatim; reversing is a tensor-loading decision, documented here rather than guessed in the parser.

**ggml type codes.** The `type` field names the storage format of the tensor data: plain types (`F32=0`, `F16=1`, `BF16=30`, integer types) and block-quantized types (`Q4_0=2`, `Q8_0=8`, the `Q*_K` K-quant family, importance-quantized `IQ*` types, …). `ggml_type_name(code)` maps the common codes and returns `UNKNOWN_<code>` for anything newer, so an inventory tool never crashes on a file from a newer converter.

## Walk the code

Everything is in [gguf.py](../src/inference_from_scratch/gguf.py). No `gguf` library is used; only `struct`.

- `read_gguf_header(path) -> GGUFHeader` ([gguf.py](../src/inference_from_scratch/gguf.py)): validates the `GGUF` magic and version (2 and 3 accepted), reads `tensor_count` and `metadata_kv_count` as `uint64`, then walks every metadata entry and tensor info with a small sequential `_Reader`. Tensor *data* is never read — the function is O(header) on a multi-GB file. Raises `ValueError` on bad magic, unsupported version, truncation, invalid UTF-8, unknown metadata value types, invalid BOOL bytes, nested arrays, and a non-integer or non-positive `general.alignment`.
- `GGUFHeader(version, tensor_count, metadata, tensors, alignment, data_start, header_end, metadata_kv_count)`: `metadata` is a plain dict of Python values (arrays become lists); `tensors` is a list of `TensorInfo(name, n_dims, dims, ggml_type, offset)` with `.type_name`, `.numel()`, and `.absolute_offset(data_start)`.
- `ggml_type_name(code) -> str`: the code→name table for plain and block-quantized ggml types, `UNKNOWN_<code>` fallback.
- `summarize(path) -> dict`: the whole header as plain JSON-serialisable data — version, counts, metadata, alignment, `header_end`, `data_start`, and one row per tensor including `absolute_offset` and `numel`.

The writer is deliberately **not** in the library. [test_gguf.py](../tests/test_gguf.py) carries `build_gguf_bytes(metadata, tensors)` — a `struct`-only serialiser for tiny valid GGUF files — so the parser is tested against the layout, not against itself. Tests cover: roundtrip of every scalar metadata type, array metadata (string/uint32/float32/bool/empty arrays), bad magic, unsupported version, truncation, unknown value types, explicit vs sequential tensor offsets, and data-start arithmetic under default (32), custom (16), and absent alignment metadata.

Run them: `.venv/bin/python -m pytest tests/test_gguf.py -q`.

## Three formats, honestly compared

| Property | safetensors | GGUF | pickle (`torch.save` / `.bin`) |
|---|---|---|---|
| Safety | Data-only; no code execution on load | Data-only; no code execution on load | **Executable**: unpickling runs arbitrary code — never load untrusted pickles |
| Header | 8-byte length + JSON | Binary: typed metadata KVs + tensor infos | Python object graph; format is the pickler, not a spec |
| Laziness / mmap | Yes: header first, tensors page in on touch | Yes: header first, whole file typically `mmap`ed | Poor: historically eager, whole-graph materialisation |
| Metadata | One optional string→string `__metadata__` | Rich typed metadata: architecture, tokenizer tokens/merges, chat template, quantization params | Anything picklable — which is the problem, not a feature |
| Tokenizer in file | No (sidecar files) | Yes | Possible, by convention only |
| Architecture in file | No (`config.json` sidecar) | Yes (e.g. `qwen2.block_count`, `*.context_length`) | By convention only |
| Quantization | Dtypes only; quantized schemes need sidecar conventions | First-class block-quantized ggml types (`Q4_0`, `Q8_0`, K-quants) with per-block scales in the data layout | Whatever the saving code invented |
| Ecosystem | HF Hub training/checkpoint exchange | llama.cpp / on-device runtimes | Legacy checkpoints; being displaced by safetensors |

**Block quantization, at concept level.** Chapter 10's per-channel INT8 gives one scale per row. GGUF's quantized types go finer: the tensor is cut into small blocks (tens to hundreds of elements), each block stores its own scale (and sometimes a minimum or a second-level scale), and elements are packed at 4–8 bits. Finer scales track local dynamic range, so error stays bounded where a per-row grid would be stretched by one outlier — the same argument as ch 10, one granularity step further. The K-quant family adds importance weighting: columns the calibration data says matter get more bits. This parser reads the type *names*; unpacking blocks is the next build (below).

## How the big systems do it

**llama.cpp / Glimmer-style on-device runtimes** convert an HF checkpoint offline (the converter performs ch 09's three translations once), quantize most matrices to a Q4/Q8-class block format, and ship the single `.gguf`. At load the runtime parses exactly the header parsed here, `mmap`s the file so untouched tensors cost no RAM, and dequantizes blocks inside its matmul kernels. One file means one download, one version check, no sidecar drift — the right shape for app bundles and intermittent connectivity.

**HF / vLLM-style serving** stays with safetensors + `config.json`: datacenter serving wants sharded, tensor-parallel loading and bf16/AWQ/GPTQ checkpoints, and the sidecar ecosystem (indexes, configs, tokenizers as separate artefacts) is a feature when different components update independently. Per sec. 4.2.4, `transformers` remains the reference implementation to verify against — ch 19's role — not the thing an on-device runtime embeds.

**pickle** survives in legacy checkpoints. Its lesson is negative: a model file that can execute code makes every download a trust decision. Both modern formats exist partly to make loading boring.

## What our parser does NOT yet do

Stated plainly, so the boundary is testable:

- **No tensor data.** `read_gguf_header` never seeks into the data section. There is no `load_tensor(name)` yet.
- **No dequantization.** `Q4_0`/`Q8_0`/K-quant block layouts are named, not unpacked; no fp32 tensor can be produced from quantized data.
- **No model construction.** Nothing maps GGUF tensor names/dims onto `DecoderLM` (the GGUF analogue of ch 09's `map_hf_state_dict`), and the ggml dim-order reversal noted above is documented, not applied.
- **No tokenizer execution.** Tokenizer metadata is parsed as generic arrays/strings; no BPE encode/decode is built from it.

The next build is: lazy tensor reads at `absolute_offset`, F32/F16 materialisation first, then Q8_0/Q4_0 dequantization checked against chapter 10's error-bound discipline — at which point a real converted Qwen GGUF can be compared token-for-token the way ch 19 compared safetensors.

## Common pitfalls

- **Ignoring alignment.** Tensor offsets are relative to the aligned data start, not to the end of the tensor infos. Off-by-padding reads are silent and catastrophic.
- **Assuming PyTorch dim order.** GGUF dims are ggml-ordered (reversed). Building shapes without reversing produces transposed-looking matrices that still multiply — sometimes.
- **Treating metadata as strings.** Unlike safetensors `__metadata__`, GGUF values are typed; an alignment read as a string, or a count read at the wrong width (`uint32` vs the v3 `uint64` counts), desynchronises every subsequent byte.
- **Trusting pickle-era habits.** Any loader that falls back to pickle for an unknown extension has reintroduced code execution. Both formats in this chapter are data-only by design; keep them that way.

## Break it / exercises

1. Build a GGUF with `build_gguf_bytes` in the tests, flip one magic byte, and confirm the parser's error names the magic. Why is failing here better than failing at tensor 40?
2. Write a file with `general.alignment = 16` and one tensor at offset 0. Compute `data_start` by hand from the header length and check `summarize()` agrees. Repeat with the alignment entry deleted (default 32).
3. A tensor info declares `dims = [896, 151936]`. What logical PyTorch shape does that describe, and which chapter-09 tensor has the transposed story?

## Check yourself

<details>
<summary>1. The tensor infos end at byte 100, alignment is 32, and a tensor has offset 48. Where are its bytes?</summary>

`data_start = ceil(100/32)*32 = 128`, so the tensor starts at file byte `128 + 48 = 176`. Bytes 100–127 are alignment padding.

</details>

<details>
<summary>2. Why can a GGUF file ship alone while a safetensors checkpoint cannot (without sidecars)?</summary>

GGUF's typed metadata carries the architecture hyperparameters, tokenizer tokens/merges, and quantization parameters alongside the tensor infos. Safetensors stores only tensors (plus an optional string map), so `config.json` and tokenizer files must travel with it.

</details>

<details>
<summary>3. What does <code>ggml_type_name(999)</code> return, and why not raise?</summary>

`"UNKNOWN_999"`. A header inventory should survive files from newer converters whose types it cannot yet dequantize — naming is not unpacking.

</details>

## Summary in 30 seconds

GGUF = typed metadata + tensor infos + aligned tensor data in one data-only, mmap-friendly file; safetensors = tensors + JSON header with sidecars for everything else; pickle = executable and unsafe. If you remember one formula, make it `absolute = align_up(header_end, alignment) + offset`; if you remember one code location, make it `read_gguf_header` in [gguf.py](../src/inference_from_scratch/gguf.py).

## Where to go next

The natural next rung is tensor materialisation: lazy reads plus F32/F16 and then block-dequantization, verified with ch 10's error bounds and ch 19's id-exact comparison on a converted real checkpoint. Revisit ch 09 if the safetensors column of the comparison table is not yet obvious — GGUF is best understood as that chapter's three translations, frozen into one file.

### Reader checkpoints

- Parse any synthetic GGUF with `summarize()` and state its version, alignment, `data_start`, and one tensor's absolute offset without running code twice.
- Explain why both modern formats are data-only, and what pickle does differently.
- Name what the parser does not do (tensor data, dequantization, model building) and the order the next build should add them.
