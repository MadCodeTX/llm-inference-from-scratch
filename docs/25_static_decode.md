# 25 — Static decode: preallocated state for CUDA-graph-friendly stepping

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), sec. 4.1 (CUDA kernels: why per-step launch and allocation overhead matters when each decode step is dozens of small kernels) and sec. 4.5 (performance benchmarking: honest medians, no promised winner). **Production question:** what must be true of a decode step's shapes, addresses, and control flow before a CUDA graph can capture it once and replay it per token — and what stays dynamic even then? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Allocate every piece of decode state once, at fixed shapes and fixed addresses, and make each step *overwrite* that state in place — then a captured graph (or a compiled step, or just a tight loop) never has to ask Python how big anything is this time.

## Why inference needs this

A CUDA graph captures a sequence of kernel launches with fixed tensor addresses and fixed shapes, then replays it without re-launching from Python. Anything that breaks that contract breaks capture:

- **New storage per step.** If a step returns a freshly allocated state tensor, the next step's kernels point at new addresses; a captured graph still points at the old ones.
- **Shape that depends on the step.** If a KV tensor grows by concatenation, its shape is `t`-dependent, so each step is a *different* graph (sec. 4.1's kernel-selection/launch problem at `T=1`: overhead rivals maths).
- **Python-level data dependence.** Branching or sizing from a tensor value read back to the host (`.item()`, `bool(finished.all())`) forces a host sync and a Python decision the graph cannot contain.

The legacy decode path in this repo violates all three on purpose — it is the readable reference (sec. 4.2.4 stance), not the capturable one. This chapter adds the static twin and proves it computes the same logits.

## Audit: what the legacy per-token path allocates

Inspected in [hybrid.py](../src/inference_from_scratch/hybrid.py), [linear_attention.py](../src/inference_from_scratch/linear_attention.py), and [kv_cache.py](../src/inference_from_scratch/kv_cache.py):

| Site | Per-token cost |
|---|---|
| `KVCache.update` (full layers) | `torch.cat` on K and V: new tensor + copy of all previous entries, every step → O(T²) total copying |
| `GatedDeltaNet.step` → `_conv_step` | `torch.cat([conv_buffer, raw])` builds a new window; `window[:, :, 1:]` becomes the new buffer |
| `GatedDeltaNet.step` → `_recurrent_token` | `S.float() * exp(g)`, `S + k ⊗ delta`: two new `S`-sized tensors per token |
| `GatedDeltaNet.snapshot_state` / `HybridState.snapshot` | `clone()` of every state tensor (correct for branching; wasteful if taken per token) |
| `TinyHybridLM.forward` | new `entries` list, new single-layer `KVCache` objects, and a new `HybridState` per call; `seq_len` is a Python int read from state |
| `generate_hybrid` | `torch.cat` of the output ids per token, `sample_next`, and `bool(finished.all())` host sync for EOS |

Activations (projections, scores, MLP intermediates) also allocate per step, but they are *not* state: a graph capture reuses their addresses too once shapes are fixed (`[B, D]` in, `[B, V]` out). State is what must additionally survive, unchanged in location, from one step to the next.

## The static path

Three additions; every legacy API is untouched and keeps its immutable semantics.

**1. `StaticHybridState` ([hybrid.py](../src/inference_from_scratch/hybrid.py))** — created once from `(config, batch_size, max_seq_len, device, dtype)`:

- Full layers share one [StaticKVCache](../src/inference_from_scratch/kv_cache.py) (`kv_cache`, shape `[n_full, B, H_kv, max_seq_len, head_dim]`), slice-written in place. Each full layer knows its slot (`full_slot(i)`).
- Linear layers each own a fixed `S [B, Hv, Dk, Dv]` and conv ring `[B, C, K-1]` (`C = 2·Hk·Dk + Hv·Dv`), in the parallel lists `linear_S` / `linear_conv_buffers` (`None` at full layers).
- **Position** is a 0-dim `torch.long` tensor (`position`, alias `position_tensor`) on the state's device: the graph-friendly choice, since a captured graph can read/advance it without any shape change. The trade is explicit — the Python `seq_len` property reads it with `.item()`, a host sync on GPU, which is why `decode_token_static` also accepts a plain Python `int` (or 0-dim tensor) as an optional `position=` argument. Either way, position never changes a tensor shape.
- `reset()` zeroes every buffer **in place** (`zero_()` / `StaticKVCache.clear()`), so `data_ptr()` is stable across steps *and* across reset — asserted in tests. `snapshot()` is the only copying operation, and only when explicitly requested: it clones into an independent `StaticHybridState`; `copy_from(snapshot)` copies one back in place.

**2. `GatedDeltaNet.step_inplace(x_t, S, conv_buf)` ([linear_attention.py](../src/inference_from_scratch/linear_attention.py))** — the recurrent token step with in-place state: the conv ring is shifted with `copy_` and the gated-delta update is `S.mul_(exp(g))` then `S.add_(k ⊗ delta)`. No `torch.cat`, no list append, no new state tensor. Against the legacy `step` chain it is bit-identical on CPU (`torch.equal` in tests): the conv window temporary is built with the same values in the same layout so the reduction order matches; only *where the result lands* changed.

**3. `HybridBlock.decode_token_static` / `TinyHybridLM.decode_token_static(token_id_1d, state) -> logits`** — one token through the stack over a `StaticHybridState` (`model.static_state(...)` allocates one). Full blocks slice-write into the shared static cache at the state's position; linear blocks call `step_inplace` on their fixed tensors. The state's position advances exactly once, after all blocks, so every layer sees the same absolute position. Returns logits `[B, vocab_size]`. Overflow is checked **before** any mutation: decoding past `max_seq_len` raises

```text
ValueError: StaticHybridState overflow: position 4 would exceed max_seq_len=4 ...
```

## Equality contract (tests)

[test_static_decode.py](../tests/test_static_decode.py) pins, on the tiny 3-linear + 1-full config:

- static token-by-token logits equal the legacy `step` path (`allclose`, atol 1e-6; measured max diff 0.0) and the single prefill forward;
- `step_inplace` equals the legacy `step` chain bit-for-bit, final `S` and conv buffer included, while the legacy `step` still provably does not mutate its input;
- `data_ptr()` of every state tensor is unchanged across steps and across `reset()`, and reset + rerun reproduces the first run with `torch.equal`;
- a snapshot is an independent clone (different storage, equal values) and `copy_from` restores it in place;
- overflow past `max_seq_len` raises the clear `ValueError` above and mutates nothing (position stays exactly at the limit, and `reset()` still works).

## What remains dynamic (honest list)

Static state removes state allocation and shape dependence from the *model step*. It does not make the whole generation loop capturable:

- **Sampling** (`sample_next`): top-k/top-p filtering and `multinomial` read logits and produce a host-visible token; greedy `argmax` is graph-shaped but the *choice* of sampler is Python configuration.
- **EOS / stopping**: `generate_hybrid`'s `bool(finished.all())` is a host sync per token by design — the caller must learn that generation ended. A captured graph can emit a fixed number of tokens; deciding to stop cannot live inside it.
- **Position readback**: reading `seq_len` in Python syncs; a fully captured loop must keep position on-device (the 0-dim tensor is the seam for that).
- **Batch churn in serving** (ch 11): requests arriving/leaving change the batch; production systems graph a fixed batch bucket or page around it. Out of scope here.
- **Prefill**: arbitrary prompt lengths are inherently variable-shape; only the `T=1` decode step is static. Prefill stays on the legacy/chunkwise paths.

## Benchmark (sec. 4.5: report, don't promise)

[benchmarks/microbench_hybrid_decode.py](../benchmarks/microbench_hybrid_decode.py) times 32 tiny CPU decode tokens, legacy vs static, honest JSON medians. One measured run on this machine (`.venv/bin/python benchmarks/microbench_hybrid_decode.py --repeats 3 --warmup 1`):

```json
{"event": "comparison", "label": "tiny", "legacy_seconds_median": 0.020542018115520477, "static_seconds_median": 0.020290337968617678, "speedup_static_vs_legacy": 1.012403940599317, "max_abs_diff_vs_legacy": 0.0}
```

That is parity within noise, and it is the expected CPU result: at tiny dims Python overhead dominates both paths, and the static path's payoff — fixed addresses and shapes a CUDA graph can capture (sec. 4.1) — is a GPU property this CPU benchmark cannot measure. No win is claimed; the script prints whichever way a given machine falls.

## Check yourself

<details>
<summary>1. Why does the legacy <code>KVCache</code> break CUDA-graph capture even though its maths is identical to the static cache's?</summary>

Each `update` concatenates into newly allocated storage with a step-dependent shape, so the next step's kernels would need new addresses and a new graph. The static cache slice-writes one preallocated buffer: same address, same shape, only contents change.
</details>

<details>
<summary>2. <code>reset()</code> zeroes buffers instead of reallocating them. What breaks if it reallocated?</summary>

Any captured graph (or compiled step, or caller-held reference) still points at the old storage. Zeroing in place keeps every `data_ptr` stable — the tests assert exactly that across steps and resets.
</details>

<details>
<summary>3. Name two things that stay dynamic in a static-decode generation loop, and why each cannot be captured.</summary>

Sampling/EOS: choosing the next token and deciding to stop are host-visible decisions (`bool(finished.all())` syncs by design). Position readback in Python likewise syncs; a captured loop must keep position and token choice on-device or run a fixed token count.
</details>

Run it: `.venv/bin/python -m pytest tests/test_static_decode.py -q` and `.venv/bin/python benchmarks/microbench_hybrid_decode.py`.
