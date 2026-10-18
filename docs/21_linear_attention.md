# 21 — Linear attention: the gated delta rule

> **Source of truth** — Read in the repo `.venv` (transformers 5.20.1, present, not absent): `transformers/models/qwen3_5/modeling_qwen3_5.py` — `Qwen3_5GatedDeltaNet` (line 490), `torch_recurrent_gated_delta_rule` (line 424), `torch_chunk_gated_delta_rule` (line 287), `causal_conv1d_fn` / `causal_conv1d_update` (lines 236/256), `l2norm` (line 280), `Qwen3_5RMSNormGated` (line 204); and `transformers/models/qwen3_5/configuration_qwen3_5.py` — `Qwen3_5TextConfig`. Our code: [linear_attention.py](../src/inference_from_scratch/linear_attention.py) (`GatedDeltaNet`, `RMSNormGated`); tests: [test_linear_attention.py](../tests/test_linear_attention.py).

## The one-sentence idea

Instead of attending over every past token, each token updates a fixed-size matrix state `S [Dk, Dv]` per head with a gated, error-correcting (delta) rank-1 write, and its output is a read from that same state — so decode state never grows with sequence length.

## Why inference needs this

Softmax attention's decode cost is dominated by streaming the KV cache: `O(T)` reads per token, per layer. A recurrent state of `Hv × Dk × Dv` floats replaces that growth with a constant. In the real Qwen3.8-27B-FP8 checkpoint (verified dims supplied for this chapter: hidden 5120, 64 layers of which 48 are linear-attention and 16 full-attention interleaved every 4th), the linear layers carry `S` of `48 × 128 × 128` fp32 values per layer regardless of the 262144-token context — the full-attention minority pays the KV price, the linear majority does not. The trade is expressiveness: `S` is a lossy, decaying summary of the past, not an exact transcript.

## The walk, token by token

Input `x_t [D]` (residual stream). All projections are bias-free.

**1. Projections.** `mixed = in_proj_qkv(x)` of width `conv_dim = 2·Hk·Dk + Hv·Dv`; `z = in_proj_z(x)` of width `Hv·Dv` (the output gate); scalars per value head `b = in_proj_b(x)`, `a = in_proj_a(x)`, each `[Hv]`. Real shapes: `in_proj_qkv.weight [10240, 5120]`, `in_proj_z.weight [6144, 5120]`, `in_proj_a/b.weight [48, 5120]`, `out_proj.weight [5120, 6144]`.

**2. Short causal depthwise conv.** Before splitting, `mixed` passes a per-channel causal conv (kernel 4, `conv1d.weight [conv_dim, 1, 4]`, left zero-pad 3) plus SiLU — HF `causal_conv1d_fn` for prefill, `causal_conv1d_update` for one-token decode. It gives each channel a 4-token local window; the recurrence below is otherwise purely pointwise in time. Decode keeps only the last `K−1 = 3` *raw* (pre-conv) vectors as the conv buffer: window `[prev3, prev2, prev1, cur]` cross-correlated with the kernel equals the padded prefill conv at the newest position.

**3. Split, reshape, expand.** `mixed` splits into `q, k` (`Hk·Dk` each) and `v` (`Hv·Dv`), reshaped to heads. When `Hv > Hk`, `q`/`k` are `repeat_interleave(Hv // Hk)`-expanded (HF `Qwen3_5GatedDeltaNet.forward`): key head `g` serves value heads `g·n_rep … (g+1)·n_rep − 1`. Real: `Hk=16, Hv=48, n_rep=3`. This mirrors GQA head sharing in chapter 04, with grouped (consecutive-copy) ordering.

**4. Normalise queries/keys.** `q, k` are L2-normalised per head vector (`l2norm`: `x · rsqrt(Σx² + 1e-6)` in fp32) and `q` is scaled by `Dk^-0.5`. Values are not normalised.

**5. Gate scalars.** `beta = sigmoid(b) ∈ (0, 1)` — how much of the error to write. `g = −exp(A_log) · softplus(a + dt_bias) ≤ 0` — a log-decay; `exp(g)` multiplies the state each step. `A_log [48]` and `dt_bias [48]` are per-value-head parameters; this is the "gated" part (a learned, input-dependent forgetting rate, as in gated linear attention / Mamba-style discretisation).

**6. The recurrence** (HF `torch_recurrent_gated_delta_rule`, ours `_recurrent_token` verbatim in structure):

```text
S ← S · exp(g_t)                          # decay the old summary
kv_mem = Σ_Dk S · k_t                     # what S already predicts for k_t
delta  = (v_t − kv_mem) · beta_t          # delta rule: keep only the error
S ← S + k_t ⊗ delta                       # rank-1 write of the correction
o_t    = Σ_Dk S · q_t                     # read AFTER the write
```

The delta term is what separates DeltaNet from plain linear attention (`S += k ⊗ v`): if `S` already maps `k_t` to `v_t`, `delta ≈ 0` and the state is left alone instead of double-counting or overwriting an unrelated direction. Writing before reading means token `t` attends to itself (causality allows `≤ t`), matching the causal mask convention of chapter 04.

**7. Gated output norm and projection.** `core [Hv, Dv]` is RMS-normalised per head-vector in fp32 (`Qwen3_5RMSNormGated`: norm first, then `· weight · silu(z)`), flattened to `[Hv·Dv]`, and mapped back by `out_proj` to `[D]`.

## Prefill, decode, and state

Our `forward_prefill` default is the explicit sequential scan: projections for the whole `[B, T, D]` are computed once (they mix no time, so this is exact), then the loop applies conv-step, recurrence, and read per token. `step(x_t, state)` runs one iteration. The chunkwise form (`method="chunkwise"`, next section) condenses a chunk's delta updates into a triangular (UT) solve plus matmuls and carries only the chunk-boundary `S`. The mathematics is identical; our parity test against HF's chunkwise prefill (copied weights, strict state-dict load) shows max-abs diff **1.19e-07** in fp32 at tiny dims, asserted at `atol=1e-5`, for both the sequential and chunkwise outputs.

State is the pair `(S [B, Hv, Dk, Dv], conv_buffer [B, conv_dim, K−1])`. `snapshot_state` / `restore_state` deep-copy both tensors (restore clones again, so one snapshot can seed many continuations) — the exact hook a prefix cache (chapter 14) needs, except the cached object here is constant-size instead of growing with the prefix. `step` never mutates its input state, so snapshotting before a step and restoring after is safe.

## Chunkwise prefill: the same recurrence, re-associated

`forward_prefill_chunkwise(x, chunk_size=64, initial_state=None)` evaluates the identical layer without stepping token by token, following HF `torch_chunk_gated_delta_rule`. `forward_prefill` exposes it as `method="chunkwise"` (with `chunk_size=`), and the default stays `method="sequential"` on purpose: old callers keep the reference scan's exact reduction order, while the chunkwise path — mathematically equal, numerically only equal to fp32 tolerance — is chosen explicitly. Textbook framing: the layer mechanics are Kiely *Inference Engineering* §2.2 territory, the reason prefill wants a different evaluation order than decode is the bottleneck analysis of §2.4–2.5, and this section is a §4.2-style reference-implementation dissection; §5.2/§5.4 (speculation, tensor parallelism) do not change the chunk math, they only consume the state it produces.

**Setup shared with the sequential path.** Projections are computed for the whole sequence at once, then one grouped causal `conv1d` runs over the raw `mixed` stream *prepended with the incoming conv buffer* — so a continuation chunk sees exactly the `K−1` raw predecessors the sequential `_conv_step` would have carried, and the returned buffer is the last `K−1` raw vectors of the extended stream. Split, grouped `repeat_interleave` expansion of Q/K to the value heads, L2 normalisation, `Dk^-0.5` query scale, `beta = sigmoid(b)` and `g = −exp(A_log)·softplus(a + dt_bias)` are unchanged; `beta`/`g` (and everything inside the chunk core) are computed in fp32, as the HF source requires.

**Inside one chunk** of `C` tokens (per batch and value head), write `cum[i] = Σ_{j≤i} g_j` for the cumulative log-decay from the chunk start. Because the sequential rule multiplies `S` by `exp(g)` before each write, a value written at position `j` reaches a read at position `i ≥ j` scaled by `exp(cum[i] − cum[j])`; call that `C×C` lower-triangular factor `pairwise[i, j]` (the strictly upper part is masked to `exp(−inf) = 0`, which is causality in matrix form). The delta rule makes tokens inside a chunk interact: token `i`'s write is `beta_i · (v_i − S·k_i)`, but once earlier chunk-mates have written, `S·k_i` also contains *their* contributions. Collecting those interactions gives a unit-lower-triangular system `A[i, j] = beta_i · (k_i · k_j) · pairwise[i, j]` for `i > j` (diagonal implied `1`: a token's own write is not predicted by itself). Solving `A u = beta ⊙ v` once per chunk — a triangular solve, HF's UT transform — condenses all `C` sequential error corrections into the corrected values `u`; solving the same system against `beta ⊙ k ⊙ exp(cum)` expresses the decayed keys in the same condensed basis. That pair of solves is the whole trick: `C` rank-1 updates become two triangular solves plus matmuls.

**Between chunks** only the boundary state `S` is sequential. With `S` from the previous chunk boundary: `v_new = u − k_cumdecay @ S` subtracts what the *old* state predicts for the chunk's keys (the delta rule, applied once per chunk instead of once per token), each output is `core = (q ⊙ exp(cum)) @ S + intra @ v_new` where `intra[i, j] = (q_i · k_j) · pairwise[i, j]` adds the within-chunk reads, and the state advances in one shot: `S ← exp(cum[C−1]) · S + (k ⊙ exp(cum[C−1] − cum))ᵀ @ v_new`. The final gated RMSNorm (`norm` then `· silu(z)`) and `out_proj` are applied to the whole `[B, T, Hv·Dv]` core at once, exactly as the sequential tail does per token. A padded tail chunk (sequence length not a multiple of `C`) is inert: its `q/k/v/beta/g` are zero, so it writes nothing, reads nothing, and leaves `S` untouched — which is why `T=65` at `C=64` and `T=1` at `C=64` both match the scan.

**Complexity and state.** Per chunk and head the dense work is the `C×C` interactions: `O(C²·(Dk + Dv))` for the pairwise products and triangular solves plus `O(C·Dk·Dv)` to read and update `S`, against the sequential scan's `O(C·Dk·Dv)` with `C` dependent steps. Chunkwise therefore does *more* arithmetic when `Dv`/`Dk` are small relative to `C`, and its reward is not fewer FLOPs but parallelism: the `C` positions become matmul rows a GPU can execute concurrently, instead of `C` serial state updates (the prefill side of the §2.4–2.5 bottleneck story — decode stays recurrent and `O(1)` in `T`). The recurrent state itself is unchanged: still the fixed `S [B, Hv, Dk, Dv]` plus the `K−1` conv buffer, still snapshotable, and a snapshot taken after chunkwise prefill continues through `step` identically (to fp32 tolerance) to one taken after sequential prefill. What grows is only *workspace*: materialising `pairwise`/`intra` costs `O(B·Hv·C²)` per chunk (this reference implementation, following HF, builds them for all chunks at once, `O(B·Hv·T·C)` in total) — production kernels tile the chunk loop to keep that at one chunk's worth. Chunk size trades that workspace and the `C²` term against fewer sequential boundaries; 64 is HF's default, not a law.

On CPU none of this has to be faster, and at tiny dims it usually is not: `benchmarks/microbench_linear_attn.py` times both methods at tiny/medium dims and prints the measured medians, tokens/s and the chunkwise-vs-sequential max output diff as JSON, whichever way the numbers fall.

## Causality, proved not assumed

The conv sees only positions `t−3…t` (left padding), the recurrence only accumulates `≤ t`, and the read happens after the current write but before any future one. The test changes only the last input token and checks every earlier output is unchanged to 1e-6.

## How the big systems do it

vLLM runs the same layer with fused FLA-style kernels: chunkwise prefill, a recurrent decode kernel that updates `S` and the conv state in place, and no materialised attention matrix (there is none to materialise). Nothing about the state is paged like KV blocks — it is one fixed tensor per layer per sequence, which is precisely why hybrid stacks interleave a few full-attention layers (exact recall) among many linear ones (cheap decode).

## Common pitfalls

- **Reading before writing** (or decaying after the write): silently shifts every output by one token's contribution.
- **Forgetting the L2 norm or the `Dk^-0.5` scale:** outputs stay shaped and plausible while magnitudes drift.
- **Scalar `beta`/`g` per layer instead of per value head:** the checkpoint has 48 of each; broadcasting wrongly still runs.
- **Storing post-conv values in the conv buffer:** prefill pads the *raw* projection; the buffer must hold raw vectors or decode diverges from step 2 onward.
- **Interleaved (rather than grouped) Q/K head expansion** when `Hv ≠ Hk`: pairs value heads with the wrong keys, as with GQA in chapter 04.

## Check yourself

<details>
<summary>1. Why does decode memory not grow with T here, and what is paid instead?</summary>

State is the fixed `S [Hv, Dk, Dv]` plus a `K−1` conv buffer per layer. The price is recall: `S` is a decayed, lossy summary, not the exact past a KV cache keeps.
</details>

<details>
<summary>2. What does the delta in the delta rule subtract, and why?</summary>

`S`'s current prediction for `k_t` (`kv_mem`). Only the prediction error `(v_t − kv_mem)·beta` is written, so re-seeing a known key does not double-count it.
</details>

<details>
<summary>3. A snapshot is restored twice and both branches continue. What guarantees they cannot corrupt each other?</summary>

`snapshot_state` clones, and `restore_state` clones again, so each branch owns independent `S` and conv-buffer tensors; `step` additionally never mutates its input state.
</details>

## Summary in 30 seconds

Project, locally convolve (kernel 4, SiLU), split into Q/K/V, expand Q/K heads by grouped repetition, L2-normalise Q/K, then per token: decay `S` by `exp(g)`, write only the beta-scaled prediction error as a rank-1 update, read with `q`, gate-RMS-norm, project out. Fixed-size state, causal by construction, and our sequential scan matches HF's chunkwise kernel to 1.19e-07 in fp32.
