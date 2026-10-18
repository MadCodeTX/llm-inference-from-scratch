# 26 — Efficient TP prefill: chunkwise static prefill

## What changed

`DistributedHybridLM.prefill_chunkwise_static(idx, state, chunk_size=128,
position_ids=None, return_hidden=False)` fills a `TPStaticState` in
chunks instead of token by token. `prefill_static` (the v1 path) calls
`decode_token_static` once per token: every projection weight is read
once *per token*, and every linear layer pays one mixer all-reduce per
token. Chunkwise prefill reads each weight once *per chunk* (M =
chunk tokens per matmul) and pays one mixer + one MLP all-reduce per
block per chunk.

New pieces (all in `src/inference_from_scratch/tp_distributed.py`):

| Piece | Role |
|---|---|
| `DistributedFullAttention.prefill_chunk_static` | M-token full attention over static KV buffers |
| `DistributedGatedDeltaNet.prefill_chunk_inplace_static` | M-token chunkwise gated-delta over static `S`/conv |
| `DistributedHybridBlock.prefill_chunk_static` | norms + mixer chunk + MLP chunk (position not advanced) |
| `DistributedHybridLM.prefill_chunkwise_static` | chunk loop, position advance, last-token logits |

## Chunk math

### Full-attention layers

For a chunk starting at slot `p = state.position` with length `M`:
Q/K/V/gate projections run once over `[B, M, D]`. RoPE rows are
`p .. p+M-1` (or the chunk's `position_ids` slice via mRoPE tables,
exactly as `forward_with_state`). K/V are written in place with
`index_copy_` at those slots in the preallocated
`[B, H_kv_local, max_seq_len, Dh]` buffers. Attention reads the whole
buffer (cached prefix + this chunk) with the causal mask
`key_pos > query_pos -> -inf` per query, so query `i` sees keys
`<= p+i`; the unwritten tail is masked out. One `o_proj` all-reduce
per chunk per layer (vs `M` in the token loop).

### Linear (gated-delta) layers

Per chunk, on this rank's value-head shard:

1. Projections once: `in_proj_qkv/z/a/b` as single `[B, M, D]`
   matmuls; `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`
   in fp32.
2. Causal depthwise conv as one grouped `F.conv1d` over the stream
   seeded with the incoming conv ring (`conv_buf` holds the last
   `K-1` raw vectors); the new ring is the last `K-1` raw vectors of
   the extended stream — identical to the sequential conv loop's end
   state.
3. Recurrence via the dense chunkwise core
   (`GatedDeltaNet._chunk_gated_delta_core`, internal chunk = M):
   inside the chunk, cumulative decay `cum[i] = sum_{j<=i} g_j`,
   pairwise decay `exp(cum[i]-cum[j])`, the unit-lower-triangular
   (UT) system `A[i,j] = beta_i (k_i·k_j) pairwise[i,j]` is solved by
   forward substitution to condense the chunk's delta writes, the
   boundary state `S` is corrected once per chunk
   (`v_new = u - k_cumdecay @ S`), reads combine
   `q_decayed @ S + intra @ v_new`, and `S` advances as
   `exp(cum[-1]) S + k_decayed^T @ v_new`. This re-associates the
   sequential recurrence, so results agree to fp32 tolerance, not
   bit-for-bit (same contract as dense `forward_prefill_chunkwise`).
4. Gated RMSNorm + `out_proj` once over `[B, M, D]`, one all-reduce.
   `S` and `conv_buf` are updated with `copy_` (pointers stable).

### Position, non-empty states, last logits

`state.position` advances by each chunk length after the chunk runs,
so a state that already holds a prefix continues at the correct KV
slots, RoPE rows, and recurrent state — chunkwise continuation equals
the one-shot prefill (proved in tests). Only the last hidden vector
goes through the final norm + `lm_head`: the return is
`logits [B, vocab_size]`, never a `[B, T, V]` tensor (at the real
config that tensor would be `T × 248320` values). `return_hidden=True`
additionally returns the pre-norm hidden `[B, T, D]`. Overflow and
invalid `chunk_size`/`position_ids` raise before any mutation.

## Measured (TP2, real 27B, 2026-10-18)

`scripts/prefill_bench_tp2.py` (resident FP8, Triton path, vocab
lm_head, filler prompts; raw rows in
`benchmarks/baseline/ours_tp2_prefill_bench.json`):

| Prompt | Method | Seconds | tok/s | Last argmax |
|---|---|---|---|---|
| 512 | `prefill_static` (token loop) | 22.124 | 23.1 | 30876 |
| 512 | chunkwise 128 | 0.389 | 1316.1 | 30876 |
| 512 | chunkwise 256 | 0.806 | 635.2 | 30876 |
| 2048 | `prefill_static` | capped at 512, not run | — | — |
| 2048 | chunkwise 128 | 1.568 | 1306.4 | 2336 |
| 2048 | chunkwise 256 | 3.235 | 633.1 | 2336 |

Last-token argmax is equal across every measured method at both
lengths. Chunkwise-128 prefill is ~57× the token loop at 512 tokens.

**FP8 M>128 implication (measured):** resident FP8 dispatches to the
Triton kernel only while flattened rows `B*M <= TRITON_MAX_M = 128`
(`fp8_linear.py`); above that it falls back to the reference path,
which materialises a dequantised BF16 weight per call. At B=1,
chunk 256 therefore runs every FP8 projection on the reference path
and is ~2.1× *slower* than chunk 128 despite half the chunks
(633 vs 1306 tok/s at 2048). Default `chunk_size=128` is both the
API default and the measured winner; larger chunks only pay off if
the Triton M cap is raised and re-measured.

This is an ours-vs-ours prefill comparison only. It is **not** a TTFT
win vs vLLM: no server, queueing, scheduling, or sampling is involved,
and vLLM Arm A TTFT (0.79s short / 5.69s medium / 14.34s long, greedy
c1) was measured through its HTTP server. Closing that comparison
needs an end-to-end run, not this bench.

## Scheduling long prompts (simulation)

Chunkwise execution fixes per-token cost; scheduling decides when
chunks run. `ChunkedPrefillScheduler` (ch 11, `serving.py`) simulates
that layer deterministically: a per-step `token_budget` is spent
decode-first (1 token per decoding request), then on FIFO prefill
chunks over `prompt_tokens - cached_prefix_tokens`. It is the future
TP server's scheduler shape, testable on CPU with an injected
step-cost model before any server exists.

Budget mapping: vLLM Arm A uses `--max-num-batched-tokens 8192`
with chunked prefill enabled (`benchmarks/baseline/arm_a_live_config.json`),
so the future server starts at `token_budget=8192` for parity and
then sweeps 1024–8192, watching TTFT (fewer, larger steps win) vs
TPOT (decode delay bounded by the budget). Do not conflate this
aggregate budget with the 128-token execution chunk above: the
scheduler may admit thousands of prefill tokens in a step, which the
model still executes in 128-token chunks on the FP8 Triton path.

## Proofs

- `tests/test_tp_prefill_chunkwise.py` — single-rank: chunkwise
  (chunk 4, plus sizes 1/3/128) vs `prefill_static` vs dense logits
  allclose fp32; state contents (KV filled region, `S`, conv)
  allclose; 6-token greedy continuation identical; non-empty-state
  continuation equals one-shot; overflow/argument errors mutate
  nothing. Includes the 2-process gloo subprocess proof.
- `scripts/tp2_chunkwise_check.py` — the same proof under real
  torchrun: gloo/CPU and NCCL/GPU both PASS (NCCL run 2026-10-18:
  state/logit errors ≤ ~1e-6 fp32, greedy tokens identical).
