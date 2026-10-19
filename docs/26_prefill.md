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

**FP8 M>128 implication (measured, superseded below):** resident FP8
originally dispatched to Triton only while `B*M <= TRITON_MAX_M = 128`;
chunk 256 fell back to the reference dequant path and measured
~2.1× *slower* than chunk 128 (633 vs 1306 tok/s at 2048). The v2
work below raised the cap with measured per-bucket tile configs.

## Prefill v2: profile, tile sweep, larger chunks (2026-10-18)

**Profile** (`scripts/profile_prefill_tp2.py`, prompt 2048 chunk 128,
TP2, leaf CUDA kernels only — wrapper ops nest the same device time;
raw top-40 in `benchmarks/profile/tp2_prefill_profile.json`): FP8
Triton GEMM 46.8% of CUDA time, NCCL collectives 29.2%, other
GEMM/elementwise 19.6%, gated-delta chunk core (incl. triangular
solve) 3.4%, norms 0.6%, full-attn-named kernels 0.3% (most attention
math lands in generic bmm/softmax kernels inside "other"). Profiler
wall (590 tok/s) is inflated vs the unprofiled 1306-1413 tok/s; only
the shares are indicative. Conclusion: FP8 GEMM is the largest single
bucket, but collectives + launch-bound elementwise work together
rival it — larger external chunks amortise all three.

**Tile sweep** (`benchmarks/fp8_prefill_sweep.py`, M=64/128/512, TP2
shard + full shapes, `benchmarks/profile/fp8_prefill_sweep.json`):
BLOCK_M=64/BLOCK_N=32/num_warps=8 won or tied everywhere (rel diff
vs dequant reference ≤ 7.8e-3 per config). BLOCK_M=128 never won and
BLOCK_M=128/BLOCK_N=128 exceeds shared memory (OutOfResources).
Dequant+cuBLAS lost to Triton by 3-12× at every M including 512, so
no cuBLAS dispatch is wired. Why: with BLOCK_M=16 a chunk of M rows
re-reads each weight tile `M/16` times per N-tile pass (e.g. 8× at
M=128); BLOCK_M=64 cuts weight re-reads 4×, and beyond 64 the
occupancy/shared-memory cost outweighs the remaining re-read saving.
Wired as `PREFILL_TUNED` for M≥64 buckets (64/128/512/1024) in
`fp8_triton.choose_config`; decode M=1 tuned config untouched.
`BlockFP8Linear` dispatch is M-based via `fp8_dispatch`: Triton for
flattened rows 1..1024 (`TRITON_MAX_M=1024`), reference above.

**End-to-end** (`scripts/prefill_bench_tp2.py --chunks 128,512,1024
--lengths 2048,8192`, `benchmarks/baseline/ours_tp2_prefill_v2.json`):

| Prompt | Chunk | tok/s | vs chunk128 | Peak GiB/rank | Argmax = chunk128 |
|---|---|---|---|---|---|
| 2048 | 128 | 1412.8 | 1.00× | 15.24 | — (baseline) |
| 2048 | 512 | 1731.6 | 1.23× | 15.41 | yes (2336) |
| 2048 | 1024 | 1596.0 | 1.13× | 15.76 | yes (2336) |
| 8192 | 128 | 1351.0 | 1.00× | 15.58 | — (baseline) |
| 8192 | 512 | 1631.3 | 1.21× | 15.75 | yes (310) |
| 8192 | 1024 | 1518.1 | 1.12× | 15.99 | yes (310) |

Chunk 512 is the measured best external chunk (+21-23%); chunk 128
itself improved 1306 → 1413 tok/s at 2048 from the BLOCK_N=32 tile
retune alone. The API default stays `chunk_size=128` (callers choose
512 explicitly).

**Prefill round 2 (2026-10-18, integrated):** ABBAAB on
`scripts/prefix_ttft_bench.py` + `scripts/prefill_bench_tp2.py`
(`benchmarks/harness/prefill_round2.json`): chunk512 vs chunk128 pure
prefill +22.1%/+21.9%/+20.0% tok/s at 512/2048/8192, argmax equal,
peak +≤0.17 GiB/rank; prefix candidate `--chunk 512
--snapshot-interval 512` beat the 128/256 baseline on every cold/warm
cell (see ch. 27). The bench defaults are now 512/512; the library
API default above is unchanged.
Remaining gap to Arm A's ~2.9-3.4k tok/s prompt rate: collectives
(29%) and non-FP8 elementwise work dominate after the GEMM fix;
batched prefill (docs/28) amortises them across sequences.

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
