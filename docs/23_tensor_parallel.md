# 23 — Tensor parallelism: partitioning the hybrid stack across two ranks

> **Source of truth** — the real checkpoint on disk (`config.json`,
> `model.safetensors.index.json`, per-layer shard headers; names and
> shapes only, weights never loaded for this chapter) and the code in
> [tensor_parallel.py](../src/inference_from_scratch/tensor_parallel.py)
> (`shard_spec_for`, `shard_state_dict`, `merge_state_dicts`,
> `partitioned_linear`, `parameter_bytes_by_rank`, `real_config_report`);
> tests: [test_tensor_parallel.py](../tests/test_tensor_parallel.py).
> Framing follows Kiely, *Inference Engineering* (2026): sec. 2.2 for
> config and mechanics, sec. 2.4–2.5 for bottlenecks and attention
> costs, sec. 4.2 for formats and reference implementations, sec. 5.2
> for the speculation (MTP) context, and sec. 5.4 for tensor
> parallelism itself. All prose below is original to this repo.

## The one-sentence idea

Cut every large matrix so each of two ranks stores and multiplies only
its slice — column-parallel slices concatenate for free, row-parallel
partials sum in one all-reduce — and prove on CPU, with exact
shard/merge roundtrips and dense-equivalent matmuls, that the cuts
never separate two rows the maths treats as a unit.

This chapter first builds and proves the partitioning *maths* on CPU.
Real two-process execution behind the same specs now also exists (see
"Simulated vs distributed" below): `tp_distributed.py` +
`scripts/tp2_tiny_check.py` run actual NCCL (GPU) and gloo (CPU)
collectives at tiny scale, and `DistributedHybridBlock` /
`DistributedHybridLM` (`scripts/tp2_model_check.py`) assemble those
blocks into a full tiny hybrid LM — 3 linear + 1 full layers, dense vs
TP logits for prefill and 6 decode steps proven on both backends. The
full 27B model has now also been run under true TP2 (see "TP2 real
run" below).

## Why the naive even split is wrong twice

Two layouts in this checkpoint (ch. 22) punish a blind "first half /
second half" cut:

1. **The fused q+gate projection.** `q_proj` stores, per query head, a
   `[query (256) | gate (256)]` pair. Cutting its 12288 rows at row
   6144 happens to be safe at world 2 only by arithmetic luck; cutting
   by arbitrary features is not a rule. The rule used here: the shard
   unit is the whole 512-row head pair, so a head's gate always travels
   with its query. The test builds identity-like rows (output feature
   *i* reports the dense row that produced it) and shows each rank's
   shard holding whole, contiguous `[q|gate]` pairs.
2. **The concatenated gated-delta projection.** `in_proj_qkv` stores
   `[q (16×128) | k (16×128) | v (48×128)]`, and q/k heads are
   repeat-interleaved onto value heads (value head *v* reads key head
   *v // 3*). The rule: shard by value-head groups — 24 value heads per
   rank at world 2 — and *derive* each rank's q/k rows from the repeat
   mapping (8 key heads per rank). Each rank's shard keeps the stored
   concatenation order `[its q | its k | its v]`, and `conv1d` channels
   follow the identical order, so the rank's depthwise conv sees exactly
   the channels its projection produced.

## Tensor-family map (world = 2, real dims)

`D = 5120`, `F = 17408`, Q/KV heads 24/4 of 256, gated-delta K/V heads
16/48 of 128, repeat factor 3, vocab 248320. Kinds are the
`ShardSpec.kind` values returned by `shard_spec_for`.

| Tensor family | Dense shape | Kind | Per-rank slice | Why |
|---|---|---|---|---|
| `embed_tokens.weight` | [248320, 5120] | replicated (future: vocab) | whole | simulation choice: every rank looks up every token; production vocab-shards rows by token id |
| `lm_head.weight` | [248320, 5120] | replicated (future: vocab) | whole | same choice; production vocab-shards and all-gathers logits |
| `input/post_attention_layernorm.weight` | [5120] | replicated | whole | tiny; every rank needs the full row reduction |
| `self_attn.q_proj.weight` (+bias) | [12288, 5120] | head_column | 12 head-pairs [6144, 5120] | per-head `[q\|gate]` pairs are indivisible |
| `self_attn.k_proj/v_proj.weight` | [1024, 5120] | head_column | 2 KV heads [512, 5120] | whole KV heads; 4 KV heads → 2/rank |
| `self_attn.q_norm/k_norm.weight` | [256] | replicated | whole | one gain per head-dim, shared by all heads |
| `self_attn.o_proj.weight` | [5120, 6144] | row | cols of its 12 Q heads [5120, 3072] | consumes exactly the heads this rank's `q_proj` produced; partials sum |
| `linear_attn.in_proj_qkv.weight` | [10240, 5120] | segmented_qkv | [q 1024 \| k 1024 \| v 3072 rows] | value-head groups; q/k rows derived via `v // 3` repeat mapping |
| `linear_attn.conv1d.weight` | [10240, 1, 4] | segmented_qkv | same channel indices as `in_proj_qkv` | conv channels are in the same concatenated q\|k\|v order |
| `linear_attn.in_proj_z.weight` | [6144, 5120] | head_column | 24 value heads [3072, 5120] | rows are per value head |
| `linear_attn.in_proj_a/b.weight` | [48, 5120] | head_column | 24 rows | one row per value head |
| `linear_attn.A_log`, `dt_bias` | [48] | head_column | 24 elements | one element per value head |
| `linear_attn.norm.weight` | [128] | replicated | whole | `[Dv]` gain shared by every value head; no per-head copy exists to split |
| `linear_attn.out_proj.weight` | [5120, 6144] | row | cols of its 24 value heads [5120, 3072] | row-parallel over the same value heads; partials sum |
| `mlp.gate_proj/up_proj.weight` | [17408, 5120] | column | 8704 rows | intermediate features concatenate; no communication yet |
| `mlp.down_proj.weight` | [5120, 17408] | row | 8704 cols | consumes the same intermediate features; partials sum |
| `norm.weight` (final) | [5120] | replicated | whole | as per-block norms |
| `mtp.layers.0.*` (attn + MLP + norms) | as layer 3 | same suffix rules | as above | the MTP block (Kiely sec. 5.2) is one more full-attention layer |
| `mtp.fc.weight` | [5120, 10240] | replicated | whole | fuses normed hidden + embedding branches; both halves stay local |
| `mtp.pre_fc_norm_*/norm.weight` | [5120] | replicated | whole | norm gains |
| `*.weight_scale_inv` | [ceil(R/128), ceil(C/128)] | follows its weight's axis (segmented qkv: replicated) | block-grid slice | one BF16 scale per [128, 128] FP8 block (Kiely sec. 4.2) |
| `model.visual.*` | various | replicated | whole | vision tower deferred (ch. 22); out of scope for text TP |

Generic names outside the hybrid stack fall back to replicated, so
`shard_state_dict` / `merge_state_dicts` stay exact for any state dict;
the fallback is stated in the spec's `reason`, never silent.

## The three proofs (all CPU, all in the tests)

1. **Exact merge roundtrip.** A tiny hybrid `TinyHybridLM` state dict
   (3 linear + 1 full layers) is sharded at world 2 and merged back;
   every tensor compares `torch.equal` to the original. Segmented
   tensors are placed by dense index, not concatenated, because a
   rank's `[q|k|v]` storage order differs from dense order. Merging
   also verifies replicated copies agree across ranks and raises when
   they do not.
2. **Dense-equivalent partitioned matmuls.** For q/k/v/o, qkv/z/out
   and gate/down projections, `partitioned_linear` on the shards
   reproduces `F.linear(x, W)`: exactly (`torch.equal`) for the
   column family, to fp tolerance for the row family (the partial sum
   re-associates floating-point addition — the all-reduce in miniature).
3. **Index-level segment tests.** At tiny dims (K 2×8, V 4×8) the
   expected index lists are written out literally: rank 0 owns q rows
   0–7, k rows 16–23, v rows 32–47. A world that would split a repeat
   group (1 value head per rank against a repeat factor of 2) raises
   instead of producing a rank that lacks its key head.

## All-reduce points and their price (PCIe, world = 2)

Column-parallel projections (q/k/v, qkv/z, gate/up) need **no**
communication: their outputs are disjoint feature slices consumed by
the next row-parallel projection on the same rank. Each row-parallel
projection closes with one all-reduce of the full-width partials.
Per hybrid block, for activations of `B × T` tokens in `elem_bytes`
(2 for BF16):

| Point | Collective | Count/block | Tensor | Bytes per rank (ring all-reduce) |
|---|---|---|---|---|
| token mixer output (`o_proj` / `out_proj`) | all-reduce (sum) | 1 | [B, T, 5120] | `2·B·T·5120·elem_bytes` |
| MLP output (`down_proj`) | all-reduce (sum) | 1 | [B, T, 5120] | `2·B·T·5120·elem_bytes` |

So a forward pass costs **2 all-reduces per block × 64 blocks = 128
all-reduces**, each moving `2·B·T·5120·elem_bytes` bytes per rank in a
ring (the factor 2 is the ring's send+receive halves; a two-rank
exchange moves the same tensor once each way). At decode (`T = 1`,
BF16) that is 20480 bytes per all-reduce per token — small messages,
so latency, not bandwidth, dominates on PCIe; at prefill the same
formula with the prompt's `T` applies (Kiely sec. 2.4–2.5: prefill is
compute-bound, decode is bandwidth/latency-bound). Norms and residual
adds between the two points are replicated compute and communicate
nothing; the gated-delta recurrence and softmax attention run entirely
within a rank because every rank owns whole heads. Embedding lookup
and the final logits are replicated under this chapter's simulation
choice, so they add no collective either — a vocab-parallel production
layout (Kiely sec. 5.4) would add one all-reduce after the embedding
and an all-gather (or vocab-parallel sampler) at the head.

## Per-rank memory: does half the model fit in 24 GiB?

`real_config_report(world=2)` counts parameters *by formula* from the
config (never by materialising the model) and prices them at the
checkpoint's storage widths:

- FP8 weights, sharded: ≈ **11.5 GiB per rank**. Cross-check: twice
  this figure reproduces the manifest's dense FP8 payload
  (24,699,207,680 bytes over 407 tensors) to within rounding — the
  test asserts that agreement when the snapshot is present.
- BF16 replicated/misc, whole on every rank: ≈ **4.8 GiB**, dominated
  by the replicated embedding and lm_head (248320 × 5120 × 2 bytes
  each); norms, conv, `A_log`/`dt_bias` and `mtp.fc` are a rounding
  term beside them. This line is the honest cost of the replicated
  simulation choice — vocab-parallel embedding/head would roughly
  halve it.
- FP8 `weight_scale_inv` overhead (BF16, one per [128, 128] block):
  ≈ **0.002 GiB** — three orders below the weights it scales.

Total ≈ **16.4 GiB per rank**, inside a 24 GiB card with room for the
16 full layers' KV cache and the 48 fixed recurrent states (priced in
ch. 21–22, not here). The report states its assumptions in its own
output: language model + MTP only (vision deferred), weights only (no
KV, activations or buffers), BF16 scales, replicated embed/head. Tests
assert ranges (FP8 10–13.5 GiB, total 12–20 GiB, `< 24 GiB`), not a
fabricated precise figure.

## Executable simulated runtime (CPU)

The partitioning maths above is now also executed as simulated
two-rank blocks in `tensor_parallel.py` (tests:
`tests/test_tp_runtime.py`, numbers executed, not just indices):

- `SimulatedProcessGroup(world)` — `all_reduce_sum(partials)` sums one
  partial per rank; `gather_concat(partials)` concatenates feature
  slices in rank order. Counters let tests assert exactly one
  all-reduce per block.
- `ColumnParallelLinear` / `RowParallelLinear` — built *from* a dense
  `nn.Linear` by cloning its weight row/column slices (no
  re-initialisation; tests assert the shards concatenate back to the
  dense weight). Column outputs gather-concat; row partials
  all-reduce-sum, bias added once.
- `TPMLP` — gate/up as column shards, down as a row shard over the
  same intermediate slices, `silu(gate_r) * up_r` per rank, then one
  all-reduce. Equals the dense `SwiGLU` output.
- `TPFullAttention` — q by whole `[q|gate]` head pairs, k/v by KV
  heads, o row-parallel over the same Q heads. Each rank attends only
  over its Q-head subset, repeating its local KV heads by the GQA
  factor (rank 0: Q heads 0–1 → KV head 0 at tiny dims); q/k norms and
  RoPE reuse the dense module. One all-reduce on the o partials equals
  dense `GatedFullAttention`, including the KV-cache path (per-rank
  K/V are gathered to update the dense-shaped cache once, then each
  rank reads its KV-head slice).
- `TPGatedDeltaNet` (linear attention, recurrent step only) — weights
  sliced per the segmented spec (qkv/conv channels, value-head
  z/a/b/`A_log`/`dt_bias`, row-parallel `out_proj`); state sharded by
  value heads via `shard_linear_state` / `merge_linear_state` (S by
  value heads, conv channels scattered back to dense `[q|k|v]` order).
  The per-rank step equals the dense `GatedDeltaNet.step` output and
  merged state. *Remaining:* TP prefill (sequential/chunkwise) is not
  implemented — only the single-token recurrent step is; a TP prefill
  would run the chunkwise triangular solve per rank over its value
  heads with the same state/boundary maths.

## Runtime state bytes: KV cache vs linear state

`state_bytes(config, seq_len, batch, kv_dtype_bytes, world)` (CLI:
`.venv/bin/python -m inference_from_scratch.tensor_parallel`) separates
state that grows with context from fixed recurrent state, per rank:

- Full KV (16 layers, KV-head sharded):
  `n_full * 2 * B * seq_len * (H_kv/world) * head_dim * kv_dtype_bytes`
- Linear recurrent S (48 layers, value-head sharded, fp32):
  `n_lin * B * (Hv/world) * Dk * Dv * recurrent_dtype_bytes`
- Linear conv buffer (48 layers, segmented channels):
  `n_lin * B * conv_dim_per_rank * (K-1) * conv_dtype_bytes`, with
  `conv_dim_per_rank = 2*(kh_per*Dk) + vh_per*Dv = 5120` at world 2.

Real config, world 2, batch 1, KV in BF16 (2 B):

| seq_len | Full KV / rank | Linear recurrent / rank | Linear conv / rank | Total / rank |
|---|---|---|---|---|
| 4096 | 134,217,728 B (0.125 GiB) | 75,497,472 B (0.0703 GiB) | 1,474,560 B (0.0014 GiB) | 211,189,760 B (0.197 GiB) |
| 32768 | 1,073,741,824 B (1.0 GiB) | 75,497,472 B | 1,474,560 B | 1,150,713,856 B (1.072 GiB) |
| 262144 | 8,589,934,592 B (8.0 GiB) | 75,497,472 B | 1,474,560 B | 8,666,906,624 B (8.072 GiB) |

The linear rows do not move with `seq_len` — that fixed state is the
hybrid model's long-context advantage — while the 16 full layers' KV
cache is the only term that grows, and TP halves it per rank.

## Simulated vs distributed

| Aspect | Simulated (existing) | Distributed (new) |
|---|---|---|
| Module | `tensor_parallel.py`: `SimulatedProcessGroup`, `ColumnParallelLinear`/`RowParallelLinear`, `TPMLP`, `TPFullAttention`, `TPGatedDeltaNet` | `tp_distributed.py`: `DistributedTPContext`, `TPColumnLinear`/`TPRowLinear`, `DistributedSwiGLU`, `DistributedFullAttention`, `DistributedGatedDeltaNet`, `DistributedHybridBlock`, `DistributedHybridLM`, `TPState` |
| Processes | one process, ranks are list positions | two OS processes under `torchrun` (`scripts/tp2_tiny_check.py`, `scripts/tp2_model_check.py`) |
| Collectives | in-process sum / concat | real `dist.all_reduce` (in-place, returns the same tensor) + `all_gather`-concat (uneven dim-0 padded); model: exactly 2 all-reduces per block per decode step (mixer `o_proj`/`out_proj` + MLP `down_proj`), norms/embed/lm_head replicated (0) |
| Weights held | all ranks' shards in one process | only this rank's shard (+ replicated embed/norms/lm_head) per process, built via `shard_state_dict` specs from the dense weights (`DistributedHybridLM.from_dense`) |
| Backends proven | CPU only | NCCL on `cuda:LOCAL_RANK` (fp32 + bf16 blocks) and gloo on CPU, both world=2 |
| Coverage | MLP, full attention (incl. KV-cache path), gated-delta step, state bytes | tiny blocks (as before) plus full tiny LM: 3 linear + 1 full layers, `prefill(idx, position_ids=None) -> (logits, TPState)` / `step(token, state)` with per-rank local KV (full) and local `(S, conv)` (linear); linear prefill is v1 token-by-token step loop (chunkwise TP prefill deferred) |
| Results | `tests/test_tp_runtime.py` | blocks: fp32 max err ≤1.2e-7 (tight assert); bf16 reported max err 1.95e-3. Model (`tests/test_tp_model.py`, `scripts/tp2_model_check.py`): prefill + 6 decode steps fp32 allclose (max err ~1.3e-7 gloo), greedy tokens equal, per-rank 119440 B < dense 212672 B at tiny dims |

Reproduce:

```bash
.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_tiny_check.py
.venv/bin/torchrun --nproc_per_node=2 scripts/tp2_tiny_check.py --backend gloo --device cpu
.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_model_check.py
.venv/bin/torchrun --nproc_per_node=2 scripts/tp2_model_check.py --backend gloo --device cpu
```

The gloo runs take ~1.5 s (blocks) / ~2 s (model), so
`tests/test_tp_distributed.py` and `tests/test_tp_model.py` keep them
as durable subprocess tests alongside unit tests of shard
construction. The gated-delta step lifted cleanly: the simulated
per-rank branch (segmented qkv shard, local conv/recurrence over its
value heads, `out_proj` partial) needed no new maths — only the
in-process sum replaced by a real all-reduce. The distributed model
adds the KV-cache path the block check lacked:
`DistributedFullAttention.forward_with_state` appends this rank's
local K/V (post-RoPE) and attends over its local history, and
`DistributedGatedDeltaNet.forward_prefill_v1` is the labelled v1
token-by-token prefill loop over `step`.

### Tiny all-reduce overhead per block (from measured microbench, not remeasured)

From `benchmarks/roofline/SUMMARY.md` / `roofline_derived.json`
(2-rank NCCL over PCIe, least-squares fit
`seconds = latency + bytes/bus`): fixed latency **8.67 µs** per
collective, bus **6.81 GB/s**. At real decode shape (hidden 5120,
`[1, 5120]` BF16 partials = 10240 B), one all-reduce costs
≈ 8.67 µs + 10240/6.81 GB/s ≈ **10.2 µs**; a block issues 2
(mixer + MLP), so ≈ **17.3 µs latency-only, ~20 µs including the byte
term, per block** — latency-dominated, as ch. 23's byte table
predicts. Over 64 blocks that is ~1.11 ms/token latency-only
(128 × 8.67 µs, as in `roofline_derived.json`), small next to the
19.17 ms TP2 weight-read floor.

## Per-rank checkpoint loading (resident FP8)

`tp_checkpoint.py` closes the gap between the tiny distributed model
and the real checkpoint. `DistributedHybridLM.empty(config, rank,
world, fp8_mode="resident")` builds the shard-shaped model on the
meta device; `load_rank_state(snapshot, config, rank, world, device)`
then streams one checkpoint tensor at a time on CPU, slices it, and
assigns only this rank's shard (peak CPU RAM: one full tensor plus
its scale — the replicated embedding, ~2.4 GiB at the real config, is
the largest). FP8 projections stay resident:
`fp8_linear.shard_fp8_pair` slices the weight at the
`tensor_parallel` spec indices and the `weight_scale_inv` grid by the
block indices those rows/columns fall in (the segmented-qkv grid is
the non-contiguous q/k/v block rows of the rank's weight), and
`BlockFP8Linear.from_pre_sliced` validates the local block grid.
Execution is `TPBlockFP8Column` (local shard output, gathered only
where semantics require) and `TPBlockFP8Row` (local partial + one
all-reduce, bias after the reduction) inside the distributed
attention / gated-delta / MLP blocks; the Triton path runs per-rank
on each rank's own device. Embeddings and `lm_head` load replicated
on both ranks (v1).

Validation never moves tensors: `validate_rank_coverage` checks from
headers that every expected tensor's rank indices partition its
dense axis (scale grids: cover), and under torchrun each rank's
`rank_shard_summary` (count, bytes, SHA-256 of `name|shape|dtype`
lines) is all-gathered on rank 0 only. On the real snapshot this
reports 1273 language tensors per rank, vision (333 entries)
excluded. `resident_memory_report` prices the real config from the
manifest with no allocation: **16.362 GiB per rank** vs the
16.364 GiB formula estimate (`real_config_report`) — the −0.002 GiB
delta is the segmented-qkv scale grid being block-sliced instead of
replicated, plus conv/A_log/dt_bias moving from the estimate's
replicated line to the sharded line; the report's `explanations`
field names each term.

Proven end-to-end at tiny scale by `scripts/tp2_checkpoint_check.py`
(tests: `tests/test_tp_checkpoint.py`): both ranks load shards of a
synthetic FP8 hybrid checkpoint and generate greedily-identical
tokens to the dense converted model over prefill + 6 decode steps,
on gloo CPU and on NCCL CUDA with `INFERENCE_FP8_TRITON=1`
(per-rank resident bytes 37242 < dense 106336 at tiny dims).

## TP2 real run (27B, measured)

`scripts/run_qwen38_tp2.py` runs the full 27B model under true TP2:
each rank builds `DistributedHybridLM` from the real config on
`cuda:LOCAL_RANK`, loads only its rank shards resident-FP8 via
`load_rank_state` (Triton by default, `--fp8-path reference` for the
reference fallback), and greedy-generates the 3 quality probes.
Two documented choices: every rank encodes the prompts itself
(deterministic tokenizer, zero communication, ids asserted against
the recorded `prompt_token_ids`), and — because embed/`lm_head` are
replicated — both ranks argmax identical replicated logits locally,
so tokens stay on-device with one batched host transfer at the end
(no per-token broadcast; `--debug-token-sync` all-gathers and
asserts the argmax every step instead). A 23 GiB/rank memory guard
aborts before load if the header estimate exceeds it. Reproduce:

```bash
.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/run_qwen38_tp2.py
```

Measured (raw record `benchmarks/baseline/ours_tp2_first_run.json`):
load 21.2 s / 21.3 s per rank (1251 tensors/rank), resident params
16.091 GiB/rank against the 16.362 GiB header estimate, peak
16.39 GiB/rank. Generated ids are **32/32 exact on all 3 probes**
against both the pipeline-bridge record
(`ours_pipeline_fastfp8.json`) and the vLLM Arm A probes
(`arm_a_quality_probes.json`), first divergence none; a separate
8-token `--debug-token-sync` run also passed its per-step
cross-rank assertion with 8/8 prefix-exact ids. Decode is
22.9–23.6 tok/s (TPOT 42.5–43.7 ms; prefill 0.61 s first probe
including Triton JIT, then 0.14–0.17 s). Read against the standing
references with no win claim: Arm A c1 TPOT is ~20.3 ms (~49 tok/s)
and the TP2 roofline floor is 19.17 ms/token (52.2 tok/s), so TP2
v1 is roughly 2.1× off Arm A and 2.2× above the floor; it only
modestly beats the pipeline bridge's 19.6–19.7 tok/s despite halving
the per-rank weight read. The remaining gap is not yet profiled.

*Profile update (same window, later task).* `scripts/profile_qwen38_tp2.py`
(rank-0 `torch.profiler` + CUDA-event timers on both ranks, 16 decode
tokens post-warmup; `benchmarks/profile/tp2_profile.json` +
`SUMMARY_tp2.md`) attributes the 44.4 ms/token clean decode wall:
GPU busy 31.3 ms/token (Triton FP8 kernel 18.9 ms over 400 calls,
lm_head GEMV 2.65 ms, ~7 ms of small elementwise/copy/reduce ops)
against a ~13.1 ms/token host gap — decode is host-launch-bound, so
kernel-only savings do not move wall time. Collectives measure
exactly the designed 128 all-reduces/token at 1.28 ms GPU total
(10.0 µs each, matching the fitted model above) with no all-gather
in decode. Paired interleaved A/B (alternating 16-token segments in
one process) of the measured candidates: fused RMSNorm −2.04 ms/token,
vocab-parallel lm_head −1.04, cached FP32 FP8 scale −0.78, and a
BLOCK_N=32 Triton tile ~0 (its −3.8 ms kernel saving is absorbed by
the host-bound stream; not shipped). The single shipped fix is the
fused norm: `QwenRMSNorm.use_fused_rmsnorm` (opt-in, manual default
kept for dense/pipeline) enabled by the runner's `--norm fused`
(default; `--norm manual` restores v1). TP2 v2
(`benchmarks/baseline/ours_tp2_v2.json`) decodes at 23.4–23.9 tok/s
(TPOT 41.8–42.8 ms) with generated ids identical to v1, 32/32 exact
on all 3 probes against both records — a ~5% step, still ~2.2× above
the floor and not a win; CUDA-graph capture remains the structural
next lever and is deliberately not built here.

## What this deliberately is not

Still deliberately not built: chunkwise TP prefill (linear layers
prefill via the labelled v1 token-by-token loop), vocab-parallel
embedding/head (both replicated here), MTP under TP, and any TP
*serving* benchmark (`run_bench.py` against a TP server). The 27B
TP2 run above is a single-stream greedy generation measurement,
not a serving comparison. Those remain gated on later approved
windows, exactly like the rest of this repo's GPU work.
