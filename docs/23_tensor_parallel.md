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

This chapter builds and proves the partitioning *maths* only. There is
no distributed runtime, no process group and no GPU work here; a later
step can put real collectives behind these same specs.

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

## What this deliberately is not

No `torch.distributed`, no processes, no NCCL/PCIe transfer is run:
`partitioned_linear` performs the all-reduce as an explicit in-process
sum so the *maths* is the thing under test. GPU execution, real
collectives and benchmarking the partitioning are gated on a later
approved window, exactly like the rest of this repo's GPU work.
