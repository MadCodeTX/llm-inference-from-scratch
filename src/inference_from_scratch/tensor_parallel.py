"""Tensor-parallel (TP) partitioning maths for the hybrid Qwen3.8 stack.

Why this exists in inference
------------------------------
A 27B-class checkpoint does not fit comfortably beside its runtime
state on one 24 GiB card, and even when the weights fit, one device's
memory bandwidth caps decode speed. Tensor parallelism (Kiely,
*Inference Engineering*, sec. 5.4) answers both problems by cutting
each large matrix across devices: every rank stores and multiplies
only its slice, and the ranks exchange the small partial results that
recombine into the full-width answer. This module proves the cutting
maths on CPU -- specs, shard/merge roundtrips, partitioned matmuls and
byte budgets -- with no distributed runtime, no processes and no GPU.
A later, separately approved step can put real collectives behind the
same specs; nothing here starts one.

The partitioning rules (Megatron-style, adapted to this checkpoint)
----------------------------------------------------------------------
* Column-parallel (weight rows split): each rank computes a disjoint
  slice of the output features; slices concatenate -- no communication
  until a later row-parallel consumer needs the full vector.
* Row-parallel (weight columns split): each rank multiplies the
  matching slice of the input; partial outputs *sum* (one all-reduce).
* Norms, per-head norm gains and small per-head vectors are replicated:
  they are tiny and every rank needs the full reduction they encode.
* Embedding and ``lm_head`` are replicated *in this simulation* (the
  spec records ``future_kind="vocab"``): a production vocab-parallel
  embedding shards rows by token id and all-reduces the looked-up
  vectors, which is a different communication pattern from a matmul
  slice and is deliberately not faked here.

Two checkpoint layouts make naive even splits silently wrong, and both
are handled explicitly:

1. Full-attention ``q_proj`` emits, *per query head*, a
   ``[query (head_dim) | gate (head_dim)]`` pair (see
   :mod:`inference_from_scratch.hybrid`). Splitting its rows at an
   arbitrary even boundary can separate a head's query from its gate.
   Here the shard unit is the whole ``2 * head_dim`` pair: a head and
   its gate always travel together, ``k_proj`` / ``v_proj`` shard by
   whole KV heads, and ``o_proj`` splits its input columns by the same
   query heads so each rank consumes exactly the heads it produced.
2. Gated-delta ``in_proj_qkv`` is one concatenated matrix
   ``[q (K*Dk) | k (K*Dk) | v (V*Dv)]`` whose q/k heads are
   repeat-interleaved onto the value heads (value head ``v`` reads key
   head ``v // n_rep``). Sharding by value-head groups and deriving
   each rank's q/k rows from that repeat mapping keeps every rank
   self-contained: it owns the q/k/v rows for exactly its value heads.
   ``conv1d`` channels follow the identical concatenated order, and
   ``in_proj_z`` / ``out_proj`` / ``in_proj_a`` / ``in_proj_b`` /
   ``A_log`` / ``dt_bias`` shard by the same value heads. The gated
   norm gain ``norm.weight`` is ``[Dv]`` shared by every value head,
   so it is replicated -- there is no per-head copy to split.

MTP tensors follow the same suffix rules as a decoder layer (the MTP
block is one more full-attention layer); only ``mtp.fc`` is replicated
outright, because it fuses the normed hidden and embedding branches
into one projection whose two halves a rank would otherwise have to
exchange before multiplying.

Byte accounting follows the same formulas (Kiely sec. 2.4-2.5 for why
bytes, not FLOPs, bound decode; sec. 4.2 for the FP8 block format):
FP8 weights at 1 byte/param, BF16 replicated/misc tensors at
2 bytes/param, and one BF16 ``weight_scale_inv`` per [128, 128] block.

All prose and code here is original; section numbers cite Kiely,
*Inference Engineering* (2026).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ShardSpec",
    "ShardKind",
    "shard_spec_for",
    "spec_for",
    "shard_state_dict",
    "merge_state_dicts",
    "partitioned_linear",
    "parameter_bytes_by_rank",
    "parameter_counts_by_rank",
    "real_config_report",
    "qwen38_tp_report",
    "allreduce_points",
    "SimulatedProcessGroup",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPMLP",
    "TPFullAttention",
    "TPGatedDeltaNet",
    "TPLinearAttention",
    "shard_linear_state",
    "merge_linear_state",
    "state_bytes",
    "state_bytes_table",
    "print_state_bytes",
]

# Shard kinds (plain strings so specs serialise and compare easily).
ShardKind = str
REPLICATED: ShardKind = "replicated"
COLUMN: ShardKind = "column"
ROW: ShardKind = "row"
HEAD_COLUMN: ShardKind = "head_column"
SEGMENTED_QKV: ShardKind = "segmented_qkv"
VOCAB: ShardKind = "vocab"

_GIB = 1024 ** 3
_FP8_BLOCK = (128, 128)


# ---------------------------------------------------------------------------
# Config dims: accept HybridConfig, ModelConfig, a mapping, or None (real).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Dims:
    hidden: int
    vocab: int
    layer_types: tuple[str, ...]
    n_q_head: int
    n_kv_head: int
    head_dim: int
    intermediate: int
    n_key_head: int
    key_dim: int
    n_value_head: int
    value_dim: int
    conv_kernel: int
    mtp_layers: int

    @property
    def key_total(self) -> int:
        return self.n_key_head * self.key_dim

    @property
    def value_total(self) -> int:
        return self.n_value_head * self.value_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_total + self.value_total

    @property
    def n_rep(self) -> int:
        return self.n_value_head // self.n_key_head


def _get(config: Any, *names: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        for n in names:
            if config.get(n) is not None:
                return config[n]
        return default
    for n in names:
        v = getattr(config, n, None)
        if v is not None:
            return v
    return default


def _dims(config: Any = None) -> _Dims:
    if config is None:
        from .hybrid import HybridConfig

        config = HybridConfig()
    hidden = int(_get(config, "hidden_size", "d_model", default=5120))
    n_q = int(_get(config, "num_attention_heads", "n_head", default=24))
    head_dim_raw = _get(config, "head_dim", default=None)
    head_dim = int(head_dim_raw) if head_dim_raw else hidden // n_q
    # A default-constructed HybridConfig derives head_dim as
    # hidden//n_head (5120//24=213) because its __init__ default is
    # None; the real checkpoint's head_dim is an independent 256.
    if hidden == 5120 and n_q == 24 and head_dim == 5120 // 24:
        head_dim = 256
    layer_types_raw = _get(config, "layer_types", default=None)
    n_layer = int(_get(config, "num_hidden_layers", "n_layer", default=64))
    if layer_types_raw:
        layer_types = tuple(
            "full_attention" if "full" in str(t) else "linear_attention"
            for t in layer_types_raw
        )
    else:
        interval = int(_get(config, "full_attention_interval", default=4))
        layer_types = tuple(
            "full_attention" if (i + 1) % max(interval, 1) == 0 else "linear_attention"
            for i in range(n_layer)
        )
    return _Dims(
        hidden=hidden,
        vocab=int(_get(config, "vocab_size", default=248320)),
        layer_types=layer_types,
        n_q_head=n_q,
        n_kv_head=int(_get(config, "num_key_value_heads", "n_kv_head", default=n_q)),
        head_dim=head_dim,
        intermediate=int(_get(config, "intermediate_size", "d_ff", default=17408)),
        n_key_head=int(_get(config, "linear_num_key_heads", "num_key_heads", default=16)),
        key_dim=int(_get(config, "linear_key_head_dim", "key_dim", default=128)),
        n_value_head=int(
            _get(config, "linear_num_value_heads", "num_value_heads", default=48)
        ),
        value_dim=int(
            _get(config, "linear_value_head_dim", "value_dim", default=128)
        ),
        conv_kernel=int(_get(config, "linear_conv_kernel_dim", "conv_kernel", default=4)),
        mtp_layers=int(_get(config, "mtp_num_hidden_layers", default=1)),
    )


# ---------------------------------------------------------------------------
# ShardSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardSpec:
    """How one named tensor is cut across TP ranks.

    Fields:
        name: tensor name as given (checkpoint or repo spelling).
        shape: full (dense) shape of the tensor.
        kind: one of ``replicated | column | row | head_column |
            segmented_qkv | vocab``. ``embed``/``lm_head`` report
            ``replicated`` for this simulation with
            ``future_kind="vocab"`` recording the production option.
        dim: axis split across ranks (``None`` when replicated).
        unit: what one shard unit is (``head``, ``kv_head``,
            ``value_head``, ``intermediate``, ``row`` ...).
        unit_size: elements along ``dim`` per unit.
        num_units: total units along ``dim``.
        reason: why this rule is correct for this tensor.
        future_kind: production alternative not used here, if any.
        segments: for ``segmented_qkv``, the dense layout as
            ``((segment, start, size), ...)`` in stored order.
        geom: for ``segmented_qkv``, ``(n_key_heads, key_dim,
            n_value_heads, value_dim)``; otherwise ``None``.
    """

    name: str
    shape: tuple[int, ...]
    kind: ShardKind
    dim: int | None = None
    unit: str = "row"
    unit_size: int = 1
    num_units: int = 0
    reason: str = ""
    future_kind: str | None = None
    segments: tuple[tuple[str, int, int], ...] = ()
    geom: tuple[int, int, int, int] | None = None

    # -- rank arithmetic ---------------------------------------------------

    def _check_rank_world(self, rank: int, world: int) -> None:
        if world <= 0:
            raise ValueError(f"world must be positive, got {world}")
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} out of range for world {world}")

    def units_for_rank(self, rank: int, world: int) -> list[int]:
        """Unit ids owned by ``rank`` (contiguous block, in order)."""
        self._check_rank_world(rank, world)
        if self.kind == REPLICATED or self.dim is None:
            return list(range(self.num_units))
        if self.kind == SEGMENTED_QKV:
            raise ValueError("segmented_qkv has no single unit list; use indices_for_rank")
        if self.num_units % world != 0:
            raise ValueError(
                f"{self.name}: {self.num_units} {self.unit} units not divisible "
                f"by world={world}"
            )
        per = self.num_units // world
        return list(range(rank * per, (rank + 1) * per))

    def indices_for_rank(self, rank: int, world: int) -> list[int]:
        """Dense-axis indices owned by ``rank``, in shard storage order.

        For ``segmented_qkv`` the order is the rank's q rows, then its
        k rows, then its v rows (the stored concatenation order); for
        every other sharded kind it is ascending dense order. For a
        replicated tensor it is the full axis.
        """
        self._check_rank_world(rank, world)
        if self.kind == REPLICATED or self.dim is None:
            return list(range(self.shape[self.dim] if self.dim is not None else 0)) \
                if self.dim is not None else []
        if self.kind == SEGMENTED_QKV:
            return _segmented_indices(self, rank, world)
        out: list[int] = []
        for u in self.units_for_rank(rank, world):
            out.extend(range(u * self.unit_size, (u + 1) * self.unit_size))
        return out

    # Aliases (same computation, alternative spellings).
    def shard_indices(self, rank: int, world: int) -> list[int]:
        return self.indices_for_rank(rank, world)

    def get_indices(self, rank: int, world: int) -> list[int]:
        return self.indices_for_rank(rank, world)

    def local_segments(self, rank: int, world: int) -> list[tuple[str, int, int, int]]:
        """``(segment, local_start, size, dense_start)`` for segmented specs."""
        if self.kind != SEGMENTED_QKV:
            return []
        return _segmented_local_segments(self, rank, world)

    def shard_shape(self, rank: int, world: int) -> tuple[int, ...]:
        """Shape of ``rank``'s shard of a tensor with this spec's shape."""
        self._check_rank_world(rank, world)
        if self.kind == REPLICATED or self.dim is None:
            return tuple(self.shape)
        shape = list(self.shape)
        shape[self.dim] = len(self.indices_for_rank(rank, world))
        return tuple(shape)

    @property
    def is_sharded(self) -> bool:
        return self.kind != REPLICATED and self.dim is not None

    def describe(self) -> str:
        if not self.is_sharded:
            extra = f" (future: {self.future_kind})" if self.future_kind else ""
            return f"{self.name}: replicated{extra} -- {self.reason}"
        return (
            f"{self.name}: {self.kind} dim={self.dim} "
            f"unit={self.unit}x{self.unit_size} ({self.num_units} units) -- {self.reason}"
        )


def _segmented_indices(spec: ShardSpec, rank: int, world: int) -> list[int]:
    geom = spec.geom
    if geom is None:
        raise ValueError(f"{spec.name}: segmented spec missing head geometry")
    n_key, d_k, n_value, d_v = geom
    n_rep = n_value // n_key
    if n_value % world != 0:
        raise ValueError(
            f"{spec.name}: {n_value} value heads not divisible by world={world}"
        )
    vh_per = n_value // world
    v_start, v_end = rank * vh_per, (rank + 1) * vh_per
    if vh_per % n_rep != 0:
        raise ValueError(
            f"{spec.name}: {vh_per} value heads/rank does not align to whole "
            f"key heads (repeat factor {n_rep}); choose a world that divides "
            "value heads in whole key-head groups"
        )
    k_start, k_end = v_start // n_rep, v_end // n_rep
    key_total = n_key * d_k
    q_idx = [i for h in range(k_start, k_end) for i in range(h * d_k, (h + 1) * d_k)]
    k_idx = [key_total + i for i in q_idx]
    v_idx = [
        2 * key_total + i
        for h in range(v_start, v_end)
        for i in range(h * d_v, (h + 1) * d_v)
    ]
    return q_idx + k_idx + v_idx


def _segmented_local_segments(
    spec: ShardSpec, rank: int, world: int
) -> list[tuple[str, int, int, int]]:
    geom = spec.geom
    if geom is None:
        raise ValueError(f"{spec.name}: segmented spec missing head geometry")
    n_key, d_k, n_value, d_v = geom
    n_rep = n_value // n_key
    if n_value % world != 0:
        raise ValueError(
            f"{spec.name}: {n_value} value heads not divisible by world={world}"
        )
    vh_per = n_value // world
    v_start = rank * vh_per
    k_start = v_start // n_rep
    key_total = n_key * d_k
    q_size = (vh_per // n_rep) * d_k
    v_size = vh_per * d_v
    return [
        ("q", 0, q_size, k_start * d_k),
        ("k", q_size, q_size, key_total + k_start * d_k),
        ("v", 2 * q_size, v_size, 2 * key_total + v_start * d_v),
    ]


# ---------------------------------------------------------------------------
# shard_spec_for: name + shape + config -> ShardSpec
# ---------------------------------------------------------------------------

_RE_LAYER_REST = re.compile(r"(?:^|\.)layers\.\d+\.(.+)$")


def _rest_of(name: str) -> str:
    """Tensor suffix after any ``layers.{i}.`` / checkpoint prefix."""
    m = _RE_LAYER_REST.search(name)
    if m:
        return m.group(1)
    rest = name
    for prefix in ("model.language_model.", "model."):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
    return rest


def _spec(name, shape, kind, dim=None, unit="row", unit_size=1, num_units=0,
          reason="", future_kind=None, segments=(), geom=None) -> ShardSpec:
    return ShardSpec(name=name, shape=tuple(int(s) for s in shape), kind=kind,
                     dim=dim, unit=unit, unit_size=int(unit_size),
                     num_units=int(num_units), reason=reason,
                     future_kind=future_kind, segments=tuple(segments), geom=geom)


def _replicated(name, shape, reason, future_kind=None) -> ShardSpec:
    return _spec(name, shape, REPLICATED, None, "replicated", 1,
                 int(shape[0]) if shape else 0, reason, future_kind)


def shard_spec_for(name: str, shape: Sequence[int] | torch.Size,
                   config: Any = None) -> ShardSpec:
    """Return the TP shard spec for one tensor.

    ``name`` may use repo spellings (``layers.3.self_attn.q_proj.weight``,
    ``embed_tokens.weight``), checkpoint spellings
    (``model.language_model.layers.3...``, ``mtp.layers.0...``) or a bare
    suffix (``self_attn.q_proj.weight``). ``shape`` is the dense shape;
    ``config`` is a ``HybridConfig`` / ``ModelConfig`` / mapping / None
    (None means the real Qwen3.8 text config). See the module docstring
    for the rules; the returned spec's ``reason`` states each one.
    """
    name = str(name)
    shape = tuple(int(s) for s in shape)
    d = _dims(config)
    rest = _rest_of(name)
    base_rest = rest

    # FP8 scale companions follow their weight's axis on the block grid.
    if rest.endswith("weight_scale_inv"):
        base_name = name[: -len("_scale_inv")]  # ...weight
        base_shape = _infer_base_shape(base_rest[: -len("weight_scale_inv")] + "weight", d)
        if base_shape is not None and len(shape) == 2 and len(base_shape) == 2:
            base_spec = shard_spec_for(base_name, base_shape, config)
            if base_spec.kind == SEGMENTED_QKV:
                return _replicated(name, shape,
                    "FP8 block-scale grid for the segmented qkv matrix is "
                    "replicated in this simulation; production shards block "
                    "rows with the derived q/k/v row groups")
            if base_spec.is_sharded and base_spec.dim is not None:
                kind = COLUMN if base_spec.dim == 0 else ROW
                return _spec(name, shape, kind, base_spec.dim, "scale_block", 1,
                             shape[base_spec.dim],
                             "weight_scale_inv block grid follows its FP8 "
                             "weight's sharded axis (one scale per [128,128] block)")
        return _replicated(name, shape, "FP8 block-scale companion replicated")

    is_bias = rest.endswith(".bias")
    w_rest = rest[: -len(".bias")] + ".weight" if is_bias else rest

    # --- embedding / head: replicated here, vocab-parallel in production ---
    if w_rest in ("embed_tokens.weight", "tok_embedding.weight") or \
            name.endswith("embed_tokens.weight") or "tok_embedding" in name:
        return _replicated(name, shape,
            "embedding replicated in this CPU simulation (every rank looks "
            "up every token); production may vocab-shard rows by token id",
            future_kind=VOCAB)
    if w_rest == "lm_head.weight" or name == "lm_head.weight" or name.endswith(".lm_head.weight"):
        return _replicated(name, shape,
            "lm_head replicated in this CPU simulation; production may "
            "vocab-shard rows and all-gather logits", future_kind=VOCAB)

    # --- MTP fusion projection: replicated with reason ---------------------
    if name.startswith("mtp.") and w_rest in ("fc.weight", "mtp.fc.weight"):
        return _replicated(name, shape,
            "mtp.fc fuses normed hidden + embedding branches ([D, 2D]); "
            "replicated so both branches stay local -- it is one matrix "
            "per MTP layer, not per decoder layer")

    # --- norms and other full-width gains: replicated ----------------------
    if (w_rest.endswith("layernorm.weight") or w_rest.endswith("norm.weight")
            or w_rest in ("norm.weight", "final_norm.weight")
            or "pre_fc_norm" in w_rest or w_rest.endswith(".norm.bias")
            or w_rest.endswith("layernorm.bias")):
        return _replicated(name, shape,
            "norm gain replicated: tiny, and every rank needs the full "
            "per-row reduction it parameterises")

    # --- full attention ------------------------------------------------------
    if w_rest.endswith("self_attn.q_proj.weight") or w_rest.endswith("attn.q_proj.weight") \
            or (".q_proj." in name and "in_proj" not in name):
        rows = shape[0]
        if rows == 2 * d.n_q_head * d.head_dim:
            return _spec(name, shape, HEAD_COLUMN, 0, "q_head_pair",
                         2 * d.head_dim, d.n_q_head,
                         "q_proj rows are per-head [query|gate] pairs; shard "
                         "by whole heads so a head's gate travels with its query")
        return _spec(name, shape, HEAD_COLUMN, 0, "q_head", d.head_dim,
                     d.n_q_head, "q_proj rows shard by whole query heads")
    if w_rest.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight",
                        "attn.k_proj.weight", "attn.v_proj.weight")) or \
            re.search(r"\.[kv]_proj\.(weight|bias)$", name):
        return _spec(name, shape, HEAD_COLUMN, 0, "kv_head", d.head_dim,
                     d.n_kv_head, "k/v_proj rows shard by whole KV heads "
                     "(world=2 on the real model: 4 KV heads -> 2 per rank)")
    if w_rest.endswith(("self_attn.o_proj.weight", "attn.o_proj.weight")) or \
            name.endswith(".o_proj.weight"):
        return _spec(name, shape, ROW, 1, "q_head", d.head_dim, d.n_q_head,
                     "o_proj input columns shard by the same query heads "
                     "this rank's q_proj produced (row-parallel: partial "
                     "outputs sum in one all-reduce)")

    # --- gated-delta (linear attention) -------------------------------------
    if w_rest.endswith("linear_attn.in_proj_qkv.weight") or \
            w_rest.endswith("linear_attn.conv1d.weight"):
        kt, vt = d.key_total, d.value_total
        segments = (("q", 0, kt), ("k", kt, kt), ("v", 2 * kt, vt))
        return _spec(name, shape, SEGMENTED_QKV, 0, "value_head", d.value_dim,
                     d.n_value_head,
                     "in_proj_qkv/conv1d channels are stored [q|k|v]; shard "
                     "by value-head groups and derive q/k rows from the "
                     "repeat mapping (value head v reads key head v//n_rep), "
                     "keeping the concatenated q|k|v order on each rank",
                     segments=segments,
                     geom=(d.n_key_head, d.key_dim, d.n_value_head, d.value_dim))
    if w_rest.endswith("linear_attn.in_proj_z.weight"):
        return _spec(name, shape, HEAD_COLUMN, 0, "value_head", d.value_dim,
                     d.n_value_head, "in_proj_z rows shard by value heads")
    if w_rest.endswith("linear_attn.out_proj.weight"):
        return _spec(name, shape, ROW, 1, "value_head", d.value_dim,
                     d.n_value_head, "out_proj input columns shard by the "
                     "same value heads (row-parallel: partials sum)")
    if w_rest.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                        "linear_attn.A_log", "linear_attn.dt_bias")):
        return _spec(name, shape, HEAD_COLUMN, 0, "value_head", 1,
                     d.n_value_head, "one row/element per value head; shard "
                     "by value heads with the rank's value-head group")

    # --- MLP ------------------------------------------------------------------
    if w_rest.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")) or \
            re.search(r"\.(gate|up)_proj\.(weight|bias)$", name):
        return _spec(name, shape, COLUMN, 0, "intermediate", 1,
                     shape[0], "gate/up rows shard by intermediate features "
                     "(column-parallel: outputs concatenate, no all-reduce "
                     "until down_proj)")
    if w_rest.endswith("mlp.down_proj.weight") or name.endswith(".down_proj.weight"):
        return _spec(name, shape, ROW, 1, "intermediate", 1, shape[1],
                     "down_proj columns shard by the same intermediate "
                     "features gate/up produced (row-parallel: partials sum)")

    # --- generic fallbacks ----------------------------------------------------
    if is_bias and len(shape) == 1:
        return _replicated(name, shape,
            "bias for a replicated or norm tensor; replicated with its owner")
    if name.startswith("model.visual."):
        return _replicated(name, shape,
            "vision tower is out of scope for text TP; replicated (deferred)")
    return _replicated(name, shape,
        "no TP rule for this tensor; replicated so shard/merge stays exact")


def _infer_base_shape(w_rest: str, d: _Dims) -> tuple[int, ...] | None:
    """Dense weight shape for a ``weight_scale_inv``'s base suffix."""
    table = {
        "self_attn.q_proj.weight": (2 * d.n_q_head * d.head_dim, d.hidden),
        "self_attn.k_proj.weight": (d.n_kv_head * d.head_dim, d.hidden),
        "self_attn.v_proj.weight": (d.n_kv_head * d.head_dim, d.hidden),
        "self_attn.o_proj.weight": (d.hidden, d.n_q_head * d.head_dim),
        "linear_attn.in_proj_qkv.weight": (d.conv_dim, d.hidden),
        "linear_attn.in_proj_z.weight": (d.value_total, d.hidden),
        "linear_attn.out_proj.weight": (d.hidden, d.value_total),
        "mlp.gate_proj.weight": (d.intermediate, d.hidden),
        "mlp.up_proj.weight": (d.intermediate, d.hidden),
        "mlp.down_proj.weight": (d.hidden, d.intermediate),
    }
    for suffix, shp in table.items():
        if w_rest.endswith(suffix):
            return shp
    return None


spec_for = shard_spec_for


# ---------------------------------------------------------------------------
# shard / merge state dicts (exact roundtrip)
# ---------------------------------------------------------------------------


def _slice_dim(t: torch.Tensor, dim: int, indices: list[int]) -> torch.Tensor:
    idx = torch.tensor(indices, dtype=torch.long, device=t.device)
    return t.index_select(dim, idx).clone()


def shard_state_dict(state_dict: Mapping[str, torch.Tensor], rank: int,
                     world: int, config: Any = None) -> dict[str, torch.Tensor]:
    """Return ``rank``'s shard of every tensor in ``state_dict``.

    Replicated tensors are cloned whole onto every rank; sharded
    tensors are sliced along their spec axis at the spec's indices
    (head pairs, KV heads, value-head groups with derived q/k rows,
    intermediate rows/columns). The input is never mutated.
    """
    if world <= 0:
        raise ValueError(f"world must be positive, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world {world}")
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            out[name] = tensor  # type: ignore[assignment]
            continue
        spec = shard_spec_for(name, tuple(tensor.shape), config)
        if not spec.is_sharded or spec.dim is None:
            out[name] = tensor.clone()
        else:
            if tensor.dim() <= spec.dim:
                out[name] = tensor.clone()
            else:
                out[name] = _slice_dim(tensor, spec.dim, spec.indices_for_rank(rank, world))
    return out


def merge_state_dicts(shards: Sequence[Mapping[str, torch.Tensor]] | Mapping[int, Mapping[str, torch.Tensor]],
                      config: Any = None) -> dict[str, torch.Tensor]:
    """Inverse of :func:`shard_state_dict`: rebuild the dense state dict.

    ``shards`` is the per-rank dicts in rank order (or a rank->dict
    mapping). Replicated tensors are taken from rank 0 after checking
    every rank agrees (``torch.equal``); sharded tensors are placed
    back at their dense indices, so the roundtrip is exact
    (``torch.equal``) for every tensor, including the segmented
    gated-delta qkv/conv tensors whose rank order differs from dense
    order. Raises ``ValueError`` on missing keys, mismatched replicated
    copies, or index sets that do not partition the dense axis.
    """
    if isinstance(shards, Mapping):
        ordered = [shards[r] for r in sorted(shards)]
    else:
        ordered = list(shards)
    if not ordered:
        raise ValueError("merge_state_dicts needs at least one shard")
    world = len(ordered)
    names = list(ordered[0].keys())
    for r, shard in enumerate(ordered):
        missing = [n for n in names if n not in shard]
        if missing:
            raise ValueError(f"rank {r} shard missing keys {missing[:3]}")
    out: dict[str, torch.Tensor] = {}
    for name in names:
        first = ordered[0][name]
        if not isinstance(first, torch.Tensor):
            out[name] = first
            continue
        # Reconstruct the dense shape first (sum of shard axis sizes).
        probe = shard_spec_for(name, tuple(first.shape), config)
        if not probe.is_sharded or probe.dim is None:
            for r in range(1, world):
                if not torch.equal(first, ordered[r][name]):
                    raise ValueError(f"replicated tensor {name!r} differs on rank {r}")
            out[name] = first.clone()
            continue
        dim = probe.dim
        dense_shape = list(first.shape)
        dense_shape[dim] = sum(int(s[name].shape[dim]) for s in ordered)
        spec = shard_spec_for(name, tuple(dense_shape), config)
        if not spec.is_sharded or spec.dim is None or spec.dim != dim:
            for r in range(1, world):
                if not torch.equal(first, ordered[r][name]):
                    raise ValueError(f"replicated tensor {name!r} differs on rank {r}")
            out[name] = first.clone()
            continue
        merged = torch.zeros(dense_shape, dtype=first.dtype, device=first.device)
        seen: list[int] = []
        for r in range(world):
            idx = spec.indices_for_rank(r, world)
            seen.extend(idx)
            shard_t = ordered[r][name]
            if int(shard_t.shape[dim]) != len(idx):
                raise ValueError(
                    f"{name}: rank {r} shard axis {int(shard_t.shape[dim])} "
                    f"!= expected {len(idx)}")
            merged.index_copy_(dim, torch.tensor(idx, dtype=torch.long), shard_t)
        if sorted(seen) != list(range(dense_shape[dim])):
            raise ValueError(f"{name}: rank indices do not partition the dense axis")
        out[name] = merged
    return out


# ---------------------------------------------------------------------------
# partitioned_linear: column concat / row sum == dense linear
# ---------------------------------------------------------------------------


def partitioned_linear(x: torch.Tensor, shards: Sequence[torch.Tensor],
                       spec: ShardSpec | None = None,
                       biases: Sequence[torch.Tensor | None] | None = None,
                       *, name: str | None = None,
                       config: Any = None) -> torch.Tensor:
    """Dense-equivalent linear from per-rank weight shards (CPU simulation).

    ``shards`` are the rank weight matrices in rank order for one
    projection, ``spec`` its :class:`ShardSpec` (or ``name``+``config``
    to derive it). Column/head/segmented kinds compute each rank's
    ``x @ W_r^T`` and place the outputs at their dense feature
    positions (segmented qkv outputs are reassembled into dense
    q|k|v order); the row kind slices ``x`` to each rank's input
    columns and sums the partials -- the all-reduce, performed here as
    an explicit sum. Replicated kinds evaluate rank 0's copy. An
    optional per-rank ``biases`` sequence is added per shard before
    recombination (row-parallel biases are added once, not per rank).
    """
    world = len(shards)
    if world == 0:
        raise ValueError("partitioned_linear needs at least one shard")
    if spec is None:
        if name is None:
            raise ValueError("partitioned_linear needs spec= or name=")
        # Dense shape: axis total is the sum of shard axis sizes; the
        # axis itself is resolved by re-deriving with a probe shape.
        probe_shape = tuple(shards[0].shape)
        spec = shard_spec_for(name, probe_shape, config)
        if spec.is_sharded and spec.dim is not None:
            dense = list(probe_shape)
            dense[spec.dim] = sum(int(s.shape[spec.dim]) for s in shards)
            spec = shard_spec_for(name, tuple(dense), config)
    bias_list: list[torch.Tensor | None] = list(biases) if biases is not None else [None] * world

    if not spec.is_sharded or spec.dim is None:
        b = bias_list[0]
        return F.linear(x, shards[0], b)

    if spec.dim == 1:  # row-parallel: slice input, sum partials
        total: torch.Tensor | None = None
        offset = 0
        for r, w in enumerate(shards):
            width = int(w.shape[1])
            cols = spec.indices_for_rank(r, world)
            if len(cols) != width:
                raise ValueError(f"rank {r}: spec width {len(cols)} != shard width {width}")
            part = F.linear(x[..., cols], w)
            total = part if total is None else total + part
        assert total is not None
        if bias_list[0] is not None:
            total = total + bias_list[0]
        return total

    # dim == 0: column-parallel family; place outputs at dense positions.
    if spec.kind == SEGMENTED_QKV:
        parts: list[tuple[int, torch.Tensor]] = []  # (dense_start, output)
        for r, w in enumerate(shards):
            y = F.linear(x, w, bias_list[r])
            for _seg, local_start, size, dense_start in spec.local_segments(r, world):
                parts.append((dense_start, y[..., local_start:local_start + size]))
        parts.sort(key=lambda p: p[0])
        return torch.cat([p[1] for p in parts], dim=-1)

    outs = [F.linear(x, w, bias_list[r]) for r, w in enumerate(shards)]
    # Head/column shards are stored in ascending dense order already,
    # so concatenation in rank order is the dense order.
    return torch.cat(outs, dim=-1)


# ---------------------------------------------------------------------------
# Executable simulated TP runtime (CPU, in-process, world ranks as slices)
# ---------------------------------------------------------------------------


def _check_world(world: int) -> int:
    world = int(world)
    if world <= 0:
        raise ValueError(f"world must be positive, got {world}")
    return world


class SimulatedProcessGroup:
    """In-process stand-in for a process group: ranks are list positions.

    ``all_reduce_sum`` sums one partial per rank (the row-parallel
    closing collective); ``gather_concat`` concatenates per-rank
    feature slices in rank order (the column-parallel recombination).
    Counters let tests assert a block performs *exactly one*
    all-reduce, not one per projection.
    """

    def __init__(self, world: int):
        self.world = _check_world(world)
        self.all_reduce_calls = 0
        self.gather_calls = 0
        self.all_reduce_elements = 0
        self.gather_elements = 0

    def all_reduce_sum(self, partials: Sequence[torch.Tensor]) -> torch.Tensor:
        parts = list(partials)
        if len(parts) != self.world:
            raise ValueError(f"all_reduce_sum needs {self.world} partials, got {len(parts)}")
        if not parts or not isinstance(parts[0], torch.Tensor):
            raise ValueError("all_reduce_sum partials must be tensors")
        shape = tuple(parts[0].shape)
        for p in parts:
            if tuple(p.shape) != shape:
                raise ValueError(f"all_reduce_sum shape mismatch {tuple(p.shape)} != {shape}")
        self.all_reduce_calls += 1
        self.all_reduce_elements += int(parts[0].numel()) * len(parts)
        total = parts[0].clone()
        for p in parts[1:]:
            total = total + p
        return total

    # Alias spelling used by some call sites.
    allreduce_sum = all_reduce_sum

    def gather_concat(self, partials: Sequence[torch.Tensor], dim: int = -1) -> torch.Tensor:
        parts = list(partials)
        if len(parts) != self.world:
            raise ValueError(f"gather_concat needs {self.world} partials, got {len(parts)}")
        self.gather_calls += 1
        self.gather_elements += sum(int(p.numel()) for p in parts)
        return torch.cat(parts, dim=dim)

    all_gather_concat = gather_concat


def _even_bounds(total: int, world: int, what: str) -> list[tuple[int, int]]:
    if total % world != 0:
        raise ValueError(f"{what}={total} not divisible by world={world}")
    per = total // world
    return [(r * per, (r + 1) * per) for r in range(world)]


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear built by slicing a dense ``nn.Linear``.

    Weight rows (output features) are split evenly; each rank computes
    ``F.linear(x, W_r, b_r)`` and outputs gather-concatenate in rank
    order, exactly reproducing the dense linear. No weight is
    re-initialised: shards are clones of dense row slices.
    """

    def __init__(self, dense: nn.Linear, world: int = 2,
                 process_group: SimulatedProcessGroup | None = None):
        super().__init__()
        if not isinstance(dense, nn.Linear):
            raise TypeError("ColumnParallelLinear must be built from nn.Linear")
        self.world = _check_world(world)
        self.process_group = process_group or SimulatedProcessGroup(self.world)
        if self.process_group.world != self.world:
            raise ValueError("process_group world mismatch")
        self.in_features = int(dense.in_features)
        self.out_features = int(dense.out_features)
        bounds = _even_bounds(self.out_features, self.world, "out_features")
        self._bounds = bounds
        self.shard_weights = nn.ParameterList(
            [nn.Parameter(dense.weight.data[s:e].clone()) for s, e in bounds])
        if dense.bias is not None:
            self.shard_biases: nn.ParameterList | None = nn.ParameterList(
                [nn.Parameter(dense.bias.data[s:e].clone()) for s, e in bounds])
        else:
            self.shard_biases = None

    @classmethod
    def from_dense(cls, dense: nn.Linear, world: int = 2,
                   process_group: SimulatedProcessGroup | None = None) -> "ColumnParallelLinear":
        return cls(dense, world=world, process_group=process_group)

    def per_rank_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [F.linear(x, w, (self.shard_biases[r] if self.shard_biases is not None else None))
                for r, w in enumerate(self.shard_weights)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process_group.gather_concat(self.per_rank_forward(x), dim=-1)


class RowParallelLinear(nn.Module):
    """Row-parallel linear built by slicing a dense ``nn.Linear``.

    Weight columns (input features) are split evenly; rank ``r``
    multiplies the matching input slice and the partials combine in
    one ``all_reduce_sum``. Bias (if any) is replicated and added once
    after the reduction, never per rank.
    """

    def __init__(self, dense: nn.Linear, world: int = 2,
                 process_group: SimulatedProcessGroup | None = None):
        super().__init__()
        if not isinstance(dense, nn.Linear):
            raise TypeError("RowParallelLinear must be built from nn.Linear")
        self.world = _check_world(world)
        self.process_group = process_group or SimulatedProcessGroup(self.world)
        if self.process_group.world != self.world:
            raise ValueError("process_group world mismatch")
        self.in_features = int(dense.in_features)
        self.out_features = int(dense.out_features)
        bounds = _even_bounds(self.in_features, self.world, "in_features")
        self._bounds = bounds
        self.shard_weights = nn.ParameterList(
            [nn.Parameter(dense.weight.data[:, s:e].clone()) for s, e in bounds])
        self.bias = nn.Parameter(dense.bias.data.clone()) if dense.bias is not None else None

    @classmethod
    def from_dense(cls, dense: nn.Linear, world: int = 2,
                   process_group: SimulatedProcessGroup | None = None) -> "RowParallelLinear":
        return cls(dense, world=world, process_group=process_group)

    def per_rank_partials(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [F.linear(x[..., s:e], w) for (s, e), w in zip(self._bounds, self.shard_weights)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.process_group.all_reduce_sum(self.per_rank_partials(x))
        if self.bias is not None:
            out = out + self.bias
        return out


class TPMLP(nn.Module):
    """Tensor-parallel SwiGLU: column gate/up, row down, one all-reduce."""

    def __init__(self, dense: Any, world: int = 2,
                 process_group: SimulatedProcessGroup | None = None):
        super().__init__()
        self.world = _check_world(world)
        self.process_group = process_group or SimulatedProcessGroup(self.world)
        if self.process_group.world != self.world:
            raise ValueError("process_group world mismatch")
        self.d_model = int(dense.gate_proj.in_features)
        self.d_ff = int(dense.gate_proj.out_features)
        self.gate_proj = ColumnParallelLinear(dense.gate_proj, self.world, self.process_group)
        self.up_proj = ColumnParallelLinear(dense.up_proj, self.world, self.process_group)
        self.down_proj = RowParallelLinear(dense.down_proj, self.world, self.process_group)
        # Aliases mirroring dense SwiGLU attribute names for shard access.
        self.gate = self.gate_proj
        self.up = self.up_proj
        self.down = self.down_proj

    @classmethod
    def from_dense(cls, dense: Any, world: int = 2,
                   process_group: SimulatedProcessGroup | None = None) -> "TPMLP":
        return cls(dense, world=world, process_group=process_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_parts = self.gate_proj.per_rank_forward(x)
        up_parts = self.up_proj.per_rank_forward(x)
        partials = []
        for r in range(self.world):
            hidden_r = F.silu(gate_parts[r]) * up_parts[r]
            s, e = self.down_proj._bounds[r]
            # hidden_r is exactly this rank's intermediate slice; the
            # down shard's columns are that same slice (no re-slicing).
            partials.append(F.linear(hidden_r, self.down_proj.shard_weights[r]))
        out = self.process_group.all_reduce_sum(partials)
        if self.down_proj.bias is not None:
            out = out + self.down_proj.bias
        return out


class TPFullAttention(nn.Module):
    """Tensor-parallel gated full attention (per-rank head subsets).

    q/k/v are head-column shards (q by whole ``[q|gate]`` head pairs),
    o is row-parallel over the same Q heads. Each rank runs attention
    only over its Q-head subset, repeating its local KV heads by the
    dense GQA factor; o partials combine in one all-reduce. Norms and
    RoPE reuse the dense module's modules/tables. KV-cache path:
    per-rank K/V are gathered to update the caller's cache once, then
    each rank reads its KV-head slice of the cached tensors -- the
    cache itself stays dense-shaped, as in ``GatedFullAttention``.
    """

    def __init__(self, dense: Any, world: int = 2,
                 process_group: SimulatedProcessGroup | None = None):
        super().__init__()
        self.dense = dense
        self.world = _check_world(world)
        self.process_group = process_group or SimulatedProcessGroup(self.world)
        if self.process_group.world != self.world:
            raise ValueError("process_group world mismatch")
        H, Hkv, Dh = int(dense.n_head), int(dense.n_kv_head), int(dense.head_dim)
        if H % self.world != 0:
            raise ValueError(f"n_head={H} not divisible by world={self.world}")
        if Hkv % self.world != 0:
            raise ValueError(f"n_kv_head={Hkv} not divisible by world={self.world}")
        self.n_head, self.n_kv_head, self.head_dim = H, Hkv, Dh
        self.n_rep = int(dense.n_rep)
        self.q_heads_per_rank = [list(range(r * H // self.world, (r + 1) * H // self.world))
                                 for r in range(self.world)]
        self.kv_heads_per_rank = [list(range(r * Hkv // self.world, (r + 1) * Hkv // self.world))
                                  for r in range(self.world)]
        # The KV mapping is consistent iff each rank's Q heads map to
        # exactly its KV heads (true when world divides both counts).
        for r in range(self.world):
            mapped = sorted({h // self.n_rep for h in self.q_heads_per_rank[r]})
            if mapped != self.kv_heads_per_rank[r]:
                raise ValueError("world splits Q heads across KV-head boundaries")
        def _rows(w: torch.Tensor, heads: list[int], per: int) -> torch.Tensor:
            idx = [i for h in heads for i in range(h * per, (h + 1) * per)]
            return w.data[idx].clone()
        def _bias(b: torch.Tensor | None, heads: list[int], per: int) -> torch.Tensor | None:
            if b is None:
                return None
            idx = [i for h in heads for i in range(h * per, (h + 1) * per)]
            return b.data[idx].clone()
        self.q_shards = [_rows(dense.q_proj.weight, hs, 2 * Dh) for hs in self.q_heads_per_rank]
        self.k_shards = [_rows(dense.k_proj.weight, hs, Dh) for hs in self.kv_heads_per_rank]
        self.v_shards = [_rows(dense.v_proj.weight, hs, Dh) for hs in self.kv_heads_per_rank]
        self.q_bias_shards = [_bias(dense.q_proj.bias, hs, 2 * Dh) for hs in self.q_heads_per_rank]
        self.k_bias_shards = [_bias(dense.k_proj.bias, hs, Dh) for hs in self.kv_heads_per_rank]
        self.v_bias_shards = [_bias(dense.v_proj.bias, hs, Dh) for hs in self.kv_heads_per_rank]
        self.o_shards = []
        for hs in self.q_heads_per_rank:
            cols = [i for h in hs for i in range(h * Dh, (h + 1) * Dh)]
            self.o_shards.append(dense.o_proj.weight.data[:, cols].clone())

    @classmethod
    def from_dense(cls, dense: Any, world: int = 2,
                   process_group: SimulatedProcessGroup | None = None) -> "TPFullAttention":
        return cls(dense, world=world, process_group=process_group)

    def forward(self, x: torch.Tensor, cos: torch.Tensor | None = None,
                sin: torch.Tensor | None = None, kv_cache: Any = None,
                layer_idx: int = 0, start_pos: int = 0,
                position_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        from .attention import manual_scaled_dot_product_attention, repeat_kv
        from .hybrid import _apply_rope_with_tables, apply_partial_rope, precompute_mrope_cos_sin
        from .kv_cache import StaticKVCache
        if position_ids is None:
            position_ids = kwargs.get("position_ids")
        if cos is None and "position_embeddings" in kwargs:
            cos, sin = kwargs["position_embeddings"]
        dense = self.dense
        B, T, D = x.shape
        Dh = self.head_dim
        partials: list[torch.Tensor] = []
        # First compute per-rank K/V so a cache is updated exactly once.
        rank_q, rank_gate, rank_k, rank_v = [], [], [], []
        for r in range(self.world):
            hs, kvs = self.q_heads_per_rank[r], self.kv_heads_per_rank[r]
            qg = F.linear(x, self.q_shards[r], self.q_bias_shards[r]).view(B, T, len(hs), 2 * Dh)
            rank_q.append(dense.q_norm(qg[..., :Dh]).transpose(1, 2))
            rank_gate.append(qg[..., Dh:].reshape(B, T, len(hs) * Dh))
            k_r = F.linear(x, self.k_shards[r], self.k_bias_shards[r]).view(B, T, len(kvs), Dh)
            v_r = F.linear(x, self.v_shards[r], self.v_bias_shards[r]).view(B, T, len(kvs), Dh)
            rank_k.append(dense.k_norm(k_r).transpose(1, 2))
            rank_v.append(v_r.transpose(1, 2))
        # RoPE (same tables/order as dense).
        if position_ids is not None:
            pos_cos, pos_sin = precompute_mrope_cos_sin(
                position_ids, dense.rotary_dim, theta=dense.rope_theta,
                mrope_section=dense.mrope_section)
            rank_q = [_apply_rope_with_tables(q, pos_cos, pos_sin, dense.rotary_dim) for q in rank_q]
            rank_k = [_apply_rope_with_tables(k, pos_cos, pos_sin, dense.rotary_dim) for k in rank_k]
        else:
            if cos is None or sin is None:
                cos, sin = dense._rope_tables(start_pos + T, x.device, x.dtype)
            rank_q = [apply_partial_rope(q, cos, sin, start_pos=start_pos, rotary_dim=dense.rotary_dim) for q in rank_q]
            rank_k = [apply_partial_rope(k, cos, sin, start_pos=start_pos, rotary_dim=dense.rotary_dim) for k in rank_k]
        if kv_cache is not None:
            k_full = torch.cat(rank_k, dim=1)
            v_full = torch.cat(rank_v, dim=1)
            if isinstance(kv_cache, StaticKVCache):
                k_full, v_full = kv_cache.update(layer_idx, k_full, v_full, start_pos)
            else:
                k_full, v_full = kv_cache.update(layer_idx, k_full, v_full)
            for r, kvs in enumerate(self.kv_heads_per_rank):
                idx = torch.tensor(kvs, dtype=torch.long, device=x.device)
                rank_k[r] = k_full.index_select(1, idx)
                rank_v[r] = v_full.index_select(1, idx)
        for r in range(self.world):
            k_rep = repeat_kv(rank_k[r], self.n_rep)
            v_rep = repeat_kv(rank_v[r], self.n_rep)
            if dense.attn_implementation == "sdpa":
                attn = dense._sdpa(rank_q[r], k_rep, v_rep, start_pos=start_pos)
            else:
                attn = manual_scaled_dot_product_attention(rank_q[r], k_rep, v_rep, is_causal=True)
            merged = attn.transpose(1, 2).contiguous().view(B, T, -1)
            if dense.gate_activation == "sigmoid":
                merged = merged * torch.sigmoid(rank_gate[r])
            elif dense.gate_activation == "silu":
                merged = merged * F.silu(rank_gate[r])
            partials.append(F.linear(merged, self.o_shards[r]))
        return self.process_group.all_reduce_sum(partials)


# --- linear (gated-delta) TP: state sharding + per-rank recurrent step -----


def _linear_geometry(S: torch.Tensor, conv: torch.Tensor) -> dict[str, int]:
    # S [B,Hv,Dk,Dv], conv [B,conv_dim,K-1]; conv_dim = 2*Kt + Hv*Dv.
    Hv, Dk, Dv = int(S.shape[1]), int(S.shape[2]), int(S.shape[3])
    conv_dim = int(conv.shape[1])
    key_total = (conv_dim - Hv * Dv) // 2
    return {"Hv": Hv, "Dk": Dk, "Dv": Dv, "conv_dim": conv_dim,
            "key_total": key_total, "n_key": key_total // Dk,
            "value_total": Hv * Dv, "n_rep": Hv // (key_total // Dk)}


def _conv_indices(geom: Mapping[str, int], rank: int, world: int) -> tuple[list[int], list[int], int, int]:
    """(conv_indices, value_heads, vh_per, kh_per_rank) for one rank."""
    Hv, Dk, Dv = geom["Hv"], geom["Dk"], geom["Dv"]
    n_rep, key_total = geom["n_rep"], geom["key_total"]
    if Hv % world != 0:
        raise ValueError(f"value heads {Hv} not divisible by world={world}")
    vh_per = Hv // world
    if vh_per % n_rep != 0:
        raise ValueError(f"{vh_per} value heads/rank does not align to repeat factor {n_rep}")
    v_heads = list(range(rank * vh_per, (rank + 1) * vh_per))
    k_heads = list(range(v_heads[0] // n_rep, v_heads[-1] // n_rep + 1))
    q_idx = [i for h in k_heads for i in range(h * Dk, (h + 1) * Dk)]
    k_idx = [key_total + i for i in q_idx]
    v_idx = [2 * key_total + i for h in v_heads for i in range(h * Dv, (h + 1) * Dv)]
    return q_idx + k_idx + v_idx, v_heads, vh_per, len(k_heads)


def shard_linear_state(state: tuple[torch.Tensor, torch.Tensor], rank: int, world: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank's slice of a gated-delta state: S by value heads, conv by
    the segmented qkv channel indices (per ``segmented_qkv`` spec)."""
    world = _check_world(world)
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world {world}")
    S, conv = state
    geom = _linear_geometry(S, conv)
    conv_idx, v_heads, _, _ = _conv_indices(geom, rank, world)
    return (S[:, v_heads].clone(),
            conv[:, torch.tensor(conv_idx, dtype=torch.long, device=conv.device)].clone())


def merge_linear_state(shards: Sequence[tuple[torch.Tensor, torch.Tensor]]
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`shard_linear_state` (exact roundtrip).

    S concatenates over value heads in rank order; conv channels are
    scattered back to their dense segmented indices (rank storage is
    ``[q_r|k_r|v_r]``, dense order is ``[q|k|v]``).
    """
    parts = list(shards)
    if not parts:
        raise ValueError("merge_linear_state needs at least one shard")
    world = len(parts)
    S = torch.cat([p[0] for p in parts], dim=1)
    B, Hv = int(S.shape[0]), int(S.shape[1])
    conv_dim_r = int(parts[0][1].shape[1])
    Kminus1 = int(parts[0][1].shape[2])
    # Recover dense conv_dim from a probe geometry built on merged S.
    probe_conv = torch.empty(B, conv_dim_r * world, Kminus1, dtype=parts[0][1].dtype)
    geom = _linear_geometry(S, probe_conv)
    # Dense conv_dim is not conv_dim_r*world in general only if world
    # misaligns; validate via indices partition below.
    conv = torch.zeros(B, geom["conv_dim"], Kminus1, dtype=parts[0][1].dtype, device=parts[0][1].device)
    seen: list[int] = []
    for r, (_, conv_r) in enumerate(parts):
        idx, _, _, _ = _conv_indices(geom, r, world)
        if len(idx) != int(conv_r.shape[1]):
            raise ValueError("conv shard width does not match segmented geometry")
        seen.extend(idx)
        conv[:, torch.tensor(idx, dtype=torch.long, device=conv.device)] = conv_r
    if sorted(seen) != list(range(geom["conv_dim"])):
        raise ValueError("conv shard indices do not partition the dense channels")
    return S, conv


class TPGatedDeltaNet(nn.Module):
    """Tensor-parallel gated-delta layer: recurrent step only.

    Weights are sliced once from the dense ``GatedDeltaNet`` (segmented
    qkv + conv channels, value-head z/a/b/A_log/dt_bias, row-parallel
    out_proj). Each rank steps only its value heads: its S slice and
    conv-channel slice evolve independently, and o partials combine in
    one all-reduce. State is held as a list of per-rank
    ``(S_r, conv_r)`` pairs; :meth:`shard_state` / :meth:`merge_state`
    (aliases ``restore_state``) convert to/from the dense pair exactly.

    Remaining (documented, not faked): prefill (sequential or chunkwise)
    is *not* sharded here -- only the single-token recurrent step is.
    A TP prefill would shard the same projections/conv over T and run
    the chunkwise triangular solve per rank over its value heads; the
    state/boundary maths is identical, but it is separate code.
    """

    def __init__(self, dense: Any, world: int = 2,
                 process_group: SimulatedProcessGroup | None = None):
        super().__init__()
        self.dense = dense
        self.world = _check_world(world)
        self.process_group = process_group or SimulatedProcessGroup(self.world)
        if self.process_group.world != self.world:
            raise ValueError("process_group world mismatch")
        net = dense
        if int(net.num_value_heads) % self.world != 0:
            raise ValueError("value heads not divisible by world")
        vh_per = int(net.num_value_heads) // self.world
        if vh_per % int(net.n_rep) != 0:
            raise ValueError("value heads/rank does not align to repeat factor")
        self.vh_per = vh_per
        # Build a probe geometry to reuse the segmented index maths.
        probe_S = torch.empty(1, int(net.num_value_heads), int(net.key_dim), int(net.value_dim))
        probe_conv = torch.empty(1, int(net.conv_dim), max(int(net.conv_kernel) - 1, 0))
        geom = _linear_geometry(probe_S, probe_conv)
        self._geom = geom
        self.rank_shards: list[dict[str, torch.Tensor]] = []
        for r in range(self.world):
            conv_idx, v_heads, _, _ = _conv_indices(geom, r, self.world)
            ci = torch.tensor(conv_idx, dtype=torch.long)
            z_idx = [i for h in v_heads for i in range(h * int(net.value_dim), (h + 1) * int(net.value_dim))]
            zi = torch.tensor(z_idx, dtype=torch.long)
            self.rank_shards.append({
                "in_proj_qkv": net.in_proj_qkv.weight.data[ci].clone(),
                "conv_weight": net.conv1d.weight.data[ci].clone(),
                "in_proj_z": net.in_proj_z.weight.data[zi].clone(),
                "in_proj_a": net.in_proj_a.weight.data[v_heads].clone(),
                "in_proj_b": net.in_proj_b.weight.data[v_heads].clone(),
                "A_log": net.A_log.data[v_heads].clone(),
                "dt_bias": net.dt_bias.data[v_heads].clone(),
                "out_proj": net.out_proj.weight.data[:, zi].clone(),
                "value_heads": torch.tensor(v_heads),
                "conv_indices": ci,
            })

    @classmethod
    def from_dense(cls, dense: Any, world: int = 2,
                   process_group: SimulatedProcessGroup | None = None) -> "TPGatedDeltaNet":
        return cls(dense, world=world, process_group=process_group)

    def shard_state(self, state: tuple[torch.Tensor, torch.Tensor], rank: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        return shard_linear_state(state, rank, self.world)

    def split_state(self, state: tuple[torch.Tensor, torch.Tensor] | None,
                    batch_size: int | None = None,
                    device: Any = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            assert batch_size is not None
            dense_state = self.dense.initial_state(batch_size, device=device, dtype=torch.float32)
            state = dense_state
        return [shard_linear_state(state, r, self.world) for r in range(self.world)]

    def merge_state(self, shards: Sequence[tuple[torch.Tensor, torch.Tensor]]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        return merge_linear_state(shards)

    restore_state = merge_state

    def step(self, x_t: torch.Tensor, state: Any = None
             ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One recurrent token across ranks.

        ``x_t`` is ``[B, D]`` or ``[B, 1, D]`` (as dense ``step``);
        ``state`` is a list of per-rank ``(S_r, conv_r)`` pairs, a dense
        ``(S, conv)`` pair (split here), or None (zero state). Returns
        ``(out, new_per_rank_state)`` equal to the dense step's output
        and (merged) new state.
        """
        from .linear_attention import _l2norm
        net = self.dense
        squeeze = False
        if x_t.dim() == 2:
            x2 = x_t
        elif x_t.dim() == 3 and x_t.shape[1] == 1:
            x2, squeeze = x_t[:, 0, :], True
        else:
            raise ValueError(f"x_t must be [B, D] or [B, 1, D], got {tuple(x_t.shape)}")
        B = int(x2.shape[0])
        if state is None:
            rank_states = self.split_state(None, batch_size=B, device=x2.device)
        elif isinstance(state, (list, tuple)) and len(state) == self.world and state and isinstance(state[0], (list, tuple)):
            rank_states = [(s[0], s[1]) for s in state]  # per-rank form
        else:
            rank_states = self.split_state(state)  # dense pair
        partials, new_states = [], []
        Dk, Dv = int(net.key_dim), int(net.value_dim)
        for r in range(self.world):
            sh = self.rank_shards[r]
            S_r, conv_r = rank_states[r]
            mixed = F.linear(x2, sh["in_proj_qkv"])
            z = F.linear(x2, sh["in_proj_z"]).reshape(B, self.vh_per, Dv)
            beta = torch.sigmoid(F.linear(x2, sh["in_proj_b"]))
            a = F.linear(x2, sh["in_proj_a"])
            g = -sh["A_log"].float().exp() * F.softplus(a.float() + sh["dt_bias"].float())
            # Conv step with this rank's channels / weight slice.
            if int(net.conv_kernel) == 1:
                window = mixed.unsqueeze(-1)
                new_buf = conv_r
            else:
                buf = conv_r.to(dtype=mixed.dtype)
                window = torch.cat([buf, mixed.unsqueeze(-1)], dim=-1)
                new_buf = window[:, :, 1:]
            w = sh["conv_weight"][:, 0, :].to(dtype=window.dtype)
            mixed_act = F.silu((window * w.unsqueeze(0)).sum(dim=-1))
            kh_per = self.vh_per // int(net.n_rep)
            q_sz, v_sz = kh_per * Dk, self.vh_per * Dv
            q_raw, k_raw, v = torch.split(mixed_act, [q_sz, q_sz, v_sz], dim=-1)
            q = q_raw.reshape(B, kh_per, Dk)
            k = k_raw.reshape(B, kh_per, Dk)
            v = v.reshape(B, self.vh_per, Dv).float()
            if int(net.n_rep) > 1:
                q = q.repeat_interleave(int(net.n_rep), dim=1)
                k = k.repeat_interleave(int(net.n_rep), dim=1)
            q = _l2norm(q) * (Dk ** -0.5)
            k = _l2norm(k)
            S = S_r.float() * torch.exp(g.float())[:, :, None, None]
            kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
            delta = (v - kv_mem) * beta.float().unsqueeze(-1)
            S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
            core = (S * q.unsqueeze(-1)).sum(dim=-2)
            core = net.norm(core.reshape(B * self.vh_per, Dv), z.reshape(B * self.vh_per, Dv))
            core = core.reshape(B, v_sz)
            partials.append(F.linear(core.to(sh["out_proj"].dtype), sh["out_proj"]))
            new_states.append((S, new_buf.to(dtype=mixed.dtype)))
        out = self.process_group.all_reduce_sum(partials).to(x_t.dtype)
        if squeeze:
            out = out.unsqueeze(1)
        return out, new_states


TPLinearAttention = TPGatedDeltaNet


# ---------------------------------------------------------------------------
# Runtime state byte accounting (KV cache + linear recurrent/conv state)
# ---------------------------------------------------------------------------


def state_bytes(config: Any = None, seq_len: int = 4096, batch: int = 1,
                kv_dtype_bytes: int | float = 2, world: int = 2,
                recurrent_dtype_bytes: int | float = 4,
                conv_dtype_bytes: int | float | None = None) -> dict[str, Any]:
    """Per-rank runtime-state bytes: full KV vs linear recurrent+conv.

    Formulas (per rank, ``world`` ranks):

    * full KV: ``n_full * 2(K,V) * B * seq * (H_kv/world) * head_dim *
      kv_dtype_bytes`` -- grows with ``seq_len``, sharded by KV heads.
    * linear recurrent: ``n_lin * B * (Hv/world) * Dk * Dv *
      recurrent_dtype_bytes`` (fp32 S, as in ``GatedDeltaNet``) --
      independent of ``seq_len``, sharded by value heads.
    * linear conv: ``n_lin * B * conv_dim_per_rank * (K-1) *
      conv_dtype_bytes`` where ``conv_dim_per_rank = 2*(kh_per*Dk) +
      vh_per*Dv`` (segmented channels) and ``conv_dtype_bytes``
      defaults to ``kv_dtype_bytes`` -- also seq-independent.
    """
    world = _check_world(world)
    if int(seq_len) <= 0 or int(batch) <= 0:
        raise ValueError("seq_len and batch must be positive")
    if conv_dtype_bytes is None:
        conv_dtype_bytes = kv_dtype_bytes
    d = _dims(config)
    n_full = sum(1 for t in d.layer_types if t == "full_attention")
    n_lin = len(d.layer_types) - n_full
    if d.n_kv_head % world != 0:
        raise ValueError(f"KV heads {d.n_kv_head} not divisible by world={world}")
    if d.n_value_head % world != 0:
        raise ValueError(f"value heads {d.n_value_head} not divisible by world={world}")
    kv_per_rank = d.n_kv_head // world
    vh_per = d.n_value_head // world
    if vh_per % d.n_rep != 0:
        raise ValueError("value heads/rank does not align to repeat factor")
    kh_per = vh_per // d.n_rep
    conv_dim_per_rank = 2 * kh_per * d.key_dim + vh_per * d.value_dim
    B, S = int(batch), int(seq_len)
    full_kv_per_rank = int(n_full * 2 * B * S * kv_per_rank * d.head_dim * float(kv_dtype_bytes))
    full_kv_dense = int(n_full * 2 * B * S * d.n_kv_head * d.head_dim * float(kv_dtype_bytes))
    lin_rec_per_rank = int(n_lin * B * vh_per * d.key_dim * d.value_dim * float(recurrent_dtype_bytes))
    lin_rec_dense = int(n_lin * B * d.n_value_head * d.key_dim * d.value_dim * float(recurrent_dtype_bytes))
    lin_conv_per_rank = int(n_lin * B * conv_dim_per_rank * max(d.conv_kernel - 1, 0) * float(conv_dtype_bytes))
    lin_conv_dense = int(n_lin * B * d.conv_dim * max(d.conv_kernel - 1, 0) * float(conv_dtype_bytes))
    lin_per_rank = lin_rec_per_rank + lin_conv_per_rank
    return {
        "world": world, "batch": B, "seq_len": S,
        "kv_dtype_bytes": kv_dtype_bytes, "recurrent_dtype_bytes": recurrent_dtype_bytes,
        "conv_dtype_bytes": conv_dtype_bytes,
        "full_layers": n_full, "linear_layers": n_lin,
        "kv_heads_per_rank": kv_per_rank, "value_heads_per_rank": vh_per,
        "conv_dim_per_rank": conv_dim_per_rank, "conv_dim_dense": d.conv_dim,
        "full_kv_bytes_per_rank": full_kv_per_rank,
        "full_kv_bytes_dense_total": full_kv_dense,
        "linear_recurrent_bytes_per_rank": lin_rec_per_rank,
        "linear_recurrent_bytes_dense_total": lin_rec_dense,
        "linear_conv_bytes_per_rank": lin_conv_per_rank,
        "linear_conv_bytes_dense_total": lin_conv_dense,
        "linear_bytes_per_rank": lin_per_rank,
        "linear_bytes_dense_total": lin_rec_dense + lin_conv_dense,
        "total_bytes_per_rank": full_kv_per_rank + lin_per_rank,
        "total_bytes_dense_total": full_kv_dense + lin_rec_dense + lin_conv_dense,
        "formulas": {
            "full_kv_per_rank": "n_full * 2 * B * seq_len * (H_kv/world) * head_dim * kv_dtype_bytes",
            "linear_recurrent_per_rank": "n_lin * B * (Hv/world) * Dk * Dv * recurrent_dtype_bytes (seq-independent)",
            "linear_conv_per_rank": "n_lin * B * conv_dim_per_rank * (K-1) * conv_dtype_bytes (seq-independent)",
        },
    }


def state_bytes_table(config: Any = None, seq_lens: Sequence[int] = (4096, 32768, 262144),
                      batch: int = 1, kv_dtype_bytes: int | float = 2, world: int = 2,
                      **kwargs: Any) -> list[dict[str, Any]]:
    """``state_bytes`` at each of ``seq_lens`` (real-config long-context table)."""
    return [state_bytes(config, seq_len=int(s), batch=batch,
                        kv_dtype_bytes=kv_dtype_bytes, world=world, **kwargs)
            for s in seq_lens]


def print_state_bytes(config: Any = None, seq_lens: Sequence[int] = (4096, 32768, 262144),
                      batch: int = 1, kv_dtype_bytes: int | float = 2, world: int = 2,
                      **kwargs: Any) -> str:
    """Human-readable state-bytes table (also the ``__main__`` CLI)."""
    rows = state_bytes_table(config, seq_lens=seq_lens, batch=batch,
                             kv_dtype_bytes=kv_dtype_bytes, world=world, **kwargs)
    gib = lambda b: b / _GIB  # noqa: E731
    lines = [f"state bytes per rank (world={world}, batch={batch}, kv={kv_dtype_bytes}B):",
             "seq_len  full_kv_GiB  linear_rec_GiB  linear_conv_GiB  total_GiB"]
    for r in rows:
        lines.append(f"{r['seq_len']:>7}  {gib(r['full_kv_bytes_per_rank']):>11.4f}  "
                     f"{gib(r['linear_recurrent_bytes_per_rank']):>14.4f}  "
                     f"{gib(r['linear_conv_bytes_per_rank']):>15.4f}  "
                     f"{gib(r['total_bytes_per_rank']):>9.4f}")
    lines.append("full KV grows with seq_len (KV-head sharded); linear recurrent+conv "
                 "state is fixed per token (value-head sharded).")
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Byte accounting by formula
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _scale_params(rows: int, cols: int) -> int:
    return _ceil_div(rows, _FP8_BLOCK[0]) * _ceil_div(cols, _FP8_BLOCK[1])


def parameter_counts_by_rank(config: Any = None, world: int = 2) -> dict[int, dict[str, int]]:
    """Per-rank parameter counts by storage kind, from config formulas.

    Kinds: ``fp8_weight`` (sharded FP8 matrices), ``bf16_sharded``
    (BF16 tensors cut across ranks: value-head a/b projections),
    ``bf16_replicated`` (embed/lm_head, norms, conv, A_log/dt_bias,
    mtp.fc, per-head norm gains -- charged whole to every rank),
    ``scale_inv`` (BF16 block scales for the FP8 matrices, sharded
    with their weights except the segmented qkv grid, replicated).
    MTP layers count as extra full-attention layers under the same
    suffix rules; vision tensors are excluded (deferred tower).
    """
    if world <= 0:
        raise ValueError(f"world must be positive, got {world}")
    d = _dims(config)
    D, F = d.hidden, d.intermediate
    n_full = sum(1 for t in d.layer_types if t == "full_attention")
    n_lin = len(d.layer_types) - n_full
    q_rows, kv_rows = 2 * d.n_q_head * d.head_dim, d.n_kv_head * d.head_dim
    o_cols = d.n_q_head * d.head_dim

    fp8_per_rank = 0
    scale_per_rank = 0

    def add_fp8(rows: int, cols: int, count: int, scale_replicated: bool = False):
        nonlocal fp8_per_rank, scale_per_rank
        fp8_per_rank += count * rows * cols // world
        s = _scale_params(rows, cols)
        scale_per_rank += count * (s if scale_replicated else s // world)

    # Per-layer FP8 matrices (identical count for full and linear MLPs).
    mlp_fp8 = (F * D + F * D + D * F)
    mlp_scale = _scale_params(F, D) * 2 + _scale_params(D, F)
    fp8_per_rank += len(d.layer_types) * mlp_fp8 // world
    scale_per_rank += len(d.layer_types) * mlp_scale // world
    # Full-attention FP8.
    fp8_per_rank += n_full * (q_rows * D + 2 * kv_rows * D + D * o_cols) // world
    scale_per_rank += n_full * (
        _scale_params(q_rows, D) + 2 * _scale_params(kv_rows, D) + _scale_params(D, o_cols)
    ) // world
    # Linear-attention FP8. in_proj_qkv's scale grid is replicated (see spec).
    fp8_per_rank += n_lin * (d.conv_dim * D + d.value_total * D + D * d.value_total) // world
    scale_per_rank += n_lin * (
        _scale_params(d.value_total, D) + _scale_params(D, d.value_total)
    ) // world
    scale_per_rank += n_lin * _scale_params(d.conv_dim, D)
    # MTP: one full-attention layer + MLP per MTP layer, same rules.
    fp8_per_rank += d.mtp_layers * (
        q_rows * D + 2 * kv_rows * D + D * o_cols + mlp_fp8) // world
    scale_per_rank += d.mtp_layers * (
        _scale_params(q_rows, D) + 2 * _scale_params(kv_rows, D)
        + _scale_params(D, o_cols) + mlp_scale) // world

    bf16_sharded = n_lin * (2 * d.n_value_head * D) // world  # in_proj_a + in_proj_b
    bf16_replicated = (
        2 * d.vocab * D                                   # embed + lm_head (untied)
        + len(d.layer_types) * 2 * D                      # input/post norms
        + n_full * 2 * d.head_dim                         # q/k norms
        + n_lin * (d.conv_dim * d.conv_kernel + 2 * d.n_value_head + d.value_dim)
        + D                                               # final norm
        + d.mtp_layers * (4 * D + 2 * d.head_dim + D * 2 * D)  # mtp norms + fc + ...
    )
    # mtp.fc is [D, 2D] BF16 replicated; the term above counts it as 2*D*D.
    counts = {
        "fp8_weight": int(fp8_per_rank),
        "bf16_sharded": int(bf16_sharded),
        "bf16_replicated": int(bf16_replicated),
        "scale_inv": int(scale_per_rank),
    }
    counts["total"] = int(sum(counts.values()))
    return {r: dict(counts) for r in range(world)}


def parameter_bytes_by_rank(config: Any = None, world: int = 2,
                            dtype_bytes_by_kind: Mapping[str, float] | None = None
                            ) -> dict[int, dict[str, int]]:
    """Per-rank bytes by storage kind, from :func:`parameter_counts_by_rank`.

    ``dtype_bytes_by_kind`` maps kind -> bytes per element; defaults
    are the checkpoint's own storage: ``fp8_weight=1`` (E4M3),
    ``bf16_*=2``, ``scale_inv=2`` (BF16 scales). Unknown kinds in the
    mapping are ignored; missing kinds fall back to those defaults.
    Each rank's dict also carries ``total_bytes`` (and ``total``).
    """
    defaults = {"fp8_weight": 1, "bf16_sharded": 2, "bf16_replicated": 2, "scale_inv": 2}
    rates = dict(defaults)
    if dtype_bytes_by_kind:
        for k, v in dtype_bytes_by_kind.items():
            rates[str(k)] = v
    out: dict[int, dict[str, int]] = {}
    for rank, counts in parameter_counts_by_rank(config, world).items():
        row: dict[str, int] = {}
        total = 0
        for kind, n in counts.items():
            if kind == "total":
                continue
            b = int(round(n * float(rates.get(kind, 2))))
            row[f"{kind}_bytes"] = b
            row[kind] = int(n)
            total += b
        row["total_bytes"] = int(total)
        row["total"] = int(total)
        out[rank] = row
    return out


def real_config_report(config: Any = None, world: int = 2) -> dict[str, Any]:
    """Real-config (Qwen3.8-27B) per-rank TP memory report.

    Assumptions, stated plainly: language model + MTP only (vision
    tower deferred, as elsewhere in this repo); FP8 weights at
    1 byte/param with BF16 [128,128]-block ``weight_scale_inv``;
    embedding and lm_head BF16 *replicated* on every rank (the
    simulation choice in :func:`shard_spec_for`); norms, conv, gated-
    delta misc and ``mtp.fc`` BF16 replicated; no KV/recurrent state,
    activations, or runtime buffers counted. Where the local
    checkpoint manifest is available its header totals are reported
    alongside as a cross-check (headers only, no tensor bytes read).
    """
    cfg = config
    if cfg is None:
        from .hybrid import HybridConfig
        cfg = HybridConfig()
    d = _dims(cfg)
    per_rank = parameter_bytes_by_rank(cfg, world)
    rank0 = per_rank[0]
    manifest_info: dict[str, Any] | None = None
    try:
        from .hybrid_checkpoint import build_manifest, find_default_snapshot
        snap = find_default_snapshot()
        if snap is not None:
            man = build_manifest(snap)
            manifest_info = {
                "snapshot": str(snap),
                "entries": len(man.entries),
                "total_bytes_by_dtype": man.total_bytes_by_dtype(),
                "counts_by_kind": man.counts_by_kind(),
            }
    except Exception:
        manifest_info = None
    total_gib = rank0["total_bytes"] / _GIB
    return {
        "world": int(world),
        "config": {
            "hidden_size": d.hidden, "layers": len(d.layer_types),
            "full_layers": sum(1 for t in d.layer_types if t == "full_attention"),
            "linear_layers": sum(1 for t in d.layer_types if t != "full_attention"),
            "vocab_size": d.vocab, "intermediate_size": d.intermediate,
        },
        "assumptions": [
            "language model + MTP only; vision tower excluded (deferred)",
            "FP8 e4m3 weights 1 byte/param; weight_scale_inv BF16 per [128,128] block",
            "embed_tokens and lm_head BF16 replicated per rank (future: vocab-parallel)",
            "norms, conv1d, A_log/dt_bias, mtp.fc BF16 replicated; in_proj_a/b BF16 sharded by value heads",
            "weights only: no KV cache, recurrent state, activations or buffers",
        ],
        "per_rank_bytes": per_rank,
        "per_rank_gib": {r: {k: (v / _GIB if k.endswith("_bytes") or k == "total" else v)
                             for k, v in row.items()} for r, row in per_rank.items()},
        "per_rank_total_gib": total_gib,
        "total_gib_rank0": total_gib,
        "fits_24gib_per_rank": bool(total_gib < 24.0),
        "manifest_cross_check": manifest_info,
    }


qwen38_tp_report = real_config_report


def allreduce_points(config: Any = None, world: int = 2) -> list[dict[str, Any]]:
    """All-reduce points per hybrid block, with byte formulas.

    One row-parallel all-reduce closes each token mixer and one closes
    each MLP, so a block costs 2 all-reduces of ``[B,T,D]`` partials
    (``2 * B*T*hidden*bytes`` moved per rank in a ring all-reduce,
    ``bytes`` = activation element size). Column-parallel halves need
    none. ``world=1`` needs none either (kept for formula symmetry).
    """
    d = _dims(config)
    per_block = [
        {"point": "token_mixer_row_parallel (o_proj | out_proj)",
         "collective": "all_reduce(sum)", "count_per_block": 1,
         "tensor": "[B, T, hidden]", "bytes_formula": "2*B*T*hidden*elem_bytes (ring)"},
        {"point": "mlp_down_proj (row-parallel)",
         "collective": "all_reduce(sum)", "count_per_block": 1,
         "tensor": "[B, T, hidden]", "bytes_formula": "2*B*T*hidden*elem_bytes (ring)"},
    ]
    return [{
        "per_block": per_block,
        "allreduces_per_block": 2,
        "blocks": len(d.layer_types),
        "allreduces_per_forward": 2 * len(d.layer_types),
        "bytes_per_allreduce_formula": "2*B*T*hidden*elem_bytes (ring, per rank)",
        "world": int(world),
    }]


if __name__ == "__main__":
    print_state_bytes()
