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
