"""Per-rank checkpoint loading for the distributed hybrid LM.

The dense streaming loader (:mod:`hybrid_checkpoint`) loads a whole
checkpoint into one model. This module loads the SAME checkpoint into
``world`` tensor-parallel ranks: each rank streams one checkpoint
tensor at a time on CPU, slices it with the
:mod:`tensor_parallel` specs (resident FP8 pairs via
:func:`fp8_linear.shard_fp8_pair`, everything else via
``shard_state_dict``), and keeps ONLY its shard. Embeddings and
``lm_head`` are replicated on both ranks in v1 (vocab-parallel is
deferred, as in docs/23).

Peak CPU RAM while loading is one full checkpoint tensor plus its
scale at a time -- the largest single tensor is the replicated
embedding (248320 x 5120 BF16, ~2.4 GiB at the real config) -- plus
the destination rank's already-loaded shards if they are kept on CPU;
the loader moves each shard to ``device`` and drops the full tensor
before streaming the next one.

Cross-rank validation never moves tensors: ranks exchange only
per-tensor shard shapes/counts and a SHA-256 of the sorted
``name|shape|dtype`` lines (see :func:`rank_shard_summary` /
:func:`gather_rank_summaries`), and :func:`validate_rank_coverage`
checks, from headers alone, that the ranks' indices partition every
dense axis.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from .fp8_linear import shard_fp8_pair
from .hybrid_checkpoint import (
    Manifest, _load_one, _model_order_targets, _resolve_config, _resolve_dir,
    _target_to_checkpoint, build_manifest, expected_language_tensors, load_tensor,
)
from .quantization import quantize_block_fp8
from .tensor_parallel import parameter_bytes_by_rank, shard_spec_for, shard_state_dict
from .tp_distributed import DistributedHybridLM, TPBlockFP8Column, TPBlockFP8Row, _LocalContext

__all__ = [
    "iter_rank_shards",
    "load_rank_state",
    "rank_shard_summary",
    "gather_rank_summaries",
    "validate_rank_coverage",
    "resident_memory_report",
]

_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4, "F64": 8}
_GIB = 1024 ** 3
_BLOCK = (128, 128)


def _scale_shard_shape(weight_name: str, weight_shape, scale_shape,
                       rank: int, world: int, config) -> tuple[int, ...]:
    """Block-grid shard shape for a ``weight_scale_inv`` (see shard_fp8_pair)."""
    spec = shard_spec_for(weight_name, tuple(weight_shape), config)
    if not spec.is_sharded or spec.dim is None:
        return tuple(int(s) for s in scale_shape)
    idx = spec.indices_for_rank(rank, world)
    block_idx = sorted({int(i) // _BLOCK[spec.dim] for i in idx})
    out = [int(s) for s in scale_shape]
    out[spec.dim] = len(block_idx)
    return tuple(out)


def _shard_shape(name: str, shape, rank: int, world: int, config,
                 expected: dict | None = None, lm_head_mode: str = "replicated") -> tuple[int, ...]:
    if lm_head_mode == "vocab" and (name == "lm_head.weight" or name.endswith(".lm_head.weight")):
        return _lm_head_shard_shape(shape, rank, world)
    if name.endswith("weight_scale_inv") and expected is not None:
        base = name[: -len("_scale_inv")]
        base_spec = expected.get(base)
        if base_spec is not None:
            return _scale_shard_shape(base, base_spec["shape"], shape, rank, world, config)
    return shard_spec_for(name, tuple(shape), config).shard_shape(rank, world)



def _vocab_shard_tensor(full: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """This rank's zero-padded vocab rows of lm_head ``[V, H]``."""
    from .tp_distributed import vocab_shard_layout
    padded, shard = vocab_shard_layout(int(full.shape[0]), int(world))
    start = int(rank) * shard
    end = min(start + shard, int(full.shape[0]))
    out = torch.zeros((shard, int(full.shape[1])), dtype=full.dtype)
    if end > start:
        out[: end - start].copy_(full[start:end])
    return out


def _lm_head_shard_shape(shape, rank: int, world: int) -> tuple[int, ...]:
    from .tp_distributed import vocab_shard_layout
    _, shard = vocab_shard_layout(int(shape[0]), int(world))
    return (shard, int(shape[1]))

def iter_rank_shards(
    snapshot_dir: Any,
    config: Any = None,
    rank: int = 0,
    world: int = 1,
    *,
    fp8_mode: str = "resident",
    include_mtp: bool = False,
    manifest: Manifest | None = None,
    lm_head_mode: str = "replicated",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(target_name, cpu_shard)`` for ONLY this rank's tensors.

    Model order (as ``hybrid_checkpoint.iter_checkpoint_state``). In
    ``fp8_mode="resident"`` every FP8-target projection yields its raw
    FP8 shard followed by its ``<target>_scale_inv`` block-grid shard
    (a dense-stored FP8 target is quantised on the fly, mirroring
    ``load_into_model``); all other tensors are dequantised/sliced as
    dense. In ``fp8_mode="dequantize"`` everything yields dense shards.
    Only one full checkpoint tensor is materialised at a time.
    """
    if fp8_mode not in ("resident", "dequantize"):
        raise ValueError(f"fp8_mode must be 'resident' or 'dequantize', got {fp8_mode!r}")
    if include_mtp:
        raise ValueError("include_mtp=True is unsupported: DistributedHybridLM "
                         "holds no MTP block in v1")
    if lm_head_mode not in ("replicated", "vocab"):
        raise ValueError(f"lm_head_mode must be 'replicated' or 'vocab', got {lm_head_mode!r}")
    root = _resolve_dir(snapshot_dir)
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else (manifest or build_manifest(root))
    cfg = _resolve_config(snapshot_dir, config=config)
    expected = expected_language_tensors(cfg)
    by_name = man.by_name()
    for target in _model_order_targets(cfg, include_mtp=False):
        if cfg.tie_word_embeddings and target == "lm_head.weight":
            continue
        ckpt = _target_to_checkpoint(target)
        entry = by_name.get(ckpt)
        if entry is None:
            continue
        is_fp8_target = expected.get(ckpt, {}).get("kind") == "fp8_weight"
        if fp8_mode == "resident" and is_fp8_target:
            raw = _load_one(root, ckpt, manifest=man)
            if raw.dtype == torch.float8_e4m3fn:
                scale = _load_one(root, ckpt + "_scale_inv", manifest=man)
            else:
                q, scale_f32 = quantize_block_fp8(raw.to(torch.float32), block=_BLOCK)
                raw, scale = q, scale_f32.to(torch.bfloat16)
            w_shard, s_shard = shard_fp8_pair(target, raw, scale, rank, world, cfg, block=_BLOCK)
            yield target, w_shard
            yield target + "_scale_inv", s_shard
        else:
            full = load_tensor(root, ckpt, dequantize=True, manifest=man)
            if lm_head_mode == "vocab" and target == "lm_head.weight":
                yield target, _vocab_shard_tensor(full, rank, world)
            else:
                yield target, shard_state_dict({target: full}, rank, world, cfg)[target]


def _destinations(model: DistributedHybridLM, config: Any) -> dict[str, tuple[Any, str]]:
    """Target name -> (owner module, attribute) for every model tensor."""
    dest: dict[str, tuple[Any, str]] = {}

    def put(name: str, owner: Any, attr: str) -> None:
        dest[name] = (owner, attr)

    put("embed_tokens.weight", model.embed_tokens, "weight")
    put("norm.weight", model.norm, "weight")
    if not config.tie_word_embeddings:
        put("lm_head.weight", model.lm_head, "weight")
    for i, block in enumerate(model.layers):
        p = f"layers.{i}"
        put(f"{p}.input_layernorm.weight", block.input_layernorm, "weight")
        put(f"{p}.post_attention_layernorm.weight", block.post_attention_layernorm, "weight")
        if block.layer_type == "full_attention":
            attn = block.self_attn
            for proj, dense_attr, fp8_attr in (
                    ("q_proj", "q_weight", "q_fp8"), ("k_proj", "k_weight", "k_fp8"),
                    ("v_proj", "v_weight", "v_fp8"), ("o_proj", "o_weight", "o_fp8")):
                fp8_mod = getattr(attn, fp8_attr)
                if fp8_mod is not None:
                    put(f"{p}.self_attn.{proj}.weight", fp8_mod, "weight")
                    put(f"{p}.self_attn.{proj}.weight_scale_inv", fp8_mod, "weight_scale_inv")
                else:
                    put(f"{p}.self_attn.{proj}.weight", attn, dense_attr)
            put(f"{p}.self_attn.q_norm.weight", attn.q_norm, "weight")
            put(f"{p}.self_attn.k_norm.weight", attn.k_norm, "weight")
        else:
            la = block.linear_attn
            for target_suffix, dense_attr, fp8_attr in (
                    ("in_proj_qkv.weight", "in_proj_qkv_w", "in_proj_qkv_fp8"),
                    ("in_proj_z.weight", "in_proj_z_w", "in_proj_z_fp8"),
                    ("out_proj.weight", "out_proj_w", "out_proj_fp8")):
                fp8_mod = getattr(la, fp8_attr)
                if fp8_mod is not None:
                    put(f"{p}.linear_attn.{target_suffix}", fp8_mod, "weight")
                    put(f"{p}.linear_attn.{target_suffix}_scale_inv", fp8_mod, "weight_scale_inv")
                else:
                    put(f"{p}.linear_attn.{target_suffix}", la, dense_attr)
            put(f"{p}.linear_attn.conv1d.weight", la, "conv_weight")
            put(f"{p}.linear_attn.in_proj_a.weight", la, "in_proj_a_w")
            put(f"{p}.linear_attn.in_proj_b.weight", la, "in_proj_b_w")
            put(f"{p}.linear_attn.A_log", la, "A_log_r")
            put(f"{p}.linear_attn.dt_bias", la, "dt_bias_r")
            put(f"{p}.linear_attn.norm.weight", la.norm, "weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(block.mlp, proj)
            if isinstance(mod, (TPBlockFP8Column, TPBlockFP8Row)):
                put(f"{p}.mlp.{proj}.weight", mod.local, "weight")
                put(f"{p}.mlp.{proj}.weight_scale_inv", mod.local, "weight_scale_inv")
            else:
                put(f"{p}.mlp.{proj}.weight", mod, "weight")
    return dest


def _assign(owner: Any, attr: str, tensor: torch.Tensor, device: torch.device) -> None:
    current = getattr(owner, attr, None)
    moved = tensor.to(device)
    if isinstance(current, nn.Parameter):
        # Meta placeholders cannot take real data via .data; replace
        # the Parameter object (registration follows the attribute).
        setattr(owner, attr, nn.Parameter(moved, requires_grad=current.requires_grad))
    elif isinstance(current, torch.Tensor):
        setattr(owner, attr, moved)  # replaces the registered buffer
    else:
        raise RuntimeError(f"no tensor destination at {type(owner).__name__}.{attr}")


def load_rank_state(
    snapshot_dir: Any,
    config: Any = None,
    rank: int = 0,
    world: int = 1,
    device: torch.device | str | None = None,
    fp8_mode: str = "resident",
    *,
    ctx: Any = None,
    strict: bool = True,
    include_mtp: bool = False,
    use_triton: bool | None = None,
    manifest: Manifest | None = None,
    lm_head_mode: str = "replicated",
    use_scale_cache: bool = False,
) -> tuple[DistributedHybridLM, dict[str, Any]]:
    """Build this rank's :class:`DistributedHybridLM` from a checkpoint.

    The model is built shard-shaped on meta
    (:meth:`DistributedHybridLM.empty`) and filled tensor-by-tensor
    from :func:`iter_rank_shards` -- only this rank's shards are ever
    held. ``ctx`` (a real ``DistributedTPContext`` under torchrun) is
    attached via ``set_ctx``; without it a world-1 local context is
    attached so a single-rank load can forward immediately. Returns
    ``(model, report)`` with the loaded target names and this rank's
    resident bytes.
    """
    root = _resolve_dir(snapshot_dir)
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else (manifest or build_manifest(root))
    cfg = _resolve_config(snapshot_dir, config=config)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model = DistributedHybridLM.empty(cfg, rank, world, fp8_mode=fp8_mode, use_triton=use_triton, lm_head_mode=lm_head_mode)
    dest = _destinations(model, cfg)
    by_name = man.by_name()
    # Every destination (scale destinations included) maps to exactly
    # one checkpoint tensor via _target_to_checkpoint.
    missing = sorted(t for t in dest if _target_to_checkpoint(t) not in by_name)
    if strict and missing:
        raise RuntimeError(f"load_rank_state: checkpoint is missing tensors for this rank: {missing[:8]}")
    loaded: list[str] = []
    with torch.no_grad():
        for target, shard in iter_rank_shards(
                root, config=cfg, rank=rank, world=world, fp8_mode=fp8_mode,
                include_mtp=include_mtp, manifest=man, lm_head_mode=lm_head_mode):
            owner_attr = dest.get(target)
            if owner_attr is None:
                if strict:
                    raise RuntimeError(f"load_rank_state: no destination for {target!r}")
                continue
            _assign(owner_attr[0], owner_attr[1], shard, dev)
            loaded.append(target)
    scale_cache_modules = 0
    if use_scale_cache:
        from .fp8_linear import apply_scale_cache
        scale_cache_modules = apply_scale_cache(model, True)
    not_loaded = sorted(set(dest) - set(loaded))
    if strict and not_loaded:
        raise RuntimeError(f"load_rank_state: destinations never loaded: {not_loaded[:8]}")
    if ctx is not None:
        model.set_ctx(ctx)
    else:
        model.set_ctx(_LocalContext(rank, world, dev))
    model.eval()
    report = {
        "rank": int(rank), "world": int(world), "fp8_mode": fp8_mode,
        "loaded": loaded, "loaded_count": len(loaded),
        "missing": missing, "not_loaded": not_loaded,
        "resident_bytes": model.local_parameter_bytes(),
        "device": str(dev), "lm_head_mode": str(lm_head_mode),
        "use_scale_cache": bool(use_scale_cache),
        "scale_cache_modules": int(scale_cache_modules),
    }
    return model, report


def rank_shard_summary(snapshot_dir: Any, config: Any = None, rank: int = 0,
                       world: int = 2, *, include_mtp: bool = True,
                       manifest: Manifest | None = None,
                       lm_head_mode: str = "replicated") -> dict[str, Any]:
    """This rank's shard shapes/count/bytes + shape-hash (headers only).

    Covers every expected language tensor (MTP included by default;
    vision excluded as deferred). No tensor data is read, and the hash
    covers only ``name|shape|dtype`` lines, so summaries are safe to
    all-gather across ranks.
    """
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else (manifest or build_manifest(_resolve_dir(snapshot_dir)))
    cfg = _resolve_config(snapshot_dir, config=config)
    expected = expected_language_tensors(cfg)
    by_name = man.by_name()
    lines: list[str] = []
    total_bytes = 0
    shapes: dict[str, list[int]] = {}
    for ckpt in sorted(expected):
        entry = by_name.get(ckpt)
        if entry is None:
            continue
        shard_shape = _shard_shape(ckpt, entry.shape, rank, world, cfg, expected, lm_head_mode=lm_head_mode)
        shapes[ckpt] = list(shard_shape)
        lines.append(f"{ckpt}|{tuple(shard_shape)}|{entry.dtype}")
        numel = 1
        for s in shard_shape:
            numel *= int(s)
        total_bytes += numel * _DTYPE_BYTES.get(entry.dtype, 2)
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"rank": int(rank), "world": int(world), "count": len(lines),
            "bytes": int(total_bytes), "hash": digest, "shapes": shapes}


def gather_rank_summaries(ctx: Any, summary: dict[str, Any]) -> list[dict[str, Any]] | None:
    """All-gather rank summaries; the list is returned on rank 0 only.

    Only counts/hashes/shapes cross the wire -- never tensors.
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        return [summary]
    gathered: list[Any] = [None] * int(ctx.world_size)
    dist.all_gather_object(gathered, summary)
    return gathered if int(ctx.rank) == 0 else None


def validate_rank_coverage(snapshot_dir: Any, config: Any = None, world: int = 2,
                           *, include_mtp: bool = True,
                           manifest: Manifest | None = None) -> dict[str, Any]:
    """Header-only check that rank shards union to the dense manifest.

    For every expected language tensor: replicated specs must keep the
    dense shape on every rank; sharded specs' per-rank indices must
    partition the dense axis (block-grid indices partition the scale
    grid for ``weight_scale_inv``). Vision tensors are counted as
    excluded (deferred), never unexpected.
    """
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else (manifest or build_manifest(_resolve_dir(snapshot_dir)))
    cfg = _resolve_config(snapshot_dir, config=config)
    expected = expected_language_tensors(cfg)
    by_name = man.by_name()
    errors: list[str] = []
    # Missing-from-expected is validate_manifest's job; coverage here
    # judges only that present tensors' shards union to the dense axis.
    missing = sorted(n for n in expected if n not in by_name)
    for ckpt, spec_info in sorted(expected.items()):
        entry = by_name.get(ckpt)
        if entry is None:
            continue
        dense_shape = [int(s) for s in spec_info["shape"]]
        if list(entry.shape) != dense_shape:
            errors.append(f"{ckpt}: manifest shape {entry.shape} != expected {dense_shape}")
            continue
        if ckpt.endswith("weight_scale_inv"):
            base = ckpt[: -len("_scale_inv")]
            base_shape = expected[base]["shape"] if base in expected else dense_shape
            wspec = shard_spec_for(base, tuple(base_shape), cfg)
            if not wspec.is_sharded or wspec.dim is None:
                continue
            dim = wspec.dim
            # Cover, not partition: a sub-block tensor's single block
            # is legitimately held by every rank whose rows fall in it.
            seen: set[int] = set()
            for r in range(world):
                idx = wspec.indices_for_rank(r, world)
                seen.update(int(i) // _BLOCK[dim] for i in idx)
            if sorted(seen) != list(range(dense_shape[dim])):
                errors.append(f"{ckpt}: scale block indices do not cover grid axis {dim}")
            continue
        spec = shard_spec_for(ckpt, tuple(dense_shape), cfg)
        if not spec.is_sharded or spec.dim is None:
            continue
        seen = []
        for r in range(world):
            seen.extend(spec.indices_for_rank(r, world))
        if sorted(seen) != list(range(dense_shape[spec.dim])):
            errors.append(f"{ckpt}: rank indices do not partition dense axis {spec.dim}")
    excluded_vision = sum(1 for e in man.entries if e.target_name is None)
    per_rank = [rank_shard_summary(man, config=cfg, rank=r, world=world,
                                   include_mtp=include_mtp, manifest=man) for r in range(world)]
    return {"ok": not errors, "world": int(world), "errors": errors,
            "missing": missing, "excluded_vision": excluded_vision,
            "per_rank": [{"rank": s["rank"], "count": s["count"],
                          "bytes": s["bytes"], "hash": s["hash"]} for s in per_rank]}


def resident_memory_report(snapshot_dir: Any, config: Any = None, world: int = 2,
                           *, manifest: Manifest | None = None,
                           lm_head_mode: str = "replicated") -> dict[str, Any]:
    """Per-rank resident bytes by category for the REAL config (headers only).

    Categories follow the checkpoint's storage: sharded FP8 weights
    (1 B), their block-sliced BF16 ``weight_scale_inv``, BF16 tensors
    sharded by the TP specs (value-head a/b, A_log/dt_bias, conv
    channels), and BF16 replicated tensors (embed/lm_head, norms,
    ``mtp.fc``). Compared against the formula estimate in
    ``tensor_parallel.parameter_bytes_by_rank`` (16.36 GiB at world 2);
    ``explanations`` names every systematic delta. No allocation.
    """
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else (manifest or build_manifest(_resolve_dir(snapshot_dir)))
    cfg = _resolve_config(snapshot_dir, config=config)
    expected = expected_language_tensors(cfg)
    cats = ("fp8_weight", "scale_inv", "bf16_sharded", "bf16_replicated")
    per_rank: dict[int, dict[str, int]] = {}
    for rank in range(world):
        row = {c: 0 for c in cats}
        for entry in man.entries:
            if entry.target_name is None:
                continue  # vision: deferred, excluded from both sides
            shard_shape = _shard_shape(entry.checkpoint_name, entry.shape, rank, world, cfg, expected, lm_head_mode=lm_head_mode)
            numel = 1
            for s in shard_shape:
                numel *= int(s)
            nbytes = numel * _DTYPE_BYTES.get(entry.dtype, 2)
            if entry.kind == "fp8_weight":
                row["fp8_weight"] += nbytes
            elif entry.kind == "scale_inv":
                row["scale_inv"] += nbytes
            else:
                spec = shard_spec_for(entry.target_name, tuple(entry.shape), cfg)
                is_sh = spec.is_sharded or (lm_head_mode == "vocab" and entry.target_name == "lm_head.weight")
                row["bf16_sharded" if is_sh else "bf16_replicated"] += nbytes
        row["total"] = int(sum(row.values()))
        per_rank[rank] = {k: int(v) for k, v in row.items()}
    estimate = parameter_bytes_by_rank(cfg, world)[0]
    est_by_cat = {
        "fp8_weight": int(estimate.get("fp8_weight_bytes", 0)),
        "scale_inv": int(estimate.get("scale_inv_bytes", 0)),
        "bf16_sharded": int(estimate.get("bf16_sharded_bytes", 0)),
        "bf16_replicated": int(estimate.get("bf16_replicated_bytes", 0)),
    }
    est_total = int(estimate.get("total_bytes", sum(est_by_cat.values())))
    actual_total = per_rank[0]["total"]
    delta = actual_total - est_total
    explanations = [
        "vision tower (model.visual.*) is excluded from both sides (deferred in docs/22-23)",
        "segmented in_proj_qkv scales: the formula estimate charges the full "
        "[ceil(10240/128), ceil(5120/128)] grid to every rank, while resident "
        "loading block-slices it (40 of 80 grid rows per rank at world 2), so "
        "scale_inv lands below the estimate",
        "conv1d and A_log/dt_bias: the estimate prices them as BF16 replicated, "
        "while the specs shard conv channels (segmented) and A_log/dt_bias by "
        "value heads, moving bytes from bf16_replicated to bf16_sharded and "
        "halving the A_log/dt_bias total",
        "the estimate divides FP8 element counts by world with integer "
        "division per matrix family; the manifest count is exact per tensor, "
        "so a sub-0.1% rounding delta remains",
        "embed_tokens/lm_head are BF16 replicated on both ranks in v1 "
        "(~4.8 GiB of the replicated line); vocab-parallel would halve it",
    ]
    return {
        "world": int(world),
        "per_rank_bytes": per_rank,
        "per_rank_gib": {r: {k: v / _GIB for k, v in row.items()} for r, row in per_rank.items()},
        "per_rank_total_gib": actual_total / _GIB,
        "fits_24gib_per_rank": bool(actual_total < 24 * _GIB),
        "estimate_bytes_by_category": est_by_cat,
        "estimate_total_bytes": est_total,
        "estimate_total_gib": est_total / _GIB,
        "delta_bytes": int(delta),
        "delta_gib": delta / _GIB,
        "explanations": explanations,
        "assumptions": [
            "language model + MTP only; vision excluded (deferred)",
            "FP8 e4m3 weights 1 byte/param; weight_scale_inv BF16 per [128,128] block, block-sliced per rank",
            "embed_tokens and lm_head BF16 replicated per rank (v1; future vocab-parallel)",
            "weights only: no KV cache, recurrent state, activations or buffers",
        ],
    }
