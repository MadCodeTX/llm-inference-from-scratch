"""Real-checkpoint streaming layer for the hybrid Qwen3.8 stack.

Header-first, CPU-only loading for the Qwen3.8-27B-FP8 checkpoint
(see Kiely, *Inference Engineering*, sec. 2.2 for config/mechanics,
sec. 2.4-2.5 for bottlenecks/attention, sec. 4.2 for formats and
reference implementations, sec. 5.2 for speculation/MTP context, and
sec. 5.4 for tensor-parallel sharding context).

Design: a 27B checkpoint must never be materialised to inspect it.
:func:`build_manifest` reads only safetensors headers (via
:func:`inference_from_scratch.weights.read_safetensors_header`), so
listing 1606 tensors costs kilobytes, not gigabytes.
:func:`load_tensor` then materialises exactly one tensor with
``safetensors.safe_open`` (lazy, mmap-backed), pairing an FP8
weight with its ``*_weight_scale_inv`` companion and dequantising
through :func:`inference_from_scratch.quantization.dequantize_block_fp8`
to BF16. :func:`iter_layer_tensors` streams one decoder layer at a
time on top of that primitive.

Target names reuse ``hybrid.list_hybrid_tensor_map``'s mapping
(checkpoint ``model.language_model.layers.{i}.*`` -> ``layers.{i}.*``,
``mtp.*`` unchanged, vision ``model.visual.*`` deferred -> ``None``).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from .fp8_linear import BlockFP8Linear
from .hybrid import HybridConfig, _map_hybrid_key
from .quantization import dequantize_block_fp8, quantize_block_fp8
from .weights import read_safetensors_header

__all__ = [
    "ManifestEntry", "Manifest", "discover_shards", "build_manifest",
    "expected_language_tensors", "validate_manifest", "load_tensor",
    "iter_layer_tensors", "find_default_snapshot",
    "iter_checkpoint_state", "validate_state_against_model", "load_into_model",
]

_BLOCK = (128, 128)
_RE_LAYER_IDX = re.compile(r"layers\.(\d+)\.")


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _scale_shape(shape: Sequence[int]) -> list[int]:
    return [_ceil_div(shape[0], _BLOCK[0]), _ceil_div(shape[1], _BLOCK[1])]


def _kind_for(checkpoint_name: str, dtype: str) -> str:
    if checkpoint_name.endswith("weight_scale_inv"):
        return "scale_inv"
    if dtype in ("F8_E4M3", "F8_E5M2"):
        return "fp8_weight"
    if dtype == "BF16":
        return "bf16"
    return "other"


@dataclass
class ManifestEntry:
    checkpoint_name: str
    target_name: str | None
    shard: str
    dtype: str
    shape: list[int]
    bytes: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_name": self.checkpoint_name,
                "target_name": self.target_name, "shard": self.shard,
                "dtype": self.dtype, "shape": list(self.shape),
                "bytes": self.bytes, "kind": self.kind}


@dataclass
class Manifest:
    snapshot_dir: Path
    entries: list[ManifestEntry] = field(default_factory=list)

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> list[str]:
        return [e.checkpoint_name for e in self.entries]

    def by_name(self) -> dict[str, ManifestEntry]:
        return {e.checkpoint_name: e for e in self.entries}

    def total_bytes_by_dtype(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.dtype] = out.get(e.dtype, 0) + int(e.bytes)
        return dict(sorted(out.items()))

    def counts_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.kind] = out.get(e.kind, 0) + 1
        return dict(sorted(out.items()))


def find_default_snapshot() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots"
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "config.json").is_file():
                return snap
    return None


def _resolve_dir(snapshot_dir: Any) -> Path:
    if isinstance(snapshot_dir, Manifest):
        return Path(snapshot_dir.snapshot_dir)
    if isinstance(snapshot_dir, (str, Path)):
        p = Path(snapshot_dir)
        if p.is_dir():
            return p
    raise FileNotFoundError(f"snapshot dir not found: {snapshot_dir!r}")


def discover_shards(snapshot_dir: str | Path) -> list[Path]:
    """Shard files of a snapshot, index-first, sorted by name.

    Uses ``model.safetensors.index.json``'s ``weight_map`` values when
    present (the authoritative shard list); otherwise globs
    ``*.safetensors``. Only existing files are returned.
    """
    root = _resolve_dir(snapshot_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            wm = data.get("weight_map") if isinstance(data, dict) else None
            if isinstance(wm, Mapping) and wm:
                shards = sorted({str(v) for v in wm.values()})
                found = [root / s for s in shards if (root / s).is_file()]
                if found:
                    return found
        except (ValueError, OSError):
            pass
    return sorted(root.glob("*.safetensors"))


def build_manifest(snapshot_dir: str | Path) -> Manifest:
    """Header-only manifest: no tensor data is loaded.

    Each entry records checkpoint/target names, shard filename,
    header dtype/shape, byte span, and kind
    (``fp8_weight|scale_inv|bf16|other``).
    """
    root = _resolve_dir(snapshot_dir)
    entries: list[ManifestEntry] = []
    for shard_path in discover_shards(root):
        header = read_safetensors_header(shard_path)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            entries.append(ManifestEntry(
                checkpoint_name=name,
                target_name=_map_hybrid_key(name),
                shard=shard_path.name,
                dtype=str(entry["dtype"]),
                shape=[int(d) for d in entry["shape"]],
                bytes=int(end) - int(begin),
                kind=_kind_for(name, str(entry["dtype"])),
            ))
    return Manifest(snapshot_dir=root, entries=entries)


def _as_config(config: Any) -> HybridConfig:
    if config is None:
        return HybridConfig()
    if isinstance(config, HybridConfig):
        return config
    return HybridConfig.from_text_config(config)


def _add_fp8(out: dict[str, dict[str, Any]], name: str, shape: list[int]) -> None:
    out[name] = {"shape": list(shape), "dtype": "F8_E4M3", "kind": "fp8_weight"}
    out[name + "_scale_inv"] = {"shape": _scale_shape(shape), "dtype": "BF16", "kind": "scale_inv"}


def _add_bf16(out: dict[str, dict[str, Any]], name: str, shape: list[int]) -> None:
    out[name] = {"shape": list(shape), "dtype": "BF16", "kind": "bf16"}


def _add_full_attn(out: dict[str, dict[str, Any]], prefix: str, cfg: HybridConfig) -> None:
    H = cfg.hidden_size
    q_out = 2 * cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    _add_fp8(out, f"{prefix}.self_attn.q_proj.weight", [q_out, H])
    _add_fp8(out, f"{prefix}.self_attn.k_proj.weight", [kv_out, H])
    _add_fp8(out, f"{prefix}.self_attn.v_proj.weight", [kv_out, H])
    _add_fp8(out, f"{prefix}.self_attn.o_proj.weight", [H, cfg.num_attention_heads * cfg.head_dim])
    _add_bf16(out, f"{prefix}.self_attn.q_norm.weight", [cfg.head_dim])
    _add_bf16(out, f"{prefix}.self_attn.k_norm.weight", [cfg.head_dim])


def _add_mlp(out: dict[str, dict[str, Any]], prefix: str, cfg: HybridConfig) -> None:
    H, I = cfg.hidden_size, cfg.intermediate_size
    _add_fp8(out, f"{prefix}.mlp.gate_proj.weight", [I, H])
    _add_fp8(out, f"{prefix}.mlp.up_proj.weight", [I, H])
    _add_fp8(out, f"{prefix}.mlp.down_proj.weight", [H, I])


def expected_language_tensors(config: Any = None) -> dict[str, dict[str, Any]]:
    """Expected language+MTP tensors by formula from ``HybridConfig``.

    Returns ``{checkpoint_name: {"shape", "dtype", "kind"}}`` including
    every FP8 weight's ``weight_scale_inv`` companion at
    ``[ceil(R/128), ceil(C/128)]``. Vision tensors are excluded.
    """
    cfg = _as_config(config)
    H, V = cfg.hidden_size, cfg.vocab_size
    out: dict[str, dict[str, Any]] = {}
    _add_bf16(out, "model.language_model.embed_tokens.weight", [V, H])
    _add_bf16(out, "model.language_model.norm.weight", [H])
    if not cfg.tie_word_embeddings:
        _add_bf16(out, "lm_head.weight", [V, H])
    key_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    value_total = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    conv_dim = key_total * 2 + value_total
    for i, layer_type in enumerate(cfg.layer_types):
        p = f"model.language_model.layers.{i}"
        _add_bf16(out, f"{p}.input_layernorm.weight", [H])
        _add_bf16(out, f"{p}.post_attention_layernorm.weight", [H])
        if layer_type == "full_attention":
            _add_full_attn(out, p, cfg)
        else:
            _add_fp8(out, f"{p}.linear_attn.in_proj_qkv.weight", [conv_dim, H])
            _add_fp8(out, f"{p}.linear_attn.in_proj_z.weight", [value_total, H])
            _add_fp8(out, f"{p}.linear_attn.out_proj.weight", [H, value_total])
            _add_bf16(out, f"{p}.linear_attn.in_proj_a.weight", [cfg.linear_num_value_heads, H])
            _add_bf16(out, f"{p}.linear_attn.in_proj_b.weight", [cfg.linear_num_value_heads, H])
            _add_bf16(out, f"{p}.linear_attn.conv1d.weight", [conv_dim, 1, cfg.linear_conv_kernel_dim])
            _add_bf16(out, f"{p}.linear_attn.A_log", [cfg.linear_num_value_heads])
            _add_bf16(out, f"{p}.linear_attn.dt_bias", [cfg.linear_num_value_heads])
            _add_bf16(out, f"{p}.linear_attn.norm.weight", [cfg.linear_value_head_dim])
        _add_mlp(out, p, cfg)
    if int(cfg.mtp_num_hidden_layers) > 0:
        _add_bf16(out, "mtp.pre_fc_norm_embedding.weight", [H])
        _add_bf16(out, "mtp.pre_fc_norm_hidden.weight", [H])
        _add_bf16(out, "mtp.fc.weight", [H, 2 * H])
        _add_bf16(out, "mtp.norm.weight", [H])
    for j in range(int(cfg.mtp_num_hidden_layers)):
        mp = f"mtp.layers.{j}"
        _add_bf16(out, f"{mp}.input_layernorm.weight", [H])
        _add_bf16(out, f"{mp}.post_attention_layernorm.weight", [H])
        _add_full_attn(out, mp, cfg)
        _add_mlp(out, mp, cfg)
    return out


def _is_languageish(name: str) -> bool:
    return (name.startswith("model.language_model.") or name.startswith("mtp.")
            or name == "lm_head.weight")


def validate_manifest(manifest: Manifest, config: Any = None) -> dict[str, Any]:
    """Compare a header manifest against formula-derived expectations.

    Vision tensors are out of scope (deferred), so they are neither
    expected nor unexpected. Returns ``{missing, unexpected,
    shape_mismatch, ok, counts, total_bytes_by_dtype}``.
    """
    cfg = _as_config(config)
    expected = expected_language_tensors(cfg)
    by_name = manifest.by_name() if isinstance(manifest, Manifest) else {}
    missing = sorted(n for n in expected if n not in by_name)
    unexpected = sorted(
        e.checkpoint_name for e in manifest.entries
        if _is_languageish(e.checkpoint_name) and e.checkpoint_name not in expected
    )
    shape_mismatch: list[dict[str, Any]] = []
    for name in sorted(expected):
        entry = by_name.get(name)
        if entry is None:
            continue
        if list(entry.shape) != list(expected[name]["shape"]):
            shape_mismatch.append({
                "checkpoint_name": name,
                "expected_shape": list(expected[name]["shape"]),
                "actual_shape": list(entry.shape),
                "shape": list(entry.shape),
                "expected": list(expected[name]["shape"]),
            })
    total_bytes_by_dtype = (manifest.total_bytes_by_dtype()
                            if isinstance(manifest, Manifest) else {})
    counts = {
        "expected": len(expected), "present": len(expected) - len(missing),
        "manifest_entries": len(manifest.entries) if isinstance(manifest, Manifest) else 0,
        "missing": len(missing), "unexpected": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
    }
    return {"missing": missing, "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
            "ok": not missing and not unexpected and not shape_mismatch,
            "counts": counts, "total_bytes_by_dtype": total_bytes_by_dtype}


def _shard_for(root: Path, checkpoint_name: str, manifest: Manifest | None = None) -> Path:
    if manifest is not None:
        entry = manifest.by_name().get(checkpoint_name)
        if entry is not None:
            return root / entry.shard
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            wm = data.get("weight_map") if isinstance(data, dict) else None
            if isinstance(wm, Mapping) and checkpoint_name in wm:
                return root / str(wm[checkpoint_name])
        except (ValueError, OSError):
            pass
    for shard_path in discover_shards(root):
        header = read_safetensors_header(shard_path)
        if checkpoint_name in header:
            return shard_path
    raise KeyError(f"unknown tensor {checkpoint_name!r} in snapshot {root}")


def _load_one(root: Path, checkpoint_name: str, manifest: Manifest | None = None) -> torch.Tensor:
    from safetensors import safe_open  # lazy: header paths need no safetensors
    shard_path = _shard_for(root, checkpoint_name, manifest=manifest)
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        if checkpoint_name not in f.keys():
            raise KeyError(f"unknown tensor {checkpoint_name!r} in shard {shard_path.name}")
        return f.get_tensor(checkpoint_name)


def load_tensor(snapshot_dir: Any, checkpoint_name: str, dequantize: bool = True,
                manifest: Manifest | None = None) -> torch.Tensor:
    """Lazily load one tensor; FP8 weights dequantise to BF16.

    ``snapshot_dir`` may be a path or a :class:`Manifest`. With
    ``dequantize=True`` (default) an FP8 weight is paired with its
    ``<name>_scale_inv`` companion (i.e. ``...weight`` ->
    ``...weight_scale_inv``) and returned BF16 via
    ``dequantize_block_fp8``; a non-FP8 tensor is returned as stored.
    With ``dequantize=False`` the raw stored tensor is returned.
    Loading a ``*_weight_scale_inv`` tensor directly, an unknown
    name, or an FP8 weight whose scale is missing raises a clear error.
    """
    if checkpoint_name.endswith("weight_scale_inv"):
        raise ValueError(
            f"cannot load scale tensor {checkpoint_name!r} directly; "
            "load its FP8 weight instead (the scale is paired automatically)")
    root = _resolve_dir(snapshot_dir)
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else manifest
    tensor = _load_one(root, checkpoint_name, manifest=man)
    if not dequantize:
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        scale_name = checkpoint_name + "_scale_inv"
        try:
            scale = _load_one(root, scale_name, manifest=man)
        except KeyError as exc:
            raise KeyError(
                f"FP8 weight {checkpoint_name!r} has no companion "
                f"scale {scale_name!r} in snapshot {root}") from exc
        return dequantize_block_fp8(tensor, scale.to(torch.float32), block=_BLOCK,
                                    out_dtype=torch.bfloat16)
    return tensor


def iter_layer_tensors(snapshot_dir: Any, layer_idx: int | None = None, *,
                       dequantize: bool = True,
                       layer_index: int | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(target_name, tensor)`` for one decoder layer at a time.

    Generator: nothing is loaded until iterated, and only the
    requested layer's tensors are materialised (FP8 weights are
    dequantised; their ``weight_scale_inv`` companions are consumed
    by the pairing and not yielded separately). ``snapshot_dir`` may
    be a path or a :class:`Manifest`.
    """
    idx = layer_idx if layer_idx is not None else layer_index
    if idx is None:
        raise TypeError("iter_layer_tensors requires layer_idx")
    idx = int(idx)
    root = _resolve_dir(snapshot_dir)
    manifest = snapshot_dir if isinstance(snapshot_dir, Manifest) else build_manifest(root)
    prefix_ckpt = f"model.language_model.layers.{idx}."
    selected = [e for e in manifest.entries
                if e.checkpoint_name.startswith(prefix_ckpt) and e.kind != "scale_inv"]
    selected.sort(key=lambda e: e.checkpoint_name)
    for entry in selected:
        tensor = load_tensor(root, entry.checkpoint_name, dequantize=dequantize, manifest=manifest)
        yield (entry.target_name or entry.checkpoint_name), tensor


# ---------------------------------------------------------------------------
# Full-model assembly: iter / validate-against-model / streaming load.
#
# Target naming is exactly TinyHybridLM/HybridLM state_dict naming:
#   checkpoint model.language_model.embed_tokens.weight -> embed_tokens.weight
#   checkpoint model.language_model.norm.weight         -> norm.weight (final norm)
#   checkpoint lm_head.weight                           -> lm_head.weight
#   checkpoint model.language_model.layers.{i}.*        -> layers.{i}.*
#   checkpoint mtp.*                                    -> mtp.* (only if include_mtp)
# Vision (model.visual.* -> None) is always out of scope.  FP8
# ``*_weight_scale_inv`` companions are consumed by load_tensor's
# dequantisation pairing and are never yielded / never model targets.
# ---------------------------------------------------------------------------


def _target_to_checkpoint(target_name: str) -> str:
    if target_name == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target_name == "norm.weight":
        return "model.language_model.norm.weight"
    if target_name == "lm_head.weight":
        return "lm_head.weight"
    if target_name.startswith("mtp."):
        return target_name
    if target_name.startswith("layers."):
        return f"model.language_model.{target_name}"
    return target_name


def _model_order_targets(cfg: HybridConfig, include_mtp: bool = False) -> list[str]:
    """Target names in deterministic model (state_dict) order.

    Order mirrors TinyHybridLM.state_dict(): embed, then per layer
    (input/post norms, mixer params in state_dict order, MLP), then
    final norm and lm_head.  MTP targets, when requested, follow in
    expected_language_tensors order (pre-fc norms, fc, norm, layers).
    """
    out: list[str] = ["embed_tokens.weight"]
    for i, layer_type in enumerate(cfg.layer_types):
        p = f"layers.{i}"
        out.append(f"{p}.input_layernorm.weight")
        out.append(f"{p}.post_attention_layernorm.weight")
        if layer_type == "full_attention":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                out.append(f"{p}.self_attn.{proj}.weight")
            out.append(f"{p}.self_attn.q_norm.weight")
            out.append(f"{p}.self_attn.k_norm.weight")
        else:
            # GatedDeltaNet direct params precede its Linear children
            # in state_dict order.
            out.append(f"{p}.linear_attn.dt_bias")
            out.append(f"{p}.linear_attn.A_log")
            for proj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                out.append(f"{p}.linear_attn.{proj}.weight")
            out.append(f"{p}.linear_attn.conv1d.weight")
            out.append(f"{p}.linear_attn.norm.weight")
            out.append(f"{p}.linear_attn.out_proj.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out.append(f"{p}.mlp.{proj}.weight")
    out.append("norm.weight")
    out.append("lm_head.weight")
    if include_mtp and int(cfg.mtp_num_hidden_layers) > 0:
        out.append("mtp.pre_fc_norm_embedding.weight")
        out.append("mtp.pre_fc_norm_hidden.weight")
        out.append("mtp.fc.weight")
        out.append("mtp.norm.weight")
        for j in range(int(cfg.mtp_num_hidden_layers)):
            mp = f"mtp.layers.{j}"
            out.append(f"{mp}.input_layernorm.weight")
            out.append(f"{mp}.post_attention_layernorm.weight")
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                out.append(f"{mp}.self_attn.{proj}.weight")
            out.append(f"{mp}.self_attn.q_norm.weight")
            out.append(f"{mp}.self_attn.k_norm.weight")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                out.append(f"{mp}.mlp.{proj}.weight")
    return out


def _resolve_config(snapshot_dir: Any, config: Any = None, model: Any = None) -> HybridConfig:
    if config is not None:
        return _as_config(config)
    if model is not None and getattr(model, "config", None) is not None:
        return _as_config(model.config)
    # Try the snapshot's own config.json before falling back to defaults.
    try:
        root = _resolve_dir(snapshot_dir)
        if (root / "config.json").is_file():
            return HybridConfig.from_text_config(root)
    except Exception:
        pass
    return HybridConfig()


def iter_checkpoint_state(
    snapshot_dir: Any,
    config: Any = None,
    include_mtp: bool = False,
    dequantize: bool = True,
    *,
    manifest: Manifest | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(target_name, tensor)`` for the whole model, one at a time.

    Deterministic model order (see :func:`_model_order_targets`):
    ``embed_tokens.weight``, per-layer tensors, final ``norm.weight``,
    ``lm_head.weight``, then MTP targets only when ``include_mtp=True``
    (MTP is excluded by default: the LM has no MTP parameters, so its
    tensors are neither yielded nor, in validation, unexpected).
    ``weight_scale_inv`` companions are consumed by FP8 dequantisation
    and never yielded.  Vision tensors are never yielded.  When
    ``config.tie_word_embeddings`` is set, ``lm_head.weight`` is not
    yielded separately (it shares ``embed_tokens.weight``).

    Generator: only headers are read until iteration, and each
    tensor is materialised (and released by the caller) one at a time
    via :func:`load_tensor`, so a 27B checkpoint is never held whole.
    ``snapshot_dir`` may be a path or a :class:`Manifest`; ``config``
    may be a :class:`HybridConfig`, a text-config mapping, or ``None``
    (snapshot ``config.json`` / defaults).
    """
    root = _resolve_dir(snapshot_dir)
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else (manifest or build_manifest(root))
    cfg = _resolve_config(snapshot_dir, config=config)
    by_target: dict[str, ManifestEntry] = {}
    for e in man.entries:
        if e.kind == "scale_inv" or e.target_name is None:
            continue
        if e.target_name.startswith("mtp.") and not include_mtp:
            continue
        by_target.setdefault(e.target_name, e)
    for target in _model_order_targets(cfg, include_mtp=include_mtp):
        if cfg.tie_word_embeddings and target == "lm_head.weight":
            continue
        entry = by_target.get(target)
        if entry is None:
            continue
        tensor = load_tensor(root, entry.checkpoint_name, dequantize=dequantize, manifest=man)
        yield target, tensor


def validate_state_against_model(
    model: Any,
    snapshot_dir: Any,
    include_mtp: bool = False,
    *,
    config: Any = None,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    """Compare a checkpoint manifest against a model's state_dict.

    ``model`` is typically a ``meta``-device ``HybridLM`` /
    ``TinyHybridLM`` instance: only its ``state_dict()`` shapes are
    read, never tensor data, so validating the 27B checkpoint costs
    no allocation.  Returns a dict with exact lists:

    * ``missing`` -- model target names (state_dict order) with no
      checkpoint tensor;
    * ``unexpected`` -- sorted checkpoint target names (vision and,
      unless ``include_mtp=True``, MTP excluded; ``scale_inv``
      excluded) with no model parameter;
    * ``shape_mismatch`` -- dicts ``{target_name, checkpoint_name,
      expected_shape, actual_shape, shape, expected}`` where
      ``expected_shape`` is the model shape;
    * ``dtype_mismatch`` -- dicts for present targets whose stored
      dtype cannot serve a BF16 target.  BF16-target policy: every
      model target is loaded as BF16 (FP8 weights dequantise to
      BF16, BF16 tensors load as stored) and ``copy_`` casts into
      the parameter dtype, so a float32 meta model is not itself a
      mismatch, and float storage (BF16/F32/F16, or FP8 for weights)
      is acceptable; non-float storage (e.g. integer) is not.

    Plus ``ok``, ``counts``, ``parameter_count`` (from shapes via
    ``model.parameters()``, tied weights counted once),
    ``include_mtp`` and ``tied_embeddings``.  A tied ``lm_head`` is
    not required to have its own checkpoint tensor.
    """
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else manifest
    if man is None:
        man = build_manifest(_resolve_dir(snapshot_dir))
    sd = model.state_dict()
    model_order = list(sd.keys())
    tied = False
    try:
        tied = bool(model.lm_head.weight is model.embed_tokens.weight)  # type: ignore[attr-defined]
    except Exception:
        tied = False
    required = [k for k in model_order if not (tied and k == "lm_head.weight")]
    required_set = set(required)
    # Resident-FP8 models carry one ``*.weight_scale_inv`` state entry
    # per BlockFP8Linear; those names line up with the checkpoint's
    # scale_inv manifest entries (same suffix via _map_hybrid_key),
    # while a dequantising (dense) model never requires them.
    scale_required = {k for k in required if k.endswith(".weight_scale_inv")}
    by_target: dict[str, ManifestEntry] = {}
    by_scale: dict[str, ManifestEntry] = {}
    for e in man.entries:
        if e.target_name is None:
            continue
        if e.target_name.startswith("mtp.") and not include_mtp:
            continue
        if e.kind == "scale_inv":
            by_scale.setdefault(e.target_name, e)
            continue
        by_target.setdefault(e.target_name, e)

    def _entry_for(target: str) -> ManifestEntry | None:
        if target in scale_required:
            return by_scale.get(target)
        return by_target.get(target)

    missing = [t for t in required if _entry_for(t) is None]
    unexpected = sorted(t for t in by_target if t not in required_set)
    shape_mismatch: list[dict[str, Any]] = []
    dtype_mismatch: list[dict[str, Any]] = []
    for target in required:
        entry = _entry_for(target)
        if entry is None:
            continue
        expected_shape = [int(d) for d in sd[target].shape]
        if list(entry.shape) != expected_shape:
            shape_mismatch.append({
                "target_name": target,
                "checkpoint_name": entry.checkpoint_name,
                "expected_shape": expected_shape,
                "actual_shape": list(entry.shape),
                "shape": list(entry.shape),
                "expected": list(expected_shape),
            })
        if entry.dtype not in ("BF16", "F32", "F16", "F8_E4M3", "F8_E5M2"):
            dtype_mismatch.append({
                "target_name": target,
                "checkpoint_name": entry.checkpoint_name,
                "expected_dtype": "BF16",
                "actual_dtype": entry.dtype,
                "kind": entry.kind,
            })
    try:
        # State-entry count: identical to parameters() for a dense
        # model, and the honest element count for a resident-FP8
        # model whose FP8 weights live in buffers, not Parameters
        # (scale entries included; tied lm_head counted once).
        parameter_count = int(sum(int(sd[k].numel()) for k in required))
    except Exception:
        parameter_count = int(sum(int(p.numel()) for p in model.parameters()))
    counts = {
        "expected": len(required),
        "present": len(required) - len(missing),
        "manifest_entries": len(man.entries),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
        "dtype_mismatch": len(dtype_mismatch),
    }
    return {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "ok": not missing and not unexpected and not shape_mismatch and not dtype_mismatch,
        "counts": counts,
        "parameter_count": parameter_count,
        "total_params": parameter_count,
        "include_mtp": bool(include_mtp),
        "tied_embeddings": tied,
    }


def load_into_model(
    model: Any,
    snapshot_dir: Any,
    strict: bool = True,
    include_mtp: bool = False,
    dequantize: bool = True,
    *,
    fp8_mode: str = "dequantize",
    config: Any = None,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    """Stream a checkpoint into ``model`` via per-tensor ``copy_``.

    Pre-validates with :func:`validate_state_against_model` before
    touching any parameter: with ``strict=True`` (default) a missing,
    unexpected, shape- or dtype-mismatched checkpoint raises
    ``RuntimeError`` *before* any tensor is copied, so a failed load
    leaves the model unmutated.  With ``strict=False``, mismatched /
    unexpected tensors are skipped and the rest are loaded.

    ``fp8_mode`` selects how FP8 checkpoint weights land:

    * ``"dequantize"`` (default, the historic behaviour): each FP8
      weight is dequantised to BF16 via :func:`iter_checkpoint_state`
      and copied into a dense parameter (``dequantize=False`` keeps
      the raw stored tensors on this path);
    * ``"resident"``: for every
      :class:`~inference_from_scratch.fp8_linear.BlockFP8Linear` in
      the model, the raw FP8 bits and their ``weight_scale_inv`` are
      copied into the module's buffers on the module's device, so
      the weights stay FP8 in memory. All other targets load on the
      dequantising path. A resident target whose checkpoint tensor
      is stored dense (BF16) is quantised on the fly instead, so a
      converted BF16 projection still loads.

    Returns the validation report plus ``loaded`` (target names, in
    load order), ``loaded_count``, ``fp8_mode`` and
    ``resident_targets`` (qualified FP8 module names, resident mode
    only). Tests exercise both modes on tiny synthetic fixtures.
    """
    if fp8_mode not in ("dequantize", "resident"):
        raise ValueError(
            f"fp8_mode must be 'dequantize' or 'resident', got {fp8_mode!r}"
        )
    resident = fp8_mode == "resident"
    cfg = _resolve_config(snapshot_dir, config=config, model=model)
    man = snapshot_dir if isinstance(snapshot_dir, Manifest) else manifest
    report = validate_state_against_model(
        model, snapshot_dir, include_mtp=include_mtp, config=cfg, manifest=man)
    if strict and not report["ok"]:
        raise RuntimeError(
            "load_into_model pre-validation failed: "
            f"missing={report['missing']} unexpected={report['unexpected']} "
            f"shape_mismatch={report['shape_mismatch']} "
            f"dtype_mismatch={report['dtype_mismatch']}")
    bad_shapes = {m["target_name"] for m in report["shape_mismatch"]}
    params: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        params.setdefault(name, p)
    for name, b in model.named_buffers():
        params.setdefault(name, b)
    # Resident targets: qualified BlockFP8Linear names -> weight target.
    resident_modules: dict[str, BlockFP8Linear] = {}
    if resident:
        for qual, module in model.named_modules():
            if isinstance(module, BlockFP8Linear):
                resident_modules[qual] = module
    resident_weight_targets = {f"{q}.weight" for q in resident_modules}
    root = _resolve_dir(snapshot_dir)
    loaded: list[str] = []
    with torch.no_grad():
        if resident:
            for qual in sorted(resident_modules):
                module = resident_modules[qual]
                weight_target = f"{qual}.weight"
                scale_target = f"{qual}.weight_scale_inv"
                if weight_target in bad_shapes or scale_target in bad_shapes:
                    continue
                ckpt_weight = _target_to_checkpoint(weight_target)
                try:
                    raw = _load_one(root, ckpt_weight, manifest=man)
                except KeyError:
                    if strict:
                        raise
                    continue
                if raw.dtype == torch.float8_e4m3fn:
                    try:
                        scale = _load_one(
                            root, ckpt_weight + "_scale_inv", manifest=man)
                    except KeyError:
                        if strict:
                            raise
                        continue
                    module.weight.copy_(
                        raw.to(device=module.weight.device))
                    module.weight_scale_inv.copy_(
                        scale.to(dtype=torch.bfloat16,
                                 device=module.weight.device))
                else:
                    # Dense-stored projection converted to resident FP8:
                    # quantise on the fly so the buffers still land.
                    q, scale_f32 = quantize_block_fp8(
                        raw.to(torch.float32), block=module.block)
                    module.weight.copy_(
                        q.to(device=module.weight.device))
                    module.weight_scale_inv.copy_(
                        scale_f32.to(dtype=torch.bfloat16,
                                     device=module.weight.device))
                loaded.append(weight_target)
                loaded.append(scale_target)
        for target, tensor in iter_checkpoint_state(
            snapshot_dir, config=cfg, include_mtp=include_mtp,
            dequantize=dequantize, manifest=man,
        ):
            if resident and target in resident_weight_targets:
                continue  # already loaded raw above
            dest = params.get(target)
            if dest is None or target in bad_shapes:
                continue
            if tuple(dest.shape) != tuple(tensor.shape):
                if strict:
                    raise RuntimeError(
                        f"shape mismatch for {target!r}: model {tuple(dest.shape)} "
                        f"vs checkpoint {tuple(tensor.shape)}")
                continue
            dest.copy_(tensor)
            loaded.append(target)
    out = dict(report)
    out["loaded"] = loaded
    out["loaded_count"] = len(loaded)
    out["fp8_mode"] = fp8_mode
    out["resident_targets"] = sorted(resident_modules)
    return out
