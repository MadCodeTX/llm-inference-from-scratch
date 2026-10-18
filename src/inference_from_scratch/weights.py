"""Ch 09 -- safetensors by hand, HF config, and Qwen weight mapping.

Why this exists in inference
------------------------------
A checkpoint on disk is not a model. Turning ``config.json`` +
``*.safetensors`` into a runnable :class:`DecoderLM` takes three
translations, and this module does each one explicitly:

1. *Bytes -> header.* The safetensors container is dissected by hand
   (no ``safetensors`` library): an 8-byte little-endian length, a
   JSON header, then a raw data buffer. The header *is* the chapter:
   for every tensor it records ``dtype``, ``shape`` and a
   ``[begin, end)`` byte range into the data buffer.
2. *HF config -> ModelConfig.* Llama / Qwen ``config.json`` field
   names (``hidden_size`` ...) are mapped onto this repo's
   :class:`ModelConfig` (``d_model`` ...) by :func:`load_hf_config`.
3. *HF names -> DecoderLM names.* HF calls the embedding
   ``model.embed_tokens.weight``; :mod:`model` calls it
   ``tok_embedding.weight``. :func:`map_hf_state_dict` renames, and
   :func:`validate_state_dict` proves the renamed dict fits a real
   model (missing / unexpected / shape-mismatched keys) *before* any
   ``load_state_dict`` is attempted.

The safetensors format, dissected
-----------------------------------
::

    | 8 bytes        | N bytes        | rest of file          |
    | uint64 LE = N  | UTF-8 JSON     | raw tensor bytes      |
    | header length  | header         | ("the data buffer")   |

The JSON header maps each tensor name to::

    {"dtype": "F32", "shape": [V, D], "data_offsets": [begin, end]}

plus one optional non-tensor entry, ``"__metadata__"`` (string ->
string, e.g. the producing framework). ``data_offsets`` are relative
to the *start of the data buffer* (file offset ``8 + N``), ``end``
exclusive. Tensors are stored contiguously, row-major, little-endian,
with no per-tensor padding requirement the reader may assume: a
correct reader honours the offsets exactly (real files may leave
gaps, and sharded checkpoints split tensors *across* files, never
within one entry). That table -- name, dtype, shape, offsets -- is
all a loader needs, and all :func:`read_safetensors_header` returns.

Lazy vs eager loading
----------------------
*Eager* (:func:`load_safetensors`) materialises every tensor now:
simple, and peak RAM is the whole checkpoint. *Lazy*
(:class:`LazySafetensorsFile`) parses only the header up front and
reads one tensor's byte range per ``load(name)`` call, so untouched
tensors never enter RAM. Production loaders go one step further and
``mmap`` the data buffer, making even the per-tensor read zero-copy
until first touch; that is what the ``safetensors`` library does, and
why loading a 27B checkpoint does not need 27B of RAM to *start*.
The dissected lazy reader here still copies the requested slice --
the deferral, not the zero-copy, is the point being shown.

HF -> DecoderLM name mapping (documented, tested)
--------------------------------------------------
Read off ``DecoderLM.state_dict()`` in ``model.py`` -- never guessed.
HF Linear weights and ours are both ``[out, in]`` and embeddings both
``[V, D]``, so renaming is pure: no transposes, no copies (the mapped
dict holds the *same* tensor objects).

=====================================  =====================================
HF (Llama / Qwen2 / Qwen3)             this repo (``DecoderLM``)
=====================================  =====================================
``model.embed_tokens.weight``          ``tok_embedding.weight``
``model.layers.{i}.input_layernorm``   ``layers.{i}.attn_norm``
``...self_attn.q_proj``                ``layers.{i}.attn.q_proj``
``...self_attn.k_proj``                ``layers.{i}.attn.k_proj``
``...self_attn.v_proj``                ``layers.{i}.attn.v_proj``
``...self_attn.o_proj``                ``layers.{i}.attn.o_proj``
``...post_attention_layernorm``        ``layers.{i}.mlp_norm``
``...mlp.gate_proj``                   ``layers.{i}.mlp.gate_proj``
``...mlp.up_proj``                     ``layers.{i}.mlp.up_proj``
``...mlp.down_proj``                   ``layers.{i}.mlp.down_proj``
``model.norm.weight``                  ``final_norm.weight``
``lm_head.weight``                     ``lm_head.weight``
=====================================  =====================================

(``...`` = ``model.layers.{i}.``; every row is a ``.weight``, and a
``.bias`` renames the same way.) One honest caveat, now resolved:

* *Biases.* Llama ships none; Qwen2 ships Q/K/V projection biases.
  When ``ModelConfig.attention_bias`` is True (see
  :func:`load_hf_config`) those biases rename to
  ``layers.{i}.attn.{q,k,v}_proj.bias`` and load strictly; with the
  default False they have no home and :func:`validate_state_dict`
  reports them as *unexpected* instead of silently dropping them.
  ``o_proj`` is bias-free in both families.
* *Tied embeddings.* HF checkpoints with ``tie_word_embeddings``
  usually omit ``lm_head.weight`` entirely (the head *is* the
  embedding). ``map_hf_state_dict(sd, tie_embeddings=True)`` then
  aliases ``lm_head.weight`` to the embedding tensor, which is what
  a tied ``DecoderLM`` needs for a strict ``load_state_dict``.

Reading a large model's config
--------------------------------
Never fabricate a 27B-class config from memory. Point
:func:`load_hf_config` at the real ``config.json`` that ships with
the checkpoint (a path, its directory, or an already-parsed dict)
and every number -- layers, heads, KV heads, ``d_ff`` -- comes from
disk at load time.

How production differs
-----------------------
Production loaders add mmap zero-copy reads, sharding indexes
(``model.safetensors.index.json``), dtype casting on load, and
device placement. None of that changes the three translations above;
it only changes *when* bytes move.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import ModelConfig

# ---------------------------------------------------------------------------
# safetensors header dissection
# ---------------------------------------------------------------------------

_HEADER_LEN = struct.Struct("<Q")  # 8-byte little-endian uint64

# safetensors dtype string -> torch dtype. The byte width comes from
# the dtype itself, so offset arithmetic can be checked exactly.
_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# numpy little-endian views for every dtype numpy can read directly
# (BF16 has no numpy dtype -- it is handled via its uint16 bit pattern).
_DTYPE_TO_NUMPY: dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "u1",
}


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _validate_header(header: Any, source: str) -> dict:
    """Check the parsed JSON has the header shape the format promises."""
    if not isinstance(header, dict):
        raise ValueError(f"{source}: safetensors header is not a JSON object")
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict):
                raise ValueError(f"{source}: __metadata__ is not an object")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: header entry {name!r} is not an object")
        dtype = entry.get("dtype")
        if dtype not in _DTYPE_TO_TORCH:
            raise ValueError(f"{source}: tensor {name!r} has unknown dtype {dtype!r}")
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape)
        ):
            raise ValueError(f"{source}: tensor {name!r} has bad shape {shape!r}")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(o, int) or isinstance(o, bool) for o in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ValueError(
                f"{source}: tensor {name!r} has bad data_offsets {offsets!r}"
            )
        itemsize = torch.tensor([], dtype=_DTYPE_TO_TORCH[dtype]).element_size()
        expected = _numel(shape) * itemsize
        if offsets[1] - offsets[0] != expected:
            raise ValueError(
                f"{source}: tensor {name!r} spans {offsets[1] - offsets[0]} bytes "
                f"but dtype {dtype} x shape {shape} needs {expected}"
            )
    return header


def read_safetensors_header(path: str | Path) -> dict:
    """Parse a safetensors header *by hand* (no safetensors library).

    Reads exactly ``8 + N`` bytes: the 8-byte little-endian header
    length ``N``, then ``N`` bytes of UTF-8 JSON, and returns the
    parsed dict verbatim -- tensor name -> ``{"dtype", "shape",
    "data_offsets"}``, plus ``"__metadata__"`` when present. The
    tensor bytes are not touched, so this stays O(header) even for a
    huge checkpoint (which is exactly how a loader can print the
    dtype/shape/offset table before deciding what to load).

    Raises ``ValueError`` on a truncated file, invalid JSON, or a
    header whose entries do not match the format (unknown dtype,
    malformed shape/offsets, or an offset span that disagrees with
    ``prod(shape) * itemsize``).
    """
    p = Path(path)
    with p.open("rb") as f:
        raw_len = f.read(_HEADER_LEN.size)
        if len(raw_len) < _HEADER_LEN.size:
            raise ValueError(
                f"{p}: file too short for the 8-byte safetensors length prefix"
            )
        (header_len,) = _HEADER_LEN.unpack(raw_len)
        raw_header = f.read(header_len)
    if len(raw_header) < header_len:
        raise ValueError(
            f"{p}: truncated safetensors header "
            f"(wanted {header_len} bytes, got {len(raw_header)})"
        )
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{p}: safetensors header is not valid JSON: {exc}") from exc
    return _validate_header(header, str(p))


def safetensors_table(header: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a parsed header into a dtype/shape/offset table.

    One row per tensor (``__metadata__`` excluded), in header order:
    ``{name, dtype, shape, numel, begin, end, nbytes}`` where
    ``[begin, end)`` is the range inside the data buffer and
    ``nbytes == end - begin``. This is the whole on-disk layout a
    loader (or a human debugging a checkpoint) ever needs.
    """
    rows: list[dict[str, Any]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        rows.append(
            {
                "name": name,
                "dtype": entry["dtype"],
                "shape": list(entry["shape"]),
                "numel": _numel(entry["shape"]),
                "begin": begin,
                "end": end,
                "nbytes": end - begin,
            }
        )
    return rows


def format_safetensors_table(header: Mapping[str, Any]) -> str:
    """Render :func:`safetensors_table` as aligned plain text."""
    rows = safetensors_table(header)
    lines = [f"{'name':48} {'dtype':5} {'shape':20} {'bytes':>10}  [begin, end)"]
    for r in rows:
        lines.append(
            f"{r['name']:48} {r['dtype']:5} {str(r['shape']):20} "
            f"{r['nbytes']:>10}  [{r['begin']}, {r['end']})"
        )
    return "\n".join(lines)


def _tensor_from_bytes(raw: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    """Materialise one tensor from its raw little-endian byte range."""
    torch_dtype = _DTYPE_TO_TORCH[dtype]
    if dtype == "BF16":
        # No numpy bfloat16: read the bits as uint16, then reinterpret.
        bits = np.frombuffer(raw, dtype="<u2").copy()
        return torch.from_numpy(bits).view(torch.bfloat16).reshape(shape).clone()
    arr = np.frombuffer(raw, dtype=np.dtype(_DTYPE_TO_NUMPY[dtype])).copy()
    if dtype == "BOOL":
        arr = arr != 0
    tensor = torch.from_numpy(arr).to(torch_dtype)
    return tensor.reshape(shape).clone()


class LazySafetensorsFile:
    """Lazy safetensors reader: header now, tensor bytes on demand.

    Opening parses (and validates) only the header and records where
    the data buffer starts; each :meth:`load` seeks to that tensor's
    ``[begin, end)`` range and materialises just it. Untouched tensors
    cost no RAM and no I/O -- the property that lets production
    loaders (which mmap instead of seek+read, so the copy below is
    also avoided until first touch) open huge checkpoints instantly.

    Use as a context manager, or call :meth:`close`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        try:
            self.header = read_safetensors_header(self.path)
        except Exception:
            self._file.close()
            raise
        # The data buffer starts right after the length prefix + JSON:
        # re-read the 8-byte prefix on this handle to locate it.
        self._file.seek(0)
        (header_len,) = _HEADER_LEN.unpack(self._file.read(_HEADER_LEN.size))
        self._data_start = _HEADER_LEN.size + header_len

    # -- introspection (header only, no tensor I/O) -------------------------
    def keys(self) -> list[str]:
        """Tensor names in header order (``__metadata__`` excluded)."""
        return [n for n in self.header if n != "__metadata__"]

    def __contains__(self, name: object) -> bool:
        return name in self.header and name != "__metadata__"

    def __len__(self) -> int:
        return len(self.keys())

    @property
    def metadata(self) -> dict[str, str]:
        """The ``__metadata__`` string map (empty dict when absent)."""
        return dict(self.header.get("__metadata__", {}))

    def table(self) -> list[dict[str, Any]]:
        """This file's dtype/shape/offset table (see safetensors_table)."""
        return safetensors_table(self.header)

    # -- loading ------------------------------------------------------------
    def load(self, name: str) -> torch.Tensor:
        """Materialise one tensor by name, reading only its byte range."""
        if name not in self:
            raise KeyError(f"{self.path}: no tensor named {name!r}")
        entry = self.header[name]
        begin, end = entry["data_offsets"]
        self._file.seek(self._data_start + begin)
        raw = self._file.read(end - begin)
        if len(raw) < end - begin:
            raise ValueError(
                f"{self.path}: truncated data buffer for tensor {name!r} "
                f"(wanted {end - begin} bytes, got {len(raw)})"
            )
        return _tensor_from_bytes(raw, entry["dtype"], list(entry["shape"]))

    def load_all(self) -> dict[str, torch.Tensor]:
        """Materialise every tensor (the eager path, via the lazy reader)."""
        return {name: self.load(name) for name in self.keys()}

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "LazySafetensorsFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Eagerly load every tensor from one safetensors file, by hand.

    Returns ``{tensor_name: Tensor}`` in header order with the dtypes
    and shapes the header declares (BF16 stays BF16, and so on).
    Peak memory is the file's tensor bytes; for on-demand access to
    a large checkpoint use :class:`LazySafetensorsFile` instead.
    """
    with LazySafetensorsFile(path) as reader:
        return reader.load_all()


# ---------------------------------------------------------------------------
# HF config -> ModelConfig
# ---------------------------------------------------------------------------


def _hf_get(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """First non-None value among ``names`` (HF name first, ours as fallback)."""
    for name in names:
        value = raw.get(name)
        if value is not None:
            return value
    return default


def _infer_attention_bias(raw: Mapping[str, Any]) -> bool:
    """Whether Q/K/V projections carry a bias, from config evidence.

    An explicit ``attention_bias`` key in ``config.json`` wins. When
    the key is absent (current Qwen2 checkpoints do not serialise it)
    the architecture is the evidence: the Qwen2 family ships Q/K/V
    biases, Llama does not. The safetensors header remains ground
    truth -- :func:`validate_state_dict` flags any disagreement.
    """
    value = raw.get("attention_bias")
    if value is not None:
        return bool(value)
    model_type = str(raw.get("model_type", "")).lower()
    if model_type in {"qwen2", "qwen2_moe"}:
        return True
    for arch in raw.get("architectures") or []:
        if isinstance(arch, str) and arch.startswith("Qwen2"):
            return True
    return False


def load_hf_config(path_or_dict: "str | Path | Mapping[str, Any] | ModelConfig") -> ModelConfig:
    """Translate a Llama/Qwen-style ``config.json`` into ``ModelConfig``.

    Accepts an already-parsed dict, a path to ``config.json``, or a
    directory containing one. Field mapping:

    ==========================  ==========================
    HF ``config.json``          ``ModelConfig``
    ==========================  ==========================
    ``vocab_size``              ``vocab_size``
    ``hidden_size``             ``d_model``
    ``num_hidden_layers``       ``n_layer``
    ``num_attention_heads``     ``n_head``
    ``num_key_value_heads``     ``n_kv_head``
    ``intermediate_size``       ``d_ff``
    ``max_position_embeddings`` ``max_seq_len``
    ``rope_theta``              ``rope_theta``
    ``rms_norm_eps``            ``rms_eps``
    ``tie_word_embeddings``     ``tie_embeddings``
    ``attention_bias``          ``attention_bias``
    ==========================  ==========================

    Defaults follow HF when a field is absent: ``num_key_value_heads``
    defaults to ``num_attention_heads`` (plain MHA -- the GQA default
    this repo's tests pin), ``rope_theta`` to 10000, ``rms_norm_eps``
    to 1e-6, ``tie_word_embeddings`` to False. ``attention_bias``
    defaults via :func:`_infer_attention_bias` (explicit key, else
    Qwen2-family detection). Architecture flavour is not an HF field:
    Llama/Qwen are RMSNorm + SwiGLU + RoPE, so
    ``norm_type``/``mlp_type``/``pos_type`` take those values. A
    ``rope_theta`` nested in ``rope_scaling`` is honoured when the
    top-level key is absent. ``max_position_embeddings`` absent falls
    back to 32768 (the Qwen2 default); real checkpoints ship it.

    The five shape-defining fields (``vocab_size``, ``hidden_size``,
    ``num_hidden_layers``, ``num_attention_heads``,
    ``intermediate_size``) are required: ``ValueError`` names every
    missing one rather than inventing a number. Nothing here is
    hard-coded per model -- point this at a checkpoint's real
    ``config.json`` and the numbers come from that file.
    """
    if isinstance(path_or_dict, ModelConfig):
        return path_or_dict
    if isinstance(path_or_dict, Mapping):
        raw: dict[str, Any] = dict(path_or_dict)
    else:
        p = Path(path_or_dict)
        config_path = p / "config.json" if p.is_dir() else p
        if not config_path.is_file():
            raise FileNotFoundError(
                f"no HF config at {config_path} "
                f"(pass a config.json path, its directory, or a dict)"
            )
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{config_path}: config.json is not a JSON object")
        raw = parsed

    required = {
        "vocab_size": _hf_get(raw, "vocab_size"),
        "hidden_size": _hf_get(raw, "hidden_size", "d_model"),
        "num_hidden_layers": _hf_get(raw, "num_hidden_layers", "n_layer"),
        "num_attention_heads": _hf_get(raw, "num_attention_heads", "n_head"),
        "intermediate_size": _hf_get(raw, "intermediate_size", "d_ff"),
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"HF config is missing required field(s): {', '.join(missing)}"
        )

    n_head = int(required["num_attention_heads"])
    rope_theta = _hf_get(raw, "rope_theta")
    if rope_theta is None and isinstance(raw.get("rope_scaling"), dict):
        rope_theta = raw["rope_scaling"].get("rope_theta")
    return ModelConfig(
        vocab_size=int(required["vocab_size"]),
        d_model=int(required["hidden_size"]),
        n_layer=int(required["num_hidden_layers"]),
        n_head=n_head,
        # GQA defaults to MHA when HF omits the KV-head count.
        n_kv_head=int(_hf_get(raw, "num_key_value_heads", "n_kv_head", default=n_head)),
        d_ff=int(required["intermediate_size"]),
        max_seq_len=int(
            _hf_get(raw, "max_position_embeddings", "max_seq_len", default=32768)
        ),
        rope_theta=float(rope_theta) if rope_theta is not None else 10000.0,
        rms_eps=float(_hf_get(raw, "rms_norm_eps", "rms_eps", default=1e-6)),
        tie_embeddings=bool(
            _hf_get(raw, "tie_word_embeddings", "tie_embeddings", default=False)
        ),
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_type="rope",
        attention_bias=_infer_attention_bias(raw),
    )


# ---------------------------------------------------------------------------
# HF state dict -> DecoderLM state dict
# ---------------------------------------------------------------------------

# Global (non-layer) renames, exact-match.
_GLOBAL_RENAMES: dict[str, str] = {
    "model.embed_tokens.weight": "tok_embedding.weight",
    "embed_tokens.weight": "tok_embedding.weight",
    "model.norm.weight": "final_norm.weight",
    "model.norm.bias": "final_norm.bias",
    "lm_head.weight": "lm_head.weight",  # same name, listed for completeness
    "lm_head.bias": "lm_head.bias",
}

# Per-layer renames. The ``model.`` prefix is optional so checkpoints
# saved from the bare model (no CausalLM wrapper) rename identically.
_RE_INPUT_NORM = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.input_layernorm\.(weight|bias)$"
)
_RE_POST_NORM = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$"
)
_RE_ATTN = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$"
)
_RE_MLP = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$"
)
_RE_FINAL_NORM = re.compile(r"^(?:model\.)?norm\.(weight|bias)$")


def _rename_hf_key(key: str) -> str:
    """Rename one HF key to its DecoderLM name (unmatched: unchanged)."""
    if key in _GLOBAL_RENAMES:
        return _GLOBAL_RENAMES[key]
    m = _RE_INPUT_NORM.match(key)
    if m:
        return f"layers.{m.group(1)}.attn_norm.{m.group(2)}"
    m = _RE_POST_NORM.match(key)
    if m:
        return f"layers.{m.group(1)}.mlp_norm.{m.group(2)}"
    m = _RE_ATTN.match(key)
    if m:
        return f"layers.{m.group(1)}.attn.{m.group(2)}.{m.group(3)}"
    m = _RE_MLP.match(key)
    if m:
        return f"layers.{m.group(1)}.mlp.{m.group(2)}.{m.group(3)}"
    m = _RE_FINAL_NORM.match(key)
    if m:
        return f"final_norm.{m.group(1)}"
    return key


def map_hf_state_dict(
    sd: Mapping[str, Any], *, tie_embeddings: bool = False
) -> dict[str, Any]:
    """Rename an HF (Llama/Qwen) state dict to ``DecoderLM`` names.

    Pure renaming: values are the *same objects* (no copy, no
    transpose -- HF and this repo both store Linear weights as
    ``[out, in]``), and input order is preserved. The mapping table
    is in this module's docstring; targets are exactly the keys of
    ``DecoderLM.state_dict()`` (``tok_embedding.weight``,
    ``layers.{i}.attn_norm/attn/mlp_norm/mlp.*``, ``final_norm.*``,
    ``lm_head.weight``), enumerated from ``model.py`` and pinned by
    ``tests/test_weights.py`` against a live model.

    Keys with no rename (e.g. a stray ``rotary_emb.inv_freq`` buffer,
    or an already-renamed key) pass through unchanged so that
    :func:`validate_state_dict` can report them as unexpected instead
    of the mapping silently hiding them.

    With ``tie_embeddings=True``, a checkpoint that omits
    ``lm_head.weight`` (the usual tied HF layout) gains
    ``lm_head.weight`` aliased to the embedding tensor -- the same
    object, mirroring the shared Parameter in a tied ``DecoderLM`` --
    and symmetrically for a missing embedding. A strict
    ``load_state_dict`` into a tied model needs that alias; without
    the flag the omission is left visible for validation to judge
    (validation itself is tie-aware, see :func:`validate_state_dict`).
    """
    mapped: dict[str, Any] = {}
    for key, value in sd.items():
        mapped[_rename_hf_key(key)] = value
    if tie_embeddings:
        if "tok_embedding.weight" in mapped and "lm_head.weight" not in mapped:
            mapped["lm_head.weight"] = mapped["tok_embedding.weight"]
        elif "lm_head.weight" in mapped and "tok_embedding.weight" not in mapped:
            mapped["tok_embedding.weight"] = mapped["lm_head.weight"]
    return mapped


def validate_state_dict(
    mapped: Mapping[str, Any], model: Any
) -> dict[str, Any]:
    """Check a (mapped) state dict against a model's real state dict.

    ``model`` is a ``DecoderLM`` (anything with ``state_dict()``; a
    plain ``{name: tensor}`` mapping is also accepted as the expected
    side). Returns::

        {"missing":        [names the model needs but `mapped` lacks],
         "unexpected":     [names in `mapped` the model does not have],
         "shape_mismatch": [(name, expected_shape, found_shape), ...],
         "ok":             True iff all three lists are empty}

    Lists are sorted by name for deterministic reports. Shapes are
    plain tuples. Tied embeddings are honoured: on a tied model, a
    present ``tok_embedding.weight`` satisfies a missing
    ``lm_head.weight`` (and vice versa), because the two names share
    one Parameter -- load with ``map_hf_state_dict(...,
    tie_embeddings=True)`` to materialise the alias before a strict
    ``load_state_dict``. Nothing is loaded or mutated here; this is
    the pre-flight check the loader script prints before copying a
    single weight.
    """
    if isinstance(model, Mapping):
        expected = {k: tuple(v.shape) for k, v in model.items() if hasattr(v, "shape")}
    else:
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    shape_mismatch: list[tuple[str, tuple, tuple | None]] = []
    for key in sorted(set(expected) & set(mapped)):
        value = mapped[key]
        found = tuple(value.shape) if hasattr(value, "shape") else None
        if found != expected[key]:
            shape_mismatch.append((key, expected[key], found))

    # Tied head/embedding share one Parameter: either name present
    # (with the shape check above already applied to it) covers both.
    config = getattr(model, "config", None)
    if bool(getattr(config, "tie_embeddings", False)):
        if "tok_embedding.weight" in mapped and "lm_head.weight" in missing:
            missing.remove("lm_head.weight")
        if "lm_head.weight" in mapped and "tok_embedding.weight" in missing:
            missing.remove("tok_embedding.weight")

    return {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "ok": not missing and not unexpected and not shape_mismatch,
    }



