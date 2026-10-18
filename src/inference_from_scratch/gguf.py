"""GGUF container dissection by hand (book sec. 4.2.2, 3.5).

Why this exists in inference
------------------------------
Safetensors (ch 09) stores tensors plus minimal string metadata and
leaves architecture, tokenizer, and quantization choices to sidecar
files (``config.json``, ``tokenizer.json``, ...). GGUF -- the format
used by llama.cpp and Glimmer-style on-device runtimes -- is a single
self-contained file: architecture hyperparameters, tokenizer, chat
template, quantization choices and the (often block-quantized) tensor
data all live in one container that a runtime can ``mmap`` and start
serving. This module dissects that container's header by hand with
``struct`` -- no ``gguf`` library -- the same way ch 09 dissected
safetensors.

The GGUF layout (version 3)
-----------------------------
All integers are little-endian. A ``string`` is a ``uint64`` length
followed by that many UTF-8 bytes (no NUL terminator)::

    | 4 B     | 4 B     | 8 B          | 8 B               |
    | "GGUF"  | version | tensor_count | metadata_kv_count |
    | magic   | (=3)    | uint64       | uint64            |

    then metadata_kv_count metadata entries:
        key:   string
        value_type: uint32
        value: ... depends on value_type (see below)

    then tensor_count tensor infos:
        name:       string
        n_dims:     uint32
        dims:       uint64 x n_dims   (ggml order, see below)
        type:       uint32            (ggml type code)
        offset:     uint64            (relative to data start)

    then zero padding up to ``alignment``, then raw tensor data.

Metadata value types (``gguf_metadata_value_type``):

    ===  ========  ==================================
    0    UINT8     ``<B``
    1    INT8      ``<b``
    2    UINT16    ``<H``
    3    INT16     ``<h``
    4    UINT32    ``<I``
    5    INT32     ``<i``
    6    FLOAT32   ``<f``
    7    BOOL      one byte, 0 or 1
    8    STRING    string as above
    9    ARRAY     uint32 elem_type + uint64 length + elems
    10   UINT64    ``<Q``
    11   INT64     ``<q``
    12   FLOAT64   ``<d``
    ===  ========  ==================================

Arrays are homogeneous; string elements are each a full GGUF string.
This reader supports scalars, strings and arrays generically (array
elements may be any scalar or string type).

Alignment and the absolute data start
--------------------------------------
Tensor data does not start immediately after the tensor infos. The
file is padded with zeros until the file offset is a multiple of
``general.alignment`` (a UINT32 metadata entry, default 32 when the
key is absent). If ``header_end`` is the file offset just after the
last tensor info::

    data_start = ceil(header_end / alignment) * alignment
    absolute_offset(tensor) = data_start + tensor.offset

``tensor.offset`` values inside the infos are relative to
``data_start``. A reader that forgets the alignment lands tens of
bytes early and reads padding as weights -- the exact analogue of
honouring safetensors ``data_offsets`` in ch 09.

ggml dimension order
---------------------
``dims`` are returned exactly as stored. ggml stores dimensions with
the fastest-varying (contiguous) dimension first, which is the
reverse of the PyTorch / safetensors convention for the same logical
tensor. Converting a GGUF tensor to a torch shape therefore reverses
``dims``; this header parser deliberately does not guess -- it reports
the on-disk order and the docs (ch 20) name the reversal.

What this module does NOT do
-----------------------------
It parses the header only. It does not read, dequantize (Q4_0, Q8_0,
K-quants, ...), or materialise tensor data, and it does not build a
``DecoderLM`` from a GGUF file. That is the next build (see ch 20).

How production differs
-----------------------
llama.cpp ``mmap``s the whole file, faults tensor pages in on first
touch, and dequantizes blocks inside fused kernels. The header walk
is identical; only when bytes move (and how they are unpacked) differs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants: magic, value types, ggml types
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

# gguf_metadata_value_type -> struct format for scalar types.
_METADATA_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",   # UINT8
    1: "<b",   # INT8
    2: "<H",   # UINT16
    3: "<h",   # INT16
    4: "<I",   # UINT32
    5: "<i",   # INT32
    6: "<f",   # FLOAT32
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}

METADATA_VALUE_TYPE_NAMES: dict[int, str] = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# ggml type code -> name. Codes follow ggml's ``enum ggml_type`` /
# gguf-py's ``GGMLQuantizationType``. Only the header name is needed
# here; block sizes / dequantization live in the next build.
_GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_2",
    5: "Q4_3",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
    34: "TQ1_0",
    35: "TQ2_0",
    36: "MXFP4",
}


def ggml_type_name(code: int) -> str:
    """Name of a ggml tensor type code (``0 -> "F32"``, ``8 -> "Q8_0"``).

    Unknown / future codes return ``"UNKNOWN_<code>"`` rather than
    raising: a header reader should be able to inventory a file written
    by a newer converter even before it can dequantize every type.
    """
    return _GGML_TYPE_NAMES.get(int(code), f"UNKNOWN_{int(code)}")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TensorInfo:
    """One GGUF tensor info entry, exactly as stored on disk.

    ``dims`` are in ggml order (fastest-varying first, the reverse of
    the PyTorch convention). ``offset`` is relative to the file's
    data start -- use :meth:`absolute_offset` with the header's
    ``data_start`` to get a file offset.
    """

    name: str
    n_dims: int
    dims: list[int]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        """Human-readable ggml type name (see :func:`ggml_type_name`)."""
        return ggml_type_name(self.ggml_type)

    # Alias matching the summarize() dict spelling.
    @property
    def ggml_type_name(self) -> str:
        return ggml_type_name(self.ggml_type)

    def absolute_offset(self, data_start: int) -> int:
        """Absolute file offset of this tensor's data."""
        return int(data_start) + int(self.offset)

    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= int(d)
        return n


@dataclass
class GGUFHeader:
    """Parsed GGUF header: version, counts, metadata and tensor infos.

    ``alignment`` comes from the ``general.alignment`` metadata entry
    (default 32). ``data_start`` is the absolute file offset where the
    tensor-data section begins, i.e. the offset just after the tensor
    infos rounded up to ``alignment``. ``header_end`` is that pre-padding
    offset, kept so the padding size is auditable.
    """

    version: int
    tensor_count: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    alignment: int = DEFAULT_ALIGNMENT
    data_start: int = 0
    header_end: int = 0
    metadata_kv_count: int = 0


# ---------------------------------------------------------------------------
# Low-level reader
# ---------------------------------------------------------------------------


class _Reader:
    """Sequential little-endian reader over a binary file object."""

    def __init__(self, f: Any, source: str):
        self._f = f
        self._source = source

    def tell(self) -> int:
        return int(self._f.tell())

    def read_exact(self, n: int, what: str) -> bytes:
        raw = self._f.read(n)
        if len(raw) < n:
            raise ValueError(
                f"{self._source}: truncated GGUF file while reading {what} "
                f"(wanted {n} bytes, got {len(raw)})"
            )
        return raw

    def unpack(self, fmt: str, what: str) -> Any:
        st = struct.Struct(fmt)
        return st.unpack(self.read_exact(st.size, what))[0]

    def read_u32(self, what: str) -> int:
        return int(self.unpack("<I", what))

    def read_u64(self, what: str) -> int:
        return int(self.unpack("<Q", what))

    def read_string(self, what: str) -> str:
        n = self.read_u64(f"{what} (string length)")
        raw = self.read_exact(n, what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{self._source}: {what} is not valid UTF-8: {exc}"
            ) from exc

    def read_metadata_value(self, value_type: int, what: str) -> Any:
        """Read one metadata value of an already-parsed type code."""
        if value_type in _METADATA_SCALAR_FORMATS:
            return self.unpack(
                _METADATA_SCALAR_FORMATS[value_type], what
            )
        if value_type == _TYPE_BOOL:
            raw = self.read_exact(1, what)[0]
            if raw not in (0, 1):
                raise ValueError(
                    f"{self._source}: {what} has invalid BOOL byte {raw}"
                )
            return bool(raw)
        if value_type == _TYPE_STRING:
            return self.read_string(what)
        if value_type == _TYPE_ARRAY:
            elem_type = self.read_u32(f"{what} (array element type)")
            if elem_type == _TYPE_ARRAY:
                raise ValueError(
                    f"{self._source}: {what} is a nested ARRAY, "
                    "which GGUF does not allow"
                )
            if (
                elem_type not in _METADATA_SCALAR_FORMATS
                and elem_type not in (_TYPE_BOOL, _TYPE_STRING)
            ):
                raise ValueError(
                    f"{self._source}: {what} has unknown array element "
                    f"type {elem_type}"
                )
            length = self.read_u64(f"{what} (array length)")
            return [
                self.read_metadata_value(elem_type, f"{what}[{i}]")
                for i in range(length)
            ]
        raise ValueError(
            f"{self._source}: {what} has unknown metadata value type "
            f"{value_type}"
        )


def _align_up(offset: int, alignment: int) -> int:
    return ((int(offset) + int(alignment) - 1) // int(alignment)) * int(alignment)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_gguf_header(path: str | Path) -> GGUFHeader:
    """Parse a GGUF header *by hand* (no ``gguf`` library).

    Reads only the header: the 24-byte fixed prefix, every metadata
    key/value, and every tensor info. Tensor data bytes are never
    touched, so this stays O(header) even for a multi-GB file -- the
    property a loader needs to inventory a checkpoint before deciding
    what to mmap or dequantize.

    Raises ``ValueError`` on bad magic, an unsupported version, a
    truncated file, invalid UTF-8 strings, an unknown metadata value
    type, or a non-positive ``general.alignment``.
    """
    p = Path(path)
    source = str(p)
    with p.open("rb") as f:
        reader = _Reader(f, source)

        magic = reader.read_exact(4, "magic")
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"{source}: bad GGUF magic {magic!r} (expected {GGUF_MAGIC!r})"
            )
        version = reader.read_u32("version")
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"{source}: unsupported GGUF version {version} "
                f"(supported: {list(_SUPPORTED_VERSIONS)})"
            )
        tensor_count = reader.read_u64("tensor_count")
        metadata_kv_count = reader.read_u64("metadata_kv_count")

        metadata: dict[str, Any] = {}
        for i in range(metadata_kv_count):
            key = reader.read_string(f"metadata key {i}")
            value_type = reader.read_u32(f"value type for {key!r}")
            metadata[key] = reader.read_metadata_value(
                value_type, f"metadata value for {key!r}"
            )

        tensors: list[TensorInfo] = []
        for i in range(tensor_count):
            name = reader.read_string(f"tensor {i} name")
            n_dims = reader.read_u32(f"n_dims for tensor {name!r}")
            dims = [
                reader.read_u64(f"dim {j} for tensor {name!r}")
                for j in range(n_dims)
            ]
            ggml_type = reader.read_u32(f"ggml type for tensor {name!r}")
            offset = reader.read_u64(f"offset for tensor {name!r}")
            tensors.append(
                TensorInfo(
                    name=name,
                    n_dims=n_dims,
                    dims=dims,
                    ggml_type=ggml_type,
                    offset=offset,
                )
            )

        header_end = reader.tell()

    alignment_raw = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    # bool is a subclass of int; a BOOL alignment is malformed.
    if isinstance(alignment_raw, bool) or not isinstance(alignment_raw, int):
        raise ValueError(
            f"{source}: general.alignment must be an integer, "
            f"got {alignment_raw!r}"
        )
    alignment = int(alignment_raw)
    if alignment <= 0:
        raise ValueError(
            f"{source}: general.alignment must be positive, got {alignment}"
        )
    data_start = _align_up(header_end, alignment)

    return GGUFHeader(
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        data_start=data_start,
        header_end=header_end,
        metadata_kv_count=metadata_kv_count,
    )


def summarize(path: str | Path) -> dict[str, Any]:
    """Header summary of a GGUF file as plain dicts/lists/ints/strs.

    Returns ``{version, tensor_count, metadata_kv_count, metadata,
    alignment, header_end, data_start, tensors}`` where each tensor row
    is ``{name, n_dims, dims, ggml_type, type_name, ggml_type_name,
    offset, absolute_offset, numel}``. Suitable for printing, diffing,
    or JSON-serialising (metadata values are already plain Python).
    """
    header = read_gguf_header(path)
    return {
        "version": header.version,
        "tensor_count": header.tensor_count,
        "metadata_kv_count": header.metadata_kv_count,
        "metadata": dict(header.metadata),
        "alignment": header.alignment,
        "header_end": header.header_end,
        "data_start": header.data_start,
        "tensors": [
            {
                "name": t.name,
                "n_dims": t.n_dims,
                "dims": list(t.dims),
                "ggml_type": t.ggml_type,
                "type_name": t.type_name,
                "ggml_type_name": t.type_name,
                "offset": t.offset,
                "absolute_offset": t.absolute_offset(header.data_start),
                "numel": t.numel(),
            }
            for t in header.tensors
        ],
    }
