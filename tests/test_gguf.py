"""Tests for GGUF header dissection by hand.

The GGUF writer lives here (in the test file, not the library):
``build_gguf_bytes`` serialises tiny valid GGUF v3 files with
``struct`` only, so the hand parser is tested against the format
layout, not against the ``gguf`` library it dissects.
"""

import struct

import pytest

from inference_from_scratch.gguf import (
    ggml_type_name,
    read_gguf_header,
    summarize,
)

# gguf_metadata_value_type codes (see gguf.py).
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

# ggml type codes used below.
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0 = 0, 1, 2, 8


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_value(type_code: int, value) -> bytes:
    if type_code in _SCALAR_FMT:
        return struct.pack(_SCALAR_FMT[type_code], value)
    if type_code == T_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if type_code == T_STRING:
        return _pack_string(value)
    if type_code == T_ARRAY:
        elem_type, elems = value
        out = struct.pack("<I", elem_type) + struct.pack("<Q", len(elems))
        for e in elems:
            out += _pack_value(elem_type, e)
        return out
    raise AssertionError(f"test writer: unhandled type {type_code}")


def build_gguf_bytes(metadata, tensors, version=3):
    """Synthesise a tiny valid GGUF file as bytes.

    ``metadata``: list of ``(key, value_type_code, value)``; for
    ``T_ARRAY`` the value is ``(elem_type_code, [elements])``.
    ``tensors``: list of dicts ``{name, dims, ggml_type, data}`` with
    optional ``offset``; offsets default to sequential packing of the
    data blobs in order. Alignment is read from a
    ``general.alignment`` metadata entry (default 32), exactly like
    the parser under test.
    """
    alignment = 32
    for key, _tc, value in metadata:
        if key == "general.alignment":
            alignment = value

    # Assign sequential offsets when the caller did not pin them.
    tensors = [dict(t) for t in tensors]
    next_offset = 0
    for t in tensors:
        if "offset" not in t:
            t["offset"] = next_offset
        next_offset = max(next_offset, t["offset"] + len(t["data"]))

    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))
    for key, type_code, value in metadata:
        out += _pack_string(key)
        out += struct.pack("<I", type_code)
        out += _pack_value(type_code, value)
    for t in tensors:
        out += _pack_string(t["name"])
        out += struct.pack("<I", len(t["dims"]))
        for d in t["dims"]:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", t["ggml_type"])
        out += struct.pack("<Q", t["offset"])

    header_end = len(out)
    data_start = ((header_end + alignment - 1) // alignment) * alignment
    out += b"\x00" * (data_start - header_end)
    # Place tensor data at data_start + offset, zero-filling gaps.
    for t in tensors:
        pos = data_start + t["offset"]
        if len(out) < pos:
            out += b"\x00" * (pos - len(out))
        out = out[:pos] + t["data"] + out[pos + len(t["data"]):]
    return out


def _write(tmp_path, blob, name="tiny.gguf"):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


def _basic_metadata():
    return [
        ("general.architecture", T_STRING, "qwen2"),
        ("general.alignment", T_UINT32, 32),
        ("qwen2.block_count", T_UINT32, 2),
        ("qwen2.context_length", T_UINT32, 512),
        ("custom.ratio", T_FLOAT32, 0.5),
        ("custom.big", T_UINT64, 2**60 + 3),
        ("custom.signed", T_INT64, -1234567890123),
        ("custom.precise", T_FLOAT64, 3.25),
        ("custom.small_u8", T_UINT8, 7),
        ("custom.small_i8", T_INT8, -9),
        ("custom.u16", T_UINT16, 65000),
        ("custom.i16", T_INT16, -32000),
        ("custom.i32", T_INT32, -70000),
        ("custom.enabled", T_BOOL, True),
    ]


def _basic_tensors():
    return [
        {"name": "token_embd.weight", "dims": [4, 3],
         "ggml_type": GGML_F32, "data": b"\x00" * 48},
        {"name": "blk.0.attn_q.weight", "dims": [4, 4],
         "ggml_type": GGML_Q8_0, "data": b"\x11" * 32},
        {"name": "output_norm.weight", "dims": [4],
         "ggml_type": GGML_F16, "data": b"\x00" * 8},
    ]


# --- roundtrip ---------------------------------------------------------------


def test_roundtrip_header_parse_all_scalars(tmp_path):
    path = _write(tmp_path, build_gguf_bytes(_basic_metadata(), _basic_tensors()))
    header = read_gguf_header(path)

    assert header.version == 3
    assert header.tensor_count == 3
    assert header.metadata_kv_count == len(_basic_metadata())
    md = header.metadata
    assert md["general.architecture"] == "qwen2"
    assert md["general.alignment"] == 32
    assert md["qwen2.block_count"] == 2
    assert md["custom.ratio"] == pytest.approx(0.5)
    assert md["custom.big"] == 2**60 + 3
    assert md["custom.signed"] == -1234567890123
    assert md["custom.precise"] == 3.25
    assert md["custom.small_u8"] == 7
    assert md["custom.small_i8"] == -9
    assert md["custom.u16"] == 65000
    assert md["custom.i16"] == -32000
    assert md["custom.i32"] == -70000
    assert md["custom.enabled"] is True

    assert [t.name for t in header.tensors] == [
        "token_embd.weight", "blk.0.attn_q.weight", "output_norm.weight",
    ]
    emb = header.tensors[0]
    assert (emb.n_dims, emb.dims, emb.ggml_type, emb.type_name) == (
        2, [4, 3], GGML_F32, "F32",
    )
    assert header.tensors[1].type_name == "Q8_0"
    assert header.tensors[2].dims == [4]
    # Sequential offsets assigned by the writer: 0, 48, 48+32.
    assert [t.offset for t in header.tensors] == [0, 48, 80]


def test_roundtrip_array_metadata(tmp_path):
    metadata = _basic_metadata() + [
        ("tokenizer.ggml.tokens", T_ARRAY,
         (T_STRING, ["<|endoftext|>", "The", " capital"])),
        ("custom.sizes", T_ARRAY, (T_UINT32, [1, 22, 333])),
        ("custom.weights", T_ARRAY, (T_FLOAT32, [0.5, -1.5])),
        ("custom.flags", T_ARRAY, (T_BOOL, [True, False, True])),
        ("custom.empty", T_ARRAY, (T_INT32, [])),
    ]
    path = _write(tmp_path, build_gguf_bytes(metadata, _basic_tensors()))
    md = read_gguf_header(path).metadata
    assert md["tokenizer.ggml.tokens"] == ["<|endoftext|>", "The", " capital"]
    assert md["custom.sizes"] == [1, 22, 333]
    assert md["custom.weights"] == pytest.approx([0.5, -1.5])
    assert md["custom.flags"] == [True, False, True]
    assert md["custom.empty"] == []


# --- offsets / data-start arithmetic -----------------------------------------


def test_offsets_and_data_start_arithmetic(tmp_path):
    blob = build_gguf_bytes(_basic_metadata(), _basic_tensors())
    path = _write(tmp_path, blob)
    header = read_gguf_header(path)

    assert header.alignment == 32
    # data_start is header_end rounded UP to a multiple of alignment.
    assert header.data_start % 32 == 0
    assert header.header_end <= header.data_start < header.header_end + 32
    for t in header.tensors:
        assert t.absolute_offset(header.data_start) == (
            header.data_start + t.offset
        )
    # The bytes at the absolute offsets are the tensor data we wrote.
    raw = path.read_bytes()
    emb = header.tensors[0]
    assert raw[emb.absolute_offset(header.data_start):][:4] == b"\x00" * 4
    attn = header.tensors[1]
    assert raw[attn.absolute_offset(header.data_start)] == 0x11


def test_custom_alignment_changes_data_start(tmp_path):
    metadata = [("general.alignment", T_UINT32, 16),
                ("general.architecture", T_STRING, "llama")]
    blob = build_gguf_bytes(metadata, _basic_tensors())
    path = _write(tmp_path, blob)
    header = read_gguf_header(path)
    assert header.alignment == 16
    assert header.data_start % 16 == 0
    assert header.header_end <= header.data_start < header.header_end + 16
    # summarize agrees with the header object.
    summary = summarize(path)
    assert summary["data_start"] == header.data_start
    assert summary["tensors"][1]["absolute_offset"] == (
        header.data_start + 48
    )


def test_default_alignment_is_32_without_metadata(tmp_path):
    metadata = [("general.architecture", T_STRING, "llama")]
    path = _write(tmp_path, build_gguf_bytes(metadata, _basic_tensors()))
    header = read_gguf_header(path)
    assert header.alignment == 32
    assert header.data_start % 32 == 0


def test_explicit_tensor_offsets_are_honoured(tmp_path):
    tensors = [
        {"name": "a.weight", "dims": [2], "ggml_type": GGML_F32,
         "data": b"AAAA", "offset": 0},
        {"name": "b.weight", "dims": [2], "ggml_type": GGML_F32,
         "data": b"BBBB", "offset": 64},  # gap between a and b
    ]
    path = _write(tmp_path, build_gguf_bytes(_basic_metadata(), tensors))
    header = read_gguf_header(path)
    assert [t.offset for t in header.tensors] == [0, 64]
    raw = path.read_bytes()
    assert raw[header.data_start + 64: header.data_start + 68] == b"BBBB"


# --- errors -------------------------------------------------------------------


def test_bad_magic_raises(tmp_path):
    blob = b"GGUX" + build_gguf_bytes(_basic_metadata(), _basic_tensors())[4:]
    path = _write(tmp_path, blob)
    with pytest.raises(ValueError, match="bad GGUF magic"):
        read_gguf_header(path)


def test_unsupported_version_raises(tmp_path):
    blob = build_gguf_bytes(_basic_metadata(), _basic_tensors(), version=99)
    path = _write(tmp_path, blob)
    with pytest.raises(ValueError, match="unsupported GGUF version 99"):
        read_gguf_header(path)


def test_version_2_is_accepted(tmp_path):
    blob = build_gguf_bytes(_basic_metadata(), _basic_tensors(), version=2)
    header = read_gguf_header(_write(tmp_path, blob))
    assert header.version == 2


def test_truncated_file_raises(tmp_path):
    blob = build_gguf_bytes(_basic_metadata(), _basic_tensors())
    path = _write(tmp_path, blob[: len(blob) // 3])
    with pytest.raises(ValueError, match="truncated"):
        read_gguf_header(path)


def test_unknown_metadata_value_type_raises(tmp_path):
    metadata = [("custom.mystery", 99, 0)]
    # _pack_value cannot encode type 99; hand-assemble instead.
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", 0) + struct.pack("<Q", 1)
    out += _pack_string("custom.mystery") + struct.pack("<I", 99)
    out += struct.pack("<B", 0)
    path = _write(tmp_path, out)
    with pytest.raises(ValueError, match="unknown metadata value type 99"):
        read_gguf_header(path)


# --- type names + summarize ---------------------------------------------------


def test_ggml_type_name_common_and_unknown():
    assert ggml_type_name(0) == "F32"
    assert ggml_type_name(1) == "F16"
    assert ggml_type_name(2) == "Q4_0"
    assert ggml_type_name(8) == "Q8_0"
    assert ggml_type_name(30) == "BF16"
    assert ggml_type_name(999) == "UNKNOWN_999"


def test_summarize_matches_header(tmp_path):
    path = _write(tmp_path, build_gguf_bytes(_basic_metadata(), _basic_tensors()))
    summary = summarize(path)
    assert summary["version"] == 3
    assert summary["tensor_count"] == 3
    assert summary["metadata_kv_count"] == len(_basic_metadata())
    assert summary["alignment"] == 32
    assert summary["metadata"]["general.architecture"] == "qwen2"
    rows = summary["tensors"]
    assert rows[0]["name"] == "token_embd.weight"
    assert rows[0]["dims"] == [4, 3]
    assert rows[0]["type_name"] == "F32"
    assert rows[0]["ggml_type_name"] == "F32"
    assert rows[0]["numel"] == 12
    assert rows[1]["type_name"] == "Q8_0"
    assert rows[1]["absolute_offset"] == summary["data_start"] + 48
