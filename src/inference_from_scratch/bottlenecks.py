"""Ch 13 -- Bottlenecks: arithmetic intensity and the roofline.

Framing follows Kiely, *Inference Engineering* (2026), sec. 2.4
(calculating bottlenecks) and sec. 2.5 (attention / KV cache), with
latency vocabulary from sec. 1.4. All prose and formulae here are
original to this repo.

Why this exists in inference
------------------------------
Two phases that run the same weights can be limited by completely
different hardware resources. Prefill pushes many token rows through
the same matrices, so each loaded weight is reused many times.
Decode pushes one row at a time, so each step re-reads the weights
(and the whole KV cache) to do comparatively little arithmetic.
The roofline model makes that intuition quantitative *before* any
benchmark is run: compare the work's arithmetic intensity
(operations per byte moved) with the device's ridge point
(operations per byte at which its compute and bandwidth ceilings
meet).

    intensity  = flops / bytes
    ridge      = peak_flops / memory_bandwidth_bytes_s
    bound      = "compute" if intensity >= ridge else "memory"
    time >= max(flops / peak_flops, bytes / bandwidth)

Assumptions (stated, not hidden)
--------------------------------
* The ``2 * params`` matmul rule: a matrix multiply that touches
  each of ``P`` weight elements once per token performs one
  multiply and one add per element, i.e. ``2 * P`` FLOPs per token.
  This counts the Q/K/V/O projections, the MLP matrices and the
  LM head. The embedding *lookup* is a gather, not a matmul, so it
  contributes no matmul FLOPs (a tied head still counts, because
  the head itself is a matmul).
* Attention score/value arithmetic is added on top: per layer and
  per generated token, ``Q @ K^T`` and ``weights @ V`` each cost
  ``2 * d_model * S`` FLOPs for context length ``S`` (every head's
  ``D_h``-wide dot products sum to ``d_model``). Decode therefore
  adds ``4 * L * d_model * S``. Prefill averages over the causal
  triangle: position ``i`` attends to ``i + 1`` keys, so the mean
  attended length is ``(S + 1) / 2`` and the per-token term is
  ``2 * L * d_model * (S + 1)``.
* Norms, RoPE, softmax and sampling are ignored in the FLOP count.
  They are elementwise or ``O(S)`` per token and are small next to
  the matmul term at the sizes this model is meant for; on tiny
  CPU models their Python/kernel overhead can in fact dominate
  wall time, which is precisely why this module predicts a
  *lower bound*, not a measured latency (see ``metrics.py``).
* Bytes are a traffic lower bound: weights are read once per
  forward (not once per element use), KV bytes are read in full
  per decode token and written once per prefill token, and
  activation traffic is ignored (activations are assumed to stay
  in registers/SRAM). Real traffic is higher; real intensity is
  therefore lower. Treat every intensity here as optimistic.

How production differs
-----------------------
Production profilers (sec. 4.5 in the book's framing) measure the
same quantities per kernel, with real achieved FLOPs/s and bytes/s
in place of peaks. The classification workflow is identical: get
a model shape from ``config.json``, get device peaks from the
vendor spec sheet, compute intensity, compare with the ridge.
This module deliberately ships *no* fabricated GPU table -- see
:func:`make_example_spec` and :func:`try_read_local_gpu`.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from .config import ModelConfig

_PHASES = ("prefill", "decode")


@dataclass
class HardwareSpec:
    """Peak device numbers, from a vendor spec sheet.

    Fields:
        name: human label for the device / example.
        peak_flops: peak arithmetic throughput, FLOPs/s (at the
            precision the workload runs; ``None`` if unknown).
        memory_bandwidth_bytes_s: peak memory bandwidth, bytes/s
            (``None`` if unknown).
        memory_bytes: device memory capacity, bytes (``None`` if
            unknown). Capacity gates whether a model fits; it does
            not enter the ridge point.

    ``peak_flops`` / ``memory_bandwidth_bytes_s`` may be ``None``
    when only name/capacity are known locally (see
    :func:`try_read_local_gpu`); :func:`ridge_point` refuses to
    guess in that case.
    """

    name: str
    peak_flops: float | None
    memory_bandwidth_bytes_s: float | None
    memory_bytes: int | None


def ridge_point(spec: HardwareSpec) -> float:
    """Ops per byte where the compute and bandwidth ceilings meet.

    ``peak_flops / memory_bandwidth_bytes_s``. Work with intensity
    below this is memory-bound; work above it is compute-bound.
    Raises ``ValueError`` when either peak is unknown or
    non-positive -- filling peaks in from a vendor spec sheet is
    the caller's job, never this module's guess.
    """
    if spec.peak_flops is None or spec.memory_bandwidth_bytes_s is None:
        raise ValueError(
            f"ridge point needs peak_flops and memory_bandwidth_bytes_s; "
            f"spec {spec.name!r} leaves them unknown -- fill them from "
            f"the vendor spec sheet for that exact device/precision"
        )
    if spec.peak_flops <= 0 or spec.memory_bandwidth_bytes_s <= 0:
        raise ValueError(
            f"peaks must be positive, got peak_flops={spec.peak_flops}, "
            f"bandwidth={spec.memory_bandwidth_bytes_s}"
        )
    return float(spec.peak_flops) / float(spec.memory_bandwidth_bytes_s)


def make_example_spec() -> HardwareSpec:
    """Illustrative round numbers ONLY -- not any real GPU.

    ``100 TFLOP/s`` and ``1 TB/s`` give a clean ridge point of
    ``100 ops/byte`` so examples and tests can be checked by hand.
    Never quote these as a device measurement; for a real device
    read its spec sheet (or start from :func:`try_read_local_gpu`
    for the name/capacity and fill the peaks in yourself).
    """
    return HardwareSpec(
        name="illustrative-example (round numbers, not a real GPU)",
        peak_flops=100e12,
        memory_bandwidth_bytes_s=1e12,
        memory_bytes=80_000_000_000,
    )


def try_read_local_gpus() -> list[HardwareSpec]:
    """Best-effort enumeration of ALL local GPUs via ``nvidia-smi``.

    Returns an empty list when ``nvidia-smi`` is absent or fails
    (the common CPU-only case). On success returns one
    :class:`HardwareSpec` per GPU line, in ``nvidia-smi`` order,
    with only ``name`` and ``memory_bytes`` filled:
    ``peak_flops`` and ``memory_bandwidth_bytes_s`` are left
    ``None`` on purpose -- fill them from the vendor spec sheet
    for the exact card and the precision you will run
    (fp16/bf16 tensor-core peaks differ from fp32, and boost
    clocks differ per SKU). ``nvidia-smi`` reports total memory
    in MiB; the conversion to bytes is exact (``MiB * 2**20``).
    Malformed lines are skipped rather than discarding the rest.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0 or not out.stdout.strip():
        return []
    specs: list[HardwareSpec] = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.rsplit(",", 1)]
        if len(parts) != 2:
            continue
        name, mem_text = parts
        try:
            mem_mib = int(float(mem_text))
        except ValueError:
            continue
        specs.append(
            HardwareSpec(
                name=name,
                peak_flops=None,
                memory_bandwidth_bytes_s=None,
                memory_bytes=mem_mib * 1024 * 1024,
            )
        )
    return specs


def try_read_local_gpu() -> HardwareSpec | None:
    """Best-effort first local GPU name/VRAM via ``nvidia-smi``.

    Backward-compatible single-GPU view of
    :func:`try_read_local_gpus`: returns the first enumerated GPU,
    or ``None`` when no GPU is visible. Prefer
    :func:`try_read_local_gpus` on multi-GPU boxes so later GPUs
    are not silently dropped.
    """
    gpus = try_read_local_gpus()
    return gpus[0] if gpus else None


# ---------------------------------------------------------------------------
# Parameter counting (analytic; must agree with DecoderLM.num_params)
# ---------------------------------------------------------------------------


def _norm_params(config: ModelConfig) -> int:
    """Params in one norm: D for RMSNorm, 2*D for LayerNorm."""
    return config.d_model if config.norm_type == "rmsnorm" else 2 * config.d_model


def count_params(config: ModelConfig) -> int:
    """Total unique parameters implied by ``config``.

    Matches ``DecoderLM(config).num_params()`` analytically:
    embedding (plus learned positions), per layer the Q/K/V/O and
    MLP matmul weights plus two norms, the final norm, and the LM
    head only when embeddings are untied. Useful when reasoning
    about a real ``config.json`` without instantiating the model.
    """
    D, L, V = config.d_model, config.n_layer, config.vocab_size
    total = V * D  # tok_embedding [V, D]
    if config.pos_type == "learned":
        total += config.max_seq_len * D
    per_layer = _matmul_params_per_layer(config) + 2 * _norm_params(config)
    total += L * per_layer
    total += _norm_params(config)  # final norm
    if not config.tie_embeddings:
        total += V * D  # separate lm_head [V, D]
    return int(total)


def _matmul_params_per_layer(config: ModelConfig) -> int:
    D = config.d_model
    kv_width = config.n_kv_head * config.head_dim  # K/V projection width
    attn = D * D + D * kv_width + D * kv_width + D * D  # Q, K, V, O
    if config.mlp_type == "swiglu":
        mlp = 3 * D * config.d_ff  # gate, up, down
    else:
        mlp = 2 * D * config.d_ff  # up, down
    return int(attn + mlp)


def matmul_params(config: ModelConfig) -> int:
    """Parameters that participate in a per-token matmul.

    Per layer Q/K/V/O + MLP, plus the LM head ``[V, D]`` (counted
    even when tied: the head matmul still runs per token). The
    embedding lookup, norms and any learned position table are
    excluded -- they are gathers/elementwise, not matmuls.
    """
    return int(
        config.n_layer * _matmul_params_per_layer(config)
        + config.vocab_size * config.d_model
    )


# ---------------------------------------------------------------------------
# FLOPs / bytes / intensity
# ---------------------------------------------------------------------------


def _validate_phase_seq(phase: str, seq_len: int) -> None:
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}, got {phase!r}")
    if int(seq_len) < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")


def estimate_flops_per_token(
    config: ModelConfig,
    seq_len: int,
    phase: str,
    batch: int = 1,
) -> float:
    """Estimated FLOPs per generated/prefilled token (per sequence).

    ``2 * matmul_params`` (the 2*params rule; see module docstring)
    plus the attention score/value term:

    * decode (one query vs ``seq_len`` cached keys):
      ``4 * L * d_model * seq_len``;
    * prefill (causal average of ``(seq_len + 1) / 2`` keys/query):
      ``2 * L * d_model * (seq_len + 1)``.

    Per-token FLOPs do not depend on ``batch`` (each sequence does
    its own arithmetic; batching shares *reads*, not math), so
    ``batch`` is accepted for call-site symmetry and ignored here.
    """
    _validate_phase_seq(phase, seq_len)
    if int(batch) < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    matmul = 2.0 * matmul_params(config)
    L, D, S = config.n_layer, config.d_model, int(seq_len)
    if phase == "decode":
        attn = 4.0 * L * D * S
    else:
        attn = 2.0 * L * D * (S + 1)
    return float(matmul + attn)


def _kv_bytes_per_token_stored(config: ModelConfig, kv_bytes: int) -> int:
    """Bytes to store one token's K+V across all layers."""
    return int(2 * config.n_layer * config.n_kv_head * config.head_dim * kv_bytes)


def estimate_bytes_per_token(
    config: ModelConfig,
    seq_len: int,
    phase: str,
    weight_bytes: int = 2,
    kv_bytes: int = 2,
    batch: int = 1,
) -> float:
    """Estimated bytes moved per token (per generated token).

    * decode: every step re-reads all weights and the full KV
      cache for its sequence. Weights are read once per *step*,
      so per generated token they are amortized over ``batch``;
      each sequence's KV read is its own. ::

          weight_total / batch + 2*L*H_kv*S*D_h*kv_bytes

    * prefill: one forward processes ``batch * seq_len`` tokens
      and reads the weights once, so per token the weight traffic
      is ``weight_total / (batch * seq_len)``; the KV term is the
      one-time *write* of this token's own K/V
      (``2*L*H_kv*D_h*kv_bytes``). Intra-prefill attention reads
      are assumed served from SRAM and not counted (a stated
      lower bound -- see module docstring).

    ``weight_bytes`` / ``kv_bytes`` are bytes per stored element
    (2 for fp16/bf16, 1 for int8, ...; ch 10 prices the options).
    """
    _validate_phase_seq(phase, seq_len)
    if int(batch) < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if weight_bytes <= 0 or kv_bytes <= 0:
        raise ValueError(
            f"weight_bytes/kv_bytes must be positive, got {weight_bytes}/{kv_bytes}"
        )
    weight_total = float(count_params(config) * int(weight_bytes))
    if phase == "decode":
        kv_read = float(_kv_bytes_per_token_stored(config, int(kv_bytes)) * int(seq_len))
        return weight_total / int(batch) + kv_read
    kv_write = float(_kv_bytes_per_token_stored(config, int(kv_bytes)))
    return weight_total / (int(batch) * int(seq_len)) + kv_write


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """Ops per byte: ``flops / bytes_moved`` (both must be sane)."""
    if bytes_moved <= 0:
        raise ValueError(f"bytes_moved must be positive, got {bytes_moved}")
    if flops < 0:
        raise ValueError(f"flops must be >= 0, got {flops}")
    return float(flops) / float(bytes_moved)


def classify(
    config: ModelConfig,
    spec: HardwareSpec,
    seq_len: int,
    phase: str,
    batch: int = 1,
    weight_bytes: int = 2,
    kv_bytes: int = 2,
) -> dict:
    """Place one phase of one model on one device.

    Returns a dict with ``intensity`` (ops/byte), ``ridge``
    (ops/byte), ``bound`` (``"compute"`` or ``"memory"``),
    ``flops`` and ``bytes`` (the per-token estimates behind the
    intensity), plus ``phase``/``seq_len``/``batch`` echoes for
    logging. ``bound`` is ``"compute"`` iff ``intensity >= ridge``.
    """
    flops = estimate_flops_per_token(config, seq_len, phase, batch=batch)
    moved = estimate_bytes_per_token(
        config, seq_len, phase, weight_bytes=weight_bytes, kv_bytes=kv_bytes, batch=batch
    )
    intensity = arithmetic_intensity(flops, moved)
    ridge = ridge_point(spec)
    return {
        "intensity": intensity,
        "ridge": ridge,
        "bound": "compute" if intensity >= ridge else "memory",
        "flops": flops,
        "bytes": moved,
        "phase": phase,
        "seq_len": int(seq_len),
        "batch": int(batch),
    }


def roofline_time_lower_bound(*args, **kwargs) -> float:
    """Seconds floor from the roofline: ``max(flops/peak, bytes/bandwidth)``.

    Two call shapes are supported:

    * ``roofline_time_lower_bound(flops, bytes_moved, spec)`` --
      raw numbers straight from the estimators;
    * ``roofline_time_lower_bound(config, spec, seq_len, phase, batch=1)``
      -- estimates per-token flops/bytes first, then bounds the
      per-token time. Keyword forms ``flops=``, ``bytes_moved=``
      (or ``bytes=``) plus ``spec=`` also work.

    The result is a lower bound: no kernel beats both ceilings at
    once, and real kernels achieve a fraction of each peak.
    """
    flops: float | None = None
    moved: float | None = None
    spec: HardwareSpec | None = None

    if args and isinstance(args[0], ModelConfig):
        if len(args) < 4:
            raise TypeError(
                "config form needs (config, spec, seq_len, phase[, batch])"
            )
        config, spec, seq_len, phase = args[0], args[1], args[2], args[3]
        batch = args[4] if len(args) > 4 else kwargs.get("batch", 1)
        flops = estimate_flops_per_token(config, seq_len, phase, batch=batch)
        moved = estimate_bytes_per_token(
            config,
            seq_len,
            phase,
            weight_bytes=kwargs.get("weight_bytes", 2),
            kv_bytes=kwargs.get("kv_bytes", 2),
            batch=batch,
        )
    else:
        if len(args) >= 3:
            flops, moved, spec = float(args[0]), float(args[1]), args[2]
        flops = kwargs.get("flops", flops)
        moved = kwargs.get("bytes_moved", kwargs.get("bytes", moved))
        spec = kwargs.get("spec", spec)

    if flops is None or moved is None or spec is None:
        raise TypeError(
            "need (flops, bytes_moved, spec) or (config, spec, seq_len, phase)"
        )
    ridge_point(spec)  # validates peaks present/positive
    assert spec.peak_flops is not None and spec.memory_bandwidth_bytes_s is not None
    return float(max(flops / spec.peak_flops, moved / spec.memory_bandwidth_bytes_s))
