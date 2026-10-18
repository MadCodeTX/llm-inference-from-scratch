"""Ch 13 / sec. 1.4 -- Latency metrics: percentiles, TTFT, TPS.

Framing follows Kiely, *Inference Engineering* (2026), sec. 1.4
(measuring latency/throughput: TTFT, TPS, percentiles, end-to-end
vs on-GPU) and sec. 2.4. All prose and code here are original.

Why this exists in inference
------------------------------
A mean latency hides the tail users remember, and a single
"tokens/sec" number hides which phase produced it. Production
latency is therefore reported as percentiles (p50/p90/p99) of
time-to-first-token (TTFT, prefill-dominated) separately from
per-token decode speed (TPS / TPOT). This module is the small,
tested vocabulary for that reporting, plus one honest timer for
this repo's own generation loop.

End-to-end vs on-device (sec. 1.4.2 distinction)
-------------------------------------------------
Everything measured here is *end-to-end wall time* on the host:
``time.perf_counter`` brackets around Python calls, so sampling,
token concatenation and interpreter overhead are all included.
That is the latency a client would feel (minus network/serving
queueing, which this toy has none of). It is *not* on-device
kernel time: a GPU profiler would report only the kernel spans,
excluding host gaps, and would give smaller numbers for the same
work. On this repo's tiny CPU models the host fraction dominates,
so the numbers here honestly show overhead, not device limits --
pair them with ``bottlenecks.py``'s roofline floor to see how far
above the bound the implementation sits, and do not compare a
wall-time TPS against a profiler TPS as if they were one quantity.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import torch

from .generate import _forward, _model_n_layer
from .kv_cache import KVCache
from .sampling import sample_next


def percentile(values: Iterable[float], p: float) -> float:
    """The ``p``-th percentile with linear interpolation.

    ``p`` is in ``[0, 100]`` (``percentile(xs, 95)`` is p95). The
    method is linear interpolation between closest ranks (the
    common default, e.g. NumPy's ``method="linear"``): sort, take
    rank ``p/100 * (n - 1)``, and interpolate between its floor
    and ceil neighbours; exact endpoints return the
    min/max. Raises ``ValueError`` for empty input or ``p``
    outside ``[0, 100]`` -- a percentile of nothing is a caller
    bug, not a zero.
    """
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("percentile needs at least one value")
    if not (0.0 <= float(p) <= 100.0):
        raise ValueError(f"p must be in [0, 100], got {p}")
    if len(vals) == 1:
        return vals[0]
    rank = (float(p) / 100.0) * (len(vals) - 1)
    lo = int(rank)  # floor (rank >= 0)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return float(vals[lo] + frac * (vals[hi] - vals[lo]))


def summarize(values: Iterable[float]) -> dict:
    """Summary stats for a latency/throughput sample.

    Returns ``{"n", "mean", "p50", "p90", "p99", "min", "max"}``.
    ``n`` is an int; the rest are floats. Empty input raises
    ``ValueError`` (via the same contract as :func:`percentile`:
    report nothing rather than fabricate zeros). The key set is
    deliberately the reporting set from sec. 1.4 -- median, tail,
    and spread -- not an exhaustive statistics package.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("summarize needs at least one value")
    return {
        "n": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def measure_generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int | None = None,
    **kwargs,
) -> dict:
    """Time one cached generation as prefill (TTFT) + decode.

    Runs the same explicit prefill-then-decode loop as
    ``generate.generate`` (one prompt forward, then one-token
    forwards against a ``KVCache``) with ``time.perf_counter``
    brackets, *without* changing ``generate``'s API: this function
    re-implements the loop for measurement, and a re-run of the
    prefill forward alone is what isolates TTFT.

    Accepted kwargs (mirroring ``generate``): ``temperature``,
    ``top_k``, ``top_p``, ``eos_id``, ``generator``; also
    ``max_new`` as an alias for the positional budget. Returns::

        {ttft_s, decode_s, tps, tokens,
         total_s, tpot_s, decode_tps, token_ids, n_new_tokens}

    where ``ttft_s`` is prefill + first-token sampling (end-to-end
    wall time, host overhead included -- see module docstring),
    ``decode_s`` is the remaining decode steps, ``tps`` is the
    end-to-end rate ``tokens / total_s``, ``decode_tps`` is
    ``(tokens - 1) / decode_s`` when at least two tokens were
    produced (else ``tps``), ``tpot_s`` is ``1 / decode_tps`` in
    seconds (0.0 when undefined), and ``tokens`` is the count of
    new tokens (``token_ids`` holds the full id list, prompt
    included). With ``max_new_tokens == 0`` every time is 0.0 and
    ``tokens`` is 0.
    """
    if max_new_tokens is None:
        max_new_tokens = kwargs.pop("max_new", None)
    if max_new_tokens is None:
        raise TypeError("measure_generate needs max_new_tokens (or max_new=...)")
    kwargs.pop("use_cache", None)  # measurement loop is the cached path
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if not isinstance(idx, torch.Tensor) or idx.dim() != 2:
        raise ValueError("idx must be a [B, T] tensor")

    temperature = kwargs.pop("temperature", 1.0)
    top_k = kwargs.pop("top_k", None)
    top_p = kwargs.pop("top_p", None)
    eos_id = kwargs.pop("eos_id", None)
    generator = kwargs.pop("generator", None)
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")

    if max_new_tokens == 0:
        return {
            "ttft_s": 0.0,
            "decode_s": 0.0,
            "tps": 0.0,
            "tokens": 0,
            "total_s": 0.0,
            "tpot_s": 0.0,
            "decode_tps": 0.0,
            "token_ids": idx[0].tolist() if idx.shape[0] else [],
            "n_new_tokens": 0,
        }

    sample_kwargs = dict(
        temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            kv_caches = KVCache(n_layer=_model_n_layer(model))
            out = idx.clone()

            t0 = time.perf_counter()
            logits, kv_caches = _forward(model, out, kv_caches=kv_caches, start_pos=0)
            next_token = sample_next(logits[:, -1], **sample_kwargs)
            t1 = time.perf_counter()

            out = torch.cat([out, next_token], dim=1)
            n_new = 1
            finished = eos_id is not None and bool(
                (next_token.squeeze(1) == eos_id).all()
            )

            decode_start = time.perf_counter()
            while n_new < max_new_tokens and not finished:
                start_pos = out.shape[1] - 1
                logits, kv_caches = _forward(
                    model, next_token, kv_caches=kv_caches, start_pos=start_pos
                )
                next_token = sample_next(logits[:, -1], **sample_kwargs)
                out = torch.cat([out, next_token], dim=1)
                n_new += 1
                if eos_id is not None and bool(
                    (next_token.squeeze(1) == eos_id).all()
                ):
                    finished = True
            decode_s = time.perf_counter() - decode_start
    finally:
        if was_training:
            model.train()

    ttft_s = float(t1 - t0)
    decode_s = float(decode_s)
    total_s = ttft_s + decode_s
    tps = float(n_new / total_s) if total_s > 0 else 0.0
    if decode_s > 0 and n_new > 1:
        decode_tps = float((n_new - 1) / decode_s)
        tpot_s = float(decode_s / (n_new - 1))
    else:
        decode_tps = tps
        tpot_s = 0.0
    return {
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "tps": tps,
        "tokens": int(n_new),
        "total_s": float(total_s),
        "tpot_s": float(tpot_s),
        "decode_tps": float(decode_tps),
        "token_ids": out[0].tolist() if out.shape[0] else [],
        "n_new_tokens": int(n_new),
    }
