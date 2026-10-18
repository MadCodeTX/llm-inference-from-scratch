"""Ch 08 -- Generation: explicit PRE-FILL then DECODE, plus no-cache reference.

Why this exists in inference
------------------------------
A served request is two different workloads wearing one coat
(Ch 00). PRE-FILL ingests the whole prompt in one parallel pass
(``T > 1``, compute-bound) and fills the KV cache. DECODE then emits
one token at a time (``T == 1`` per step, memory-bandwidth-bound:
each step re-reads all weights and the whole cache to produce a
single token). Writing the loop out longhand -- instead of calling
a library ``generate()`` -- makes the phase boundary, the
``start_pos`` bookkeeping and the EOS rule impossible to miss.

The two paths
-------------
``use_cache=True`` (the real path): one PRE-FILL forward over the
prompt, then DECODE forwards of a single token each, with the
:class:`~inference_from_scratch.kv_cache.KVCache` accumulating
every layer's K/V. Cost per decode step is O(1) *new* K/V work.

``use_cache=False`` (the reference path): every step re-runs the
*entire* sequence so far with no cache. O(T) forwards of growing
width -- quadratically wasteful, but trivially correct, which is
why it exists: under greedy decoding the two paths must emit
*identical* token ids, and the tests hold them to exactly that.
Any divergence is a cache / mask / position bug, never "sampling
noise" (greedy has no randomness).

Sampling and EOS
----------------
Each step's ``logits[:, -1]`` ``[B, V]`` goes through
:func:`~inference_from_scratch.sampling.sample_next`
(temperature / top-k / top-p; ``temperature=0`` is greedy). If
``eos_id`` is given, generation stops as soon as *every* batch row
has emitted it (the EOS token itself is included in the output --
callers that do not want it can strip it). With one sequence, the
common case here, that is simply "stop at the first EOS".

How production differs
----------------------
Serving engines (vLLM, TGI) run this same loop for thousands of
requests at once with continuous batching (Ch 11): rows enter and
leave the batch at every step, finished rows are retired
individually instead of waiting for the whole batch, and the
scheduler -- not the model -- owns the loop. The per-request maths
(prefill once, decode one-by-one, stop at EOS) is unchanged.
"""

from __future__ import annotations

import torch

from .kv_cache import KVCache
from .sampling import sample_next


def _model_n_layer(model: torch.nn.Module) -> int:
    """Layer count for the cache, from the model's own config/shape.

    Duck-typed so tests can pass a small stub model: prefer
    ``model.config.n_layer``, fall back to ``len(model.layers)``,
    then to a single-layer cache (enough for a stub whose forward
    ignores the cache entirely).
    """
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "n_layer"):
        return int(config.n_layer)
    layers = getattr(model, "layers", None)
    if layers is not None:
        try:
            return len(layers)
        except TypeError:
            pass
    return 1


def _forward(model, idx, kv_caches=None, start_pos=0):
    """Call ``model`` and normalise to ``(logits, kv_caches)``.

    The real :class:`DecoderLM` returns the pair; a stub that
    returns bare logits is accepted so unit tests can fake the
    model without implementing the cache contract.
    """
    out = model(idx, kv_caches=kv_caches, start_pos=start_pos)
    if isinstance(out, tuple):
        logits, caches = out
        return logits, caches
    return out, kv_caches


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate up to ``max_new_tokens`` new tokens after ``idx``.

    ``idx`` is ``[B, T_prompt]`` prompt ids; the return is
    ``[B, T_prompt + n]`` with ``n <= max_new_tokens`` (smaller only
    when EOS stops generation early). The prompt itself is never
    modified (the output is a fresh tensor).

    ``temperature=0`` selects greedy decoding; cache and no-cache
    paths then produce identical token ids (tested). ``generator``
    seeds stochastic sampling for reproducibility.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if idx.dim() != 2:
        raise ValueError(f"idx must be [B, T], got shape {tuple(idx.shape)}")

    # Sampling kwargs shared by every step (and both paths, so the
    # greedy-equivalence property compares paths, not samplers).
    sample_kwargs = dict(
        temperature=temperature, top_k=top_k, top_p=top_p, generator=generator,
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if use_cache:
                return _generate_cached(
                    model, idx, max_new_tokens, eos_id, sample_kwargs,
                )
            return _generate_no_cache(
                model, idx, max_new_tokens, eos_id, sample_kwargs,
            )
    finally:
        if was_training:
            model.train()


def _generate_cached(model, idx, max_new_tokens, eos_id, sample_kwargs):
    """PRE-FILL once, then DECODE one token per step (KV-cache path)."""
    if max_new_tokens == 0:
        return idx.clone()

    # One cache object for the whole model; every layer writes its
    # own slice (indexed by layer_idx inside the blocks).
    kv_caches = KVCache(n_layer=_model_n_layer(model))

    # ==================================================================
    # PRE-FILL: the whole prompt in one parallel forward (T > 1).
    # Every prompt token's K/V lands in the cache at positions
    # 0..T_prompt-1; only the last position's logits are sampled.
    # ==================================================================
    out = idx.clone()  # [B, T_prompt]
    logits, kv_caches = _forward(model, out, kv_caches=kv_caches, start_pos=0)
    # logits: [B, T_prompt, V] -> next-token logits [B, V]
    next_token = sample_next(logits[:, -1], **sample_kwargs)  # [B, 1]
    out = torch.cat([out, next_token], dim=1)  # [B, T_prompt + 1]

    # EOS after prefill's token: every row must have finished.
    finished = (
        (next_token.squeeze(1) == eos_id)
        if eos_id is not None
        else torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
    )
    if bool(finished.all()):
        return out

    # ==================================================================
    # DECODE: one token per forward (T == 1), cache does the memory.
    # Each step feeds only the token sampled in the previous step;
    # its absolute position is len(out) - 1 (the newest token sits
    # at the end of the sequence built so far).
    # ==================================================================
    for _ in range(max_new_tokens - 1):
        start_pos = out.shape[1] - 1  # absolute position of next_token
        logits, kv_caches = _forward(
            model, next_token, kv_caches=kv_caches, start_pos=start_pos,
        )
        # logits: [B, 1, V] -> [B, V]
        next_token = sample_next(logits[:, -1], **sample_kwargs)  # [B, 1]
        out = torch.cat([out, next_token], dim=1)  # [B, T + 1]
        if eos_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_id)
            if bool(finished.all()):
                break
    return out


def _generate_no_cache(model, idx, max_new_tokens, eos_id, sample_kwargs):
    """No-cache reference: re-run the full sequence at every step."""
    out = idx.clone()  # [B, T] grows by one token per step
    finished = torch.zeros(
        out.shape[0], dtype=torch.bool, device=out.device,
    )
    for _ in range(max_new_tokens):
        # NO-CACHE REFERENCE STEP: the entire sequence (prompt + all
        # tokens generated so far) goes through the model again --
        # no cache, start_pos always 0. Deliberately wasteful; this
        # is the ground truth the cached path is checked against.
        logits, _ = _forward(model, out, kv_caches=None, start_pos=0)
        next_token = sample_next(logits[:, -1], **sample_kwargs)  # [B, 1]
        out = torch.cat([out, next_token], dim=1)  # [B, T + 1]
        if eos_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_id)
            if bool(finished.all()):
                break
    return out
