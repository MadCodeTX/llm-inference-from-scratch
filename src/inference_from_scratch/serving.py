"""Ch 11 -- Serving: continuous batching + paged KV (dissected).

Why this exists in inference
----------------------------
``generate.py`` (Ch 08) runs one fixed batch to completion: every
row waits for the slowest row, and a new request waits for the whole
batch. A serving engine cannot do that -- requests arrive
continuously, finish at different steps, and must join the running
batch the moment a slot frees. vLLM's answer is *continuous
(iteration-level) batching*: the scheduler owns the loop, and at
**every decode step** it retires finished requests and admits
waiting ones. This module dissects that scheduler with a
model-independent sampler so the scheduling itself -- not the model
-- is what is tested.

The second dissection is memory. Ch 05's caches reserve one
contiguous ``[B, H_kv, T_max, D_h]`` slab per request; real serving
carves the KV cache into fixed-size *blocks* handed out on demand
and tracked per request in a block table (PagedAttention).
:class:`BlockAllocator` is that allocator stripped to its essence:
blocks are interchangeable, so a request's blocks need not be
contiguous -- which is exactly why paging eliminates the external
fragmentation that contiguous slabs suffer (demonstrated in the
tests).

Latency vocabulary (measured, not asserted)
-------------------------------------------
TTFT (time to first token) is prefill + first-token sampling: it
grows with prompt length and is what a user perceives as "the model
started answering". TPOT (time per output token) is the decode
step time after the first token: it grows with batch and cache
length and sets the reading speed of the stream. Throughput
(tokens/second, usually aggregated over all concurrent requests)
trades against both: bigger batches raise throughput and TPOT
together. :func:`measure_prefill_decode` measures all three with
``time.perf_counter`` around an explicit prefill-then-decode loop
(mirroring ``generate.py``); :func:`compute_tpot` /
:func:`compute_throughput` are the bare definitions for timings
measured elsewhere.

How production differs
----------------------
vLLM adds preemption (recompute/swap when blocks run out), chunked
prefill (long prompts sliced so decode steps are not starved),
prefix caching (shared prompt blocks), and CUDA-graph decode. The
loop shape -- admit, step everyone once, retire, repeat -- and the
block-table allocator are unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .kv_cache import KVCache
from .sampling import sample_next


# ---------------------------------------------------------------------------
# Requests and continuous (iteration-level) batching
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """One generation request tracked by the scheduler.

    ``prompt_ids`` are the prompt token ids (already tokenized);
    ``generated`` accumulates output ids one per scheduler step and
    starts empty. ``finished`` is a *derived* property, never stored:
    a request is finished once it has generated ``max_new_tokens``
    tokens, or -- when ``eos_id`` is set -- once it has generated the
    EOS token (which stays in ``generated``, matching
    ``generate.py``'s convention). ``request_id`` is an optional
    caller label for logging; the scheduler never interprets it.
    """

    prompt_ids: list[int]
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)
    eos_id: int | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be >= 0, got {self.max_new_tokens}"
            )

    @property
    def finished(self) -> bool:
        """True when no more steps should be spent on this request."""
        if len(self.generated) >= self.max_new_tokens:
            return True
        return (
            self.eos_id is not None
            and len(self.generated) > 0
            and self.generated[-1] == self.eos_id
        )

    @property
    def total_len(self) -> int:
        """Prompt + generated length (the KV-cache footprint in tokens)."""
        return len(self.prompt_ids) + len(self.generated)


# A sampler maps a request (plus the scheduler step counter) to the
# next token id. Model-independent on purpose: tests drive the
# scheduler with counting/rule samplers, production would run the
# model over the running batch and sample per row.
SamplerFn = Callable[..., int]


def _default_next_token(request: Request) -> int:
    """Deterministic stand-in sampler: last token + 1 (mod 256).

    Used when ``step`` gets no sampler -- a "simple rule" that makes
    the scheduler runnable (and testable) with no model at all, in
    the spirit of the contract's "sampler fn or simple rule".
    """
    last = request.generated[-1] if request.generated else (
        request.prompt_ids[-1] if request.prompt_ids else 0
    )
    return (int(last) + 1) % 256


class ContinuousBatcher:
    """Iteration-level scheduler: admit / step / retire every step.

    State: ``waiting`` (added, not yet running), ``running`` (being
    stepped), ``finished`` (retired, in completion order). Each
    :meth:`step` (1) admits waiting requests into free slots --
    including requests added *while others are mid-flight*, the
    defining continuous-batching behaviour; (2) generates exactly
    one token per running request via the sampler; (3) moves
    newly-finished requests to ``finished``. A request with
    ``max_new_tokens=0`` finishes on the first step that admits it,
    without consuming a sampler call.

    ``max_batch_size=None`` means unbounded (admit everything at
    once); a finite value caps concurrency, so admission order is
    FIFO and a long request never blocks short ones behind it beyond
    one step.
    """

    def __init__(
        self,
        max_batch_size: int | None = None,
        eos_id: int | None = None,
    ):
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive or None, got {max_batch_size}"
            )
        self.max_batch_size = max_batch_size
        self.eos_id = eos_id
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.step_count = 0

    def add(self, request: Request) -> Request:
        """Enqueue ``request``; it becomes runnable at the next step.

        If the batcher has a default ``eos_id`` and the request does
        not set its own, the batcher's is applied. Returns the
        request for chaining/logging convenience.
        """
        if request.eos_id is None and self.eos_id is not None:
            request.eos_id = self.eos_id
        self.waiting.append(request)
        return request

    @property
    def has_work(self) -> bool:
        """True while any request is waiting or running."""
        return bool(self.waiting or self.running)

    def _admit(self) -> None:
        """Move waiting requests into free running slots (FIFO)."""
        while self.waiting:
            if (
                self.max_batch_size is not None
                and len(self.running) >= self.max_batch_size
            ):
                break
            self.running.append(self.waiting.pop(0))

    def step(self, sampler: SamplerFn | None = None) -> int:
        """Run one scheduler iteration; returns tokens generated.

        ``sampler`` is called as ``sampler(request)``; callables
        accepting ``(request, step_count)`` also work (the extra
        argument is passed and a ``TypeError`` from a 1-arg callable
        falls back to the 1-arg call). With ``sampler=None`` the
        deterministic :func:`_default_next_token` rule is used.
        Finished requests are retired *within this step*, so their
        slots can be refilled by the *next* step's admission.
        """
        self._admit()
        self.step_count += 1
        n_generated = 0
        for request in self.running:
            if request.finished:  # e.g. max_new_tokens == 0
                continue
            if sampler is None:
                token = _default_next_token(request)
            else:
                try:
                    token = sampler(request, self.step_count)
                except TypeError:
                    token = sampler(request)
            request.generated.append(int(token))
            n_generated += 1
        still_running: list[Request] = []
        for request in self.running:
            if request.finished:
                self.finished.append(request)
            else:
                still_running.append(request)
        self.running = still_running
        return n_generated

    def run(
        self,
        sampler: SamplerFn | None = None,
        max_steps: int = 10_000,
    ) -> list[Request]:
        """Step until no work remains; returns ``self.finished``.

        ``max_steps`` is a runaway guard (a sampler that never lets a
        request finish would otherwise loop forever); exceeding it
        raises ``RuntimeError`` with the requests still tracked.
        """
        steps = 0
        while self.has_work:
            if steps >= max_steps:
                raise RuntimeError(
                    f"ContinuousBatcher.run exceeded max_steps={max_steps} "
                    f"with {len(self.running)} running / "
                    f"{len(self.waiting)} waiting requests unfinished"
                )
            self.step(sampler)
            steps += 1
        return self.finished


# ---------------------------------------------------------------------------
# Paged KV: block allocator (PagedAttention dissection)
# ---------------------------------------------------------------------------


class BlockAllocator:
    """Fixed-size block allocator over a paged KV pool.

    The pool is ``num_blocks`` blocks of ``block_size`` token slots
    each (one block holds the K (or K+V, accounting is the caller's)
    rows for ``block_size`` tokens of one request in one layer --
    the dissection keeps a single token-count dimension; bytes come
    from ``quantization.kv_cache_bytes``). :meth:`allocate` returns
    the block ids for a token count, or ``None`` -- allocating
    *nothing* -- when the pool cannot satisfy the request (atomic
    refusal: callers must wait, preempt, or reject, never receive a
    partial allocation). Ids need not be contiguous: that is the
    point of paging. :meth:`free` returns ids to the pool and raises
    on double-free / unknown ids, because silently accepting them
    would corrupt the free list in exactly the way production
    allocators must not.

    ``free_blocks`` / ``allocated_blocks`` count blocks; internal
    fragmentation (a partly-filled last block) is inherent to paging
    and bounded by ``block_size - 1`` tokens per request -- the
    :meth:`wasted_slots` helper makes it countable.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        # Free ids as a stack (LIFO reuse keeps recent blocks hot in
        # real systems; here it simply makes behaviour deterministic).
        self._free: list[int] = list(range(self.num_blocks - 1, -1, -1))
        self._allocated: set[int] = set()

    @property
    def free_blocks(self) -> int:
        """Number of blocks currently available for allocation."""
        return len(self._free)

    @property
    def allocated_blocks(self) -> int:
        """Number of blocks currently handed out."""
        return len(self._allocated)

    def blocks_for_tokens(self, n_tokens: int) -> int:
        """Blocks needed for ``n_tokens``: ``ceil(n / block_size)``."""
        if n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
        return -(-int(n_tokens) // self.block_size)  # ceil div

    def wasted_slots(self, n_tokens: int) -> int:
        """Token slots paid for but unused by ``n_tokens`` (internal
        fragmentation): ``blocks * block_size - n_tokens``."""
        return self.blocks_for_tokens(n_tokens) * self.block_size - int(n_tokens)

    def allocate(self, n_tokens: int) -> list[int] | None:
        """Allocate enough blocks for ``n_tokens`` tokens.

        Returns the block ids (order is allocation order, ids are
        *not* guaranteed contiguous), ``[]`` for ``n_tokens == 0``,
        or ``None`` -- with the pool untouched -- when fewer than the
        required blocks are free (over-allocation is refused, never
        partially satisfied).
        """
        needed = self.blocks_for_tokens(n_tokens)
        if needed == 0:
            return []
        if needed > len(self._free):
            return None
        ids = [self._free.pop() for _ in range(needed)]
        self._allocated.update(ids)
        return ids

    def free(self, block_ids: list[int]) -> None:
        """Return ``block_ids`` to the pool.

        Raises ``ValueError`` for an id that is out of range or not
        currently allocated (double free): both are caller bugs that
        a real allocator must surface, not absorb. Freeing is atomic
        -- validation happens before any id is returned.
        """
        ids = list(block_ids)
        for block_id in ids:
            if block_id not in self._allocated:
                raise ValueError(
                    f"cannot free block {block_id}: not currently allocated "
                    f"(unknown id or double free)"
                )
        for block_id in ids:
            self._allocated.discard(block_id)
            self._free.append(block_id)

    def is_allocated(self, block_id: int) -> bool:
        """True if ``block_id`` is currently handed out."""
        return block_id in self._allocated


# ---------------------------------------------------------------------------
# Latency measurement: TTFT / TPOT / throughput
# ---------------------------------------------------------------------------


def compute_tpot(decode_seconds: float, n_decode_tokens: int) -> float:
    """Time per output token: decode seconds / decode tokens.

    ``n_decode_tokens`` counts tokens generated *after* the first
    (the first token's cost is TTFT, not TPOT). Returns 0.0 when
    there is nothing to average (non-positive token count or
    non-positive elapsed time) rather than dividing by zero.
    """
    if n_decode_tokens <= 0 or decode_seconds <= 0:
        return 0.0
    return decode_seconds / n_decode_tokens


def compute_throughput(n_tokens: int, elapsed_seconds: float) -> float:
    """Tokens per second: ``n_tokens / elapsed_seconds`` (0.0 if the
    elapsed time is not positive, so smoke runs never divide by 0)."""
    if elapsed_seconds <= 0 or n_tokens <= 0:
        return 0.0
    return n_tokens / elapsed_seconds


def _forward(model, idx, kv_caches=None, start_pos=0):
    """Call ``model`` and normalise to ``(logits, kv_caches)``.

    Mirrors ``generate.py``'s duck-typing: the real ``DecoderLM``
    returns the pair; a stub returning bare logits is accepted.
    """
    out = model(idx, kv_caches=kv_caches, start_pos=start_pos)
    if isinstance(out, tuple):
        logits, caches = out
        return logits, caches
    return out, kv_caches


def _model_n_layer(model) -> int:
    """Layer count for the cache, duck-typed like ``generate.py``."""
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


def measure_prefill_decode(
    model,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float | int | list[int]]:
    """Measure TTFT / TPOT / throughput around prefill + decode.

    Runs the same explicit loop as ``generate.py`` (one prefill
    forward over the whole prompt, then one-token decode forwards
    against a :class:`KVCache`) with ``time.perf_counter`` brackets:

    - ``ttft_s``: prefill forward + first-token sampling;
    - ``decode_s`` / ``tpot_s``: all subsequent decode steps, and
      their per-token average (0.0 with fewer than 2 new tokens);
    - ``throughput_tok_s``: new tokens / total wall time;
    - ``total_s``, ``n_new_tokens``, ``token_ids`` (prompt + output).

    Single-sequence (batch=1) dissection helper: batch throughput in
    production aggregates over the whole running batch. Timing
    includes Python/sampling overhead -- at tiny CPU dims that
    overhead dominates, which the numbers will honestly show.
    """
    if isinstance(prompt_ids, torch.Tensor):
        idx = prompt_ids.detach().clone().to(torch.long)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
    else:
        idx = torch.tensor([list(prompt_ids)], dtype=torch.long)
    if idx.shape[0] != 1:
        raise ValueError(
            f"measure_prefill_decode is batch=1, got batch {idx.shape[0]}"
        )
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    sample_kwargs = dict(
        temperature=temperature, top_k=top_k, top_p=top_p, generator=generator,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if max_new_tokens == 0:
                return {
                    "ttft_s": 0.0, "decode_s": 0.0, "tpot_s": 0.0,
                    "throughput_tok_s": 0.0, "total_s": 0.0,
                    "n_new_tokens": 0,
                    "token_ids": idx.squeeze(0).tolist(),
                }
            kv_caches = KVCache(n_layer=_model_n_layer(model))
            out = idx.clone()

            # PREFILL (timed): whole prompt, start_pos=0 -> first token.
            t0 = time.perf_counter()
            logits, kv_caches = _forward(
                model, out, kv_caches=kv_caches, start_pos=0
            )
            next_token = sample_next(logits[:, -1], **sample_kwargs)
            t1 = time.perf_counter()
            out = torch.cat([out, next_token], dim=1)
            n_new = 1
            finished = eos_id is not None and int(next_token.item()) == eos_id

            # DECODE (timed): one token per forward until budget/EOS.
            decode_start = time.perf_counter()
            while n_new < max_new_tokens and not finished:
                start_pos = out.shape[1] - 1
                logits, kv_caches = _forward(
                    model, next_token, kv_caches=kv_caches,
                    start_pos=start_pos,
                )
                next_token = sample_next(logits[:, -1], **sample_kwargs)
                out = torch.cat([out, next_token], dim=1)
                n_new += 1
                if eos_id is not None and int(next_token.item()) == eos_id:
                    finished = True
            decode_s = time.perf_counter() - decode_start

            ttft_s = t1 - t0
            total_s = ttft_s + decode_s
            return {
                "ttft_s": ttft_s,
                "decode_s": decode_s,
                "tpot_s": compute_tpot(decode_s, n_new - 1),
                "throughput_tok_s": compute_throughput(n_new, total_s),
                "total_s": total_s,
                "n_new_tokens": n_new,
                "token_ids": out.squeeze(0).tolist(),
            }
    finally:
        if was_training:
            model.train()
