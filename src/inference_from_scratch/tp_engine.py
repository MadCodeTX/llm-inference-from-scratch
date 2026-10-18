"""Real (non-HTTP) continuous-batching engine core on the batched TP path.

``TPEngineCore`` manages fixed slots over ONE batched
:class:`~inference_from_scratch.tp_distributed.TPStaticState`
(``batch_size == max_slots``). Requests arrive via :meth:`submit`,
are admitted FIFO into free slots, prefilled chunkwise, then decoded
with one batched :meth:`DistributedHybridLM.decode_token_static` call
per engine step for all decoding rows. Greedy sampling only (v1).

Bookkeeping note: ``serving.ChunkedPrefillScheduler`` is a pure
token-count simulation -- its ``step()`` invents generated tokens
without a model, has no slot limit, and cannot observe real EOS from
logits. Importing it as the engine's bookkeeping would double-track
state that can diverge from the model. This module therefore mirrors
its minimal accounting concepts (decode-first budget reservation,
FIFO prefill chunks, ``ttft_steps``/``finish_step`` step accounting,
per-step token totals) in slot-local records, and says so here rather
than pretending the simulation drives the model.

Prefix-cache seam: admission currently always ``reset(rows=[slot])``
and prefills the whole prompt chunkwise. A prefix cache would hook
exactly there -- look up the prompt, restore/skip the cached prefix,
and start ``prefill_done`` at the cached length (see
``tp_prefix_cache.py``, which requires uniform positions and so must
be applied per-row at admission, before any divergent decode). No
prefix integration is done in v1.

Determinism: FIFO queues ordered by submission sequence, lowest free
slot first, greedy argmax only, no host randomness -- the same
submission/step interleaving always yields the same token ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .tp_distributed import TPStaticState

__all__ = ["TPEngineCore", "EngineRequest", "EngineStepRecord"]


@dataclass
class EngineRequest:
    request_id: int
    prompt_ids: list[int]
    max_new: int
    arrival_step: int
    submit_seq: int
    generated: list[int] = field(default_factory=list)
    status: str = "waiting"  # waiting|prefilling|decoding|finished
    slot: int | None = None
    prefill_done: int = 0
    first_token_step: int | None = None
    finish_step: int | None = None

    @property
    def ttft_steps(self) -> int | None:
        if self.first_token_step is None:
            return None
        return self.first_token_step - self.arrival_step + 1


@dataclass(frozen=True)
class EngineStepRecord:
    step: int
    admitted_request_ids: tuple[int, ...]
    prefill_tokens: int
    decode_tokens: int
    first_token_request_ids: tuple[int, ...]
    finished_request_ids: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens


class TPEngineCore:
    """Continuous-batching engine over one batched TP static state.

    ``model`` is a ``DistributedHybridLM``; ``ctx`` its TP context
    (kept for symmetry with the model constructors and validated
    against ``model.ctx`` device when both are present). All ranks
    must submit the same requests in the same order and call
    :meth:`step` in lockstep (collectives inside the model require it).
    """

    def __init__(self, model, ctx, max_slots: int, max_seq_len: int,
                 token_budget: int, *, eos_id: int | None = None):
        if isinstance(max_slots, bool) or int(max_slots) <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots!r}")
        if isinstance(max_seq_len, bool) or int(max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len!r}")
        if isinstance(token_budget, bool) or int(token_budget) <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget!r}")
        self.model = model
        self.ctx = ctx if ctx is not None else model.ctx
        self.max_slots = int(max_slots)
        self.max_seq_len = int(max_seq_len)
        self.token_budget = int(token_budget)
        self.eos_id = None if eos_id is None else int(eos_id)
        self.state: TPStaticState = model.static_state(
            batch_size=self.max_slots, max_seq_len=self.max_seq_len)
        self.step_count = 0
        self.history: list[EngineStepRecord] = []
        self._next_id = 0
        self._submit_seq = 0
        self._requests: dict[int, EngineRequest] = {}
        self._waiting: list[EngineRequest] = []
        self._slots: list[EngineRequest | None] = [None] * self.max_slots

    # -- submission / results -------------------------------------------
    def submit(self, prompt_ids, max_new: int, sampling: str = "greedy") -> int:
        """Enqueue a greedy request; returns its integer request_id."""
        if sampling != "greedy":
            raise ValueError(f"only greedy sampling in v1, got {sampling!r}")
        if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 0:
            raise ValueError(f"max_new must be a non-negative int, got {max_new!r}")
        if isinstance(prompt_ids, torch.Tensor):
            ids = [int(v) for v in prompt_ids.reshape(-1).tolist()]
        else:
            ids = [int(v) for v in list(prompt_ids)]
        if not ids:
            raise ValueError("prompt_ids must be non-empty")
        vocab = int(getattr(self.model.config, "vocab_size", 0) or 0)
        if vocab and any(t < 0 or t >= vocab for t in ids):
            raise ValueError(f"prompt token ids must be in [0, {vocab})")
        need = len(ids) + max(int(max_new) - 1, 0)
        if need > self.max_seq_len:
            raise ValueError(
                f"prompt ({len(ids)}) + generated ({max_new}) needs {need} "
                f"positions > max_seq_len={self.max_seq_len}")
        req = EngineRequest(request_id=self._next_id, prompt_ids=ids,
                            max_new=int(max_new), arrival_step=self.step_count,
                            submit_seq=self._submit_seq)
        self._next_id += 1
        self._submit_seq += 1
        self._requests[req.request_id] = req
        self._waiting.append(req)
        return req.request_id

    def poll(self, request_id: int) -> dict:
        """Result snapshot: ``{status, token_ids, ttft_steps, finish_step}``."""
        req = self._requests[request_id]  # KeyError for unknown ids, by design
        return {"status": req.status, "token_ids": list(req.generated),
                "ttft_steps": req.ttft_steps, "finish_step": req.finish_step}

    @property
    def has_work(self) -> bool:
        return bool(self._waiting) or any(s is not None for s in self._slots)

    # -- internals --------------------------------------------------------
    def _row_state(self, slot: int) -> TPStaticState:
        """B=1 view (sliced buffers/position) of one slot's row."""
        st, src = object.__new__(TPStaticState), self.state
        st.config, st.batch_size, st.batch = src.config, 1, 1
        st.max_seq_len, st.device, st.dtype = src.max_seq_len, src.device, src.dtype
        st.ctx, st.layer_types = src.ctx, list(src.layer_types)
        st.hkv_local, st.vh_per, st.conv_dim_local = src.hkv_local, src.vh_per, src.conv_dim_local
        sl = slice(slot, slot + 1)
        st.kv_k = [t[sl] if t is not None else None for t in src.kv_k]
        st.kv_v = [t[sl] if t is not None else None for t in src.kv_v]
        st.linear_S = [t[sl] if t is not None else None for t in src.linear_S]
        st.linear_conv_buffers = [t[sl] if t is not None else None for t in src.linear_conv_buffers]
        st.position = src.position[sl]
        st.position_tensor = st.position
        return st

    def _snapshot_rows(self, rows: list[int]):
        if not rows:
            return None
        idx = torch.tensor(rows, dtype=torch.long, device=self.state.device)
        snap = {"idx": idx, "position": self.state.position[idx].clone()}
        for name in ("kv_k", "kv_v", "linear_S", "linear_conv_buffers"):
            snap[name] = [t[idx].clone() if t is not None else None
                          for t in getattr(self.state, name)]
        return snap

    def _restore_rows(self, snap) -> None:
        if snap is None:
            return
        idx = snap["idx"]
        for name in ("kv_k", "kv_v", "linear_S", "linear_conv_buffers"):
            for t, saved in zip(getattr(self.state, name), snap[name]):
                if t is not None and saved is not None:
                    t[idx] = saved
        self.state.position[idx] = snap["position"]

    def _finish(self, req: EngineRequest, step: int, finished: list[int]) -> None:
        req.status = "finished"
        req.finish_step = step
        if req.slot is not None:
            self._slots[req.slot] = None  # row reset deferred to next admission
        finished.append(req.request_id)

    def _is_eos(self, token: int) -> bool:
        return self.eos_id is not None and int(token) == self.eos_id

    # -- main loop ----------------------------------------------------------
    def step(self) -> EngineStepRecord:
        """Advance one engine step: admit, prefill (budgeted), decode once."""
        step = self.step_count
        admitted: list[int] = []
        first_tokens: list[int] = []
        finished: list[int] = []

        # (a) Admit waiting FIFO into free slots (lowest slot first), resetting the row now.
        free = [i for i, s in enumerate(self._slots) if s is None]
        while self._waiting and free:
            req, slot = self._waiting.pop(0), free.pop(0)
            self.state.reset(rows=[slot])
            req.slot, req.status, req.prefill_done = slot, "prefilling", 0
            self._slots[slot] = req
            admitted.append(req.request_id)

        # Budget: decode-first reservation (mirrors ChunkedPrefillScheduler).
        decoding = sorted((r for r in self._slots if r is not None and r.status == "decoding"),
                          key=lambda r: r.submit_seq)
        decode_planned = decoding[: self.token_budget]
        decode_tokens = len(decode_planned)
        remaining = self.token_budget - decode_tokens

        # (b) Chunkwise prefill FIFO, one row-sliced call per chunk.
        prefilling = sorted((r for r in self._slots if r is not None and r.status == "prefilling"),
                            key=lambda r: r.submit_seq)
        prefill_tokens = 0
        with torch.no_grad():
            for req in prefilling:
                if req.status != "prefilling":
                    continue
                left = len(req.prompt_ids) - req.prefill_done
                if left <= 0:
                    raise RuntimeError("prefilling request with no prefill left")
                if remaining <= 0:
                    break
                else:
                    chunk = min(left, remaining)
                    idx = torch.tensor([req.prompt_ids[req.prefill_done: req.prefill_done + chunk]],
                                       dtype=torch.long, device=self.ctx.device)
                    logits = self.model.prefill_chunkwise_static(
                        idx, self._row_state(int(req.slot)), chunk_size=chunk)
                    req.prefill_done += chunk
                    prefill_tokens += chunk
                    remaining -= chunk
                if req.prefill_done >= len(req.prompt_ids):
                    if int(req.max_new) == 0:
                        self._finish(req, step, finished)
                    else:
                        token = int(logits.float().argmax(dim=-1).reshape(-1)[0].item())
                        req.generated.append(token)
                        req.first_token_step = step
                        first_tokens.append(req.request_id)
                        if len(req.generated) >= req.max_new or self._is_eos(token):
                            self._finish(req, step, finished)
                        else:
                            req.status = "decoding"

            # (c) One batched decode step for all (budget-planned) decoding rows.
            if decode_planned:
                planned = {id(r) for r in decode_planned}
                protected = [i for i, r in enumerate(self._slots)
                             if r is None or id(r) not in planned]
                # Finished-this-step slots are already None in _slots but their
                # requests left the slot; every non-planned row is protected.
                snap = self._snapshot_rows(protected)
                inp = torch.zeros(self.max_slots, dtype=torch.long, device=self.ctx.device)
                for r in decode_planned:
                    inp[int(r.slot)] = int(r.generated[-1])
                logits = self.model.decode_token_static(inp, self.state)
                self._restore_rows(snap)
                nxt = logits.float().argmax(dim=-1).tolist()
                for req in decode_planned:
                    token = int(nxt[int(req.slot)])
                    req.generated.append(token)
                    if len(req.generated) >= req.max_new or self._is_eos(token):
                        self._finish(req, step, finished)
        record = EngineStepRecord(step=step, admitted_request_ids=tuple(admitted),
                                  prefill_tokens=prefill_tokens, decode_tokens=decode_tokens,
                                  first_token_request_ids=tuple(first_tokens),
                                  finished_request_ids=tuple(finished))
        assert record.total_tokens <= self.token_budget, record
        self.history.append(record)
        self.step_count += 1
        return record

    def run(self, max_steps: int = 10_000) -> list[int]:
        """Step until no work remains; returns finished request ids in poll order."""
        steps = 0
        while self.has_work:
            if steps >= max_steps:
                raise RuntimeError(f"TPEngineCore.run exceeded max_steps={max_steps}")
            self.step()
            steps += 1
        return sorted(self._requests, key=lambda rid: (self._requests[rid].finish_step or 0, rid))
