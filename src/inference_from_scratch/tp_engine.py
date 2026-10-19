"""Real (non-HTTP) continuous-batching engine core on the batched TP path.

``TPEngineCore`` manages fixed slots over ONE batched
:class:`~inference_from_scratch.tp_distributed.TPStaticState`
(``batch_size == max_slots``). Requests arrive via :meth:`submit`,
are admitted FIFO into free slots, prefilled chunkwise, then decoded
with one batched :meth:`DistributedHybridLM.decode_token_static` call
per engine step for all decoding rows. Greedy by default; per-request
sampled decoding (temperature/top_p/top_k/seed) is supported on the
eager path (see :meth:`submit`).

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
slot first -- the same submission/step interleaving always yields the
same token ids. Sampled requests carry their own CPU generator seeded
from their per-request ``seed`` (or, when unseeded, deterministically
from the submission sequence, which is identical on every TP rank
because all ranks submit the same requests in the same order). Every
rank holds the full vocab logits (``_lm_head_logits`` all-gathers the
vocab shards), so identical generators over identical logits make
identical sampling decisions on every rank.

Sampled decode never replays the CUDA graph: the graph path reduces
its static output buffer with ``argmax`` only, so any step containing
a sampled row falls back to the eager path, which samples from the
returned logits. Greedy-only steps keep the graph path unchanged.

Graphed decode (``decode_mode="graph"``; the default ``None`` resolves
to ``"graph"`` on a CUDA context and ``"eager"`` elsewhere, because the
round-1 measurement was exact and faster at c1/c4/c8 with no shape
regression -- see ``benchmarks/harness/throughput_round1.json``.
Explicit ``"eager"`` preserves the exact v1 behaviour): on steps where every occupied slot
is a decode-planned row, the batched decode is ONE replay of a CUDA
graph captured over :meth:`DistributedHybridLM.decode_token_static`
with static input/output buffers (the runner's B=1 pattern in
``scripts/run_qwen38_tp2.py:greedy_generate_graph_tp``, lifted to the
fixed ``max_slots`` batch). Steps with any prefilling/just-finished
row fall back to the eager snapshot/restore step, token-identically.
Free slots are replayed too; their state is discarded by the
``reset(rows=[slot])`` at next admission, which is the only reason
skipping snapshot/restore on the graph path is exact. Capture is
lazy (first eligible step), after a side-stream warmup and a TP
barrier, mirroring the runner. By default (``graph_capture_mode=
"real-state"``) the warmup runs on the real state with an exact
snapshot/restore of occupied rows only -- no second full state is
allocated -- and a sizing guard (``graph_capture_guard``) falls back
to token-identical eager decode instead of capturing into OOM;
``graph_capture_mode="scratch"`` restores the legacy full-scratch
warmup. Eager steps likewise snapshot only live non-planned rows and
reset stateless (free / not-yet-prefilled) rows afterwards. Graph replay skips the
in-kernel overflow check, so the engine relies on the ``submit``
bound (``prompt + max_new - 1 <= max_seq_len``) plus finishing at
``max_new``/EOS -- a decoding row can never be fed past it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .sampling import sample_next
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
    ignore_eos: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    generator: torch.Generator | None = field(default=None, repr=False)

    @property
    def is_sampled(self) -> bool:
        return self.temperature != 0.0
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

    DECODE_MODES = ("eager", "graph")

    GRAPH_CAPTURE_MODES = ("real-state", "scratch")

    def __init__(self, model, ctx, max_slots: int, max_seq_len: int,
                 token_budget: int, *, eos_id: int | None = None,
                 decode_mode: str | None = None,
                 graph_capture_mode: str = "real-state",
                 graph_capture_guard: bool = True,
                 graph_capture_reserve_bytes: int = 512 * 1024 * 1024):
        if isinstance(max_slots, bool) or int(max_slots) <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots!r}")
        if isinstance(max_seq_len, bool) or int(max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len!r}")
        if isinstance(token_budget, bool) or int(token_budget) <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget!r}")
        if decode_mode is not None and decode_mode not in self.DECODE_MODES:
            raise ValueError(
                f"decode_mode must be one of {self.DECODE_MODES}, got {decode_mode!r}")
        if graph_capture_mode not in self.GRAPH_CAPTURE_MODES:
            raise ValueError(
                f"graph_capture_mode must be one of {self.GRAPH_CAPTURE_MODES}, "
                f"got {graph_capture_mode!r}")
        if int(graph_capture_reserve_bytes) < 0:
            raise ValueError("graph_capture_reserve_bytes must be >= 0")
        self.graph_capture_mode = str(graph_capture_mode)
        self.graph_capture_guard = bool(graph_capture_guard)
        self.graph_capture_reserve_bytes = int(graph_capture_reserve_bytes)
        self.graph_capture_blocked = False
        self.graph_capture_block_reason: str | None = None
        self.model = model
        self.ctx = ctx if ctx is not None else model.ctx
        if decode_mode is None:
            dev = getattr(self.ctx, "device", None)
            decode_mode = ("graph" if dev is not None
                           and torch.device(dev).type == "cuda"
                           and torch.cuda.is_available() else "eager")
        if decode_mode == "graph":
            dev = getattr(self.ctx, "device", None)
            if dev is None or torch.device(dev).type != "cuda" \
                    or not torch.cuda.is_available():
                raise ValueError(
                    "decode_mode='graph' requires a CUDA TP context, "
                    f"got device {dev!r}; use decode_mode='eager' there")
        self.decode_mode = str(decode_mode)
        self._graph = None
        self._graph_input: torch.Tensor | None = None
        self._graph_out: torch.Tensor | None = None
        self.graph_captures = 0
        self.graph_replays = 0
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
    def submit(self, prompt_ids, max_new: int, sampling: str = "greedy",
               ignore_eos: bool = False, temperature: float = 0.0,
               top_p: float = 1.0, top_k: int | None = None,
               seed: int | None = None) -> int:
        """Enqueue a request; returns its integer request_id.

        Default (``temperature=0``) is the exact greedy path. With
        ``temperature > 0`` the request is sampled per token with
        :func:`~inference_from_scratch.sampling.sample_next` applied to
        the full logits (temperature, then top-k, then top-p), using a
        per-request generator seeded by ``seed``; unseeded requests
        derive their seed from the submission sequence so TP ranks,
        which submit identically, still agree. ``top_p``/``top_k``/
        ``seed`` with ``temperature=0`` are validated but do not alter
        the greedy result.
        """
        if sampling != "greedy":
            raise ValueError(f"only greedy sampling in v1, got {sampling!r}")
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            raise ValueError(
                f"temperature must be a number, got {temperature!r}") from None
        if temperature != temperature or temperature in (float("inf"), float("-inf")) \
                or temperature < 0:
            raise ValueError(f"temperature must be finite and >= 0, got {temperature!r}")
        try:
            top_p = float(top_p)
        except (TypeError, ValueError):
            raise ValueError(f"top_p must be a number, got {top_p!r}") from None
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p!r}")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"top_k must be an int >= 1 or None, got {top_k!r}")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) \
                    or not (0 <= seed < 2 ** 63):
                raise ValueError(f"seed must be an int in [0, 2**63) or None, got {seed!r}")
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
        effective_seed = (int(seed) if seed is not None
                          else (0x5EED + self._submit_seq) % (2 ** 63))
        generator = (torch.Generator().manual_seed(effective_seed)
                     if temperature != 0.0 else None)
        req = EngineRequest(request_id=self._next_id, prompt_ids=ids,
                            max_new=int(max_new), arrival_step=self.step_count,
                            submit_seq=self._submit_seq,
                            ignore_eos=bool(ignore_eos),
                            temperature=float(temperature), top_p=float(top_p),
                            top_k=top_k, seed=seed, generator=generator)
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

    def _state_bytes(self, rows: list[int] | None = None) -> int:
        """Bytes of state buffers (all rows, or only ``rows``)."""
        total = 0
        for name in ("kv_k", "kv_v", "linear_S", "linear_conv_buffers"):
            for t in getattr(self.state, name):
                if t is None:
                    continue
                n = t.numel() if rows is None else t[rows].numel() if rows else 0
                total += int(n) * int(t.element_size())
        pos = self.state.position
        total += (pos.numel() if rows is None else len(rows)) * pos.element_size()
        return int(total)

    def _eager_protected_rows(self, decode_planned):
        """Split non-planned rows into (snapshot_rows, reset_rows).

        A batched eager decode mutates every row. Rows with live state
        that this step must not advance (partially prefilled rows, or
        decoding rows excluded by the token budget) are snapshotted and
        restored. Free rows and prefilling rows with ``prefill_done ==
        0`` hold no live state -- admission reset them and every prior
        eager step either restored or reset them -- so resetting them
        after the step is exact and avoids cloning up to a full state
        per mixed step (the c16 OOM mechanism).
        """
        planned = {id(r) for r in decode_planned}
        snap_rows, reset_rows = [], []
        for i, r in enumerate(self._slots):
            if r is None:
                reset_rows.append(i)
            elif id(r) in planned:
                continue
            elif r.status == "prefilling" and r.prefill_done == 0:
                reset_rows.append(i)
            else:
                snap_rows.append(i)
        return snap_rows, reset_rows

    def _finish(self, req: EngineRequest, step: int, finished: list[int]) -> None:
        req.status = "finished"
        req.finish_step = step
        if req.slot is not None:
            self._slots[req.slot] = None  # row reset deferred to next admission
        finished.append(req.request_id)

    def _pick_token(self, logits_row: torch.Tensor, req: EngineRequest) -> int:
        """Next token for one row: exact argmax when greedy, else sampled.

        ``logits_row`` is ``[1, V]`` (or ``[V]``) full-vocab logits,
        identical on every TP rank; the per-request generator is in the
        same state on every rank, so the sampled id agrees everywhere.
        """
        if not req.is_sampled:
            return int(logits_row.float().argmax(dim=-1).reshape(-1)[0].item())
        out = sample_next(logits_row.float(), temperature=req.temperature,
                          top_k=req.top_k, top_p=req.top_p,
                          generator=req.generator)
        return int(out.reshape(-1)[0].item())

    def _is_eos(self, token: int, req: EngineRequest | None = None) -> bool:
        if req is not None and req.ignore_eos:
            return False
        return self.eos_id is not None and int(token) == self.eos_id

    # -- opt-in graphed decode -------------------------------------------
    def _graph_eligible(self, decode_planned) -> bool:
        """True when a replay cannot touch a live non-decoding row.

        Every occupied slot must be decode-planned; free slots may be
        replayed because admission resets them before any use.
        """
        if self.decode_mode != "graph" or not decode_planned:
            return False
        if any(r.is_sampled for r in decode_planned):
            return False  # graph output is reduced with argmax only
        planned = {id(r) for r in decode_planned}
        return all(r is None or id(r) in planned for r in self._slots)

    def _graph_capture_plan(self) -> dict:
        """Memory plan for capture: no second full state in real-state mode."""
        occupied = [i for i, r in enumerate(self._slots) if r is not None]
        vocab = int(self.model.config.vocab_size)
        out_bytes = self.max_slots * vocab * int(self.state.dtype.itemsize) \
            if hasattr(self.state.dtype, "itemsize") else \
            self.max_slots * vocab * torch.tensor(0, dtype=self.state.dtype).element_size()
        if self.graph_capture_mode == "scratch":
            transient = self._state_bytes()
            snap_rows: list[int] = []
        else:
            transient = self._state_bytes(occupied)
            snap_rows = occupied
        # CUDA-graph private pools observed at 0.78-1.51 GiB scale with
        # state size; budget a third of the full state for the pool so
        # the guard does not under-count capture (activations + pool).
        pool_est = max(int(self._state_bytes()) // 3, int(out_bytes))
        return {"mode": self.graph_capture_mode, "occupied_rows": occupied,
                "snapshot_rows": snap_rows, "scratch_bytes": (
                    transient if self.graph_capture_mode == "scratch" else 0),
                "transient_bytes": int(transient) + int(out_bytes),
                "graph_pool_estimate_bytes": int(pool_est),
                "graph_out_bytes": int(out_bytes)}

    def _graph_capture_allowed(self, plan: dict | None = None) -> bool:
        """Fail-closed sizing guard: block capture (eager fallback) pre-OOM."""
        if self.graph_capture_blocked:
            return False
        if not self.graph_capture_guard:
            return True
        plan = plan or self._graph_capture_plan()
        try:
            free, _total = torch.cuda.mem_get_info(self.ctx.device)
        except Exception:
            # Fail closed: no memory oracle -> do not capture blind.
            self.graph_capture_blocked = True
            self.graph_capture_block_reason = (
                "graph capture blocked: CUDA free-memory query failed; "
                "falling back to eager decode")
            return False
        # Agree on the decision across TP ranks (min free): one rank
        # capturing while another falls back would deadlock the
        # capture barrier while both still step in lockstep.
        if int(getattr(self.ctx, "world_size", 1)) > 1 \
                and hasattr(self.ctx, "all_gather"):
            try:
                vals = self.ctx.all_gather(torch.tensor(
                    [int(free)], dtype=torch.long, device=self.state.device))
                free = min(int(v.reshape(-1)[0].item()) for v in vals)
            except Exception:
                self.graph_capture_blocked = True
                self.graph_capture_block_reason = (
                    "graph capture blocked: TP free-memory agreement "
                    "failed; falling back to eager decode")
                return False
        needed = (int(plan["transient_bytes"])
                  + int(plan.get("graph_pool_estimate_bytes", 0))
                  + self.graph_capture_reserve_bytes)
        if int(free) < needed:
            self.graph_capture_blocked = True
            self.graph_capture_block_reason = (
                f"graph capture needs ~{needed} bytes transient+reserve "
                f"but only {int(free)} free; falling back to eager decode")
            return False
        return True

    def _warmup_on_real_state(self, inp: torch.Tensor, runs: int = 3) -> None:
        """Warm kernels on the real state, then restore it exactly.

        Occupied rows are snapshot/restored (only those rows are
        cloned -- never a second full state); free rows are reset.
        Raises ValueError before mutating when a warmup run would
        overflow an occupied row, so the caller can fall back eager.
        """
        occupied = [i for i, r in enumerate(self._slots) if r is not None]
        if occupied:
            positions = self.state.position.tolist()
            if max(int(positions[i]) for i in occupied) + runs > self.max_seq_len:
                raise ValueError("graph warmup would overflow an occupied row")
        snap = self._snapshot_rows(occupied)
        free_rows = [i for i, r in enumerate(self._slots) if r is None]
        warm_in = inp.clone()
        dev = self.ctx.device
        try:
            if torch.device(dev).type == "cuda" and torch.cuda.is_available():
                side = torch.cuda.Stream(device=dev)
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):
                    for _ in range(runs):
                        self.model.decode_token_static(warm_in, self.state)
                torch.cuda.current_stream(dev).wait_stream(side)
            else:
                for _ in range(runs):
                    self.model.decode_token_static(warm_in, self.state)
        finally:
            self._restore_rows(snap)
            if free_rows:
                self.state.reset(rows=free_rows)

    def _capture_graph(self, inp: torch.Tensor) -> None:
        dev = self.ctx.device
        if self.graph_capture_mode == "scratch":
            scratch = self.model.static_state(
                batch_size=self.max_slots, max_seq_len=self.max_seq_len)
            warm_in = inp.clone()
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.model.decode_token_static(warm_in, scratch)
            torch.cuda.current_stream(dev).wait_stream(side)
            del scratch
        else:
            self._warmup_on_real_state(inp)
        if int(getattr(self.ctx, "world_size", 1)) > 1:
            self.ctx.barrier()
        self._graph_input = inp.clone()
        self._graph_out = torch.empty(
            self.max_slots, int(self.model.config.vocab_size),
            device=dev, dtype=self.state.dtype)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.model.decode_token_static(
                self._graph_input, self.state, out=self._graph_out)
        torch.cuda.synchronize(dev)
        if int(getattr(self.ctx, "world_size", 1)) > 1:
            self.ctx.barrier()
        self._graph = graph
        self.graph_captures += 1

    def _decode_graphed(self, inp: torch.Tensor) -> list[int] | None:
        """Replay the graph; ``None`` means capture was blocked -> eager."""
        # Free rows are replayed with dummy input; keep their positions
        # inside the buffers (admission resets them before real use).
        positions = self.state.position.tolist()
        stale = [i for i, r in enumerate(self._slots)
                 if r is None and int(positions[i]) >= self.max_seq_len - 1]
        if stale:
            self.state.reset(rows=stale)
        if self._graph is None:
            if not self._graph_capture_allowed():
                return None
            try:
                self._capture_graph(inp)
            except (ValueError, RuntimeError) as exc:
                # ValueError: warmup unsafe at current positions.
                # RuntimeError includes capture-time CUDA OOM: without
                # this catch it would fail every active request and
                # risk wedging TP ranks. Fail closed to eager instead.
                self.graph_capture_blocked = True
                self.graph_capture_block_reason = (
                    f"graph capture failed ({type(exc).__name__}); "
                    "falling back to eager decode")
                self._graph = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return None
        assert self._graph is not None and self._graph_input is not None
        assert self._graph_out is not None
        self._graph_input.copy_(inp)
        self._graph.replay()
        self.graph_replays += 1
        return self._graph_out.float().argmax(dim=-1).tolist()

    def close(self) -> None:
        """Drop the captured graph (sync first; NCCL teardown safety)."""
        if self._graph is not None:
            dev = getattr(self.ctx, "device", None)
            if dev is not None and torch.device(dev).type == "cuda":
                torch.cuda.synchronize(dev)
            del self._graph
            self._graph = None
            if dev is not None and torch.device(dev).type == "cuda":
                torch.cuda.synchronize(dev)

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
                        token = self._pick_token(logits, req)
                        req.generated.append(token)
                        req.first_token_step = step
                        first_tokens.append(req.request_id)
                        if len(req.generated) >= req.max_new or self._is_eos(token, req):
                            self._finish(req, step, finished)
                        else:
                            req.status = "decoding"

            # (c) One batched decode step for all (budget-planned) decoding rows.
            if decode_planned:
                inp = torch.zeros(self.max_slots, dtype=torch.long, device=self.ctx.device)
                for r in decode_planned:
                    inp[int(r.slot)] = int(r.generated[-1])
                nxt: list[int] | None = None
                if self._graph_eligible(decode_planned):
                    nxt = self._decode_graphed(inp)
                if nxt is None:
                    # Eager (also the fail-closed fallback when graph
                    # capture is blocked): snapshot only live
                    # non-planned rows, reset stateless rows after.
                    snap_rows, reset_rows = self._eager_protected_rows(
                        decode_planned)
                    snap = self._snapshot_rows(snap_rows)
                    logits = self.model.decode_token_static(inp, self.state)
                    self._restore_rows(snap)
                    if reset_rows:
                        self.state.reset(rows=reset_rows)
                for req in decode_planned:
                    token = (int(nxt[int(req.slot)]) if nxt is not None
                             else self._pick_token(
                                 logits[int(req.slot):int(req.slot) + 1], req))
                    req.generated.append(token)
                    if len(req.generated) >= req.max_new or self._is_eos(token, req):
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
