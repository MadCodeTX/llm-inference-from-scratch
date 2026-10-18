"""Ch 14 -- Speculative decoding: draft cheaply, verify exactly.

Framing follows Kiely, *Inference Engineering* (2026), sec. 5.2.1
(draft-target) with the wider family in sec. 5.2.

Why this exists
---------------
Greedy decode is serial: one target forward yields one token, and at
batch 1 that forward is a memory-bound pass over every target weight.
If a cheaper draft model can guess the next few tokens, the target can
check a whole guessed run in one parallel forward (the same shape of
work as a short prefill). Guesses that match the target's own greedy
choices are kept, the first disagreement is replaced by the target's
choice, and -- when every guess matches -- one extra "bonus" token
comes from the target logits just past the run. Each verification
forward therefore finalises at least one, and up to ``gamma + 1``,
tokens.

The invariant that makes this safe
----------------------------------
Nothing here is allowed to change the text. The returned sequence must
be *exactly* what greedy decoding with the target alone would have
produced (compare ``generate(..., temperature=0)``). The draft only
influences how much work each target forward does, never which tokens
survive: a proposal token survives only when it equals the target's
greedy token at that position. That is the greedy form of the general
rule -- under sampling, acceptance uses the target's probabilities so
the output law stays the target's.

Why this version uses no KV cache
---------------------------------
Drafting with a cache, then verifying, would leave rejected draft
tokens sitting in both caches. Removing them needs truncation or
rollback bookkeeping (easy with a preallocated cache sliced by length,
awkward with this repo's concatenating cache). To keep the correctness
argument one paragraph long, both models are called cache-free on the
full sequence each time. The forward-count saving is still real and
measurable (``target_forwards``); the constant factor is not the point
at this scale. Production systems keep caches and roll them back to
the accepted length.
"""

from __future__ import annotations

import time

import torch


def _forward_logits(model, idx):  # type: ignore[no-untyped-def]
    out = model(idx, kv_caches=None, start_pos=0)
    if isinstance(out, tuple):
        return out[0]
    return out


def _greedy_token(logits_last: torch.Tensor) -> torch.Tensor:
    """``[1, V] -> [1, 1]`` argmax token."""
    return torch.argmax(logits_last, dim=-1, keepdim=True)


def speculative_generate(
    target,
    draft,
    idx: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
):  # type: ignore[no-untyped-def]
    """Greedy speculative generation; returns ``(tokens, stats)``.

    Alias ``greedy_speculative_generate`` is the same function.

    ``target`` and ``draft`` are models callable as
    ``model(idx, kv_caches=None, start_pos=0) -> (logits, cache)``
    (bare-logits stubs are accepted) sharing a vocabulary. ``idx`` is
    ``[1, T_prompt]``; the return's first element is the full
    ``[1, T_prompt + max_new_tokens]`` sequence (prompt included,
    mirroring ``generate``), and the second is
    ``{"accepted": int, "drafted": int, "target_forwards": int}``:

    - ``drafted``: proposal tokens the draft produced;
    - ``accepted``: proposal tokens kept (the target agreed greedily);
    - ``target_forwards``: verification forwards run on the target.
      One forward can finalise several tokens, so with a good draft
      this is far smaller than ``max_new_tokens``; with a draft that
      always disagrees immediately it is one per token produced and
      the drafted work is overhead.

    Each round: the draft greedily proposes up to ``gamma`` tokens
    (cache-free full forwards), then exactly one target forward over
    prompt-plus-proposals verifies them. The longest agreeing prefix
    is kept and is followed by one target-chosen token (the correction
    at the first disagreement, or a bonus token when all agree). The
    result is truncated to the requested budget, and ``accepted``
    counts only proposal tokens actually appended.
    """
    if not isinstance(idx, torch.Tensor) or idx.dim() != 2:
        raise ValueError("idx must be a [B, T] tensor")
    if idx.shape[0] != 1:
        raise ValueError(
            f"speculative_generate is batch=1 in this dissection, got batch {idx.shape[0]}"
        )
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")

    stats = {"accepted": 0, "drafted": 0, "target_forwards": 0}
    out = idx.clone()
    if max_new_tokens == 0:
        return out, stats

    was_training_target = target.training
    was_training_draft = draft.training
    target.eval()
    draft.eval()
    try:
        with torch.no_grad():
            generated = 0
            while generated < max_new_tokens:
                remaining = max_new_tokens - generated
                n_draft = min(int(gamma), remaining)

                # --- draft: gamma greedy proposals, cache-free -------
                proposals: list[int] = []
                draft_seq = out
                for _ in range(n_draft):
                    d_logits = _forward_logits(draft, draft_seq)
                    tok = _greedy_token(d_logits[:, -1])  # [1, 1]
                    proposals.append(int(tok.item()))
                    draft_seq = torch.cat([draft_seq, tok], dim=1)
                stats["drafted"] += len(proposals)

                # --- target: ONE forward verifies the whole run -----
                candidate = draft_seq  # out + proposals
                t_logits = _forward_logits(target, candidate)
                stats["target_forwards"] += 1

                # Target greedy tokens for positions after ``out``:
                # logits at position len(out)-1 predict the first new
                # token, and each following position predicts the next.
                start = out.shape[1] - 1
                window = t_logits[0, start : start + n_draft + 1]  # [gamma+1, V]
                target_tokens = [
                    int(x) for x in torch.argmax(window, dim=-1).tolist()
                ]

                n_accept = 0
                for proposed, wanted in zip(proposals, target_tokens):
                    if proposed != wanted:
                        break
                    n_accept += 1

                # Kept proposals plus the next target token (correction
                # or bonus), truncated to the remaining budget.
                append = target_tokens[: n_accept + 1][:remaining]
                stats["accepted"] += min(n_accept, len(append))
                append_t = torch.tensor([append], dtype=out.dtype, device=out.device)
                out = torch.cat([out, append_t], dim=1)
                generated += len(append)
    finally:
        if was_training_target:
            target.train()
        if was_training_draft:
            draft.train()
    return out, stats


# Name the task / chapter vocabulary explicitly.
greedy_speculative_generate = speculative_generate


# ---------------------------------------------------------------------------
# Acceptance accounting (shared by draft-model and MTP drafting, sec. 5.2).
#
# Pure counting over caller-supplied token ids: given what a draft
# proposed at each verification step and what the target itself chose,
# how long were the accepted prefixes? Nothing here runs a model, and
# no real-model acceptance number is baked in -- callers supply ids
# they actually observed (or synthetic ids in tests).
# ---------------------------------------------------------------------------


def _ids_of(step) -> list[int]:  # type: ignore[no-untyped-def]
    if isinstance(step, torch.Tensor):
        return [int(x) for x in step.reshape(-1).tolist()]
    return [int(x) for x in step]


def accepted_prefix_len(proposed, target) -> int:  # type: ignore[no-untyped-def]
    """Longest common prefix length of one drafted run vs the target.

    ``proposed`` / ``target`` are 1-D id sequences for a single
    verification step (lists or tensors). The count stops at the first
    disagreement and never exceeds either length.
    """
    p, t = _ids_of(proposed), _ids_of(target)
    n = 0
    for a, b in zip(p, t):
        if a != b:
            break
        n += 1
    return n


def acceptance_stats(target_ids, draft_ids_per_step) -> dict:  # type: ignore[no-untyped-def]
    """Acceptance accounting over verification steps: pure token math.

    ``draft_ids_per_step`` is one proposal run per target forward: a
    sequence of steps, each a sequence of drafted token ids (a flat
    id sequence is read as a single step). ``target_ids`` mirrors it:
    per step, the target's own greedy ids for the positions the draft
    tried to fill -- conventionally ``gamma + 1`` of them (the accepted
    prefix plus the correction/bonus position); only the longest
    common prefix with the draft counts as accepted.

    Returns ``{"accept_rate", "mean_accepted_len",
    "expected_tokens_per_target_forward"}``:

    - ``accept_rate``: accepted / drafted over all steps (0.0 when
      nothing was drafted);
    - ``mean_accepted_len``: accepted tokens per target forward;
    - ``expected_tokens_per_target_forward``: ``mean_accepted_len + 1``
      per step that ran -- every verification forward finalises its
      accepted prefix plus one target-chosen token (the correction at
      the first disagreement, or the bonus when all agree). With no
      steps at all, all three values are 0.0.

    Raises ``ValueError`` when the two inputs disagree on the number
    of steps. No model is run and no acceptance figure is assumed.
    """
    def _steps(value, name):  # type: ignore[no-untyped-def]
        if isinstance(value, torch.Tensor):
            if value.dim() <= 1:
                return [_ids_of(value)]
            return [_ids_of(row) for row in value]
        seq = list(value)
        if not seq:
            return []
        if all(isinstance(x, (int,)) or (isinstance(x, torch.Tensor) and x.dim() == 0) for x in seq):
            return [[int(x) for x in seq]]
        return [_ids_of(step) for step in seq]

    draft_steps = _steps(draft_ids_per_step, "draft_ids_per_step")
    target_steps = _steps(target_ids, "target_ids")
    if len(draft_steps) != len(target_steps):
        raise ValueError(
            f"target_ids has {len(target_steps)} step(s) but draft_ids_per_step "
            f"has {len(draft_steps)}; give one target run per drafted run"
        )
    n_steps = len(draft_steps)
    if n_steps == 0:
        return {"accept_rate": 0.0, "mean_accepted_len": 0.0,
                "expected_tokens_per_target_forward": 0.0}
    accepted = sum(
        accepted_prefix_len(d, t) for d, t in zip(draft_steps, target_steps)
    )
    drafted = sum(len(d) for d in draft_steps)
    mean_accepted = accepted / n_steps
    return {
        "accept_rate": (accepted / drafted) if drafted else 0.0,
        "mean_accepted_len": mean_accepted,
        "expected_tokens_per_target_forward": mean_accepted + 1.0,
    }


# MTP drafting (docs/24) reports acceptance through this same accounting.
mtp_acceptance_stats = acceptance_stats


# ---------------------------------------------------------------------------
# Greedy speculative decoding with the native MTP head (gamma=1) on the
# TP static path (docs/24 Phase 3, Kiely sec. 5.2).
#
# Design (one cycle, batch=1, eager static decode -- no CUDA graph):
#   (a) Draft: the MTP module drafts ONE token from the last target
#       hidden (at the position of the token before ``last_accepted``)
#       fused with ``last_accepted`` itself. The draft call is
#       non-mutating: it returns a TENTATIVE MTP state.
#   (b) Verify: ONE target forward over the 2-token chunk
#       ``[last_accepted, draft]`` via
#       ``DistributedHybridLM.decode_chunk_static`` on the committed
#       ``TPStaticState`` (full-attention writes both K/V and attends
#       causally; linear layers read each weight once over M=2 and
#       step only conv/recurrence sequentially). Before the chunk, a
#       pre-cycle snapshot of the static state is taken.
#   (c) Accept iff the target argmax at the chunk position of
#       ``last_accepted`` equals the draft. On accept, the target
#       argmax after the draft (the bonus token) is also emitted, the
#       chunk state stands committed and the tentative MTP state is
#       committed. On reject, the target argmax at ``last_accepted``
#       (the correction) is emitted and the state rolls back:
#       ``copy_from`` the pre-cycle snapshot (this rewinds the linear
#       recurrent/conv state, which has no position to truncate, and
#       resets the position so the draft's full-attention KV slots are
#       overwritten by the next chunk), the tentative MTP state is
#       dropped, and ``last_accepted`` is re-fed with one single-token
#       static step. The re-feed is required for exactness: restoring
#       the pre-cycle snapshot also removes ``last_accepted``'s own
#       recurrent contribution, and its logits are already known from
#       the chunk, so the re-feed only re-establishes that one token
#       in the linear state (counted in ``target_forwards``). Without
#       it the next cycle would place the correction at
#       ``last_accepted``'s position and lose its history.
#
# Invariant (Kiely sec. 5.2): the emitted ids equal non-speculative
# target greedy token-for-token, whatever the draft does; the draft
# only changes how much work each target forward finalises. The next
# draft hidden after a cycle is the chunk hidden at the token before
# the new ``last_accepted`` (chunk index 1 on accept, 0 on reject).
# ---------------------------------------------------------------------------


class _PhaseTimer:
    """Per-phase timing for one speculative run (CUDA events or CPU wall).

    Phases: ``draft_ms`` (MTP draft step / draft-graph replay),
    ``verify_ms`` (2-token chunk forward / verify-graph replay),
    ``snapshot_ms`` (pre-cycle state snapshot), ``rollback_copy_ms``
    (``copy_from`` restore on reject) and ``refeed_ms`` (reject-path
    single-token re-feed). ``finish`` adds ``host_overhead_ms`` =
    wall minus the summed phases (argmax/host syncs, bookkeeping).
    """

    PHASES = ("draft_ms", "verify_ms", "snapshot_ms",
              "rollback_copy_ms", "refeed_ms")

    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self._events: list = []
        self._cpu: dict = {}

    def start(self, phase: str):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return (phase, ev)
        return (phase, time.perf_counter())

    def stop(self, token) -> None:
        phase, mark = token
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events.append((phase, mark, ev))
        else:
            self._cpu[phase] = self._cpu.get(phase, 0.0) + (
                time.perf_counter() - mark) * 1000.0

    def finish(self, profile: dict, wall_ms: float) -> dict:
        sums = {p: 0.0 for p in self.PHASES}
        if self.cuda:
            torch.cuda.synchronize(self.device)
            for phase, start_ev, end_ev in self._events:
                sums[phase] += float(start_ev.elapsed_time(end_ev))
        else:
            for phase, ms in self._cpu.items():
                sums[phase] = float(ms)
        profile.update(sums)
        profile["host_overhead_ms"] = max(
            float(wall_ms) - sum(sums.values()), 0.0)
        profile["wall_ms"] = float(wall_ms)
        return profile


SEGMENT_NAMES = (
    "draft_replay_launch_ms", "draft_token_readback_ms",
    "verify_input_fill_ms", "verify_replay_ms",
    "argmax_pair_readback_ms", "snapshot_ms", "rollback_refeed_ms",
    "loop_bookkeeping_ms",
)

_SYNC_KINDS = ("item", "tolist", "synchronize", "copy_", "fill_",
               "setitem", "snapshot_calls", "copy_from_calls")


class SegmentAccumulator:
    """Off-by-default wall-clock segment + host-sync counting.

    Segments are host wall time (``time.perf_counter``) around each
    named region of one speculative cycle. A segment that ends in a
    ``.item()``/``.tolist()`` therefore includes any GPU-queue drain
    wait; a segment with no synchronisation is pure launch/bookkeeping
    wall. ``loop_bookkeeping_ms`` is the residual: loop wall minus the
    explicitly timed segments. Sync counters are bumped only at the
    call sites in :func:`speculative_generate_tp` / graph ``run``.
    """

    def __init__(self):
        self.totals: dict = {n: 0.0 for n in SEGMENT_NAMES}
        self.sync_counts: dict = {k: 0 for k in _SYNC_KINDS}
        self._explicit_ms = 0.0
        self._loop_wall_ms = 0.0

    def add(self, name: str, ms: float) -> None:
        if name not in self.totals:
            raise ValueError(f"unknown segment {name!r}, expected {SEGMENT_NAMES}")
        self.totals[name] += float(ms)
        if name != "loop_bookkeeping_ms":
            self._explicit_ms += float(ms)

    def bump(self, kind: str, n: int = 1) -> None:
        if kind not in self.sync_counts:
            raise ValueError(f"unknown sync kind {kind!r}, expected {_SYNC_KINDS}")
        self.sync_counts[kind] += int(n)

    def add_loop_wall(self, ms: float) -> None:
        self._loop_wall_ms += float(ms)

    def finish_into(self, out: dict, cycles: int) -> dict:
        book = max(self._loop_wall_ms - self._explicit_ms, 0.0)
        self.totals["loop_bookkeeping_ms"] = book
        n = max(int(cycles), 1)
        out["segment_totals_ms"] = dict(self.totals)
        out["segment_per_cycle_ms"] = {k: v / n for k, v in self.totals.items()}
        out["sync_counts"] = dict(self.sync_counts)
        out["sync_per_cycle"] = {k: v / n for k, v in self.sync_counts.items()}
        out["segment_loop_wall_ms"] = float(self._loop_wall_ms)
        return out


def summarize_segments(segment_per_cycle: dict) -> list:
    """Rank segments by ms/cycle descending (pure; for diagnosis docs)."""
    return sorted(
        ({"segment": k, "ms_per_cycle": float(v)}
         for k, v in segment_per_cycle.items()),
        key=lambda r: -r["ms_per_cycle"])


def pack_acceptance_triple(draft_token, target_at_last, bonus_token) -> tuple:
    """Pack one cycle's outcome as ``(draft, correction, bonus)`` ints."""
    out = []
    for name, v in (("draft_token", draft_token),
                    ("target_at_last", target_at_last),
                    ("bonus_token", bonus_token)):
        if isinstance(v, bool):
            raise ValueError(f"{name} must be an int token id, got {v!r}")
        if not isinstance(v, (int,)):
            try:
                iv = int(v)  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError(f"{name} must be an int token id, got {v!r}") from exc
            if not isinstance(v, torch.Tensor) and iv != v:
                raise ValueError(f"{name} must be an int token id, got {v!r}")
            v = iv
        if int(v) < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
        out.append(int(v))
    return (out[0], out[1], out[2])


def decode_acceptance_triple(triple) -> dict:
    """Validate/unpack a packed ``(draft, correction, bonus)`` triple."""
    if isinstance(triple, torch.Tensor):
        vals = [int(x) for x in triple.reshape(-1).tolist()]
    else:
        vals = list(triple)  # type: ignore[arg-type]
    if len(vals) != 3:
        raise ValueError(f"acceptance triple must have 3 values, got {len(vals)}")
    draft, correction, bonus = pack_acceptance_triple(vals[0], vals[1], vals[2])
    return {"draft_token": draft, "correction_token": correction,
            "bonus_token": bonus, "accepted": draft == correction}


def _barrier(model) -> None:
    ctx = getattr(model, "ctx", None)
    if ctx is not None and int(getattr(ctx, "world_size", 1)) > 1:
        ctx.barrier()


class _ChunkGraph:
    """One captured 2-token verification chunk over the live state.

    Static input ``[1, 2]`` token ids are copied in before each
    ``replay()``; static logits/hidden buffers are read after it.
    Accept/reject branching and the snapshot/rollback copies stay
    outside the graph (they are host/data-dependent by design).
    """

    def __init__(self, model, state, warmup_steps: int = 3):
        dev = model.ctx.device
        if torch.device(dev).type != "cuda":
            raise ValueError("CUDA-graph verification requires a CUDA device")
        self.model, self.state = model, state
        vocab = int(model.config.vocab_size)
        hidden = int(model.config.hidden_size)
        dtype = model.embed_tokens.weight.dtype
        self.input = torch.zeros(1, 2, dtype=torch.long, device=dev)
        self.logits = torch.empty(1, 2, vocab, device=dev, dtype=dtype)
        self.hidden = torch.empty(1, 2, hidden, device=dev, dtype=dtype)
        scratch = model.static_state(
            batch_size=1, max_seq_len=state.max_seq_len)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(int(warmup_steps)):
                model.decode_chunk_static(self.input, scratch)
        torch.cuda.current_stream(dev).wait_stream(side)
        del scratch
        _barrier(model)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            model.decode_chunk_static(
                self.input, state, return_hidden=True,
                out=self.logits, hidden_out=self.hidden)
        torch.cuda.synchronize(dev)
        _barrier(model)

    def run(self, token_a: int, token_b: int, _seg=None):
        if _seg is None:
            self.input[0, 0] = int(token_a)
            self.input[0, 1] = int(token_b)
            self.graph.replay()
            return self.logits, self.hidden
        t0 = time.perf_counter()
        self.input[0, 0] = int(token_a)
        self.input[0, 1] = int(token_b)
        _seg.bump("setitem", 2)
        _seg.add("verify_input_fill_ms", (time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        self.graph.replay()
        _seg.add("verify_replay_ms", (time.perf_counter() - t0) * 1000.0)
        return self.logits, self.hidden

    def close(self) -> None:
        dev = self.model.ctx.device
        torch.cuda.synchronize(dev)
        del self.graph
        torch.cuda.synchronize(dev)


class _DraftGraph:
    """One captured MTP draft step over the static MTP KV state.

    Static hidden/token/rope-position buffers are copied in before
    each ``replay()``; the captured step writes K/V at the current
    sequential slot and advances ``state.position`` by 1 *inside*
    the graph. On rejection the caller rewinds ``state.position``
    outside the graph (the stale slot is overwritten by the next
    replay before it can be read). The draft argmax stays outside.
    """

    def __init__(self, model, mtp, mtp_state, embedding, lm_head,
                 warmup_steps: int = 3):
        dev = model.ctx.device
        if torch.device(dev).type != "cuda":
            raise ValueError("CUDA-graph draft requires a CUDA device")
        self.model, self.mtp, self.state = model, mtp, mtp_state
        self.embedding, self.lm_head = embedding, lm_head
        vocab = int(model.config.vocab_size)
        hidden = int(model.config.hidden_size)
        dtype = model.embed_tokens.weight.dtype
        self.hidden_in = torch.zeros(1, hidden, device=dev, dtype=dtype)
        self.token_in = torch.zeros(1, dtype=torch.long, device=dev)
        self.rope_pos = torch.zeros((), dtype=torch.long, device=dev)
        self.logits = torch.empty(1, vocab, device=dev, dtype=dtype)
        scratch = mtp.initial_static_state(
            batch_size=1, max_seq_len=mtp_state.max_seq_len)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(int(warmup_steps)):
                mtp.draft_token_static(
                    self.hidden_in, self.token_in, scratch,
                    rope_position=self.rope_pos, embedding=embedding,
                    lm_head=lm_head, out=self.logits)
                scratch.position.add_(1)
        torch.cuda.current_stream(dev).wait_stream(side)
        del scratch
        _barrier(model)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            mtp.draft_token_static(
                self.hidden_in, self.token_in, mtp_state,
                rope_position=self.rope_pos, embedding=embedding,
                lm_head=lm_head, out=self.logits)
            mtp_state.position.add_(1)
        torch.cuda.synchronize(dev)
        _barrier(model)

    def run(self, hidden_t, token_id: int, rope_pos: int, _seg=None):
        if _seg is None:
            self.hidden_in.copy_(hidden_t.reshape(1, -1).to(self.hidden_in.dtype))
            self.token_in.fill_(int(token_id))
            self.rope_pos.fill_(int(rope_pos))
            self.graph.replay()
            return self.logits
        t0 = time.perf_counter()
        self.hidden_in.copy_(hidden_t.reshape(1, -1).to(self.hidden_in.dtype))
        self.token_in.fill_(int(token_id))
        self.rope_pos.fill_(int(rope_pos))
        _seg.bump("copy_", 1)
        _seg.bump("fill_", 2)
        self.graph.replay()
        _seg.add("draft_replay_launch_ms", (time.perf_counter() - t0) * 1000.0)
        return self.logits

    def close(self) -> None:
        dev = self.model.ctx.device
        torch.cuda.synchronize(dev)
        del self.graph
        torch.cuda.synchronize(dev)


class _MergedSpecGraph:
    """graph2: ONE captured graph per cycle (snapshot+draft+verify).

    Captured body, in order: (1) ``snapshot_into`` copies of the target
    linear/conv/position and MTP position into preallocated buffers;
    (2) MTP draft step writing draft logits, advancing MTP position;
    (3) GPU-side handoff: draft argmax ``copy_`` into slot 1 of the
    verify chunk input (slot 0 from ``token_in``) -- no host read of
    the draft token; (4) 2-token verify chunk; (5) packed result
    ``[draft, target_at_last, bonus]`` written to ``result_gpu``.
    The host reads only that triple via one pinned async copy + one
    sync per cycle. Rollback (restore + refeed) stays outside, reject
    cycles only.
    """

    def __init__(self, model, state, mtp, mtp_state, embedding, lm_head,
                 warmup_steps: int = 3):
        dev = model.ctx.device
        if torch.device(dev).type != "cuda":
            raise ValueError("CUDA-graph verification requires a CUDA device")
        self.model, self.state, self.mtp, self.mtp_state = model, state, mtp, mtp_state
        self.embedding, self.lm_head = embedding, lm_head
        vocab = int(model.config.vocab_size)
        hidden = int(model.config.hidden_size)
        dtype = model.embed_tokens.weight.dtype
        self.hidden_in = torch.zeros(1, hidden, device=dev, dtype=dtype)
        self.token_in = torch.zeros(1, dtype=torch.long, device=dev)
        self.rope_pos = torch.zeros((), dtype=torch.long, device=dev)
        self.chunk_input = torch.zeros(1, 2, dtype=torch.long, device=dev)
        self.draft_logits = torch.empty(1, vocab, device=dev, dtype=dtype)
        self.logits = torch.empty(1, 2, vocab, device=dev, dtype=dtype)
        self.hidden = torch.empty(1, 2, hidden, device=dev, dtype=dtype)
        self.result_gpu = torch.zeros(3, dtype=torch.long, device=dev)
        self.result_pin = torch.empty(3, dtype=torch.long, pin_memory=True)
        self.snap_state = state.alloc_snapshot()
        self.snap_mtp_pos = torch.zeros_like(mtp_state.position)
        # Warmup on scratch states (side stream), then capture on live.
        scratch = model.static_state(batch_size=1, max_seq_len=state.max_seq_len)
        scratch_mtp = mtp.initial_static_state(
            batch_size=1, max_seq_len=mtp_state.max_seq_len)
        scratch_snap = scratch.alloc_snapshot()
        scratch_mtp_pos = torch.zeros_like(scratch_mtp.position)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(int(warmup_steps)):
                self._body(scratch, scratch_mtp, scratch_snap, scratch_mtp_pos)
        torch.cuda.current_stream(dev).wait_stream(side)
        del scratch, scratch_mtp, scratch_snap, scratch_mtp_pos
        _barrier(model)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body(state, mtp_state, self.snap_state, self.snap_mtp_pos)
        torch.cuda.synchronize(dev)
        _barrier(model)

    def _body(self, state, mtp_state, snap_state, snap_mtp_pos):
        state.snapshot_into(snap_state)
        snap_mtp_pos.copy_(mtp_state.position)
        draft_id, _ = self.mtp.draft_token_static(
            self.hidden_in, self.token_in, mtp_state,
            rope_position=self.rope_pos, embedding=self.embedding,
            lm_head=self.lm_head, out=self.draft_logits)
        mtp_state.position.add_(1)
        flat = self.chunk_input.view(-1)
        flat[0].copy_(self.token_in.reshape(-1)[0])
        flat[1].copy_(torch.as_tensor(draft_id).reshape(-1)[0])
        self.model.decode_chunk_static(
            self.chunk_input, state, return_hidden=True,
            out=self.logits, hidden_out=self.hidden)
        pair = torch.argmax(self.logits, dim=-1).reshape(-1)
        self.result_gpu[0].copy_(torch.as_tensor(draft_id).reshape(-1)[0])
        self.result_gpu[1].copy_(pair[0])
        self.result_gpu[2].copy_(pair[1])

    def run(self, hidden_t, token_id: int, rope_pos: int):
        """Fill static inputs, replay once, read packed triple (1 sync)."""
        self.hidden_in.copy_(hidden_t.reshape(1, -1).to(self.hidden_in.dtype))
        self.token_in.fill_(int(token_id))
        self.rope_pos.fill_(int(rope_pos))
        self.graph.replay()
        self.result_pin.copy_(self.result_gpu, non_blocking=True)
        torch.cuda.synchronize(self.model.ctx.device)
        return decode_acceptance_triple(
            [int(x) for x in self.result_pin.tolist()])

    def restore_on_reject(self, committed_slot: int) -> None:
        self.state.restore_linear_from(self.snap_state)
        self.mtp_state.position.fill_(int(committed_slot))

    def close(self) -> None:
        dev = self.model.ctx.device
        torch.cuda.synchronize(dev)
        del self.graph
        torch.cuda.synchronize(dev)


class _RefeedGraph:
    """Captured single-token refeed (reject cycles only)."""

    def __init__(self, model, state, warmup_steps: int = 3):
        dev = model.ctx.device
        vocab = int(model.config.vocab_size)
        dtype = model.embed_tokens.weight.dtype
        self.model, self.state = model, state
        self.input = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.empty(1, vocab, device=dev, dtype=dtype)
        scratch = model.static_state(batch_size=1, max_seq_len=state.max_seq_len)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(int(warmup_steps)):
                model.decode_token_static(self.input, scratch, out=self.out)
        torch.cuda.current_stream(dev).wait_stream(side)
        del scratch
        _barrier(model)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            model.decode_token_static(self.input, state, out=self.out)
        torch.cuda.synchronize(dev)
        _barrier(model)

    def run(self, token_id: int):
        self.input.fill_(int(token_id))
        self.graph.replay()
        return self.out

    def close(self) -> None:
        dev = self.model.ctx.device
        torch.cuda.synchronize(dev)
        del self.graph
        torch.cuda.synchronize(dev)


def speculative_generate_tp(model, mtp, idx, max_new_tokens, *,
                            state_factory=None, embedding=None, lm_head=None,
                            graph_verify: bool = False,
                            graph_draft: bool = False,
                            graph2: bool = False,
                            profile: dict | None = None,
                            segment_profile: dict | None = None):
    """Greedy MTP speculative generation on the TP static path.

    ``model`` is a ``DistributedHybridLM`` (eager static decode);
    ``mtp`` is a draft module with the ``MTPDraftModule`` /
    ``TPMTPDraftModule`` contract (``initial_state()`` +
    non-mutating ``draft_token(hidden, token_id, state, position)``);
    when ``mtp is None``, ``model.mtp_draft`` is used. ``idx`` is
    ``[1, T_prompt]`` (batch=1). ``state_factory(batch_size,
    max_seq_len) -> TPStaticState`` allocates the committed state
    (default ``model.static_state``). ``embedding`` / ``lm_head``,
    when given, are forwarded to ``draft_token`` (the TP runner
    passes the replicated embedding and a full-logits head callable).

    Returns ``(ids, stats)``: ``ids`` is the full ``[1, T_prompt +
    max_new_tokens]`` sequence (prompt included, mirroring
    :func:`speculative_generate`) and ``stats`` is
    ``{"proposed", "accepted", "target_forwards", "mtp_forwards",
    "rollbacks", "cycles"}`` -- proposal/accept counts over draft
    tokens, verification forwards (chunk forwards plus reject
    re-feeds), MTP draft forwards, rejected cycles and total cycles.

    ``graph_verify=True`` captures the 2-token verification chunk
    once (after prefill) and replays it per cycle; ``graph_draft=True``
    additionally captures the MTP draft step (static KV form) as a
    second graph. Accept/reject branching, the snapshot and the
    rollback copies stay outside both graphs. ``profile``, when a
    dict is passed, is filled with per-phase milliseconds (see
    :class:`_PhaseTimer`).
    """
    if mtp is None:
        mtp = getattr(model, "mtp_draft", None)
        if mtp is None:
            raise ValueError("mtp is None and model has no mtp_draft attached")
    if not isinstance(idx, torch.Tensor) or idx.dim() != 2 or idx.shape[0] != 1:
        raise ValueError("idx must be a [1, T] tensor (batch=1 speculative decode)")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    stats = {"proposed": 0, "accepted": 0, "target_forwards": 0,
             "mtp_forwards": 0, "rollbacks": 0, "cycles": 0}
    out = idx.clone()
    if max_new_tokens == 0:
        return out, stats
    dev = model.ctx.device
    prompt_len = int(idx.shape[1])
    if state_factory is None:
        state = model.static_state(batch_size=1, max_seq_len=prompt_len + int(max_new_tokens) + 2)
    else:
        state = state_factory(1, prompt_len + int(max_new_tokens) + 2)
    was_training = model.training
    model.eval()
    chunk_graph = None
    draft_graph = None
    merged_graph = None
    refeed_graph = None
    seg = SegmentAccumulator() if segment_profile is not None else None
    timer = _PhaseTimer(dev) if profile is not None else None
    wall_start = time.perf_counter()
    try:
        with torch.no_grad():
            logits, hidden_pre = model.prefill_static(
                idx.to(dev), state, return_hidden=True)
            generated: list[int] = [int(logits[0, -1].argmax().item())]
            last_accepted = generated[0]
            last_hidden_pre = hidden_pre[0, -1]  # [D], position prompt_len-1
            mtp_pos = prompt_len - 1
            mtp_state = mtp.initial_state() if hasattr(mtp, "initial_state") else None
            wiring = getattr(mtp, "wiring", None) or {}
            hidden_source = wiring.get("hidden_source", "post_final_norm")
            mtp_static = None
            committed_slot = 0
            host_pos = prompt_len  # state.position tracked on host (no per-cycle .item())
            if graph2 and (graph_verify or graph_draft):
                raise ValueError("graph2 is exclusive with graph_verify/graph_draft")
            if graph_verify or graph_draft or graph2:
                if torch.device(dev).type != "cuda":
                    raise ValueError("graph speculative modes require CUDA")
                if (graph_draft or graph2) and not hasattr(mtp, "draft_token_static"):
                    raise ValueError("graph modes require a draft module with "
                                     "draft_token_static (TPMTPDraftModule)")
            if graph_verify:
                chunk_graph = _ChunkGraph(model, state)
            if graph_draft:
                from .mtp import _POSITION_OFFSET
                mtp_static = mtp.initial_static_state(
                    batch_size=1,
                    max_seq_len=prompt_len + int(max_new_tokens) + 2)
                draft_graph = _DraftGraph(
                    model, mtp, mtp_static,
                    embedding if embedding is not None else getattr(mtp, "_embedding", None),
                    lm_head if lm_head is not None else getattr(mtp, "_lm_head", None))
                stats["graph_draft"] = True
            if graph_verify:
                stats["graph_verify"] = True
            if graph2:
                from .mtp import _POSITION_OFFSET as _PO2  # noqa: F401
                mtp_static = mtp.initial_static_state(
                    batch_size=1,
                    max_seq_len=prompt_len + int(max_new_tokens) + 2)
                merged_graph = _MergedSpecGraph(
                    model, state, mtp, mtp_static,
                    embedding if embedding is not None else getattr(mtp, "_embedding", None),
                    lm_head if lm_head is not None else getattr(mtp, "_lm_head", None))
                refeed_graph = _RefeedGraph(model, state)
                stats["graph2"] = True
                stats["host_syncs"] = 0
            while len(generated) < int(max_new_tokens):
                _iter_t0 = time.perf_counter() if seg is not None else 0.0
                hidden_t = (last_hidden_pre if hidden_source == "pre_final_norm"
                            else model.norm(last_hidden_pre))
                if merged_graph is not None:
                    from .mtp import _POSITION_OFFSET
                    abs_pos = int(mtp_pos) + _POSITION_OFFSET[
                        wiring.get("position_rule", "hidden_pos")]
                    pos_before = host_pos
                    tok_t = timer.start("verify_ms") if timer else None
                    triple = merged_graph.run(hidden_t, last_accepted, abs_pos)
                    if timer:
                        timer.stop(tok_t)
                    stats["host_syncs"] = int(stats.get("host_syncs", 0)) + 1
                    stats["mtp_forwards"] += 1
                    stats["proposed"] += 1
                    stats["target_forwards"] += 1
                    stats["cycles"] += 1
                    draft_int = triple["draft_token"]
                    target_at_last = triple["correction_token"]
                    bonus_or_next = triple["bonus_token"]
                    if triple["accepted"]:
                        stats["accepted"] += 1
                        committed_slot += 1
                        generated.append(draft_int)
                        if len(generated) < int(max_new_tokens):
                            generated.append(bonus_or_next)
                            last_accepted = bonus_or_next
                        else:
                            last_accepted = draft_int
                        last_hidden_pre = merged_graph.hidden[0, 1]
                        mtp_pos = pos_before + 1
                        host_pos = pos_before + 2
                    else:
                        stats["rollbacks"] += 1
                        tok_t = timer.start("rollback_copy_ms") if timer else None
                        merged_graph.restore_on_reject(committed_slot)
                        if timer:
                            timer.stop(tok_t)
                        tok_t = timer.start("refeed_ms") if timer else None
                        assert refeed_graph is not None
                        refeed_graph.run(last_accepted)
                        if timer:
                            timer.stop(tok_t)
                        stats["target_forwards"] += 1
                        generated.append(target_at_last)
                        last_accepted = target_at_last
                        last_hidden_pre = merged_graph.hidden[0, 0]
                        mtp_pos = pos_before
                        host_pos = pos_before + 1
                    if seg is not None:
                        seg.add_loop_wall((time.perf_counter() - _iter_t0) * 1000.0)
                    continue
                draft_kwargs = {}
                if embedding is not None:
                    draft_kwargs["embedding"] = embedding
                if lm_head is not None:
                    draft_kwargs["lm_head"] = lm_head
                tok_t = timer.start("draft_ms") if timer else None
                tentative = None
                if draft_graph is not None:
                    from .mtp import _POSITION_OFFSET
                    abs_pos = int(mtp_pos) + _POSITION_OFFSET[
                        wiring.get("position_rule", "hidden_pos")]
                    d_logits = draft_graph.run(hidden_t, last_accepted, abs_pos, _seg=seg)
                    if seg is not None:
                        _t = time.perf_counter()
                        draft_int = int(d_logits[0].argmax().item())
                        seg.bump("item")
                        seg.add("draft_token_readback_ms",
                                (time.perf_counter() - _t) * 1000.0)
                    else:
                        draft_int = int(d_logits[0].argmax().item())
                else:
                    draft_id, _draft_logits, tentative = mtp.draft_token(
                        hidden_t, last_accepted, mtp_state, position=mtp_pos,
                        **draft_kwargs)
                    draft_int = int(torch.as_tensor(draft_id).reshape(-1)[0].item())
                    if seg is not None:
                        seg.bump("item")
                if timer:
                    timer.stop(tok_t)
                stats["mtp_forwards"] += 1
                stats["proposed"] += 1
                pos_before = host_pos
                tok_t = timer.start("snapshot_ms") if timer else None
                _snap_t = time.perf_counter() if seg is not None else 0.0
                snapshot = state.snapshot()
                if seg is not None:
                    seg.bump("snapshot_calls")
                    seg.add("snapshot_ms", (time.perf_counter() - _snap_t) * 1000.0)
                if timer:
                    timer.stop(tok_t)
                tok_t = timer.start("verify_ms") if timer else None
                if chunk_graph is not None:
                    chunk_logits, chunk_hidden = chunk_graph.run(
                        last_accepted, draft_int, _seg=seg)
                else:
                    chunk = torch.tensor([[last_accepted, draft_int]],
                                         dtype=torch.long, device=dev)
                    chunk_logits, chunk_hidden = model.decode_chunk_static(
                        chunk, state, return_hidden=True)
                if timer:
                    timer.stop(tok_t)
                stats["target_forwards"] += 1
                stats["cycles"] += 1
                if seg is not None:
                    _t = time.perf_counter()
                    _pair = chunk_logits[0].argmax(dim=-1).tolist()
                    seg.bump("tolist")
                    seg.add("argmax_pair_readback_ms",
                            (time.perf_counter() - _t) * 1000.0)
                else:
                    _pair = chunk_logits[0].argmax(dim=-1).tolist()
                target_at_last, bonus_or_next = int(_pair[0]), int(_pair[1])
                if draft_int == target_at_last:
                    stats["accepted"] += 1
                    if draft_graph is not None:
                        committed_slot += 1  # graph already advanced position
                    else:
                        mtp_state = tentative  # commit MTP state
                    generated.append(draft_int)
                    if len(generated) < int(max_new_tokens):
                        generated.append(bonus_or_next)
                        last_accepted = bonus_or_next
                    else:
                        last_accepted = draft_int
                    last_hidden_pre = chunk_hidden[0, 1]
                    mtp_pos = pos_before + 1
                    host_pos = pos_before + 2
                else:
                    stats["rollbacks"] += 1
                    _rb_t = time.perf_counter() if seg is not None else 0.0
                    if draft_graph is not None and mtp_static is not None:
                        mtp_static.position.fill_(committed_slot)
                        if seg is not None:
                            seg.bump("fill_")
                    tok_t = timer.start("rollback_copy_ms") if timer else None
                    state.copy_from(snapshot)  # roll back (see design note)
                    if seg is not None:
                        seg.bump("copy_from_calls")
                    if timer:
                        timer.stop(tok_t)
                    del snapshot
                    # Re-establish last_accepted in the linear state;
                    # its logits are already known from the chunk.
                    tok_t = timer.start("refeed_ms") if timer else None
                    model.decode_token_static(
                        torch.tensor([last_accepted], dtype=torch.long, device=dev), state)
                    if timer:
                        timer.stop(tok_t)
                    if seg is not None:
                        seg.add("rollback_refeed_ms",
                                (time.perf_counter() - _rb_t) * 1000.0)
                    stats["target_forwards"] += 1
                    generated.append(target_at_last)
                    last_accepted = target_at_last
                    last_hidden_pre = chunk_hidden[0, 0]
                    mtp_pos = pos_before
                    host_pos = pos_before + 1  # snapshot restore + one re-feed
                if seg is not None:
                    seg.add_loop_wall((time.perf_counter() - _iter_t0) * 1000.0)
            if seg is not None and segment_profile is not None:
                seg.finish_into(segment_profile, int(stats["cycles"]))
            gen_t = torch.tensor([generated], dtype=out.dtype, device=out.device)
            out = torch.cat([out, gen_t], dim=1)
            if timer is not None and profile is not None:
                if torch.device(dev).type == "cuda":
                    torch.cuda.synchronize(dev)
                timer.finish(profile, (time.perf_counter() - wall_start) * 1000.0)
    finally:
        if chunk_graph is not None:
            chunk_graph.close()
        if draft_graph is not None:
            draft_graph.close()
        if refeed_graph is not None:
            refeed_graph.close()
        if merged_graph is not None:
            merged_graph.close()
        if was_training:
            model.train()
    return out, stats
