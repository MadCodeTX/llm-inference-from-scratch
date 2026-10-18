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
