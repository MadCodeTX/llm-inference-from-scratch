"""Ch 08 -- Sampling: turning logits [B, V] into the next token id [B, 1].

Why this exists in inference
-----------------------------
The model emits one logit vector per sequence; *decoding strategy* is
the separate, model-independent decision of how to pick the next id
from it. Greedy (argmax) is deterministic but repetitive; temperature /
top-k / top-p sampling trades determinism for diversity. Serving
systems (vLLM, TGI) expose exactly these knobs per-request, applied to
the same logits -- which is why they live in their own module here,
pure functions of a logits tensor plus an RNG.

Conventions
-----------
All functions take logits of shape [B, V] (batch, vocab); most also
accept a bare [V] vector by treating it as B=1 where noted. Filters
(``top_k_filter`` / ``top_p_filter``) never change shape and never
renormalise: they set removed logits to -inf so a later softmax gives
them probability exactly 0. Filtering and renormalising are separate
steps so filters compose (temperature -> top-k -> top-p -> softmax).

How production differs
-----------------------
vLLM performs these ops fused on GPU over thousands of concurrent
requests, with per-request temperatures and penalties (repetition /
frequency / presence) also applied to logits, and samples with
rejection-sampling speculative decoding on top. The maths per request
is exactly what is dissected below.
"""

from __future__ import annotations

import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Pick the highest-logit token per row: [B, V] -> LongTensor [B, 1].

    Argmax has no randomness and no hyper-parameters; it is the
    reference every stochastic setting is compared against
    (temperature=0 and top_k=1 must both reduce to this).
    """
    # keepdim=True keeps the result rank-2 so callers can always cat it
    # onto an [B, T] id sequence regardless of B.
    return torch.argmax(logits, dim=-1, keepdim=True)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature T: [B, V] -> [B, V] (same shape).

    T < 1 sharpens the softmax (peaks dominate), T > 1 flattens it
    (tail tokens get relatively more probability), T == 1 is identity.
    Division happens *before* softmax, so it is a true rescaling of the
    distribution, not a post-hoc tweak.

    T must be > 0. T == 0 is not handled here -- it is the greedy limit
    (division by zero); ``sample_next`` special-cases it. Raising here
    keeps the "temperature 0 means greedy" policy in exactly one place.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0 for apply_temperature, got {temperature}; "
            "use sample_next(..., temperature=0) for greedy decoding"
        )
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep the k highest logits per row, set the rest to -inf.

    [B, V] -> [B, V]. k == 1 reduces sampling to greedy; k >= V is a
    no-op. Returns a new tensor (callers may reuse the original logits,
    e.g. to compare filtered vs unfiltered sampling).
    """
    if k is None:
        return logits
    if k <= 0:
        raise ValueError(f"top_k must be >= 1, got {k}")
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits.clone()
    # kthvalue gives the k-th largest when we index from the top:
    # threshold = value at position V-k in ascending order.
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose
    cumulative probability is >= p; set all others to -inf.

    [B, V] -> [B, V]. "Smallest set >= p" is implemented by sorting
    probabilities descending, taking the cumulative sum, and keeping
    tokens up to and *including* the one that crosses p (at least the
    single most likely token is always kept, even for tiny p). The
    result adapts to distribution shape: peaked distributions keep few
    tokens, flat ones keep many -- the property top-k lacks.

    p must be in (0, 1]; p == 1 keeps everything.
    """
    if p is None:
        return logits
    if not (0.0 < p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits.clone()

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    # Remove tokens *after* the cumulative mass reaches p. Shifting the
    # mask right by one keeps the crossing token itself; position 0 is
    # never removed (always keep the argmax token).
    remove_sorted = cum_probs > p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False

    # Scatter the sorted-order mask back to original vocab order.
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_idx, remove_sorted)
    return torch.where(remove, torch.full_like(logits, float("-inf")), logits)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one next token per row: logits [B, V] -> LongTensor [B, 1].

    Pipeline (order matters, mirrors vLLM):
      1. temperature == 0  -> greedy argmax (generator ignored).
         temperature < 0   -> ValueError. This is the user-facing
         validation point: 0 is the documented greedy sentinel, any
         other non-positive value is a caller bug.
      2. Divide by temperature (if != 1).
      3. top-k filter, then top-p filter (each optional).
      4. softmax -> probabilities [B, V]; draw one id per row with
         ``torch.multinomial``, honouring ``generator`` so seeded runs
         are exactly reproducible (tests and eval harnesses depend on
         this; production serving deliberately does not seed).
    A bare [V] input is treated as B=1 and still returns [1, 1].
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0 (0 means greedy), got {temperature}")
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, V] or [V], got shape {tuple(logits.shape)}")

    if temperature == 0:
        return greedy(logits)

    out = apply_temperature(logits, temperature)
    out = top_k_filter(out, top_k)
    out = top_p_filter(out, top_p)
    probs = torch.softmax(out, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
