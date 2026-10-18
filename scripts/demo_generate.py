"""Demo (Ch 00/07/08): train a TINY char-level LM, then generate.

This repo is about *inference*, not training. The few hundred
training steps below exist for exactly one reason: to put
non-random weights into the model so the generated sample and the
cache / no-cache timing comparison mean something. No training
framework, no dataset pipeline -- one built-in string, plain
PyTorch, a few seconds on CPU.

What it shows:

* PRE-FILL vs DECODE (Ch 00): generation runs the prompt once in
  parallel, then emits one token per step (see generate.py).
* The KV cache working: greedy output with the cache must match
  the no-cache reference token-for-token (asserted, not eyeballed),
  and decode tokens/sec with vs without the cache is printed.

Run: .venv/bin/python scripts/demo_generate.py [--steps N]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.generate import generate
from inference_from_scratch.model import DecoderLM
from inference_from_scratch.tokenizer import CharTokenizer

# Small built-in corpus: repetitive on purpose, so a tiny model can
# actually learn visible structure in a few hundred steps.
TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the lazy dog sleeps under the warm sun. "
    "the quick brown fox runs through the green forest. "
    "inference is prefill then decode, one token at a time. "
    "the kv cache remembers every key and value. "
) * 4


def train_tiny(steps: int = 300, seed: int = 0):
    """Train a tiny char-level DecoderLM on TEXT; return (model, tok, losses)."""
    torch.manual_seed(seed)
    tok = CharTokenizer.build(TEXT)
    data = torch.tensor(tok.encode(TEXT), dtype=torch.long)  # [N]
    config = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_ff=128,
        max_seq_len=128,
    )
    model = DecoderLM(config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq_len = 64
    losses: list[float] = []
    model.train()
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        # One random window per step: inputs [1, T], targets shifted by 1.
        start = int(torch.randint(0, len(data) - seq_len - 1, (1,), generator=g))
        chunk = data[start : start + seq_len + 1]  # [T+1]
        x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)  # [1, T] each
        logits, _ = model(x)  # [1, T, V]
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return model, tok, losses


def timed_generate(model, prompt_ids: torch.Tensor, max_new: int, use_cache: bool):
    """Greedy-generate; return (output ids, tokens/sec over the whole call)."""
    start = time.perf_counter()
    out = generate(model, prompt_ids, max_new_tokens=max_new,
                   temperature=0, use_cache=use_cache)
    elapsed = time.perf_counter() - start
    n_new = out.shape[1] - prompt_ids.shape[1]
    return out, (n_new / elapsed if elapsed > 0 else float("inf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300,
                        help="tiny training steps (default 300)")
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--prompt", default="the quick brown ")
    args = parser.parse_args()

    print(f"training tiny char LM for {args.steps} steps "
          f"(minimal training: only to get non-random weights)...")
    t0 = time.perf_counter()
    model, tok, losses = train_tiny(steps=args.steps)
    print(f"trained in {time.perf_counter() - t0:.1f}s | vocab={tok.vocab_size} "
          f"params={model.num_params()} | loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    cached, tps_cache = timed_generate(model, prompt_ids, args.max_new, True)
    plain, tps_plain = timed_generate(model, prompt_ids, args.max_new, False)

    # The cache must not change a single greedy token.
    assert torch.equal(cached, plain), "cache / no-cache greedy mismatch!"
    print(f"prompt: {args.prompt!r}")
    print(f"sample (greedy, cache): {tok.decode(cached[0].tolist())!r}")
    print(f"cache == no-cache (greedy): True ({cached.shape[1] - prompt_ids.shape[1]} new tokens)")
    print(f"tokens/sec with cache:    {tps_cache:.1f}")
    print(f"tokens/sec without cache: {tps_plain:.1f}")


if __name__ == "__main__":
    main()
