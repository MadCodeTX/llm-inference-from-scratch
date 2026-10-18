"""Load a REAL small Qwen checkpoint into DecoderLM and prove it vs HF.

With ``--model-dir PATH`` (a Hugging Face snapshot directory holding
``config.json`` + ``model.safetensors`` + tokenizer files) this:

1. builds ``ModelConfig`` from the real ``config.json`` via
   ``weights.load_hf_config`` (numbers come from disk, never constants),
2. dissects the safetensors header by hand, loads tensors with the
   ``weights.py`` machinery, renames with ``map_hf_state_dict`` and
   requires ``validate_state_dict`` to report zero missing /
   unexpected / shape errors before a strict load (fp32 cast),
3. encodes one fixed plain-completion prompt with ``AutoTokenizer``
   from the same snapshot (raw ids, no chat template) and runs OUR
   ``generate()`` greedy for 20 tokens with the KV cache,
4. runs the HF reference (``AutoModelForCausalLM``, fp32, eval) greedy
   on the *same* raw ids and compares: per-step top-1 agreement,
   exact generated-id equality, and first-forward logits max-abs-diff,
5. times the real model with ``metrics.measure_generate`` and places
   it on the roofline with ``bottlenecks.classify`` against the
   explicitly illustrative example spec.

HF greedy trap, learned the hard way: the checkpoint's
``generation_config.json`` carries sampling defaults
(``repetition_penalty`` etc.) that ``generate(do_sample=False)``
still inherits. The reference below therefore does its greedy loop
by hand (argmax over full forwards), so no sampling config can leak
in; both sides see identical raw ids and identical rules.

Without ``--model-dir`` it prints how to use it and exits (offline).

Run:
    .venv/bin/python scripts/load_qwen_small.py
    .venv/bin/python scripts/load_qwen_small.py --model-dir <snapshot>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from inference_from_scratch.bottlenecks import classify, make_example_spec
from inference_from_scratch.generate import generate
from inference_from_scratch.metrics import measure_generate
from inference_from_scratch.model import DecoderLM
from inference_from_scratch.weights import (
    format_safetensors_table,
    load_hf_config,
    load_safetensors,
    map_hf_state_dict,
    read_safetensors_header,
    safetensors_table,
    validate_state_dict,
)

FIXED_PROMPT = "The capital of France is"


def _find_safetensors(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        print(f"error: no *.safetensors files in {model_dir}", file=sys.stderr)
        raise SystemExit(2)
    return files


def _greedy_top1s(model_forward, prompt: torch.Tensor, steps: int) -> list[int]:
    """Per-step greedy top-1 ids via repeated full forwards.

    ``model_forward(ids)`` returns next-token logits ``[V]`` for the
    last position. Teacher-forces its own picks, so step k's top-1 is
    conditioned on the same prefix on both sides of the comparison.
    """
    seq = prompt.clone()
    picks: list[int] = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = int(model_forward(seq).argmax())
            picks.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
    return picks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="local HF snapshot dir (config.json + *.safetensors)")
    parser.add_argument("--prompt", default=FIXED_PROMPT,
                        help="plain completion prompt (raw ids, no chat template)")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--metrics-tokens", type=int, default=5,
                        help="shorter budget for the timed measure_generate run")
    parser.add_argument("--max-params", type=int, default=1_000_000_000)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)

    if args.model_dir is None:
        print("no --model-dir given: nothing downloaded, nothing loaded.\n"
              "Point --model-dir at a LOCAL Qwen2.5 snapshot directory "
              "(config.json + model.safetensors + tokenizer files).")
        return 0
    model_dir: Path = args.model_dir
    if not model_dir.is_dir():
        print(f"error: --model-dir {model_dir} is not a directory", file=sys.stderr)
        return 2

    # 1) Config from the real config.json on disk.
    config = load_hf_config(model_dir)
    print(f"config (from {model_dir / 'config.json'}): vocab={config.vocab_size} "
          f"d_model={config.d_model} layers={config.n_layer} heads={config.n_head} "
          f"kv_heads={config.n_kv_head} d_ff={config.d_ff} "
          f"max_seq={config.max_seq_len} rope_theta={config.rope_theta} "
          f"tie={config.tie_embeddings} attention_bias={config.attention_bias} "
          f"head_dim={config.head_dim}")

    # 2) Headers first, size guard, then load + rename + validate.
    files = _find_safetensors(model_dir)
    total_params = 0
    for path in files:
        header = read_safetensors_header(path)
        rows = safetensors_table(header)
        total_params += sum(r["numel"] for r in rows)
        print(f"\n{path.name}: {len(rows)} tensors")
        lines = format_safetensors_table(header).splitlines()
        print("\n".join(lines[:12]))
        if len(lines) > 12:
            print(f"  ... ({len(lines) - 12} more rows in the header table)")
    print(f"\ntotal params (from headers): {total_params:,}")
    if total_params > args.max_params and not args.allow_large:
        print(f"error: {total_params:,} params exceeds --max-params "
              f"{args.max_params:,}", file=sys.stderr)
        return 2

    raw: dict[str, torch.Tensor] = {}
    for path in files:
        for name, tensor in load_safetensors(path).items():
            if name in raw:
                print(f"error: duplicate tensor {name!r}", file=sys.stderr)
                return 2
            raw[name] = tensor
    mapped = map_hf_state_dict(raw, tie_embeddings=config.tie_embeddings)
    # Checkpoint dtype is bf16; the engine runs fp32 on CPU here so the
    # comparison against the fp32 HF reference isolates architecture
    # bugs from rounding (see docs/19). Cast before loading.
    mapped = {k: v.to(torch.float32) for k, v in mapped.items()}

    torch.manual_seed(0)
    model = DecoderLM(config).eval()
    report = validate_state_dict(mapped, model)
    print(f"validate: missing={report['missing']} "
          f"unexpected={report['unexpected']} "
          f"shape_mismatch={report['shape_mismatch']}")
    if not report["ok"]:
        print("error: checkpoint does not fit this DecoderLM; not loading",
              file=sys.stderr)
        return 3
    model.load_state_dict(mapped, strict=True)
    print(f"loaded {len(mapped)} tensors into DecoderLM "
          f"({model.num_params():,} params, fp32)")

    # 3) Tokenize the fixed prompt (raw ids; no chat template either side).
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("error: transformers is required for the tokenizer + HF "
              "reference (pip install -e '.[hf]')", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    print(f"\nprompt: {args.prompt!r} -> ids {prompt[0].tolist()}")

    with torch.no_grad():
        our_logits, _ = model(prompt)
    our_gen = generate(model, prompt, max_new_tokens=args.max_new_tokens,
                       temperature=0, use_cache=True)
    our_picks = _greedy_top1s(lambda ids: model(ids)[0][0, -1],
                              prompt, args.max_new_tokens)

    # 4) HF reference: same raw ids, hand-rolled greedy (see docstring).
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        hf_logits = hf_model(prompt).logits
    hf_picks = _greedy_top1s(lambda ids: hf_model(ids).logits[0, -1],
                             prompt, args.max_new_tokens)
    hf_gen = torch.cat([prompt, torch.tensor([hf_picks])], dim=1)

    max_abs_diff = float((our_logits - hf_logits).abs().max())
    agree = sum(a == b for a, b in zip(our_picks, hf_picks))
    ids_equal = our_gen.tolist() == hf_gen.tolist()
    print(f"\nfirst-forward logits max-abs-diff: {max_abs_diff:.3e}")
    print(f"per-step top-1 agreement: {agree}/{args.max_new_tokens}")
    print(f"generated ids exactly equal: {ids_equal}")
    print(f"ours ids: {our_gen[0].tolist()}")
    print(f"hf   ids: {hf_gen[0].tolist()}")
    print(f"ours text: {tokenizer.decode(our_gen[0].tolist())!r}")
    print(f"hf   text: {tokenizer.decode(hf_gen[0].tolist())!r}")

    # 5) Timed short run + roofline on the REAL config dims.
    timed = measure_generate(model, prompt, args.metrics_tokens, temperature=0)
    print(f"\nmeasure_generate ({args.metrics_tokens} tokens, CPU fp32): "
          f"TTFT={timed['ttft_s']:.3f}s decode={timed['decode_s']:.3f}s "
          f"TPS={timed['tps']:.2f} decode_TPS={timed['decode_tps']:.2f}")
    spec = make_example_spec()  # illustrative round numbers, NOT a real GPU
    seq_len = prompt.shape[1] + args.max_new_tokens
    for phase in ("prefill", "decode"):
        c = classify(config, spec, seq_len, phase, weight_bytes=4, kv_bytes=4)
        print(f"classify[{phase}] (illustrative spec, fp32): "
              f"intensity={c['intensity']:.2f} ops/byte ridge={c['ridge']:.1f} "
              f"bound={c['bound']} flops/token={c['flops']:.3e} "
              f"bytes/token={c['bytes']:.3e}")

    if not ids_equal or agree != args.max_new_tokens:
        print("error: our engine disagrees with the HF reference",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
