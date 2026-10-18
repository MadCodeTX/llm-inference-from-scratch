"""Load a LOCAL small Qwen checkpoint into DecoderLM (Ch 09).

OPTIONAL and fully offline: this script downloads nothing, ever.
Without ``--model-dir`` it prints how to use it and exits. With
``--model-dir PATH`` pointing at a local Qwen2.5 / Qwen3 (or Llama)
checkpoint directory -- one containing ``config.json`` plus
``*.safetensors`` -- it:

1. reads ``config.json`` from that directory (numbers come from the
   file on disk, never from constants here),
2. dissects each safetensors header by hand and prints the tensor
   table + total parameter count *before* loading any weights
   (refusing implausibly large checkpoints unless ``--allow-large``),
3. loads the tensors, renames HF keys to ``DecoderLM`` names,
   validates (missing / unexpected / shape mismatches), and loads,
4. runs one forward pass and prints the top-1 next-token id,
5. if ``transformers`` is installed, loads the same directory with
   HF (``local_files_only=True``) and compares its top-1 next token.

Known dissection limits, printed rather than hidden when hit: Qwen2
Q/K/V projection *biases* have no home in this repo's bias-free
projections, so they are reported as unexpected and dropped before
the strict load (a small, real fidelity gap vs the HF model).

Run:
    .venv/bin/python scripts/load_qwen_small.py
    .venv/bin/python scripts/load_qwen_small.py --model-dir /path/to/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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


def _find_safetensors(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        print(f"error: no *.safetensors files in {model_dir}", file=sys.stderr)
        raise SystemExit(2)
    return files


def _compare_with_hf(model_dir: Path, prompt: torch.Tensor, our_top1: int) -> None:
    """Top-1 comparison vs HF, only when transformers is available."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("HF comparison: skipped (transformers not installed)")
        return
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float32
        ).eval()
        with torch.no_grad():
            hf_logits = hf_model(prompt).logits  # [B, T, V]
        hf_top1 = int(hf_logits[0, -1].argmax())
        verdict = "MATCH" if hf_top1 == our_top1 else "MISMATCH"
        print(f"HF comparison: top-1 ours={our_top1} hf={hf_top1} -> {verdict}")
    except Exception as exc:  # offline / format issues must not crash the demo
        print(f"HF comparison: skipped ({type(exc).__name__}: {exc})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="local checkpoint dir (config.json + *.safetensors); "
             "omitting this downloads nothing and only prints this help text",
    )
    parser.add_argument(
        "--prompt-ids", default="1,2,3,4",
        help="comma-separated token ids for the one forward pass",
    )
    parser.add_argument(
        "--max-params", type=int, default=1_000_000_000,
        help="refuse checkpoints larger than this (from headers) unless --allow-large",
    )
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)

    if args.model_dir is None:
        print(
            "no --model-dir given: nothing downloaded, nothing loaded.\n"
            "Point --model-dir at a LOCAL small Qwen2.5/Qwen3 checkpoint "
            "directory (config.json + *.safetensors) to run the Ch 09 demo."
        )
        return 0
    model_dir: Path = args.model_dir
    if not model_dir.is_dir():
        print(f"error: --model-dir {model_dir} is not a directory", file=sys.stderr)
        return 2

    # 1) Config: every shape number is read from the real config.json.
    config = load_hf_config(model_dir)
    print(
        f"config (from {model_dir / 'config.json'}): vocab={config.vocab_size} "
        f"d_model={config.d_model} layers={config.n_layer} heads={config.n_head} "
        f"kv_heads={config.n_kv_head} d_ff={config.d_ff} "
        f"max_seq={config.max_seq_len} rope_theta={config.rope_theta} "
        f"tie={config.tie_embeddings}"
    )

    # 2) Headers first: table + size guard before any tensor is loaded.
    files = _find_safetensors(model_dir)
    total_params = 0
    for path in files:
        header = read_safetensors_header(path)
        rows = safetensors_table(header)
        total_params += sum(r["numel"] for r in rows)
        print(f"\n{path.name}: {len(rows)} tensors")
        table = format_safetensors_table(header)
        lines = table.splitlines()
        print("\n".join(lines[:12]))  # head of the table; headers hold the rest
        if len(lines) > 12:
            print(f"  ... ({len(lines) - 12} more rows in the header table)")
    print(f"\ntotal params (from headers): {total_params:,}")
    if total_params > args.max_params and not args.allow_large:
        print(
            f"error: {total_params:,} params exceeds --max-params "
            f"{args.max_params:,}; this demo targets SMALL checkpoints "
            f"(pass --allow-large only if the machine can hold it)",
            file=sys.stderr,
        )
        return 2

    # 3) Load + rename + validate, then a strict load of what fits.
    raw: dict[str, torch.Tensor] = {}
    for path in files:
        for name, tensor in load_safetensors(path).items():
            if name in raw:
                print(f"error: duplicate tensor {name!r} across shards", file=sys.stderr)
                return 2
            raw[name] = tensor
    mapped = map_hf_state_dict(raw, tie_embeddings=config.tie_embeddings)

    torch.manual_seed(0)  # init values are fully overwritten on success
    model = DecoderLM(config).eval()
    report = validate_state_dict(mapped, model)
    print(
        f"validate: missing={report['missing']} "
        f"unexpected={report['unexpected']} "
        f"shape_mismatch={report['shape_mismatch']}"
    )
    if report["shape_mismatch"] or report["missing"]:
        print("error: checkpoint does not fit this DecoderLM; not loading",
              file=sys.stderr)
        return 3
    loadable = mapped
    if report["unexpected"]:
        print(
            "warning: dropping unexpected keys with no DecoderLM home "
            f"(e.g. Qwen2 attention biases vs bias-free projections): "
            f"{report['unexpected']}"
        )
        loadable = {k: v for k, v in mapped.items() if k not in set(report["unexpected"])}
    model.load_state_dict(loadable, strict=True)
    print(f"loaded {len(loadable)} tensors into DecoderLM "
          f"({model.num_params():,} params)")

    # 4) One forward pass on the requested prompt ids.
    try:
        ids = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    except ValueError:
        print(f"error: --prompt-ids {args.prompt_ids!r} is not comma-separated ints",
              file=sys.stderr)
        return 2
    if not ids or any(i < 0 or i >= config.vocab_size for i in ids):
        print(f"error: prompt ids {ids} outside vocab [0, {config.vocab_size})",
              file=sys.stderr)
        return 2
    if len(ids) > config.max_seq_len:
        print("error: prompt longer than max_seq_len", file=sys.stderr)
        return 2
    prompt = torch.tensor([ids], dtype=torch.long)  # [1, T]
    with torch.no_grad():
        logits, _ = model(prompt)  # [1, T, V]
    top1 = int(logits[0, -1].argmax())
    print(f"forward: prompt ids {ids} -> logits {tuple(logits.shape)}, "
          f"top-1 next token id: {top1}")

    # 5) Optional HF oracle (local files only, never a download).
    _compare_with_hf(model_dir, prompt, top1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
