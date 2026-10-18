#!/usr/bin/env python3
"""Quality/equivalence probes against an OpenAI-compatible server.

Stdlib urllib only. Sends three fixed neutral prompts (defined below) to
``/v1/completions`` at temperature 0 and records, per probe: the prompt
text, its exact token ids from the real tokenizer, the returned text,
``finish_reason``, ``usage``, and per-output-token ids. Output-token ids
come from the server's ``logprobs.token_ids`` when the API returns them;
otherwise the completion text is re-tokenized locally and the ids are
labelled ``retokenized_local`` (never presented as server ids).

Usage:
  python benchmarks/quality_probe.py \
      --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 \
      --tokenizer-dir <snapshot> \
      --output benchmarks/baseline/arm_a_quality_probes.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

# Three fixed neutral prompts; changing them breaks comparability.
PROBES: list[dict] = [
    {"id": "capital", "prompt": "The capital of France is"},
    {"id": "water", "prompt": "Water is made of hydrogen and"},
    {"id": "counting", "prompt": "One, two, three, four,"},
]

MAX_TOKENS = 32


def completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/completions"
    return base + "/v1/completions"


def build_payload(model: str, prompt: str, max_tokens: int = MAX_TOKENS,
                  seed: int | None = 42) -> dict:
    """One non-streaming greedy completion request body."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": False,
        "logprobs": 1,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def encode_ids(tokenizer, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(t) for t in ids]


def parse_completion(body: dict, tokenizer) -> dict:
    """Extract text/finish/usage/output-ids from one response body."""
    choice = (body.get("choices") or [{}])[0]
    text = str(choice.get("text") or "")
    server_ids = None
    logprobs = choice.get("logprobs") or {}
    if isinstance(logprobs, dict) and logprobs.get("token_ids"):
        server_ids = [int(t) for t in logprobs["token_ids"]]
    if server_ids is not None:
        out_ids, source = server_ids, "server_logprobs"
    else:
        out_ids, source = encode_ids(tokenizer, text), "retokenized_local"
    return {
        "text": text,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "output_token_ids": out_ids,
        "output_token_ids_source": source,
    }


def run_probe(base_url: str, model: str, probe: dict, tokenizer,
              timeout: int = 120) -> dict:
    payload = build_payload(model, probe["prompt"])
    req = urllib.request.Request(
        completions_url(base_url),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    parsed = parse_completion(body, tokenizer)
    return {
        "id": probe["id"],
        "prompt": probe["prompt"],
        "prompt_token_ids": encode_ids(tokenizer, probe["prompt"]),
        "request": {k: payload[k] for k in
                    ("temperature", "max_tokens", "seed",
                     "repetition_penalty", "ignore_eos") if k in payload},
        **parsed,
    }


def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=False
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    results = [run_probe(args.base_url, args.model, p, tokenizer)
               for p in PROBES]
    doc = {"base_url": args.base_url, "model": args.model,
           "max_tokens": MAX_TOKENS, "probes": results}
    text = json.dumps(doc, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
        print(f"# wrote {args.output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
