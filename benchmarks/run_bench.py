#!/usr/bin/env python3
"""OpenAI-compatible benchmark harness (stdlib urllib only, no new deps).

Streams ``/v1/completions`` and measures per-request TTFT, inter-token
latencies (TPOT distribution), token counts and wall time; aggregates
tokens/minute and percentiles via the repo's ``metrics.summarize``.

Prompt text is procedural: a neutral filler paragraph written for this
repo. With ``--tokenizer-dir`` prompts are built to *exact* target token
counts by repeating/truncating the filler token ids; without it they are
repeated/truncated by *words* and target token counts are approximate.
``prompt_tokens_actual`` is recorded per request: the server's streaming
usage when present, else the locally counted token ids. No incumbent
prompt text is used.

Usage:
  python benchmarks/run_bench.py --dry-run
  python benchmarks/run_bench.py --dry-run --tokenizer-dir <snapshot>
  python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen3.8-27B-FP8 --limit-concurrency 4 --output out.json
  python benchmarks/run_bench.py --smoke   # one 16-token greedy request

Safety: --dry-run never touches the network. Live runs send real load;
only run them inside an approved GPU window (see benchmarks/WIN.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

FILLER_PARAGRAPH = (
    "The valley library keeps its doors open through the winter, and "
    "people come in from the cold to read, to write letters, and to "
    "watch the river move slowly under the bridge. A good map does not "
    "hurry the traveller. It shows the roads, the hills, and the places "
    "where the path narrows, and then it trusts the walker to choose. "
    "In the workshop behind the hall, careful hands repair clocks, "
    "bind books, and mend the sails of small boats, keeping notes on "
    "every task so the next worker can follow the same steady method. "
    "Knowledge grows the same way: one clear observation at a time, "
    "recorded plainly, checked twice, and shared without ornament."
)

# Rough English words-per-token ratio used only to size word counts in
# dry-run plans; real runs report the server's usage.prompt_tokens.
WORDS_PER_TOKEN = 0.75


def filler_words() -> list[str]:
    return FILLER_PARAGRAPH.split()


def words_for_tokens(target_tokens: int) -> int:
    return max(1, int(round(int(target_tokens) * WORDS_PER_TOKEN)))


def build_text(n_words: int, seed: int = 0) -> str:
    """Deterministic filler text of exactly ``n_words`` words.

    Words cycle through the filler paragraph, offset by ``seed`` so
    unique suffixes differ while staying reproducible.
    """
    words = filler_words()
    if n_words <= 0:
        return ""
    off = int(seed) % len(words)
    seq = words[off:] + words[:off]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(seq)
    return " ".join(out[:n_words])


def count_words(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def load_tokenizer(tokenizer_dir):
    """Load a HF tokenizer on CPU from ``tokenizer_dir`` (no network use
    beyond what ``from_pretrained`` does for a local directory)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=False
    )


def _encode_ids(tokenizer, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(t) for t in ids]


def _decode_ids(tokenizer, ids) -> str:
    ids = [int(t) for t in ids]
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(ids)


def filler_token_ids(tokenizer) -> list[int]:
    ids = _encode_ids(tokenizer, FILLER_PARAGRAPH)
    if not ids:
        raise ValueError("tokenizer produced no filler token ids")
    return ids


def build_token_ids(target_tokens: int, tokenizer, seed: int = 0) -> list[int]:
    """Exactly ``target_tokens`` filler token ids, deterministically.

    The filler paragraph's ids are repeated (rotated by ``seed``) and
    truncated; no word-count approximation is involved.
    """
    target = int(target_tokens)
    if target <= 0:
        return []
    base = filler_token_ids(tokenizer)
    off = int(seed) % len(base)
    seq = base[off:] + base[:off]
    out: list[int] = []
    while len(out) < target:
        out.extend(seq)
    return out[:target]


def build_shape_prompt_ids(shape, concurrency, repeat, tokenizer):
    """Per-request token ids for one batch at exact target lengths.

    Shared-prefix shapes share the exact token-id prefix across the
    batch; each request then carries its own unique suffix. The total
    per request is exactly ``prompt_tokens`` (shared + unique for the
    frozen shapes).
    """
    shared = int(shape.get("shared_prefix_tokens") or 0)
    unique = int(shape.get("unique_tokens") or shape.get("prompt_tokens"))
    target = int(shape["prompt_tokens"])
    prefix_ids = build_token_ids(shared, tokenizer, seed=0) if shared else []
    prompts = []
    for i in range(concurrency):
        unique_ids = build_token_ids(
            unique, tokenizer, seed=1000 * (repeat + 1) + i
        )
        ids = list(prefix_ids) + unique_ids
        if len(ids) < target:
            ids.extend(
                build_token_ids(
                    target - len(ids), tokenizer,
                    seed=7000 + 1000 * (repeat + 1) + i,
                )
            )
        prompts.append(ids[:target])
    return prompts


def build_shape_prompts(shape: dict, concurrency: int, repeat: int,
                        tokenizer=None) -> list[str]:
    """Prompts for one batch; shared-prefix shapes share the prefix.

    With ``tokenizer`` the prompts are decoded from exact-length token
    ids (see :func:`build_shape_prompt_ids`); without it the legacy
    word-count fallback is used and lengths are only approximate.
    """
    if tokenizer is not None:
        return [
            _decode_ids(tokenizer, ids)
            for ids in build_shape_prompt_ids(
                shape, concurrency, repeat, tokenizer
            )
        ]
    shared = int(shape.get("shared_prefix_tokens") or 0)
    unique = int(shape.get("unique_tokens") or shape.get("prompt_tokens"))
    prompts = []
    prefix_text = build_text(words_for_tokens(shared), seed=0) if shared else ""
    for i in range(concurrency):
        unique_text = build_text(
            words_for_tokens(unique), seed=1000 * (repeat + 1) + i
        )
        prompts.append((prefix_text + " " + unique_text).strip())
    return prompts


def load_suite(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        suite = json.load(fh)
    if not suite.get("shapes"):
        raise ValueError("suite has no shapes")
    if not suite.get("concurrency"):
        raise ValueError("suite has no concurrency levels")
    return suite


def effective_concurrencies(suite: dict, limit: int | None) -> list[int]:
    levels = [int(c) for c in suite["concurrency"]]
    if limit is not None:
        levels = sorted({min(c, int(limit)) for c in levels})
    return levels


def selected_arms(suite: dict, arms: list[str] | None = None) -> list[str]:
    """Arm names to run: all suite arms, or the ``--arms`` subset."""
    available = list(suite.get("arms") or {"greedy": {}})
    if not arms:
        return available
    wanted = [a.strip() for entry in arms for a in str(entry).split(",")
              if a.strip()]
    unknown = [a for a in wanted if a not in available]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; suite arms: {available}")
    return wanted


def build_plan(
    suite: dict, limit_concurrency: int | None = None,
    shapes: list[str] | None = None, tokenizer=None,
    arms: list[str] | None = None,
) -> dict:
    """Dry-run plan: every batch and request count, without any network."""
    arm_names = selected_arms(suite, arms)
    repeats = int(suite.get("repeats", 1))
    selected = [s for s in suite["shapes"] if not shapes or s["id"] in shapes]
    exact = tokenizer is not None
    batches = []
    for shape in selected:
        for arm in arm_names:
            for conc in effective_concurrencies(suite, limit_concurrency):
                for rep in range(repeats):
                    if exact:
                        id_lists = build_shape_prompt_ids(
                            shape, conc, rep, tokenizer
                        )
                        prompts = [
                            _decode_ids(tokenizer, ids) for ids in id_lists
                        ]
                        actual = [len(ids) for ids in id_lists]
                    else:
                        prompts = build_shape_prompts(shape, conc, rep)
                        actual = [None] * len(prompts)
                    batches.append({
                        "shape": shape["id"],
                        "arm": arm,
                        "concurrency": conc,
                        "repeat": rep,
                        "requests": conc,
                        "target_prompt_tokens": int(shape["prompt_tokens"]),
                        "target_tokens_approximate": not exact,
                        "prompt_tokens_actual": actual,
                        "output_tokens": int(shape["output_tokens"]),
                        "prompt_words_first": count_words(prompts[0]) if prompts else 0,
                    })
    return {
        "dry_run": True,
        "shapes": [s["id"] for s in selected],
        "arms": arm_names,
        "concurrencies": effective_concurrencies(suite, limit_concurrency),
        "repeats": repeats,
        "batches": batches,
        "total_batches": len(batches),
        "total_requests": sum(b["requests"] for b in batches),
        "tokenizer_available": exact,
        "target_tokens_approximate": not exact,
        "note": (
            "Token counts are exact: prompts built from filler token ids "
            "at the target lengths; prompt_tokens_actual is locally counted."
            if exact else
            "Word counts are procedural (WORDS_PER_TOKEN sizing); "
            "target_tokens are approximate until a tokenizer "
            "(--tokenizer-dir) or server usage reports actuals."
        ),
    }


def summarize_values(values):
    """Percentiles via repo metrics.summarize, with a stdlib fallback."""
    vals = [float(v) for v in values]
    if not vals:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from inference_from_scratch.metrics import summarize

        return summarize(vals)
    except Exception:
        s = sorted(vals)

        def pct(p):
            if len(s) == 1:
                return s[0]
            rank = (p / 100.0) * (len(s) - 1)
            lo = int(rank)
            hi = min(lo + 1, len(s) - 1)
            return s[lo] + (rank - lo) * (s[hi] - s[lo])

        return {"n": len(s), "mean": sum(s) / len(s), "p50": pct(50),
                "p90": pct(90), "p99": pct(99), "min": s[0], "max": s[-1]}


def metrics_from_timestamps(token_times, t0):
    """TTFT + inter-token latencies from token arrival timestamps.

    ``token_times`` are perf_counter stamps, one per streamed token
    chunk that carried text; ``t0`` is the request start stamp.
    """
    if not token_times:
        return {"ttft_s": None, "inter_token_s": [], "latency_s": None}
    ttft = float(token_times[0] - t0)
    itls = [float(b - a) for a, b in zip(token_times, token_times[1:])]
    return {"ttft_s": ttft, "inter_token_s": itls,
            "latency_s": float(token_times[-1] - t0)}


def completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/completions"
    return base + "/v1/completions"


def _chunk_text(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    ch = choices[0]
    if ch.get("text"):
        return str(ch["text"])
    delta = ch.get("delta") or {}
    return str(delta.get("content") or delta.get("text") or "")


def one_request(base_url, model, prompt, output_tokens, arm_cfg,
                timeout=1800, prompt_tokens_local=None):
    """Stream one /v1/completions request; record TTFT/TPOT/tokens.

    ``prompt_tokens_actual`` is the server's streaming
    ``usage.prompt_tokens`` when present, else ``prompt_tokens_local``
    (the locally counted token ids, when a tokenizer built the prompt).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(output_tokens),
        "temperature": float(arm_cfg.get("temperature", 0.0)),
        "top_p": float(arm_cfg.get("top_p", 1.0)),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if arm_cfg.get("seed") is not None:
        payload["seed"] = int(arm_cfg["seed"])
    rec = {"ok": False, "ttft_s": None, "inter_token_s": [],
           "latency_s": None, "completion_tokens": 0, "prompt_tokens": 0,
           "prompt_tokens_actual": prompt_tokens_local,
           "prompt_words": count_words(prompt), "error": None}
    token_times: list[float] = []
    t0 = time.perf_counter()
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            completions_url(base_url), data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                if _chunk_text(chunk):
                    token_times.append(time.perf_counter())
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    rec["completion_tokens"] = int(usage["completion_tokens"])
                if usage.get("prompt_tokens"):
                    rec["prompt_tokens"] = int(usage["prompt_tokens"])
                    rec["prompt_tokens_actual"] = int(usage["prompt_tokens"])
        m = metrics_from_timestamps(token_times, t0)
        rec["ttft_s"] = m["ttft_s"]
        rec["inter_token_s"] = m["inter_token_s"]
        rec["latency_s"] = float(time.perf_counter() - t0)
        if not rec["completion_tokens"] and token_times:
            rec["completion_tokens"] = len(token_times)
        rec["ok"] = True
    except Exception as exc:  # report, never crash the sweep
        rec["latency_s"] = float(time.perf_counter() - t0)
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def run_batch(base_url, model, suite, shape, arm, concurrency, repeat,
              tokenizer=None):
    """Run one (shape, arm, concurrency, repeat) batch with threads."""
    arm_cfg = (suite.get("arms") or {}).get(arm, {})
    if tokenizer is not None:
        id_lists = build_shape_prompt_ids(shape, concurrency, repeat, tokenizer)
        prompts = [_decode_ids(tokenizer, ids) for ids in id_lists]
        local_counts = [len(ids) for ids in id_lists]
    else:
        prompts = build_shape_prompts(shape, concurrency, repeat)
        local_counts = [None] * len(prompts)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(prompt, local_count=None):
        rec = one_request(base_url, model, prompt,
                          int(shape["output_tokens"]), arm_cfg,
                          prompt_tokens_local=local_count)
        with lock:
            results.append(rec)

    threads = [threading.Thread(target=worker, args=(p, c))
               for p, c in zip(prompts, local_counts)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    total_out = sum(r["completion_tokens"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    itls = [x for r in ok for x in r["inter_token_s"]]
    lats = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    agg_tps = (total_out / wall) if wall else 0.0
    return {
        "shape": shape["id"],
        "arm": arm,
        "concurrency": concurrency,
        "run": repeat,
        "wall_s": round(wall, 4),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "prompt_tokens": ok[0]["prompt_tokens"] if ok else 0,
        "prompt_tokens_actual": (
            ok[0].get("prompt_tokens_actual") if ok else None
        ),
        "target_prompt_tokens": int(shape["prompt_tokens"]),
        "target_tokens_approximate": tokenizer is None,
        "prompt_words": ok[0]["prompt_words"] if ok else 0,
        "total_output_tokens": total_out,
        "aggregate_output_tps": round(agg_tps, 2),
        "tokens_per_minute": round(agg_tps * 60.0, 1),
        "per_stream_tps": round(agg_tps / max(1, len(ok)), 2) if ok else 0.0,
        "ttft": summarize_values(ttfts),
        "tpot": summarize_values(itls),
        "latency": summarize_values(lats),
        "errors": [r["error"] for r in results if r["error"]][:3],
        "requests_detail": [
            {"ttft_s": r["ttft_s"], "latency_s": r["latency_s"],
             "completion_tokens": r["completion_tokens"],
             "prompt_tokens": r["prompt_tokens"],
             "prompt_tokens_actual": r.get("prompt_tokens_actual"),
             "prompt_words": r["prompt_words"], "ok": r["ok"]}
            for r in results
        ],
    }


def run_smoke(base_url: str, model: str) -> dict:
    """Exactly one tiny 16-token greedy completion (harness proof)."""
    prompt = "The capital of France is"
    rec = one_request(base_url, model, prompt, 16,
                      {"temperature": 0.0, "top_p": 1.0}, timeout=120)
    itl = summarize_values(rec["inter_token_s"])
    return {"smoke": True, "model": model, "base_url": base_url,
            "prompt": prompt, "max_tokens": 16, "ok": rec["ok"],
            "ttft_s": rec["ttft_s"], "latency_s": rec["latency_s"],
            "completion_tokens": rec["completion_tokens"],
            "prompt_tokens": rec["prompt_tokens"],
            "prompt_tokens_actual": rec.get("prompt_tokens_actual"),
            "tpot": itl, "error": rec["error"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--suite", default=str(Path(__file__).with_name("suite.json")))
    ap.add_argument("--limit-concurrency", type=int, default=None)
    ap.add_argument("--shape", action="append", default=None)
    ap.add_argument("--arms", action="append", default=None,
                    help="arm name(s) to run (repeatable or comma-separated); "
                         "default: all suite arms")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned requests without calling the server")
    ap.add_argument("--smoke", action="store_true",
                    help="run exactly one 16-token greedy completion, then stop")
    ap.add_argument("--output", default=None, help="write results JSON here")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="HF tokenizer dir/snapshot; prompts are built to "
                         "exact target token counts (CPU tokenization)")
    args = ap.parse_args(argv)

    tokenizer = load_tokenizer(args.tokenizer_dir) if args.tokenizer_dir else None

    if args.smoke:
        result = run_smoke(args.base_url, args.model)
        print(json.dumps(result, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["ok"] else 1

    suite = load_suite(args.suite)
    if args.dry_run:
        plan = build_plan(suite, args.limit_concurrency, args.shape,
                          tokenizer=tokenizer, arms=args.arms)
        plan["base_url"] = args.base_url
        plan["model"] = args.model
        if args.tokenizer_dir:
            plan["tokenizer_dir"] = args.tokenizer_dir
        print(json.dumps(plan, indent=2))
        return 0

    shapes = [s for s in suite["shapes"]
              if not args.shape or s["id"] in args.shape]
    rows = []
    for shape in shapes:
        for arm in selected_arms(suite, args.arms):
            for conc in effective_concurrencies(suite, args.limit_concurrency):
                for rep in range(int(suite.get("repeats", 1))):
                    row = run_batch(args.base_url, args.model, suite,
                                    shape, arm, conc, rep,
                                    tokenizer=tokenizer)
                    row["label"] = "run_bench"
                    row["model"] = args.model
                    rows.append(row)
                    ttft = row["ttft"]["mean"] if row["ttft"] else None
                    print(f"{shape['id']:20s} arm={arm} c={conc} run={rep} "
                          f"wall={row['wall_s']:.2f}s "
                          f"agg_tps={row['aggregate_output_tps']} "
                          f"tok/min={row['tokens_per_minute']} "
                          f"ttft_mean={ttft} fail={row['failed']}", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"# wrote {args.output}")
    else:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


