# 30 — OpenAI-compatible serving surface

## What shipped

`src/inference_from_scratch/openai_server.py` is a stdlib-only
(`http.server.ThreadingHTTPServer`) OpenAI-shaped HTTP layer so
`benchmarks/run_bench.py` can measure this engine in the same harness
as vLLM. It implements `POST /v1/completions` (streaming SSE, one
chunk per token, final chunk carries `usage` when
`stream_options.include_usage` is set, then `[DONE]`; plus the
non-streaming form), `GET /health`, and `GET /v1/models`. Prompts may
be strings or token-id lists; usage counts come from the tokenizer /
engine, never from re-counting output text.

Greedy only: `temperature > 0` returns HTTP 400. Sampled arms are
explicitly unsupported, never silently ignored — `run_bench.py` will
record per-request errors for the sampled arm until a sampling
backend exists. Logprobs are accepted and returned as `null` (the
quality probe re-tokenizes locally and labels ids
`retokenized_local`).

Three backends sit behind one seam (`encode` / `decode` /
`stream_token_ids(prompt_ids, max_tokens, cancelled, ignore_eos)`):

- `FakeBackend` — deterministic char-level fake for CPU tests.
- `HFBackend` — real HF tokenizer plus an injected generator.
- `BatchedEngineBackend` — one driver thread steps a
  `TPEngineCore`-shaped engine (`submit` / `poll` / `step`) so
  concurrent HTTP requests batch in the engine. An optional
  `sync_hook` broadcasts submissions and steps to TP worker ranks;
  only rank 0 serves HTTP, workers run `participate()`.

`ignore_eos` is honoured end to end: the server passes it through the
backend into `TPEngineCore.submit(..., ignore_eos=True)`, so the
engine itself does not finish the request at EOS (the earlier
server-only flag could not stop the engine from finishing early).
Engine failures (for example `prompt + max_tokens > max_seq_len`)
fail the affected request with an HTTP 500 instead of hanging its
consumer, and `BatchedEngineBackend.close()` wakes and joins the
idle driver and, under TP, sends the shutdown broadcast that frees
rank-1 `participate()`. Invalid `finish_reason` values
(`cancelled` / `timeout`) are never emitted in an SSE chunk; they
are mapped to `stop`, and a pre-first-token timeout is an HTTP 504.

## Exact commands (later controlled head-to-head)

Fake smoke (CPU, no weights):

```bash
.venv/bin/python scripts/serve_openai.py --port 8000 --model fake-model
.venv/bin/python benchmarks/run_bench.py --smoke --base-url http://127.0.0.1:8000 --model fake-model
```

Real 27B TP2 server (approved GPU window, incumbent stopped first):

```bash
.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/serve_qwen38_tp2.py --port 8000 --model Qwen/Qwen3.8-27B-FP8
```

Same-harness bench against it:

```bash
.venv/bin/python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <snapshot> --output <artifact>.json
```

The TP2 script applies the adopted local decode defaults explicitly
(no library default changed by the script): graphed decode,
graph-scoped scale-cache on, `--fp8-config gemv-bk512`, gated-norm
fused, norm fused, delta fused, vocab lm_head. `TPEngineCore`
itself now resolves its default decode mode to `graph` on a CUDA
context and `eager` elsewhere (round-1 graphed batched decode was
exact and faster at c1/c4/c8; see
`benchmarks/harness/throughput_round1.json`), and explicit
`decode_mode="graph"` still fails closed off CUDA.

## Readiness blockers (no vLLM win claimed)

- No same-harness suite result exists yet. The only real-27B serving
  evidence is a tiny-prompt smoke (16 tokens, TPOT p50 ~88 ms
  including graph capture and per-step broadcast overhead). **No win
  vs vLLM Arm A or Arm B is claimed until a `run_bench.py` suite run
  on `benchmarks/suite.json` passes the `benchmarks/WIN.md`
  thresholds and quality gate.**
- TP lockstep uses `broadcast_object_list` once per engine step,
  which adds host latency; a tensor-broadcast or persistent
  worker-step protocol is needed before attributing head-to-head
  TPOT to the engine.
- Static KV sizing: defaults are 8 slots × 16384 tokens. The suite
  medium shape (16384+1024) and long shape (49152+1024), and c16,
  need larger `--max-slots` / `--max-seq-len`, which may OOM on
  24 GiB/rank next to the ~14.91 GiB resident weights. A sizing
  sweep is required before those cells can run.
- Cancellation of an in-flight engine request discards its output
  but cannot preempt its engine slot until it finishes (bounded by
  `max_tokens`).
- The sampled arm returns HTTP 400 by design (greedy-only engine).

Local decode context, not a serving claim: best local TPOT is
21.867 ms (graphed TP2, fused defaults), still above vLLM Arm A
20.3 ms. Prefix bench defaults are chunk 512 / snapshot 512 on
`scripts/prefix_ttft_bench.py` only; library defaults in
`tp_prefix_cache` / `prefill_chunkwise_static` are unchanged.
