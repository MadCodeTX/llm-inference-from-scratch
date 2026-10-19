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

Sampling: `FakeBackend` and `BatchedEngineBackend` accept
`temperature` / `top_p` / `top_k` / `seed` (the fields
`benchmarks/run_bench.py` sends: temperature, top_p, seed; top_k is
accepted for the engine's sampler). `temperature=0` remains the exact
greedy path. Greedy-only backends (e.g. `HFBackend`) still return
HTTP 400 for `temperature > 0`, never silently greedy. Invalid
`top_p` / `top_k` / `seed` values return 400 naming the field.
Logprobs are accepted and returned as `null` (the quality probe
re-tokenizes locally and labels ids `retokenized_local`).

`TPEngineCore` samples from the full returned logits on the eager
path with a per-request generator seeded by the request `seed`
(unseeded: derived from the submission sequence). All TP ranks hold
identical full logits and submit identically, so sampling decisions
agree across ranks. A step containing a sampled request never
replays the CUDA graph (its output is reduced with `argmax` only);
it uses eager sampling instead. Greedy-only steps keep the graph
path and exact greedy behaviour unchanged.

Three backends sit behind one seam (`encode` / `decode` /
`stream_token_ids(prompt_ids, max_tokens, cancelled, ignore_eos,
sampling)`):

- `FakeBackend` — deterministic char-level fake for CPU tests
  (seeded fake sampling when `temperature > 0`).
- `HFBackend` — real HF tokenizer plus an injected generator
  (greedy only).
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

Real 27B TP2 server (approved GPU window, incumbent stopped first;
defaults are the measured-feasible 8×8192 sizing):

```bash
.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/serve_qwen38_tp2.py --port 8000 --model Qwen/Qwen3.8-27B-FP8 --max-slots 8 --max-seq-len 8192
```

Same-harness bench against it:

```bash
.venv/bin/python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <snapshot> --output <artifact>.json
```

Final head-to-head (exact commands, unique ports, one server at a
time, shut down only the server you started): run the server command
above, then the bench command, then repeat the pair for the incumbent
under the same `benchmarks/suite.json`; compare against
`benchmarks/WIN.md` thresholds. No head-to-head has been run yet.

The TP2 script applies the adopted local decode defaults explicitly
(no library default changed by the script): graphed decode,
graph-scoped scale-cache on, `--fp8-config gemv-bk512`, gated-norm
fused, norm fused, delta fused, vocab lm_head. `TPEngineCore`
itself now resolves its default decode mode to `graph` on a CUDA
context and `eager` elsewhere (round-1 graphed batched decode was
exact and faster at c1/c4/c8; see
`benchmarks/harness/throughput_round1.json`), and explicit
`decode_mode="graph"` still fails closed off CUDA.

## Readiness verdict (no vLLM win claimed)

Measured in `benchmarks/harness/serving_readiness_round1.json`
(27B TP2, 2×24 GiB, short greedy, repeats=1):

| Config | Resident / peak MiB/rank | Result |
|---|---|---|
| 8×16384 | 20928 / 24050 | INFEASIBLE — graph-capture OOM, both protocols |
| 8×8192 (now the default) | 18880 / 23304 | FEASIBLE — c1/c4 short all OK |
| 16×4096 | 19468 / 24042 | NOT READY — c16 bench 15/16 HTTP 500, cause unresolved |
| 1×65536 | 18292 / 20410 | health/smoke/greedy only — long shape opt-in |
| 2×32768 | 18454 / 19998 | health/smoke/greedy only — medium shape at c≤2, opt-in |

- **No same-harness suite result exists yet**, so **no win vs vLLM
  Arm A or Arm B is claimed** until a `run_bench.py` run on
  `benchmarks/suite.json` passes the `benchmarks/WIN.md` thresholds
  and quality gate. Serving TPOT at 8×8192 was ≈113.9 ms (c1) —
  context only, not comparable to the 21.867 ms local-runner best.
- Compact lockstep (`--tp-sync compact`, opt-in): real-27B ABBA at
  8×8192 showed no measurable benefit (TPOT p50 113.82–113.89 ms,
  tokens/min c4 1868.7–1869.8, both protocols, ~0.1%). It stays
  opt-in; the default `object` protocol is unchanged. Sampling
  floats cross the compact codec as exact float64 bits.
- Greedy exactness: identical 32-token greedy hash prefixes
  (`bb54dbc4548dad57`, `d5b0c235b4e0dda3`) across every feasible
  config, both protocols, and ABBA repeats (64-bit prefixes only;
  full hashes were not stored).
- SSE streams send `Connection: close` and close after `[DONE]`,
  so raw clients (e.g. `curl`) see EOF; parsers that stop at
  `[DONE]` (such as `run_bench.py`) are unaffected.
- `BatchedEngineBackend.close()` is idempotent, fails in-flight and
  queued jobs with `backend closed` instead of hanging them, and
  sends the TP shutdown broadcast only after the driver stops.
- Cancellation of an in-flight engine request discards its output
  but cannot preempt its engine slot until it finishes (bounded by
  `max_tokens`).
- The sampled arm is implemented (temperature/top_p/top_k/seed) but
  has no real-27B benchmark yet; sampled steps run eager, so sampled
  TPOT is not comparable to the graphed greedy TPOT. c16 is not
  ready (above), and graceful TP shutdown exit was not demonstrated
  (round-1 recorded `shutdown_rc_torchrun=1`; only port-free +
  GPU-idle after-state was verified).

Local decode context, not a serving claim: best local TPOT is
21.867 ms (graphed TP2, fused defaults), still above vLLM Arm A
20.3 ms. Prefix bench defaults are chunk 512 / snapshot 512 on
`scripts/prefix_ttft_bench.py` only; library defaults in
`tp_prefix_cache` / `prefill_chunkwise_static` are unchanged.
