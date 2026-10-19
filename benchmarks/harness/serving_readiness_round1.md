# Serving readiness round 1 — TP protocol candidate vs shipped

Head: `f803996`. Model: Qwen/Qwen3.8-27B-FP8, TP2, 2×24 GiB. Serial real-server probe of
`scripts/serve_qwen38_tp2.py` (`--tp-sync {object,compact}`, default `object` preserved; no source edits).
Bench: `benchmarks/run_bench.py --smoke` plus short-shape greedy subsets (repeats=1, exact tokenizer).
Unique localhost ports 18211–18226; GPUs verified idle before, and port-free + GPU 1 MiB after every config.
No vLLM or external process touched. Nothing committed.

## Headline findings

1. **Current default sizing 8×16384 is infeasible on 24 GiB/rank.** Both protocols OOM on rank 1 at the
   first graphed decode: `_capture_graph` allocates a second full `static_state` (capture scratch KV),
   `torch.OutOfMemoryError` (tried 128 MiB, 31.75 MiB free). Smoke passes misleadingly (1-token reply);
   the first 32-token greedy probe returns HTTP 500, bench then hits `RemoteDisconnected`/connection refused.
2. **Compact shows no measurable benefit.** At the feasible 8×8192 sizing, ABBA short-shape TPOT p50 is
   113.82–113.89 ms (c1) and tokens/min at c4 is 1868.7–1869.8 for both protocols — within ~0.1%, single repeat.
   No win claim.
3. **Greedy exactness holds.** Identical 32-token greedy text hashes (`bb54dbc4548dad57`, `d5b0c235b4e0dda3`)
   across every feasible config, both protocols, and ABBA repeats.
4. **c16 is not ready.** 16×4096: health/smoke/single greedy OK, but c16 short bench went 1/16 OK with
   15× HTTP 500 (`generation failed: …`; run_bench does not capture the body, exact cause unresolved),
   peak 24042 MiB, and shutdown needed SIGKILL escalation.

## Feasible sizing matrix

| Config | Capacity (tok) | Resident MiB/rank | Peak MiB/rank | Result |
|---|---|---|---|---|
| 8×16384 (current) | 131072 | 20928 | 24050 | INFEASIBLE — graph-capture OOM, both protocols |
| 8×8192 | 65536 | 18880 | 23304 | FEASIBLE — c1/c4 short all OK |
| 16×4096 | 65536 | 19468 | 24042 | NOT READY — c16 bench 15/16 HTTP 500 |
| 1×65536 | 65536 | 18292 | 20410 | FEASIBLE (health/smoke/greedy) — long geometry fits (50176 ≤ 65536) |
| 2×32768 | 65536 | 18454 | 19998 | FEASIBLE (health/smoke/greedy) — medium geometry fits at c≤2 (17408 ≤ 32768) |

Notes: weights resident 14.91 GiB/rank; startup 21–24 s per config. Medium (17408) and agent-shared-prefix
(9216) do not fit 8×8192 seq len; long (50176) needs the 1×65536 shape. Graph capture/replay counts are not
exposed over HTTP or logs (engine-internal counters only) — not recorded.

## ABBA at 8×8192 (short, greedy, repeats=1)

| Run | Protocol | TPOT p50 c1 (ms) | Tokens/min c4 |
|---|---|---|---|
| A1 | object | 113.89 | 1868.7 |
| B1 | compact | 113.82 | 1869.8 |
| B2 | compact | 113.84 | 1869.6 |
| A2 | object | 113.88 | 1868.9 |

Context only: best local TPOT 21.867 ms (prior microbench) vs ≈113.9 ms serving TPOT here; incumbent
Arm A/B numbers were not re-measured. No win/loss claim until the final same-harness phase.

## Recommendation

- **Compact protocol: reject as default, keep as opt-in** (`--tp-sync compact`). Exact greedy behavior
  confirmed, but no measured TPOT/throughput benefit on the real server.
- **8×16384 default sizing: reject on 24 GiB/rank** (graph-capture OOM). **Adopt 8×8192** as the largest
  proven all-OK sizing for short c1/c4; treat 1×65536 / 2×32768 as opt-in long/medium shapes; c16 not ready.
  Status: adopted — `scripts/serve_qwen38_tp2.py` now defaults to 8×8192, compact stays opt-in, and the
  compact codec carries sampling floats exactly (float64 bits). `BatchedEngineBackend.close()` now fails
  in-flight jobs and is idempotent. No vLLM win claimed; no same-harness suite run yet.

Full per-config data: `serving_readiness_round1.json` in this directory.
