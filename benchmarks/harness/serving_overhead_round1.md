# Serving overhead round 1 — ABBA probe of prior fixes vs shipped 7fbb4a7

Shipped: `7fbb4a7` (clean worktree). Fixed: uncommitted working tree over `7fbb4a7`
(prior result 1: driver idle-poll, eager snapshot, real-state graph capture + guard).
No source edited in this lane; nothing committed. Model Qwen/Qwen3.8-27B-FP8, TP2,
2×24 GiB, `tp-sync object`, SSE `Connection: close`. Serial configs, unique ports
18511–18518, each server SIGTERM-shut down with GPU 1 MiB + port down verified.
Raw data: [serving_overhead_round1.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/harness/serving_overhead_round1.json).

## ABBA at 8×8192 (short, greedy, repeats=1, exact tokenizer)

| Run | Tree | TPOT p50 c1 (ms) | c4 tok/min | Peak MiB |
|---|---|---|---|---|
| A1 | shipped | 113.89 | 1868.8 | 23310 |
| B1 | fixed (real-state) | 63.98 | 3036.2 | 23300 |
| B-scratch | fixed (scratch) | 63.88 | — | 22526 |
| A2 | shipped | 113.97 | — | 22526 |

Greedy exactness: identical 32-token hashes in every config
(`capital bb54dbc4548dad57`, `water bf740ec80592c722`, `counting 7ffcc751cb5d7dfc`).
TTFT c1 unchanged (~1.87 s). Fixed c1 TPOT 63.98 ms is still 2.9× the 21.867 ms
local B=1 figure — the remaining gap is graphed B=8 geometry (diagnosis bucket 2),
not transport. No vLLM comparison run; no win claim.

## Larger sizing on the fixed tree

| Config | Result |
|---|---|
| 8×16384 | INFEASIBLE still: smoke/quality OK, short c1 and medium both HTTP 500. Curl repro body: `generation failed: CUDA out of memory. Tried to allocate 406.00 MiB … 181.75 MiB free … 1.51 GiB in private pools (CUDA Graphs)` on a ~2100-token prompt |
| 16×4096 c16 | FEASIBLE now: two consecutive c16 short batches, 16/16 OK both, 7453.1 and 7450.6 tok/min, TPOT p50 71.65 ms, peak 23132 MiB (shipped: 1/16, 15× HTTP 500) |
| 1×65536 | Feasible (health/smoke/quality only; full long bench not run), peak 20410 MiB |

Sampled arm works on both trees; fixed c1 sampled TPOT 71.36 vs shipped 132.44 ms.

## Recommendations per separable fix

- Driver idle-poll fix: **adopt** (−50 ms/token ABBA, exact greedy).
- Eager snapshot fix: **adopt** (c16 fixed; inseparable from driver fix in this ABBA, supported by prior unit tests).
- Graph capture real-state: **opt-in / default-candidate** — TPOT identical to scratch at 8×8192; value is memory headroom only, and it does not rescue 8×16384.
- Graph capture guard: **keep** (no false block at feasible sizes).
- Per-occupancy/right-sized graphs: **not implemented — reject for now**; remaining c1 gap needs this or M=1 graph work.
- Incremental detokenization: **skip** (negligible).
- 8×16384 sizing: **reject** on 24 GiB/rank. 16×4096: **adopt as feasible** on the fixed tree.
