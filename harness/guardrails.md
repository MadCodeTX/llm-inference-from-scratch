# Guardrails (c2) — non-negotiables

These rules bind every agent in this tree, proposers and composers included.
A proposal that violates one is discarded whole, however promising it looks.

1. **Measured numbers only.** Every speed, latency, throughput, or acceptance
   figure you state must come from an artifact in this repo
   (`benchmarks/baseline/*.json`, `benchmarks/profile/*.json`,
   `benchmarks/mtp/*.json`, `benchmarks/roofline/*`) or from a probe you ran
   in this step whose artifact you name. No estimated, extrapolated, or
   "should be about" numbers. Roofline figures are bounds, never results:
   label them `bound only` (e.g. the 19.17 ms TPOT floor / 52.2 tok/s ceiling
   in `benchmarks/WIN.md`).
2. **Exact agreement before speed claims.** Greedy token ids must match the
   reference (bridge / Arm A ids recorded in the artifact; the runner's
   `exact_match` fields) before any timing from that run may be compared,
   quoted, or adopted. An inexact run is evidence about correctness only.
   Cross-shape numbers are never wins: a local-runner tok/min at one
   prompt/batch shape does not beat a `run_bench.py` serving number at
   another (see `benchmarks/WIN.md`, "Throughput ladder").
3. **Never stop external GPU workloads.** The bigbox normally serves
   production traffic. Do not stop, restart, kill, or preempt any process
   you did not start in this step — in particular the incumbent vLLM service.
4. **Controlled vLLM stop only in approved windows.** The incumbent may be
   stopped (via its own `launch.sh --stop`) only inside a user-approved GPU
   window, and must be restored with
   `/home/nick/llm/vllm/launch.sh qwen3.8-27b-fp8 --no-cc` afterwards, with
   `/v1/models` verified answering. Outside a window, the only permitted
   live call is the single tiny smoke in `benchmarks/WIN.md`
   (`run_bench.py --smoke`, 16 tokens, greedy).
5. **Serial bigbox model/GPU use.** One GPU probe at a time: a probe owns
   both RTX 4090s (TP2) for its whole run. Never run two model loads, two
   `torchrun` jobs, or a probe alongside a serving benchmark concurrently.
   Parallelism lives in CPU-side proposing/composing, never on the GPUs.
6. **No secrets in git.** No API keys, tokens, credentials, `.env` contents,
   or private endpoints in any file under version control. Snapshot paths
   under `~/.cache/huggingface/` are local filesystem paths, not secrets,
   but say so when you quote them.
7. **Disjoint file ownership for parallel agents.** Parallel candidates must
   touch disjoint file sets, declared up front. Two candidates that need the
   same file are serialised (run one, then the other) or merged into one
   candidate — never edited concurrently. The coordinator owns
   `harness/library/` and `benchmarks/WIN.md`; proposers never write there.
