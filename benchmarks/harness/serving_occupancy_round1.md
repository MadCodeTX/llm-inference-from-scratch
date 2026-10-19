# Serving occupancy round 1 — ABBA probe

Candidate: `--graph-buckets` (per-occupancy decode graphs) over HEAD `8b18680`. Control: shipped `8b18680` serving (same tree, flag off). Measured in the ABBA lane; adopted as the serving-script default on integration.

Protocol: serial ABBA on unique ports 18521–18524 (8×8192), then c16 pair at 16×4096 (18527/18528). Per server: health, smoke, 32-token greedy quality probes (`sha256(text)[:16]`), short `run_bench` (exact tokenizer, 512 prompt / 128 output, repeats=1), sampled smoke.

## Quality — exact everywhere

All six server runs returned identical hashes: capital `bb54dbc4548dad57`, water `bf740ec80592c722`, counting `7ffcc751cb5d7dfc`. No greedy mismatch. Sampled smoke OK in all runs.

## TPOT p50 (ms) — ABBA means

| Shape | Shipped A | Candidate B | Change |
|---|---|---|---|
| c1 8×8192 | 63.86 | 25.77 | −59.7% |
| c4 8×8192 | 64.20 | 46.95 | −26.9% |
| c8 8×8192 | 64.91 | 64.89 | flat |
| c16 16×4096 | infeasible | 74.29 | capability gained |

Tokens/min follow the same pattern (c1 909 → 2129). TTFT unchanged (~0.34 s c1). Peak memory comparable: ~23.3–23.7 GiB/rank both arms. Context only (not like-for-like): prior shipped serving c1 63.98 ms, local B=1 21.867 ms; no vLLM comparison made.

## Failures retained

- Shipped control at 16×4096: c16 bench 1 ok / 15 HTTP 500; 16-way concurrent probe 16/16 timed out. This is pre-existing shipped incapability at that geometry — the candidate instead passes 16/16 (bench and probe, HTTP 200).
- First candidate c16 start failed with torchrun master-port 29500 `EADDRINUSE` during control shutdown; rerun with a unique `--master_port` succeeded.
- Graph capture/replay/bucket counters exist in `tp_engine.py` but are not exposed via `/health`, `/v1/models`, or server logs, so capture/replay counts were not measured.

## Verdict

Adopted: greedy exact everywhere, c1 materially faster, no c4/c8 regression (>1% threshold), c16 capability gained rather than lost. Integrated as the `scripts/serve_qwen38_tp2.py` default (`--graph-buckets` on; `--no-graph-buckets` restores the full-width graph); the `TPEngineCore` library default is unchanged. All own servers were cleaned up (GPUs at 1 MiB, ports free); no vLLM or external process was touched. No vLLM win is claimed — no same-harness suite run exists yet.
