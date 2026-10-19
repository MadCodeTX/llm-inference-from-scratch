# Scripts (c7) — deterministic helpers

No new scripts ship with this tree. The deterministic helpers are the repo's
existing runners, used as-is (see
[`../tool_descriptions/tool_guidance.md`](../tool_descriptions/tool_guidance.md)
for the command table):

- `scripts/run_qwen38_tp2.py` — TP2 decode/speculative probe (the decode,
  speculative, and serving-candidate workhorse).
- `scripts/prefill_bench_tp2.py`, `scripts/prefix_ttft_bench.py`,
  `scripts/batched_decode_bench.py` — prefill, prefix-TTFT, batching probes.
- `scripts/profile_*.py` — profilers; run these before proposing kernel levers.
- `scripts/ab_microopts_tp2.py` — the paired-ABBA pattern for A/B claims.
- `benchmarks/run_bench.py` — the only serving head-to-head harness; the only
  numbers comparable to Arm A / Arm B bars.

If a future step needs a new helper, it must be deterministic, take its
parameters as flags, write a JSON artifact with the full command line
recorded, and be named here in the same change. Ad-hoc timing loops in a
sub-agent's scratch space are never evidence.
