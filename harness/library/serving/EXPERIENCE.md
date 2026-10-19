# EXPERIENCE — serving

## V1 — Cross-shape / cross-harness numbers are context, never wins

- **cue**: a local-runner number (different prompt shape, batch, or
  harness) that numerically passes an Arm A / Arm B serving bar.
- **lesson**: the only win is `benchmarks/run_bench.py` on
  `benchmarks/suite.json`, same shapes, same box, quality gate passed
  (`benchmarks/WIN.md` thresholds). Everything else is labelled
  ours-vs-ours or context-only at the moment it is written down.
- **evidence**: `benchmarks/WIN.md` throughput ladder: ours eager B=8
  10,746 / B=16 20,001 tok/min vs Arm A c4 8,530 / Arm B c4 12,113–12,136
  tok/min — recorded there as "passes numerically only", verdict "no win
  claimed"; per-stream decode (23.67 vs 20.3 ms), prefill rate, and prefix
  TTFT all recorded as losses or non-comparable in the same file.
- **confidence**: high (this is the repo's own recorded verdict, applied
  consistently across every ladder row).
- **applicability limits**: none — this rule binds every domain's
  reporting, including future serving candidates.
