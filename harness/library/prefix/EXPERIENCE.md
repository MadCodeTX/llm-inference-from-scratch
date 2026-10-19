# EXPERIENCE — prefix

## X1 — Snapshot interval 256 is the prefix-TTFT fix; verify reuse before celebrating

- **cue**: block prefix cache (block 16) whose snapshots are too coarse to
  reuse the shared prefix cheaply.
- **lesson**: store state snapshots at interval boundaries
  (`snapshot_interval_tokens=256`, a multiple of block_size 16) plus the
  prompt end, and judge warm TTFT only together with `n_reused` and
  cold==warm argmax — speed without the reuse count is not evidence.
- **evidence**: v1 `benchmarks/baseline/ours_prefix_ttft.json` (2048+256=2304):
  cold median 10.66 s / warm 1.31 s, 8.14×, n_reused=2048. v2
  `benchmarks/baseline/ours_prefix_ttft_v2.json` (interval 256): GPU cold
  1.643 s / warm 0.184 s; CPU cold 1.926 s / warm 0.252 s; n_reused=2048;
  last argmax cold == warm == 12356.
- **confidence**: high for the ours-vs-ours contrast (3 reps, medians,
  argmax + reuse verified); no confidence expressed vs vLLM.
- **applicability limits**: 2304-token scaled-agent shape only. Arm A's
  1.81 s is at 8704 tokens (`benchmarks/baseline/arm_a_exact_greedy_c1_c4.json`)
  — different shape, no TTFT win claimed. One v1 warm rep (1,339.99 s) is an
  unattributed outlier recorded in the v2 artifact; do not average it away.
