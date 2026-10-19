# EXPERIENCE — speculative

## S1 — Acceptance is not speed: the M=2 verify chunk eats the MTP gain

- **cue**: native MTP (γ=1) with teacher-forced/live acceptance ~0.9 —
  tempting to default on.
- **lesson**: judge speculation on effective TPOT against the same-run
  nonspec reference, never on acceptance. Here the graphed M=2 verify
  chunk costs 34.6 ms vs the 24.5 ms M=1 graph step, and host/capture
  overhead consumes the rest; merging to one graph / one sync per cycle
  (graph2) did not fix it because the dominant cost is GPU draft+verify.
- **evidence**: `benchmarks/baseline/ours_tp2_mtp_matrix.json`: exact
  everywhere (32/32 ×3 vs bridge+Arm A; capital 128/128 vs graph-nonspec),
  acceptance 0.88–0.94; capital ×128 effective TPOT mtp1-eager 38.68 ms /
  mtp1-graph 32.51 ms vs graph-nonspec 24.49 ms. Profile per cycle:
  draft 5.8 ms, verify 52.8 ms eager / 34.6 ms graph, host/capture
  12.7–18.8 ms. `benchmarks/baseline/ours_tp2_mtp_graph2.json`: merged
  graph 32.90 ms @capital×128, decision recorded "KEEP default nonspec".
  Host diagnosis `benchmarks/profile/SUMMARY_spec.md`: ~35 ms/cycle is GPU
  draft+verify surfaced as sync waits; genuine host overhead ~8 ms/cycle.
- **confidence**: high (matrix + merged-graph follow-up + profile agree;
  exactness held in every mode).
- **applicability limits**: γ=1, this MTP head, TP2 on 2× RTX 4090.
  Arm B's 98.8% acceptance (3574/3617, `benchmarks/arms/arm-b-mtp.json`)
  shows a serving-grade MTP can pay off inside vLLM; it does not transfer
  to this runner's verify cost.
