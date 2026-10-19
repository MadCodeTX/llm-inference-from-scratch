# EXPERIENCE — prefill

## P1 — Chunkwise prefill wins; the FP8 dispatch path decides the chunk ranking

- **cue**: prefill through the token-by-token static path.
- **lesson**: prefill in chunks so each weight is read once per chunk, and
  keep every chunk on the Triton FP8 dispatch — a bigger chunk on the
  reference path (chunk256, 635 tok/s) loses to a smaller chunk on Triton.
- **evidence**: `benchmarks/baseline/ours_tp2_prefill_bench.json`:
  static 23.14 tok/s @512 vs chunk128 1,316.07 tok/s @512 vs chunk256
  (reference, M>128) 635.17 tok/s; last-argmax 30876 equal across methods.
  v2 (`benchmarks/baseline/ours_tp2_prefill_v2.json`, TRITON_MAX_M=1024):
  @2048 chunk128 1,412.8 / chunk512 1,731.6 / chunk1024 1,596.0 tok/s;
  @8192 chunk512 1,631.3 tok/s; argmax equal across chunk sizes.
- **confidence**: high (two measured sweeps; argmax equality held).
- **applicability limits**: ours-vs-ours local-runner rates only — Arm A's
  ~2.9–3.4k prompt tok/s is TTFT-derived context, so no TTFT-vs-vLLM claim
  follows from this entry.
