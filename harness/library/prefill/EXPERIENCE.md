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

## P2 — Prefill num_stages=2 (prefill-s2) rejected and removed

- **cue**: prefill sweep had fixed `num_stages=3`; stages=2 might raise
  occupancy at chunk M.
- **lesson**: do not retry `num_stages=2` on this kernel/box without
  new evidence — it loses K-loop latency hiding at every shape.
- **evidence**: `benchmarks/harness/prefill_round1.json` (ABBAAB, 3
  runs/config): s2 slower everywhere (-1.3% to -7.5%; chunk512 2048:
  1703.4 vs 1728.3 tok/s, 8192: 1608.5 vs 1629.9 tok/s), last-token
  argmax equal across configs. Code removed on integration.
- **confidence**: high (all 9 length/chunk shapes agree; exactness held).
- **applicability limits**: ours-vs-ours local-runner rates only; no
  TTFT-vs-vLLM claim follows.

## P3 — prefix-chunk512 promoted on the bench surface (prefill round 2)

- **cue**: chunk-profile at 2048 showed collectives ~invariant and
  the addressable cost in host launch/gap + FP8 launch count,
  favouring chunk512 (v2 unprofiled: 1731.6 vs 1412.8 tok/s @2048).
- **lesson**: default `scripts/prefix_ttft_bench.py` to `--chunk 512
  --snapshot-interval 512` (effective chunk = min(chunk, interval),
  so both must rise together); reproduce the v2 baseline with
  `--chunk 128 --snapshot-interval 256`. Library defaults
  (`prefill_chunkwise_static` 128, `TPBlockPrefixCache` 256) stay.
- **evidence**: `benchmarks/harness/prefill_round2.json` (ABBAAB, 3
  runs/arm/shape): pure-prefill +22.1%/+21.9%/+20.0% tok/s at
  512/2048/8192; prefix cold/warm -17.5%/-15.8% (512+256),
  -18.1%/-14.8% (2048+256), -15.0%/-12.4% (8192+512 cpu storage);
  warm n_reused == shared and argmax identical everywhere; peak
  +≤0.17 GiB/rank; base OOMs at 8192+512 gpu storage (3/3) while
  the candidate completes.
- **confidence**: high (ABBA, exactness held on every run; verifier
  recomputed medians from raw artifacts).
- **applicability limits**: ours-vs-ours local B=1 TTFT proxy only —
  no serving/vLLM TTFT claim; non-512-aligned shared prefixes may
  recompute <512 extra tokens vs interval 256; prefix bench uses
  `use_scale_cache=False` both arms.
