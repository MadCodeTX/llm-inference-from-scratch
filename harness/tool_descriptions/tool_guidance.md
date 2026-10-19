# Tool guidance (c4) — how to use the probes

The tools of this harness are the repo's runners. Use them as-is; do not
write a new timing loop when one exists.

| Need | Command (run from repo root) | Writes |
|---|---|---|
| TP2 decode / speculative probe | `.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/run_qwen38_tp2.py --decode-mode graph --delta fused --fp8-config new --probe capital` (flags per task card) | `--out` JSON (default `benchmarks/baseline/ours_tp2_first_run.json`) |
| Prefill sweep | `.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/prefill_bench_tp2.py --lengths 512,2048 --chunks 128,256,512,1024` | `benchmarks/baseline/ours_tp2_prefill_bench.json` |
| Prefix TTFT | `.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/prefix_ttft_bench.py --snapshot-interval 256 --storage both` | `benchmarks/baseline/ours_prefix_ttft_v2.json` |
| Batched decode | `.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/batched_decode_bench.py --batches 1,2,4,8,16 --decode-tokens 64` | `benchmarks/baseline/ours_tp2_batched_decode.json` |
| Serving head-to-head | `.venv/bin/python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <snapshot> --output <out.json>` | the `--output` JSON |
| Profile (decode) | `.venv-gpu/bin/torchrun --nproc_per_node=2 scripts/profile_qwen38_tp2.py` / `scripts/profile_tp2_graph.py` / `scripts/profile_spec_tp2.py` | `benchmarks/profile/*.json` + `SUMMARY*.md` |

Rules of use:

- **Exact flags matter.** Quote the full command line with the artifact.
  `--fp8-config new`, `--delta fused`, `--decode-mode graph`, and
  `--speculative off` are the current best decode configuration; changing
  one flag is one candidate.
- **ABBA for A/B claims.** Alternate old/new within one process or in
  interleaved cross-process runs (as in `scripts/ab_microopts_tp2.py` and
  `benchmarks/baseline/ours_tp2_tile_ab.json`); absolute tok/s drifts with
  GPU clocks between processes, paired deltas do not.
- **`run_bench.py --dry-run` first** for any serving plan (48 batches /
  336 requests at full suite); live runs only in an approved window.
- **Never hand-edit an artifact JSON** to "fix" a number. If an artifact is
  wrong, rerun the probe and say what changed.
