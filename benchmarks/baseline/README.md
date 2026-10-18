# Incumbent baseline (pre-existing measurements)

`incumbent_existing.json` in this directory holds **aggregate numeric
fields only**, extracted from three pre-existing bench result files that
were measured earlier on this box (bigbox, vLLM serving
`Qwen/Qwen3.8-27B-FP8`):

- `/home/nick/llm/vllm/bench-results/baseline.json` (label `baseline-262k-seqs1`)
- `/home/nick/llm/vllm/bench-results/S5-nativefp8.json` (label `S5-nativefp8`)
- `/home/nick/llm/vllm/bench-results/S9-FINAL-seqs16-262k.json` (label `S9-FINAL-seqs16-262k`)

The harness that produced them is `/home/nick/llm/vllm/bench_vllm.py`
(streaming `/v1/chat/completions`, `temperature=0.7`, `top_p=0.8`,
`ignore_eos`, thinking disabled; shapes `patient_gen` 2048→2048,
`chart_summary` 16384→1024, `research_synth` 49152→4096 tokens).

**Prompt text was NOT copied into this public repo.** The incumbent
prompts contain domain-specific text; only the numeric fields listed in
`provenance` (shape, concurrency, `prompt_tokens`, `wall_s`,
`aggregate_output_tps`, `per_stream_tps`, `mean_ttft_s`, latency fields)
were carried over, with the source file and run label kept per row so
every number can be traced back to its origin.

These are reference points, not new measurements. Re-measure with
`benchmarks/run_bench.py` (see `benchmarks/suite.json` and
`benchmarks/WIN.md`) before claiming any win.
