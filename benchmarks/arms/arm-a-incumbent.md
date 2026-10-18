# Arm A — incumbent vLLM as-is (version 1)

Config record for the GPU window. Not executed here; no live server was queried.

- Engine / model: vLLM serving `Qwen/Qwen3.8-27B-FP8` at `http://127.0.0.1:8000` (evidence: [WIN.md](../WIN.md), [suite.json](../suite.json), [incumbent_existing.json](../baseline/incumbent_existing.json)).
- Evidenced harness: `/home/nick/llm/vllm/bench_vllm.py`, streaming `/v1` chat/completions with `include_usage`, `temperature=0.7`, `top_p=0.8`, `ignore_eos`, `enable_thinking=false`; labels `baseline`, `S5-nativefp8`, `S9-FINAL-seqs16-262k`.
- Launch flags: only model and host/port are recorded in this repo. Tensor-parallel size, GPU-memory utilisation, max model length, and other `vllm serve` flags are **not in the repo** and are not invented here — record them at the start of the approved window, before any change, so the exact config can be restored.
- Machine-readable record: [arm-a-incumbent.json](arm-a-incumbent.json).
- Bench in the window: `python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <real-snapshot> --output benchmarks/results-incumbent.json`
