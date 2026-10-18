# Arm B — vLLM + MTP speculative decoding (version 1, UNMEASURED)

Config record for the GPU window. Not executed; no win may be claimed against Arm B until a sweep entry is actually measured.

- Base: Arm A ([arm-a-incumbent.md](arm-a-incumbent.md)) unchanged, plus MTP speculative config.
- Sweep (separate entries/runs): `num_speculative_tokens` in `[1, 2, 3]`, each with `--speculative-config {"method": "mtp", "num_speculative_tokens": <K>}`.
- Status: **UNMEASURED** for every entry.
- Machine-readable record: [arm-b-mtp.json](arm-b-mtp.json), with one entry per `K`.
- Bench per entry in the window: restart vLLM once with that entry's speculative config, then `python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <real-snapshot> --output benchmarks/results-mtp-k<K>.json`
