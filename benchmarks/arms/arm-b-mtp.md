# Arm B — vLLM + MTP speculative decoding (version 1, FAILED_TO_START)

Attempted 2026-10-18 in isolated container `vllm-armb-qwen38-mtp`. No throughput numbers exist; no win may be claimed against Arm B.

- Base: Arm A ([arm-a-incumbent.md](arm-a-incumbent.md)) unchanged, plus MTP speculative config.
- Method discovery (image `vllm/vllm-openai:v0.30.0`, no GPU): `SpeculativeMethod` includes `qwen3_5_mtp` and `mtp`; `vllm/config/speculative.py` maps `qwen3_5`/`qwen3_5_text` with `mtp_num_hidden_layers=1` to `qwen3_5_mtp`/`Qwen3_5MTP` and deprecates `qwen3_5_mtp` -> `mtp`. Correct string: `mtp`.
- K=1 run: exact Arm A flags plus `--speculative-config {"method": "mtp", "num_speculative_tokens": 1}`. Log reached `Resolved architecture: Qwen3_5MTP` and `Capturing model for speculator...`, then exited (exit 1) before serving: `ValueError: To serve at least one request with the model's max seq len (262144), (4.49 GiB KV cache is needed, which is larger than the available KV cache memory (4.4 GiB). ... estimated maximum model length is 256608.`
- Host port: task port 8001 was occupied by an unrelated `python3 -m http.server` (prompt-lab report); used `127.0.0.1:8003` -> container 8000. Container removed after capture; GPUs back to 1 MiB, `~/llm/run/current` absent, original container still exited, litellm untouched.
- Status: K=1 **FAILED_TO_START**; K=2/K=3 **NOT_ATTEMPTED** (gated on K1 working). Machine-readable record: [arm-b-mtp.json](arm-b-mtp.json).
- Next step (needs a decision, breaks exact-flags comparability): retry with reduced `--max-model-len` (<=256608) or higher `--gpu-memory-utilization`, then rerun `quality_probe.py` + `run_bench.py` greedy c1 short + agent-shared-prefix.
