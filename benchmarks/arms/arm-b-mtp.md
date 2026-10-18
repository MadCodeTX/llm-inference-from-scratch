# Arm B — vLLM + MTP speculative decoding (version 1, FAILED_TO_START)

Attempted 2026-10-18 in isolated container `vllm-armb-qwen38-mtp`. No throughput numbers exist; no win may be claimed against Arm B.

- Base: Arm A ([arm-a-incumbent.md](arm-a-incumbent.md)) unchanged, plus MTP speculative config.
- Method discovery (image `vllm/vllm-openai:v0.30.0`, no GPU): `SpeculativeMethod` includes `qwen3_5_mtp` and `mtp`; `vllm/config/speculative.py` maps `qwen3_5`/`qwen3_5_text` with `mtp_num_hidden_layers=1` to `qwen3_5_mtp`/`Qwen3_5MTP` and deprecates `qwen3_5_mtp` -> `mtp`. Correct string: `mtp`.
- K=1 run: exact Arm A flags plus `--speculative-config {"method": "mtp", "num_speculative_tokens": 1}`. Log reached `Resolved architecture: Qwen3_5MTP` and `Capturing model for speculator...`, then exited (exit 1) before serving: `ValueError: To serve at least one request with the model's max seq len (262144), (4.49 GiB KV cache is needed, which is larger than the available KV cache memory (4.4 GiB). ... estimated maximum model length is 256608.`
- Host port: task port 8001 was occupied by an unrelated `python3 -m http.server` (prompt-lab report); used `127.0.0.1:8003` -> container 8000. Container removed after capture; GPUs back to 1 MiB, `~/llm/run/current` absent, original container still exited, litellm untouched.
- Status: K=1 **FAILED_TO_START**; K=2/K=3 **NOT_ATTEMPTED** (gated on K1 working). Machine-readable record: [arm-b-mtp.json](arm-b-mtp.json).
- Next step (needs a decision, breaks exact-flags comparability): retry with reduced `--max-model-len` (<=256608) or higher `--gpu-memory-utilization`, then rerun `quality_probe.py` + `run_bench.py` greedy c1 short + agent-shared-prefix.

## Retry 2026-10-18 — MEASURED-WITH-DEVIATION

- **Deviation (only one): `--max-model-len 256608` instead of Arm A `262144`** — the value vLLM itself estimated as maximum in the first failure. All other flags identical to Arm A (`arm_a_live_config.json`) plus `--speculative-config {"method": "mtp", "num_speculative_tokens": 1}`. Isolated `vllm-armb-qwen38-mtp` on `127.0.0.1:8003` (8001 still occupied by unrelated http.server). Status: **MEASURED-WITH-DEVIATION** — not exact-flags comparable on max len.
- Health: `/v1/models` 200 in ~3.2 min, `max_model_len: 256608`.
- Quality: `quality_probe.py` 3/3 text + retokenized ids exact vs Arm A (`arm_b_quality_probes.json`).
- Metrics (raw in `benchmarks/baseline/arm_b_metrics_*.txt`): before all `spec_decode` counters 0; after c1 drafts 1057 / accepted 1014 (95.9%); final drafts 3617 / accepted 3574 (98.8%).
- Bench greedy c1 (`arm_b_mtp_k1_256608.json`, failed=0): short TTFT p50 0.76–0.78 s, 4272–4291 tok/min; agent-shared-prefix TTFT p50 3.33 s cold / 0.90 s warm, 3144–3814 tok/min. Short c4 (`..._c4.json`): 12113–12136 tok/min, failed=0.
- Caveat: `run_bench` TPOT is per streamed chunk; MTP chunks carry multiple tokens, so ~24.7 ms/chunk is not comparable to Arm A's 20.3 ms/token. Effective c1 short ≈71 tok/s wall-clock. No TPOT win/loss claimed from chunk TPOT.
- Cleanup: only `vllm-armb-qwen38-mtp` removed; GPUs 1 MiB, `current` absent, watchdog idle.
