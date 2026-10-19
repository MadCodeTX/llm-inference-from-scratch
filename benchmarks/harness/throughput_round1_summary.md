# Throughput round 1 — graphed batched decode candidate

Candidate: `TPEngineCore(decode_mode="graph")` (integrated; the engine default now resolves to `graph` on CUDA in
[tp_engine.py](/home/nick/muse/llm-inference-from-scratch/src/inference_from_scratch/tp_engine.py))
vs baseline `decode_mode="eager"`, same tree (HEAD `4cc73c6`), same model.
No source was edited by this round; the driver ran from `/tmp`.
Raw record: [throughput_round1.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/harness/throughput_round1.json).

## Verdict

Integrated: the engine default resolves to graph on CUDA (explicit eager preserved; explicit graph still fails closed off CUDA). Graph is exact vs eager at every concurrency and faster on whole-run
wall at the suite-short geometry. Medians (ABBA; c1 n=3, c4/c8 n=2 per mode):

| Concurrency | Eager tok/min | Graph tok/min | Eager wall | Graph wall | Per-stream graph |
|---|---|---|---|---|---|
| 1 | 1,397 | 2,334 | 22.00 s | 13.16 s | 38.9 tok/s |
| 4 | 4,462 | 5,109 | 27.54 s | 24.05 s | 21.3 tok/s |
| 8 | 6,979 | 7,298 | 35.22 s | 33.67 s | 15.2 tok/s |

Gain shrinks with batch (+67% / +15% / +5% aggregate), as expected:
the graph removes host launch overhead, which dominates at B=1 and is
amortised at B=8. Graph captures = 1 and replays = 511 on every graph run.

## Shape and defaults

Suite `short` geometry exactly: 2048 filler tokens (exact counts via
`run_bench.build_token_ids`, seed = row index) → 512 greedy tokens,
ignore-EOS (`eos_id=None`); every request finished by length at 512.
Adopted defaults applied to the model handed to the engine:
`gemv-bk512`, scale-cache on, gated-norm fused, norm fused, delta
fused, vocab lm_head. GEMV is M=1 only, so at c4/c8 the graph is the
only varied lever. Engine `token_budget=512`, `max_seq_len=2560`.

## Exactness

- Graph vs eager engine: token-identical on all 14 measured runs,
  every row, c1/c4/c8. No inexact or unfinished-as-finished result.
- Batched rows vs same-prompt c1 single-stream engine: seeds 1–3 exact
  (512/512) at c4 and c8; seed 0 diverges at token 17 under batching
  in both modes identically (M>1 numerics, not the candidate).
- Engine c1 vs the runner B=1 graph oracle on seed 0: 51-token prefix
  match, then divergence. A control probe showed changing only the
  engine chunk budget (512→4096) also flips token 66, so greedy ids
  on this 2048-token filler prompt are prefill-chunk-sensitive across
  harnesses. This does not reject the candidate, whose criterion is
  graph-vs-eager identity; it does bar any cross-harness id claim.

## Caveats

- This is the local engine harness, not `run_bench.py` serving
  semantics. No win is claimed vs vLLM Arm A (2,745 tok/min c1 short)
  or Arm B, and none may be inferred from the table above.
- TTFT/TPOT are not observable: the engine records step indices only
  (`ttft_steps` 4/9/13/17 at c4). Timing is whole-run wall with sync
  at both ends; no TPOT was fabricated.
- c16 was not attempted (optional in the task). Peak memory at c8 was
  17.7 GiB/rank. No failures or OOMs occurred.
- GPUs were free before and after (1 MiB used, no processes); no
  process remains. Nothing was committed.
