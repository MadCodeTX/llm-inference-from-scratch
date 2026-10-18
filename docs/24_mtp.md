# 24 — Native MTP head: one target, one extra layer, drafted ahead

> **Source of truth** — the real checkpoint's `mtp.safetensors` shard headers (22 `mtp.*` tensors, names/dtypes/shapes only, weights never loaded), the checkpoint `config.json` (`text_config.mtp_num_hidden_layers = 1`, `mtp_use_dedicated_embeddings = false`), and the code in [mtp.py](../src/inference_from_scratch/mtp.py) (`MTPConfig`, `MTPHead`, `mtp_tensor_spec`) with acceptance accounting in [speculation.py](../src/inference_from_scratch/speculation.py) (`acceptance_stats`); tests: [test_mtp.py](../tests/test_mtp.py). Framing follows Kiely, *Inference Engineering* (2026): sec. 2.2 for config and mechanics, sec. 2.4–2.5 for bottlenecks and attention costs, sec. 4.2 for formats and reference implementations, sec. 5.2 for speculation, and sec. 5.4 for the tensor-parallel context. All prose below is original to this repo.

## The one-sentence idea

Bolt one extra full-attention decoder layer onto the target model itself so that, from today's hidden state fused with tomorrow's token embedding, it drafts the token after next — then let the unchanged target verify that draft, so the extra layer costs a fraction of a second model and can never change the output text.

## Why transformers cannot run this head

There is no reference MTP implementation to copy in the installed transformers 5.20.1, and this chapter does not pretend otherwise. A case-insensitive search for `mtp` over `transformers/models/qwen3_5/` (modeling, configuration and modular files) finds exactly two statements, both `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", ...]` in `modeling_qwen3_5.py` (the `Qwen3_5PreTrainedModel` and `Qwen3_5ForConditionalGeneration` class bodies): on load, Hugging Face explicitly ignores every `mtp.*` tensor. `Qwen3_5TextConfig` declares no `mtp_*` field at all, no class with `MTP` in its name exists in the modeling file, and importing one raises `ImportError`. The test suite asserts exactly this absence, so that if a future transformers release ships a real MTP class the test fails loudly and a genuine parity test replaces it — no fake parity claim is made here.

The structure below therefore comes from the checkpoint itself. Reading only the safetensors header of `mtp.safetensors` enumerates all 22 real tensors:

| Tensor | Dtype | Shape | Role |
|---|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | BF16 | [5120] | norm on the next-token embedding branch |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | [5120] | norm on the hidden-state branch |
| `mtp.fc.weight` | BF16 | [5120, 10240] | fuses the concatenated branches 10240 → 5120 |
| `mtp.layers.0.input_layernorm.weight` | BF16 | [5120] | block pre-norm |
| `mtp.layers.0.self_attn.q_proj.weight` (+ scale [96, 40]) | F8_E4M3 | [12288, 5120] | fused query+gate, as decoder layer 3 |
| `mtp.layers.0.self_attn.k_proj / v_proj.weight` (+ scales [8, 40]) | F8_E4M3 | [1024, 5120] | GQA key/value projections |
| `mtp.layers.0.self_attn.o_proj.weight` (+ scale [40, 48]) | F8_E4M3 | [5120, 6144] | attention output projection |
| `mtp.layers.0.self_attn.q_norm / k_norm.weight` | BF16 | [256] | per-head norms |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 | [5120] | MLP pre-norm |
| `mtp.layers.0.mlp.gate_proj / up_proj.weight` (+ scales [136, 40]) | F8_E4M3 | [17408, 5120] | SwiGLU gate/up |
| `mtp.layers.0.mlp.down_proj.weight` (+ scale [40, 136]) | F8_E4M3 | [5120, 17408] | SwiGLU down |
| `mtp.norm.weight` | BF16 | [5120] | final norm before the shared `lm_head` |

Count them: 15 module tensors plus 7 FP8 `weight_scale_inv` companions is 22. The `weight_scale_inv` tensors are storage companions in the [128, 128] block format (Kiely sec. 4.2), consumed by the dequantising loader of chapter 22; they are never module parameters, and the test pins the tiny module's state-dict keys to exactly the other 15.

## The layer subtype, identified not assumed

`mtp.layers.0` holds `self_attn.*` and an SwiGLU `mlp`, and holds no `linear_attn.*` tensor whatsoever. The MTP layer is therefore a **full-attention** layer — gated full attention with per-head Q/K norms, partial RoPE and a sigmoid output gate, shape-identical to decoder layer 3 — not a gated-delta linear layer. `MTPHead` builds it by reusing the existing `HybridBlock` with `layer_types = ["full_attention"]`, so every attribute already carries its checkpoint name and a dequantised checkpoint state dict will load strictly at matching dims. Kiely sec. 2.2's discipline applies directly: the config (`MTPConfig`, parsed from `HybridConfig` or straight from a `text_config` mapping) is the single bundle every width is read from, and nothing is sized from memory.

What the shapes prove is the skeleton: two separately normed branches, one [5120, 10240] fusion, one full-attention block, one final norm, and — by the absence of any vocab-sized tensor — a head that must reuse the target's embedding and `lm_head` (the config's `mtp_use_dedicated_embeddings = false` confirms the embedding half). What the shapes cannot prove is recorded in one place in the code, `INFERRED_JOINS`, and repeated here so it can be challenged: the concatenation order inside `fc` (hidden-first is implemented; both branches are [5120], so the shape is silent), the norm-to-branch pairing beyond the names, that the fused vector feeds the block directly, the RoPE position rule (caller-supplied `position_ids` / `start_pos` in this foundation), and that `mtp.norm` is the last norm before the shared head. Each is the name-ordered reading; none is dressed up as measured fact.

## Measured wiring sweep (2026-10-18): no single wiring dominates

The inferred joins above were challenged empirically. Teacher trajectories (greedy, 64 tokens x 6 prompts: the 3 quality probes plus code/list/prose neutrals) were captured with the pipeline bridge (`scripts/capture_target_hidden.py`; tensors stay in `/tmp`, summary in `benchmarks/mtp/capture_summary.json`), and all 48 live candidate wirings (`mtp.enumerate_wiring_candidates()`; the dedicated-embedding option is pruned by config + tensor absence) were run teacher-forced on one real dequantized MTP block (`scripts/mtp_acceptance_sweep.py`, artifact `benchmarks/mtp/wiring_sweep.json`): draft at `t` fuses `hidden[t]` with `embed(token[t+1])` and predicts `token[t+2]` (`mtp.teacher_forced_draft_indices`). Baselines: repeat-previous-token 0.000, random 1/248320. Best mean top-1 acceptance is 0.872 (430 draft steps) — far above either baseline — but the top candidates tie, so **no single wiring is resolved and the code default is unchanged**:

| Axis | Finding (mean acceptance range over candidates) |
|---|---|
| fc concat order | **embedding-first 0.665–0.872 vs hidden-first 0.005–0.016 — decisive; the inferred hidden-first reading is measured wrong** |
| norm pairing | named 0.862 vs swapped 0.691 within embedding-first — strong |
| hidden source | post-final-norm best 0.872 vs pre-final-norm 0.861 — not decisive (the pre-fc norm absorbs most of the difference) |
| position rule | hidden/input/predicted positions tie (top rows identical) — a constant RoPE offset is a gauge freedom for this block's own KV, unidentifiable by acceptance |
| lm_head input | mtp.norm 0.870 vs raw block 0.872 — not decisive at top-1 over 430 steps |

Top cluster (all `fc=embedding_first, norm=named`): 0.845–0.872 across the remaining axes. Follow-up must separate the tied axes with a more sensitive metric (e.g. draft-logit margins) or sequential-KV position handling before any default is promoted; picking among the ties now would be vibes, not measurement.

## Reusable draft modules (dense + TP) and module-level acceptance

`MTPDraftModule` (in [mtp.py](../src/inference_from_scratch/mtp.py)) turns the head into a reusable one-step drafter: `draft_token(hidden_t, token_id_t1, state, position) -> (draft_token_id, draft_logits_or_topk, tentative_state)`, where `hidden_t` is the target's final hidden at `t` and `token_id_t1` is the target's chosen token for `t+1`; the draft predicts `t+2`. The wiring is a parameter (`wiring=`), defaulting to `LEADING_WIRING` — the sweep's first-ranked candidate (`fc=embedding_first, norm=named, hidden=post_final_norm, pos=hidden_pos, head=raw_block`) — because the sweep above is unresolved, not because that wiring is proven. Target hidden states come from the new hook-free capture: `TinyHybridLM` / `DistributedHybridLM` gained `forward_with_hidden(...)` and a default-off `return_hidden` flag on `forward` / `prefill` / `step` (the returned hidden is the final norm's input; `norm(hidden)` is what `lm_head` reads — proven against a manual block-by-block recomputation in `tests/test_mtp_draft.py`, with default-off bit-identical logits).

**State semantics.** The MTP block is a proven full-attention layer, so the draft state is its own KV cache (one `(k, v)` pair per MTP layer; a hypothetical linear-attention MTP layer would carry `(S, conv)` under the same contract). `draft_token` never mutates its input state: the returned state is tentative and is committed only when the target accepts the token run; on rejection the caller keeps the old state (functional commit-on-accept, with `snapshot_state` / `restore_state` as the explicit deep-copy form). The MTP KV therefore advances **only on accepted tokens** — the property that keeps speculative greedy output token-for-token equal to target greedy (Kiely sec. 5.2).

**TP version.** `TPMTPDraftModule` shards the MTP block with the existing TP wrappers: `fc` and the norms are replicated per the docs/23 decision, and the layer is a `DistributedHybridBlock`, so its collectives are exactly a normal block's. Tiny 2-process equality against the dense draft logits is proven by `scripts/mtp_tp_check.py` under both gloo (CPU, durable test) and NCCL (2x RTX 4090: max abs err ~1e-5, draft ids equal). The checkpoint loader now supports it: `tp_checkpoint.load_rank_state(..., include_mtp=True)` loads `mtp.*` into `DistributedHybridLM.mtp_draft` (synthetic-fixture test pins all 22 resident yields per rank; the real manifest contributes 22 `mtp.*` tensors, headers only).

**Module-level acceptance (measured 2026-10-18).** Re-measured *through `MTPDraftModule`* (not the sweep script) on the same captured trajectories (`scripts/mtp_module_acceptance.py`, artifact `benchmarks/mtp/module_acceptance.json`): **0.8698 (374/430)** versus the sweep winner's 0.8721 (375/430) — a one-token difference from bf16 single-step-KV vs sequence-pass numerics, i.e. the module reproduces the sweep winner within noise. The wiring remains unresolved; this number validates the module, not the wiring choice.

The forward itself is deliberately narrow. `MTPHead.forward(hidden, token_embedding_or_ids, embedding=..., lm_head=..., position_ids/start_pos, state)` takes the target's hidden state ([B, D] or [B, T, D]) plus the next token as an embedding or as ids resolved through the caller-provided embedding, and returns `(draft_hidden, logits, new_state)`: norm both branches, concatenate, fuse, run the block(s), apply `mtp.norm`, and — only when the caller passes its `lm_head` callable — produce draft logits. `draft_token` is the greedy special case (argmax of those logits). State is the MTP layers' own KV pairs in `initial_state()` form; inputs are never mutated, so a pre-step state replays bit-identically, which the tests assert with `torch.equal`.

## Native MTP versus a draft model (Kiely sec. 5.2)

Chapter 14's speculative decoding and this head solve the same problem — decode is memory-bound, so guess several tokens cheaply and verify them in one parallel target forward — with different machinery:

| Aspect | Draft-model speculation (ch. 14) | Native MTP head (this chapter) |
|---|---|---|
| Drafter | separate small model, own weights | one extra layer of the target itself |
| Extra memory | a whole second model + its KV cache | ~1 layer of weights; embedding and `lm_head` shared |
| Draft input | token ids only | target hidden state + next-token embedding |
| Draft depth per step | several autoregressive steps | one layer pass per drafted token |
| Output guarantee | target-greedy exact (verified) | identical guarantee once verified the same way |
| Failure cost | wasted draft-model forwards | wasted single-layer passes (smaller) |

The verification contract does not change between the two columns: a drafted token survives only when the target agrees, the first disagreement is replaced by the target's own token, and a fully accepted run earns one bonus token. That is why the acceptance accounting is shared code rather than an MTP-specific copy. `acceptance_stats(target_ids, draft_ids_per_step)` in speculation.py counts, per verification step, the longest common prefix of draft and target ids, and returns `accept_rate` (accepted / drafted), `mean_accepted_len` (accepted per target forward) and `expected_tokens_per_target_forward` (`mean_accepted_len + 1`: the accepted prefix plus the one correction-or-bonus token every verification forward finalises). It is pure counting over caller-supplied ids — empty inputs return defined zeros, mismatched step counts raise — and no acceptance figure for the real 27B model is stated anywhere in this repo, because none has been measured. Any future number must come from running the real head, not from this foundation.

## Speculative generation loop (built, eager static TP path)

`speculative_generate_tp` in [speculation.py](../src/inference_from_scratch/speculation.py) runs greedy MTP speculation with gamma=1 on the TP2 static path (`scripts/run_qwen38_tp2.py --speculative mtp1` eager, `--speculative mtp1-graph` with CUDA-graph verify+draft). Each cycle drafts one token from the last target hidden plus the last accepted token, verifies with ONE 2-token chunk forward (`DistributedHybridLM.decode_chunk_static`: full-attention writes both K/V causally, linear layers read each weight once over M=2 and step only conv/recurrence sequentially), and accepts iff the target argmax at the last-accepted position equals the draft — emitting the bonus token after the draft on accept. On reject it emits the correction, restores the pre-cycle `TPStaticState` snapshot via `copy_from` (the linear recurrent/conv rollback), drops the tentative MTP state, and re-feeds the last accepted token once so its recurrent contribution is re-established exactly. Output ids equal non-speculative target greedy token-for-token (Kiely sec. 5.2), pinned by `tests/test_speculative_tp.py` including rigged always-wrong (acceptance 0) and target-shifted (all-accepted) drafts and a 2-process gloo check.

**Measured (2026-10-18, real 27B TP2 eager, artifact `benchmarks/baseline/ours_tp2_mtp_eager.json`):** exact 32/32 against both the bridge and Arm-A references on all 3 probes; acceptance 0.94 / 0.88 / 0.94 (capital / water / counting) over 16–17 cycles each. Effective TPOT is 42–59 ms/token — *slower* than the non-speculative TP2 graph baseline (24.48 ms/token): at eager M=2 the chunk, per-cycle state snapshot and MTP forward cost more than the accepted drafts save. No speedup is claimed; beating the 19.17 ms/token roofline floor requires the graph-captured variant (a later task), not this eager loop.

## Graph-captured speculation and the honest matrix (2026-10-18)

`--speculative mtp1-graph` captures the 2-token verification chunk (`decode_chunk_static` with static logits/hidden buffers) and the MTP draft step (`TPMTPDraftModule.draft_token_static`: sequential static KV slots, RoPE decoupled via `rope_position`) as two CUDA graphs; accept/reject branching, the pre-cycle snapshot and rollback copies stay outside both graphs. Tiny 2-rank NCCL equality is proven by `scripts/tp2_spec_graph_check.py` (chunk and draft graph logits bit-equal to eager, full graph-speculative ids equal to plain greedy). The draft's KV slot advances inside the draft graph and is rewound outside it on rejection.

**Profile (CUDA events, capital x128, per cycle):** MTP draft 5.77 ms eager / 5.64 ms graph; verify chunk 52.79 ms eager / 34.61 ms graph; snapshot 0.46 ms; rollback `copy_from` ~0.03 ms averaged over cycles (1-5 rejects per run); reject re-feed ~3.3 ms averaged; host overhead (argmax syncs, capture, prefill) 12.66 ms eager / 18.75 ms graph per cycle.

**Matrix (artifact `benchmarks/baseline/ours_tp2_mtp_matrix.json`):** all modes exact — 32/32 x3 probes vs bridge+Arm-A, and capital x128 speculative ids 128/128 equal to the graph-nonspec reference ids generated in the same task. Acceptance: 0.94 / 0.88 / 0.94 (32 tokens) and 0.924 (capital x128, 61/66). Effective TPOT on capital x128: graph-nonspec 24.49 ms, mtp1-eager 38.68 ms, mtp1-graph 32.51 ms.

**Effective-TPOT math.** At gamma=1 the expected tokens per cycle are `1 + alpha` (the draft token when accepted, plus the bonus/correction token every cycle finalises): alpha = 0.924 gives 1.924 tokens/cycle. Summing only the profiled GPU phases for mtp1-graph (5.64 + 34.61 + 0.47 + 0.03 + 3.44 = 44.19 ms/cycle) predicts 44.19 / 1.924 = **22.96 ms/token** — already barely below the 24.49 ms graph-nonspec TPOT, because the M=2 verify chunk (34.61 ms) costs ~10 ms more than the M=1 graph step it replaces. Measured effective TPOT is **32.51 ms/token**: host syncs and graph capture/prefill overhead consume the remaining ~9.5 ms/token. MTP speculation is therefore exact but **not faster** on this stack today; the runner speculative default stays `off` and no speedup is claimed. Beating the 19.17 ms/token roofline floor requires cutting the M=2 verify cost and the per-cycle host round-trips, not just capturing them.

## Phase 3 integration design (not yet built)

Wiring this head into generation is the next phase; the design below is the contract that phase must satisfy, and the reasons it is not already wired in.

**The loop.** Prefill the target once over the prompt, keeping the last hidden state (the hybrid model must expose hidden states, not only logits — a small, explicit addition to `TinyHybridLM`). Draft with the MTP head from that hidden state and the just-emitted token's embedding. Verify with one target forward over the drafted run, accept the longest agreeing prefix plus the correction/bonus token (chapter 14's rule), and feed the accepted run's last hidden state into the next draft. Acceptance is reported through `acceptance_stats` on the ids actually observed.

**State rollback, the hard part.** Verification gambles on state that rejection then invalidates, and the hybrid stack makes this sharper than a plain KV-cache model:

- The 16 full-attention layers append KV per token. Rolling back is a length truncation: keep the cache preallocated and slice by the accepted length, exactly the bookkeeping chapter 14 avoided by going cache-free.
- The 48 linear layers keep a recurrent matrix `S` and a conv buffer that are *overwritten*, not appended. There is no prefix of `S` to slice back to: the only correct rollback is a snapshot taken before the drafted run (the existing `snapshot_state` / `restore_state` deep-copy, proven bit-identical in chapter 22's branching test) restored on rejection. Snapshotting 48 recurrent states per draft step is a real memory-bandwidth cost (Kiely sec. 2.4–2.5), so Phase 3 must measure snapshot cost against the tokens the draft saves before claiming any speedup.
- The MTP layer's own KV cache has the same truncation question as the full layers, plus one subtlety: its entries were computed from *drafted* positions, so on partial acceptance only the accepted prefix's entries survive.
- Batching multiplies all of the above per row: rows accept different lengths, so rollback points diverge per sequence. The typed `HybridState` (chapter 22) is the seam where per-row snapshot/restore belongs.

**Status (2026-10-18, measured; supersedes the pending-work framing above where it conflicts).** The wiring sweep over 48 candidates (`benchmarks/mtp/wiring_sweep.json`) is resolved=false: best teacher-forced acceptance 0.8721 with tied top rows, so no winner was written into `mtp.py` as the default. The TP2 speculative loop shipped in eager and graph-captured forms (`benchmarks/baseline/ours_tp2_mtp_matrix.json`): exact everywhere (32/32 x3 vs bridge+Arm-A; capital 128/128 vs graph-nonspec), acceptance 0.88–0.94, but slower than graph-nonspec (effective TPOT capital x128: mtp1-eager 38.68 ms, mtp1-graph 32.51 ms, graph-nonspec 24.49 ms) — no speedup is claimed and the runner default stays `off`. Final smoke in this workflow: `scripts/run_qwen38_tp2.py --speculative mtp1-graph --probe capital --max-new-tokens 16` → 16/16 tokens matching bridge and Arm-A references, acceptance 1.0 (8/8 drafts accepted, 8 cycles, 0 rollbacks; `benchmarks/baseline/ours_tp2_mtp_smoke_capital16.json`). Arm B (vLLM + MTP) is still **UNMEASURED** and is the next workflow's first job: restart the incumbent with the MTP config record from `benchmarks/arms/`.

**Explicit non-goals for Phase 3's first cut:** no GPU work while the live vLLM service owns both cards, no FP8 execution of the MTP weights (dequantise to BF16 through the chapter-22 loader, as elsewhere), and no acceptance or speed claim until both are measured on the real checkpoint in an approved window.

## Check yourself

<details>
<summary>1. Why does this head own no embedding table and no vocab projection?</summary>

The checkpoint proves it: no vocab-sized tensor exists under `mtp.*`, and `mtp_use_dedicated_embeddings = false`. Both arrive per-call from the target (`embedding=`, `lm_head=`), which is also what keeps the draft tied to the target's representation.
</details>

<details>
<summary>2. Which part of the MTP forward is inferred rather than proven, and where is that recorded?</summary>

The joins, not the skeleton: concatenation order inside `fc`, norm pairing beyond names, direct block input, the RoPE position rule, and `mtp.norm` as the pre-head norm. They are listed as data in `INFERRED_JOINS` in mtp.py and repeated in this chapter.
</details>

<details>
<summary>3. Why is rollback for the 48 linear layers fundamentally different from the 16 full layers?</summary>

Full layers append KV, so rollback truncates by length. Linear layers overwrite a recurrent state, so there is nothing to truncate — rollback means restoring a pre-draft snapshot, with a bandwidth cost that must be measured, not assumed away.
</details>

## Summary in 30 seconds

Transformers 5.20.1 ignores the checkpoint's `mtp.*` tensors, so the native MTP head is built from the 22 header-verified tensors themselves: two pre-fusion norms, one [5120, 10240] fusion, one proven full-attention block reusing `HybridBlock`, and a final norm feeding the caller's shared `lm_head` — with the shape-silent joins marked inferred, acceptance kept as pure shared counting with no fabricated numbers, and Phase 3's rollback design (truncate KV, snapshot recurrent state) written down before any of it is wired into generation.
