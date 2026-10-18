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

## Phase 3 integration design (not yet built)

Wiring this head into generation is the next phase; the design below is the contract that phase must satisfy, and the reasons it is not already wired in.

**The loop.** Prefill the target once over the prompt, keeping the last hidden state (the hybrid model must expose hidden states, not only logits — a small, explicit addition to `TinyHybridLM`). Draft with the MTP head from that hidden state and the just-emitted token's embedding. Verify with one target forward over the drafted run, accept the longest agreeing prefix plus the correction/bonus token (chapter 14's rule), and feed the accepted run's last hidden state into the next draft. Acceptance is reported through `acceptance_stats` on the ids actually observed.

**State rollback, the hard part.** Verification gambles on state that rejection then invalidates, and the hybrid stack makes this sharper than a plain KV-cache model:

- The 16 full-attention layers append KV per token. Rolling back is a length truncation: keep the cache preallocated and slice by the accepted length, exactly the bookkeeping chapter 14 avoided by going cache-free.
- The 48 linear layers keep a recurrent matrix `S` and a conv buffer that are *overwritten*, not appended. There is no prefix of `S` to slice back to: the only correct rollback is a snapshot taken before the drafted run (the existing `snapshot_state` / `restore_state` deep-copy, proven bit-identical in chapter 22's branching test) restored on rejection. Snapshotting 48 recurrent states per draft step is a real memory-bandwidth cost (Kiely sec. 2.4–2.5), so Phase 3 must measure snapshot cost against the tokens the draft saves before claiming any speedup.
- The MTP layer's own KV cache has the same truncation question as the full layers, plus one subtlety: its entries were computed from *drafted* positions, so on partial acceptance only the accepted prefix's entries survive.
- Batching multiplies all of the above per row: rows accept different lengths, so rollback points diverge per sequence. The typed `HybridState` (chapter 22) is the seam where per-row snapshot/restore belongs.

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
