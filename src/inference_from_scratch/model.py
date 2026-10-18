"""Ch 07 -- Full model: TransformerBlock + DecoderLM.

Why this exists in inference
------------------------------
Every earlier chapter built one piece: norms + RoPE (Ch 02-03),
attention with GQA and a KV cache (Ch 04-05), the MLP (Ch 06). This
module wires them into the decoder-only Transformer that inference
actually runs: embed -> L x (pre-norm attention + pre-norm MLP, each
with a residual add) -> final norm -> LM head. There is no encoder
and no cross-attention; the only sequence mixing is causal
self-attention, which is what makes the KV cache (Ch 05) and the
prefill/decode split (Ch 00/08) possible at all.

Pre-norm, and why
-----------------
Each sublayer reads a *normalised copy* of the residual stream and
adds its output back: ``x = x + attn(norm(x))``, then
``x = x + mlp(norm(x))``. The residual stream itself is never
normalised inside the stack (only once, by the final norm, before
the head), so it is an uninterrupted identity highway from embedding
to head. Post-norm (normalise after the add) puts a norm on that
highway in every block; it trains less stably at depth and is not
what Llama/Qwen ship, so this repo is pre-norm throughout.

Shapes
------
``idx`` is ``[B, T]`` token ids. The residual stream ``x`` is
``[B, T, d_model]`` from the embedding lookup until the head. The
head emits logits ``[B, T, vocab_size]`` -- one distribution per
input position, each predicting the *next* token. Generation
(Ch 08) only ever reads the last position's logits.

Tied embeddings
---------------
When ``config.tie_embeddings`` is true the LM head *is* the
embedding matrix (``lm_head.weight is tok_embedding.weight``): the
same ``[V, D]`` table maps ids -> vectors on the way in and scores
vectors -> ids on the way out. Small models (GPT-2, Qwen small)
do this; it saves ``V * D`` parameters at negligible quality cost.

How production differs
----------------------
A production checkpoint (Llama/Qwen) is exactly this graph, wider
and deeper, sometimes with sliding-window attention, MoE MLPs or
a bias on the Q/K/V projections. The dissected version keeps every
choice in ``ModelConfig`` (norm / MLP / position flavour) so the
same code instantiates the GPT-2-style and Llama-style variants
the tests compare -- and, via Ch 09, real Qwen weights.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .kv_cache import KVCache, StaticKVCache
from .layers import LayerNorm, RMSNorm, precompute_rope_cache
from .mlp import GeluMLP, SwiGLU


def _make_norm(config: ModelConfig) -> nn.Module:
    """Norm module for ``config.norm_type`` (single construction point)."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, eps=config.rms_eps)
    return LayerNorm(config.d_model, eps=config.rms_eps)


def _make_mlp(config: ModelConfig) -> nn.Module:
    """MLP module for ``config.mlp_type`` (single construction point)."""
    if config.mlp_type == "swiglu":
        return SwiGLU(config.d_model, config.d_ff)
    return GeluMLP(config.d_model, config.d_ff)


class TransformerBlock(nn.Module):
    """One pre-norm decoder block: attention sublayer + MLP sublayer.

    Structure (both sublayers pre-norm, residual adds around each)::

        x = x + attn(attn_norm(x))     # tokens talk to each other
        x = x + mlp(mlp_norm(x))       # tokens think, independently

    ``attn_norm`` / ``mlp_norm`` are RMSNorm or LayerNorm per
    ``config.norm_type``; ``mlp`` is SwiGLU or GeluMLP per
    ``config.mlp_type``. No dropout and no biases in the attention
    or MLP projections: inference is deterministic, and the
    cache-equivalence tests require prefill and decode to agree
    exactly.

    State-dict names (Ch 09 maps HF checkpoints onto these)::

        attn_norm.weight, attn.{q,k,v,o}_proj.weight,
        mlp_norm.weight,  mlp.{gate,up,down}_proj.weight  (SwiGLU)
                          mlp.{up,down}_proj.weight       (GELU)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.attn_norm = _make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = _make_norm(config)
        self.mlp = _make_mlp(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        kv_cache: KVCache | StaticKVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """``[B, T, D] -> [B, T, D]``; see class docstring for the graph.

        ``cos`` / ``sin`` are the ``[T_max, head_dim]`` RoPE tables
        (``None`` for the learned-position variant). ``kv_cache`` /
        ``layer_idx`` / ``start_pos`` are forwarded verbatim to
        attention: ``layer_idx`` selects this block's slice of the
        shared cache object, ``start_pos`` is the absolute position
        of ``x``'s first token (0 prefill, >0 decode).
        """
        # --- Attention sublayer: x [B, T, D] --------------------------
        # attn_norm(x): [B, T, D]; attn(...): [B, T, D].
        x = x + self.attn(
            self.attn_norm(x), cos, sin,
            kv_cache=kv_cache, layer_idx=layer_idx, start_pos=start_pos,
        )  # [B, T, d_model] (residual add keeps the shape)
        # --- MLP sublayer: per-token, no cross-token maths ------------
        # mlp_norm(x): [B, T, D]; mlp(...): [B, T, d_model].
        x = x + self.mlp(self.mlp_norm(x))  # [B, T, d_model]
        return x


class DecoderLM(nn.Module):
    """Decoder-only language model: embeddings -> blocks -> head.

    State-dict names (the exact targets Ch 09 maps HF weights to)::

        tok_embedding.weight                     [V, D]
        pos_embedding.weight                     [T_max, D]  (learned only)
        layers.{i}.attn_norm.weight             [D]
        layers.{i}.attn.{q,k,v,o}_proj.weight
        layers.{i}.mlp_norm.weight              [D]
        layers.{i}.mlp.{gate,up,down}_proj.weight / {up,down} (GELU)
        final_norm.weight                        [D]  (+ bias if LayerNorm)
        lm_head.weight                           [V, D] (tied: shared, see below)

    When ``config.tie_embeddings`` is true, ``lm_head.weight`` is
    the *same Parameter object* as ``tok_embedding.weight``
    (``lm_head.weight is tok_embedding.weight``) and is stored once
    in the state dict (under ``tok_embedding.weight``; PyTorch drops
    the duplicate by default). Otherwise the head owns its matrix.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        V, D = config.vocab_size, config.d_model  # scalars

        # ids [B, T] -> x [B, T, D]
        self.tok_embedding = nn.Embedding(V, D)

        # Learned positions (GPT-2 style): one [D] row per absolute
        # position, added to the token embedding. Only built for
        # pos_type='learned'; the RoPE variant has no position table
        # at all (position lives in the Q/K rotations instead).
        if config.pos_type == "learned":
            self.pos_embedding: nn.Embedding | None = nn.Embedding(
                config.max_seq_len, D
            )
        else:
            self.pos_embedding = None

        # cos/sin: [max_seq_len, head_dim] each. Registered as
        # non-persistent buffers: they are pure functions of the
        # config, so they must not pollute the checkpoint (Ch 09
        # loads real weights into a fresh model and would otherwise
        # have to special-case them). Placeholders keep the attribute
        # present for the learned variant (None means "no RoPE").
        if config.pos_type == "rope":
            cos, sin = precompute_rope_cache(
                config.max_seq_len,
                config.head_dim,
                theta=config.rope_theta,
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        else:
            self.cos = None  # type: ignore[assignment]
            self.sin = None  # type: ignore[assignment]

        # The stack: L identical-shape blocks, [B, T, D] throughout.
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )

        # Final norm: the residual stream is normalised exactly once,
        # here, before the head (pre-norm invariant, see module docs).
        self.final_norm = _make_norm(config)

        # Head: [B, T, D] -> logits [B, T, V], bias-free.
        self.lm_head = nn.Linear(D, V, bias=False)
        if config.tie_embeddings:
            # Weight tying: one shared [V, D] Parameter. Assignment
            # makes lm_head.weight the embedding's Parameter object.
            self.lm_head.weight = self.tok_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        # Small-std normal init keeps early logits near-uniform, so
        # the tiny training demo (scripts/demo_generate.py) starts
        # from a sane loss (~ln V) instead of saturated softmaxes.
        # This repo is about inference; the init only needs to be
        # good enough for a few hundred demo steps to learn from.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # lm_head shares the embedding weight when tied; re-applying
        # the loop above already initialised that one Parameter once
        # (modules() deduplicates), so nothing extra is needed here.

    def num_params(self) -> int:
        """Total parameter count (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        kv_caches: KVCache | StaticKVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | StaticKVCache | None]:
        """Token ids -> (logits, cache): ``[B, T] -> [B, T, V]``.

        ``kv_caches`` is a *single* cache object for the whole model
        (a :class:`KVCache` or :class:`StaticKVCache` built with
        ``n_layer=config.n_layer``), or ``None`` for the no-cache
        reference path. It is threaded through every block with the
        block's ``layer_idx`` and returned unchanged (same object),
        so the caller (``generate.py``) can keep feeding it back:
        prefill once with ``start_pos=0``, then one ``[B, 1]`` step
        per decode token with ``start_pos`` = that token's absolute
        position.

        Learned positions: rows ``start_pos .. start_pos + T - 1``
        of ``pos_embedding`` are added, so a decode step must pass
        its absolute position -- the same bookkeeping RoPE needs.
        RoPE positions need no work here: ``cos``/``sin`` are handed
        to attention, which slices them at ``start_pos``.
        """
        B, T = idx.shape  # idx: [B, T] token ids
        # ids [B, T] -> residual stream [B, T, D]
        x = self.tok_embedding(idx)  # [B, T, d_model]

        if self.config.pos_type == "learned":
            assert self.pos_embedding is not None
            # positions [T] (absolute) -> pos rows [T, D] -> [1, T, D]
            positions = torch.arange(
                start_pos, start_pos + T, device=idx.device
            )  # [T]
            x = x + self.pos_embedding(positions).unsqueeze(0)  # [B, T, D]
            cos = sin = None  # learned variant: no rotary tables
        else:
            cos, sin = self.cos, self.sin  # [T_max, D_h] each

        # Blocks: [B, T, D] -> [B, T, D] each, cache appended in place.
        for layer_idx, block in enumerate(self.layers):
            x = block(
                x, cos, sin,
                kv_cache=kv_caches, layer_idx=layer_idx, start_pos=start_pos,
            )  # [B, T, d_model]

        x = self.final_norm(x)  # [B, T, d_model]
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits, kv_caches
