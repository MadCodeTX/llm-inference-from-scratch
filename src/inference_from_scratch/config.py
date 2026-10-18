"""Ch 00/07 -- ModelConfig: the single source of truth for model shape.

Why this exists in inference
-----------------------------
Every inference component (embedding table, attention, KV-cache sizing,
RoPE cache, sampler) needs the same handful of integers: how wide the
residual stream is, how many layers / heads exist, and so on. Passing
twelve loose ints around guarantees they will drift apart, so they are
bundled into one frozen-ish dataclass that is constructed once and
handed to every submodule -- exactly how HF ``PretrainedConfig`` or
vLLM's ``ModelConfig`` work, minus the hundreds of extra fields.

How production differs
-----------------------
Qwen / Llama ``config.json`` files carry the same fields under different
names (``hidden_size``, ``num_hidden_layers`` ...); ``weights.py``
translates them into this dataclass (see ``load_hf_config``). Production
configs also carry dtype, quantization, sliding-window and MoE routing
settings. We deliberately keep only what the dissected forward pass
needs, so each field below maps to exactly one tensor shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Shape / hyper-parameter bundle for the DecoderLM in ``model.py``.

    Fields (no tensor shapes yet -- this is pure metadata):
        vocab_size:  number of token ids, V. Embedding is [V, d_model].
        d_model:     residual-stream width, D. Almost every weight is
                     sized in terms of D.
        n_layer:     number of Transformer blocks, L.
        n_head:      query heads per layer, H. Must divide ``d_model``.
        n_kv_head:   key/value heads per layer, H_kv. ``n_kv_head == n_head``
                     is classic multi-head attention (MHA);
                     ``n_kv_head == 1`` is multi-query attention (MQA);
                     anything in between is grouped-query attention (GQA),
                     which is what Qwen / Llama-3 ship because it shrinks
                     the KV cache by ``n_head // n_kv_head``.
        d_ff:        MLP hidden width, F. SwiGLU / GELU MLP is
                     [d_model -> d_ff -> d_model].
        max_seq_len: maximum positions, T_max. Sizes the RoPE cache
                     [T_max, head_dim] and bounds the KV cache.
        rope_theta:  RoPE base frequency (10000 for Llama-2, 1e6 for
                     Qwen2 long-context). Only used when pos_type='rope'.
        rms_eps:     epsilon inside RMSNorm / LayerNorm variance, guards
                     division by ~0 on the first tokens.
        tie_embeddings: if True the LM head reuses the embedding matrix
                     ([V, D] transposed) instead of owning a second
                     [V, D] matrix -- saves V*D params, standard for
                     small models (GPT-2, Qwen small sizes).
        norm_type:   'rmsnorm' (Llama/Qwen) or 'layernorm' (GPT-2).
        mlp_type:    'swiglu' (Llama/Qwen) or 'gelu' (GPT-2).
        pos_type:    'rope' (rotary, no learned table) or 'learned'
                     (adds an embedding row per position, GPT-2 style).
    """

    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_ff: int
    max_seq_len: int
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    norm_type: str = "rmsnorm"  # 'rmsnorm' | 'layernorm'
    mlp_type: str = "swiglu"  # 'swiglu' | 'gelu'
    pos_type: str = "rope"  # 'rope' | 'learned'

    def __post_init__(self) -> None:
        # Fail fast at construction: a config that cannot produce a
        # valid head_dim would otherwise explode deep inside attention
        # with a far less readable reshape error.
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_head={self.n_head}"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head={self.n_head} must be divisible by n_kv_head={self.n_kv_head} "
                "(GQA requires each KV head to serve an integer number of Q heads)"
            )
        for name in ("vocab_size", "d_model", "n_layer", "n_head", "n_kv_head", "d_ff", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm_type must be 'rmsnorm'|'layernorm', got {self.norm_type!r}")
        if self.mlp_type not in ("swiglu", "gelu"):
            raise ValueError(f"mlp_type must be 'swiglu'|'gelu', got {self.mlp_type!r}")
        if self.pos_type not in ("rope", "learned"):
            raise ValueError(f"pos_type must be 'rope'|'learned', got {self.pos_type!r}")

    @property
    def head_dim(self) -> int:
        """Per-head width D_h = d_model // n_head. RoPE and QK^T scaling
        are sized by this, never by d_model directly."""
        return self.d_model // self.n_head
