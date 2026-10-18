"""Native MTP head foundation for the hybrid Qwen3.8 stack (Kiely sec. 5.2).

Why this exists in inference
------------------------------
Multi-token prediction (MTP) attaches one small extra decoder layer to
the *target* model itself. After the target's 64 layers produce a
hidden state ``h_t`` for position ``t``, the MTP head fuses ``h_t``
with the embedding of the *next* token ``t+1`` and predicts token
``t+2``. At inference time that prediction is a draft: the target
verifies it in parallel, exactly as in draft-model speculative decoding
(:mod:`speculation`), but the draft machinery shares the target's
embedding, ``lm_head`` and hidden stream instead of being a second
model. This module builds that head's forward only -- verification,
state rollback and serving integration are Phase 3 design (docs/24).

Source of truth, and what is *not* a source
--------------------------------------------
There is no runnable HF MTP module to copy. In the installed
transformers 5.20.1 (``.venv``), a case-insensitive search for ``mtp``
over ``transformers/models/qwen3_5/{modeling,configuration,modular}_
qwen3_5.py`` finds only two hits, both
``_keys_to_ignore_on_load_unexpected = [r"^mtp.*", ...]``
(``modeling_qwen3_5.py`` lines 911 and 1666): HF explicitly *ignores*
the checkpoint's ``mtp.*`` tensors on load. ``Qwen3_5TextConfig``
declares no ``mtp_*`` field (its field list is
``configuration_qwen3_5.py`` lines 82--108), no class with ``MTP`` in
its name exists in the modeling file, and
``from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MTP``
raises ``ImportError``. The real checkpoint's own ``config.json`` does
carry ``text_config.mtp_num_hidden_layers = 1`` and
``mtp_use_dedicated_embeddings = false``.

So the structure below is derived strictly from the 22 real ``mtp.*``
tensors in ``mtp.safetensors`` (safetensors *headers only* -- names,
dtypes, shapes; tensor bytes are never read here). The layer subtype
is identified from those tensors, not assumed: ``mtp.layers.0`` holds
``self_attn.{q,k,v,o}_proj`` (+ FP8 ``weight_scale_inv`` companions),
``self_attn.{q,k}_norm`` and an SwiGLU ``mlp`` -- a *full-attention*
layer identical in shape to decoder layer 3 -- and holds no
``linear_attn.*`` tensor at all. The MTP layer is therefore built from
the existing :class:`~inference_from_scratch.hybrid.HybridBlock` with
``layer_type="full_attention"`` (gated full attention + SwiGLU), and
every module attribute mirrors its checkpoint name so a dequantised
checkpoint state dict loads with ``strict=True`` at matching dims.

What the shapes prove, and what they cannot prove
---------------------------------------------------
Proven by shapes alone (real dims, ``H = 5120``): two pre-fusion norms
(``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden``, each ``[H]``), one
fusion projection ``fc.weight [H, 2H]`` consuming a ``2H`` concatenation
and emitting ``H``, one full-attention decoder block, and one final
norm ``mtp.norm [H]`` before the shared (untied) ``lm_head`` -- the
head owns no vocab-sized tensor, so it must reuse the target's head,
and ``mtp_use_dedicated_embeddings = false`` says it likewise reuses
the target's embedding table.

Not provable from shapes (recorded in :data:`INFERRED_JOINS` and in
``docs/24_mtp.md``): the concatenation order inside ``fc`` (hidden
first vs embedding first -- both are ``[H]``, so the shape is silent),
which norm pairs with which branch beyond the names, whether the
block input is the fused vector directly, and the position the MTP
layer's RoPE should use. This implementation picks the name-ordered
reading -- ``fc(cat([norm_hidden(h), norm_embedding(e)]))``, the fused
vector fed straight into the block, RoPE at the caller-supplied
``position_ids`` / ``start_pos`` -- and marks each choice as inferred.
No acceptance number for the real model is claimed anywhere here:
:func:`inference_from_scratch.speculation.acceptance_stats` is pure
counting over caller-supplied token ids.

Tensor spec (22 checkpoint tensors, real dims)
-------------------------------------------------
``mtp.pre_fc_norm_embedding.weight`` BF16 ``[5120]``;
``mtp.pre_fc_norm_hidden.weight`` BF16 ``[5120]``;
``mtp.fc.weight`` BF16 ``[5120, 10240]``;
``mtp.layers.0.input_layernorm.weight`` BF16 ``[5120]``;
``mtp.layers.0.self_attn.q_proj.weight`` F8_E4M3 ``[12288, 5120]``
(+ ``weight_scale_inv`` BF16 ``[96, 40]``);
``k_proj`` / ``v_proj`` F8_E4M3 ``[1024, 5120]`` (+ ``[8, 40]``);
``o_proj`` F8_E4M3 ``[5120, 6144]`` (+ ``[40, 48]``);
``q_norm`` / ``k_norm`` BF16 ``[256]``;
``mtp.layers.0.post_attention_layernorm.weight`` BF16 ``[5120]``;
``mlp.gate_proj`` / ``up_proj`` F8_E4M3 ``[17408, 5120]``
(+ ``[136, 40]``); ``mlp.down_proj`` F8_E4M3 ``[5120, 17408]``
(+ ``[40, 136]``); ``mtp.norm.weight`` BF16 ``[5120]``.
That is 15 module tensors + 7 FP8 scale companions = 22.
The ``weight_scale_inv`` tensors are storage companions consumed by
the dequantising loader (ch. 22), never module parameters.

How production differs
------------------------
vLLM-class engines run the same head FP8 with fused kernels, keep an
MTP KV cache alongside the target's, and roll both back on rejection.
Here every step is explicit tiny-dims PyTorch on CPU so the fusion,
the block and the draft argmax are provable in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .hybrid import HybridBlock, HybridConfig, QwenRMSNorm

__all__ = [
    "INFERRED_JOINS",
    "MTPConfig",
    "MTPHead",
    "mtp_tensor_spec",
    "expected_mtp_tensors",
]

# Choices the tensor shapes cannot fix. Kept as data so docs/tests and
# future parity work have exactly one place to challenge them.
INFERRED_JOINS: tuple[str, ...] = (
    "fc concatenation order is [norm_hidden(h) | norm_embedding(e)] "
    "(both branches are [H]; the [H, 2H] shape cannot order them)",
    "pre_fc_norm_hidden pairs with the hidden branch and "
    "pre_fc_norm_embedding with the embedding branch (name reading only)",
    "the fused fc output feeds layers.0 directly (no extra projection "
    "tensor exists that could sit between them)",
    "the MTP block's RoPE position is the caller-supplied position "
    "(position_ids / start_pos); no mtp.* tensor encodes a position rule",
    "mtp.norm is the final norm before the shared lm_head (it is the "
    "only [H] norm left after the block; the head owns no vocab tensor)",
)


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


# ---------------------------------------------------------------------------
# MTPConfig: the MTP-relevant slice of HybridConfig / text_config.
# ---------------------------------------------------------------------------


@dataclass
class MTPConfig:
    """Shape bundle for the MTP head, parsed from the hybrid config.

    Canonical values come from :class:`HybridConfig` (itself parsed
    from the real ``text_config``): the MTP block is a full-attention
    layer, so it inherits the target's attention/MLP widths unchanged.
    ``mtp_num_hidden_layers`` is the checkpoint's own field (1 in the
    real model); ``mtp_use_dedicated_embeddings`` is recorded verbatim
    (``False`` in the real checkpoint: the provided embedding is the
    target's table, which is why this head owns no embedding).
    """

    hidden_size: int = 5120
    vocab_size: int = 248320
    mtp_num_hidden_layers: int = 1
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 17408
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] | None = field(default_factory=lambda: [11, 11, 10])
    attention_bias: bool = False
    mtp_use_dedicated_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in ("hidden_size", "vocab_size", "mtp_num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim",
                     "intermediate_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads (GQA)")
        if self.mrope_section is not None:
            self.mrope_section = [int(v) for v in self.mrope_section]

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_hybrid_config(cls, config: HybridConfig) -> "MTPConfig":
        """Copy the MTP-relevant fields out of a :class:`HybridConfig`."""
        if not isinstance(config, HybridConfig):
            raise TypeError(f"expected HybridConfig, got {type(config).__name__}")
        return cls(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            mtp_num_hidden_layers=int(config.mtp_num_hidden_layers),
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            mrope_section=list(config.mrope_section) if config.mrope_section else None,
            attention_bias=config.attention_bias,
        )

    @classmethod
    def from_text_config(cls, source: Any) -> "MTPConfig":
        """Parse a ``text_config`` dict / full ``config.json`` / path.

        Delegates to :meth:`HybridConfig.from_text_config`, then reads
        ``mtp_use_dedicated_embeddings`` from the raw mapping when the
        caller supplied one (``HybridConfig`` does not carry that flag;
        the real value is ``False``).
        """
        cfg = HybridConfig.from_text_config(source)
        out = cls.from_hybrid_config(cfg)
        raw: Mapping[str, Any] | None = None
        if isinstance(source, Mapping):
            text = source.get("text_config")
            raw = text if isinstance(text, Mapping) else source
        if raw is not None and raw.get("mtp_use_dedicated_embeddings") is not None:
            out.mtp_use_dedicated_embeddings = bool(raw["mtp_use_dedicated_embeddings"])
        return out

    @classmethod
    def from_config(cls, source: Any) -> "MTPConfig":
        """Accept an ``MTPConfig`` / ``HybridConfig`` / mapping / path."""
        if isinstance(source, MTPConfig):
            return source
        if isinstance(source, HybridConfig):
            return cls.from_hybrid_config(source)
        return cls.from_text_config(source)

    # -- aliases ------------------------------------------------------------

    @property
    def num_hidden_layers(self) -> int:
        """Number of MTP layers (alias of ``mtp_num_hidden_layers``)."""
        return self.mtp_num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.hidden_size

    def to_hybrid_config(self) -> HybridConfig:
        """A HybridConfig whose stack is exactly the MTP block(s).

        ``layer_types`` is all ``full_attention`` -- the subtype the
        ``mtp.layers.0`` tensors prove -- so :class:`HybridBlock`
        builds the gated full-attention + SwiGLU block under the
        checkpoint's own attribute names.
        """
        return HybridConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.mtp_num_hidden_layers,
            layer_types=["full_attention"] * self.mtp_num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            mrope_section=list(self.mrope_section) if self.mrope_section else None,
            attention_bias=self.attention_bias,
            mtp_num_hidden_layers=self.mtp_num_hidden_layers,
        )


# ---------------------------------------------------------------------------
# Checkpoint tensor spec (names/shapes only; mirrors hybrid_checkpoint's
# formula arithmetic so tests can pin module keys against the 22 tensors).
# ---------------------------------------------------------------------------


def mtp_tensor_spec(config: Any = None) -> dict[str, dict[str, Any]]:
    """Expected ``mtp.*`` checkpoint tensors by formula: name -> spec.

    Each entry is ``{"shape", "dtype", "kind"}`` with ``kind`` in
    ``bf16 | fp8_weight | scale_inv``. FP8 weights carry their
    ``weight_scale_inv`` companion at ``[ceil(R/128), ceil(C/128)]``
    (Kiely sec. 4.2 block format), exactly as the real headers show.
    At the real config this returns the 22 tensors of
    ``mtp.safetensors``; at tiny dims the same names carry tiny shapes.
    """
    cfg = MTPConfig.from_config(config) if config is not None else MTPConfig()
    H = cfg.hidden_size
    out: dict[str, dict[str, Any]] = {}

    def bf16(name: str, shape: list[int]) -> None:
        out[name] = {"shape": list(shape), "dtype": "BF16", "kind": "bf16"}

    def fp8(name: str, shape: list[int]) -> None:
        out[name] = {"shape": list(shape), "dtype": "F8_E4M3", "kind": "fp8_weight"}
        out[name + "_scale_inv"] = {
            "shape": [_ceil_div(shape[0], 128), _ceil_div(shape[1], 128)],
            "dtype": "BF16", "kind": "scale_inv",
        }

    if cfg.mtp_num_hidden_layers > 0:
        bf16("mtp.pre_fc_norm_embedding.weight", [H])
        bf16("mtp.pre_fc_norm_hidden.weight", [H])
        bf16("mtp.fc.weight", [H, 2 * H])
        bf16("mtp.norm.weight", [H])
    q_out = 2 * cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    for j in range(cfg.mtp_num_hidden_layers):
        p = f"mtp.layers.{j}"
        bf16(f"{p}.input_layernorm.weight", [H])
        bf16(f"{p}.post_attention_layernorm.weight", [H])
        fp8(f"{p}.self_attn.q_proj.weight", [q_out, H])
        fp8(f"{p}.self_attn.k_proj.weight", [kv_out, H])
        fp8(f"{p}.self_attn.v_proj.weight", [kv_out, H])
        fp8(f"{p}.self_attn.o_proj.weight", [H, cfg.num_attention_heads * cfg.head_dim])
        bf16(f"{p}.self_attn.q_norm.weight", [cfg.head_dim])
        bf16(f"{p}.self_attn.k_norm.weight", [cfg.head_dim])
        fp8(f"{p}.mlp.gate_proj.weight", [cfg.intermediate_size, H])
        fp8(f"{p}.mlp.up_proj.weight", [cfg.intermediate_size, H])
        fp8(f"{p}.mlp.down_proj.weight", [H, cfg.intermediate_size])
    return out


# Same entry point under the hybrid_checkpoint naming.
expected_mtp_tensors = mtp_tensor_spec


# ---------------------------------------------------------------------------
# MTPHead: pre-fc norms -> fc fusion -> full-attention block(s) -> norm.
# ---------------------------------------------------------------------------


class MTPHead(nn.Module):
    """Native MTP head mirroring the checkpoint's ``mtp.*`` names.

    Attribute names are the checkpoint names with the ``mtp.`` prefix
    dropped (the head *is* the ``mtp`` subtree): ``pre_fc_norm_embedding``,
    ``pre_fc_norm_hidden``, ``fc``, ``layers.0`` (a full-attention
    :class:`HybridBlock`: ``input_layernorm``, ``self_attn.*``,
    ``post_attention_layernorm``, ``mlp.*``) and ``norm``. A state dict
    therefore has exactly the 15 non-scale names of :func:`mtp_tensor_spec`
    at matching dims.

    The head owns neither an embedding nor an ``lm_head``: the real
    config's ``mtp_use_dedicated_embeddings = false`` and the absence
    of any vocab-sized ``mtp.*`` tensor make both the caller's. They
    arrive per-call (``embedding=`` / ``lm_head=``), so the same head
    drafts against whatever target stack produced ``hidden``.
    """

    layer_type = "full_attention"  # proven by mtp.layers.0 tensors; see docs
    concat_order = ("hidden", "embedding")  # INFERRED_JOINS[0]

    def __init__(self, config: Any = None):
        super().__init__()
        self.config = MTPConfig.from_config(config) if config is not None else MTPConfig()
        cfg = self.config
        H = cfg.hidden_size
        self.pre_fc_norm_embedding = QwenRMSNorm(H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = QwenRMSNorm(H, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        hybrid_cfg = cfg.to_hybrid_config()
        self.layers = nn.ModuleList(
            HybridBlock(hybrid_cfg, layer_idx=i) for i in range(cfg.mtp_num_hidden_layers)
        )
        self.norm = QwenRMSNorm(H, eps=cfg.rms_norm_eps)

    # -- state --------------------------------------------------------------

    def initial_state(self) -> list[Any]:
        """Fresh per-layer state: one ``None`` (empty KV) per MTP layer."""
        return [None] * len(self.layers)

    # -- input resolution -----------------------------------------------------

    def _resolve_embedding(
        self,
        token_embedding_or_ids: Any,
        embedding: Callable[..., torch.Tensor] | nn.Module | None,
        token_embedding: torch.Tensor | None,
        token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_embedding is not None:
            emb = token_embedding
        elif token_ids is not None:
            if embedding is None:
                raise ValueError("token_ids given but no embedding= callable/module provided")
            emb = embedding(token_ids)
        elif isinstance(token_embedding_or_ids, torch.Tensor) and token_embedding_or_ids.dtype in (
            torch.long, torch.int, torch.int32, torch.short,
        ):
            if embedding is None:
                raise ValueError(
                    "token ids given but no embedding= callable/module provided; "
                    "pass the target's embedding (mtp_use_dedicated_embeddings=false)"
                )
            emb = embedding(token_embedding_or_ids)
        elif isinstance(token_embedding_or_ids, torch.Tensor):
            emb = token_embedding_or_ids
        else:
            raise ValueError(
                "pass token embeddings [.., H] or token ids via "
                "token_embedding_or_ids / token_ids= (with embedding=)"
            )
        if not isinstance(emb, torch.Tensor):
            raise ValueError(f"embedding must produce a tensor, got {type(emb).__name__}")
        return emb

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        token_embedding_or_ids: Any = None,
        *,
        embedding: Callable[..., torch.Tensor] | nn.Module | None = None,
        token_embedding: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        lm_head: Callable[..., torch.Tensor] | nn.Module | None = None,
        position_ids: torch.Tensor | None = None,
        start_pos: int = 0,
        state: Sequence[Any] | Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[Any]]:
        """One MTP draft step: ``(draft_hidden, logits, new_state)``.

        ``hidden`` is the target's hidden state ``[B, D]`` or
        ``[B, T, D]``; the token input (positional, or ``token_ids=`` /
        ``token_embedding=``) is the *next* token's embedding ``[B, D]``
        / ``[B, T, D]``, or its ids resolved through ``embedding``
        (a callable ``ids -> [..., D]``, typically the target's
        ``nn.Embedding``). The two branches are normed separately,
        concatenated hidden-first (inferred; see :data:`INFERRED_JOINS`),
        fused by ``fc``, run through the full-attention MTP block(s)
        and the final ``norm``. ``logits`` is ``lm_head(draft_hidden)``
        when a callable head is given, else ``None`` -- the head owns
        no vocab projection of its own.

        ``position_ids`` / ``start_pos`` give the MTP block's RoPE
        position (the target position the draft is *for*); ``state``
        threads per-layer KV pairs (``initial_state()`` form) and the
        returned ``new_state`` is the extended one. Inputs are never
        mutated, so a pre-step state can be replayed identically.
        """
        if not isinstance(hidden, torch.Tensor) or hidden.dim() not in (2, 3):
            raise ValueError(f"hidden must be [B, D] or [B, T, D], got {tuple(hidden.shape) if isinstance(hidden, torch.Tensor) else type(hidden).__name__}")
        H = self.config.hidden_size
        if hidden.shape[-1] != H:
            raise ValueError(f"hidden last dim must be hidden_size={H}, got {hidden.shape[-1]}")
        emb = self._resolve_embedding(token_embedding_or_ids, embedding, token_embedding, token_ids)
        squeeze = hidden.dim() == 2
        h3 = hidden.unsqueeze(1) if squeeze else hidden
        e3 = emb.unsqueeze(1) if emb.dim() == h3.dim() - 1 else emb
        if e3.shape != h3.shape:
            # Allow [B, D] embeddings against [B, 1, D] hidden and vice versa.
            try:
                e3 = torch.broadcast_to(e3, h3.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"token embedding shape {tuple(emb.shape)} cannot align with hidden {tuple(hidden.shape)}"
                ) from exc
        if e3.shape[-1] != H:
            raise ValueError(f"token embedding last dim must be hidden_size={H}, got {e3.shape[-1]}")

        fused = self.fc(torch.cat(
            [self.pre_fc_norm_hidden(h3), self.pre_fc_norm_embedding(e3)], dim=-1
        ))

        if state is None:
            layer_states: list[Any] = self.initial_state()
        elif isinstance(state, (list, tuple)) and len(state) == len(self.layers) and not (
            len(state) == 2 and all(isinstance(s, torch.Tensor) for s in state)
        ):
            layer_states = list(state)
        elif len(self.layers) == 1:
            layer_states = [state]  # single bare (k, v) pair / KVCache
        else:
            raise ValueError(f"state must hold {len(self.layers)} per-layer entries")
        x = fused
        new_state: list[Any] = []
        for block, layer_state in zip(self.layers, layer_states):
            x, new_layer_state = block.forward_with_state(
                x, layer_state, start_pos=int(start_pos), position_ids=position_ids
            )
            new_state.append(new_layer_state)

        draft_hidden = self.norm(x)
        if squeeze:
            draft_hidden = draft_hidden[:, 0, :]
        logits = lm_head(draft_hidden) if lm_head is not None else None
        return draft_hidden, logits, new_state

    def draft_token(
        self, *args: Any, lm_head: Callable[..., torch.Tensor] | nn.Module, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
        """Greedy draft: ``(token_ids, draft_hidden, new_state)``.

        ``token_ids`` is the argmax of the draft logits (``[B]`` for a
        ``[B, D]`` hidden, ``[B, T]`` otherwise). Deterministic for a
        fixed head/input pair -- the property the tests pin.
        """
        draft_hidden, logits, new_state = self.forward(*args, lm_head=lm_head, **kwargs)
        assert logits is not None  # lm_head was required above
        return torch.argmax(logits, dim=-1), draft_hidden, new_state
