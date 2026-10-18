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
    "WIRING_AXES",
    "PRUNED_WIRING",
    "LEADING_WIRING",
    "resolve_wiring",
    "enumerate_wiring_candidates",
    "teacher_forced_draft_indices",
    "clone_mtp_state",
    "mtp_state_length",
    "MTPConfig",
    "MTPHead",
    "MTPDraftModule",
    "TPMTPDraftModule",
    "mtp_tensor_spec",
    "expected_mtp_tensors",
]

# Candidate wiring space, enumerated explicitly as data. Each axis lists
# every shape-compatible reading; an option is *live* until the checkpoint
# (names / shapes / config) rules it out, in which case it moves to
# PRUNED_WIRING with its reason and never reaches the acceptance sweep.
# Axes (see docs/24):
#   fc_concat_order  -- which normed branch occupies fc's first H columns
#   norm_pairing     -- which pre_fc norm feeds which branch
#   hidden_source    -- target hidden fed to the head, before/after the
#                       target's final norm (both are [H]; shape-silent)
#   embedding_source -- token embedding table (see PRUNED_WIRING)
#   position_rule    -- absolute RoPE position of one MTP draft step:
#                       the hidden's own position t, the fed token's
#                       position t+1, or the predicted token's t+2
#   lm_head_input    -- shared lm_head reads mtp.norm's output or the
#                       raw block output (mtp.norm would be unused then)
WIRING_AXES: dict[str, tuple[str, ...]] = {
    "fc_concat_order": ("hidden_first", "embedding_first"),
    "norm_pairing": ("named", "swapped"),
    "hidden_source": ("pre_final_norm", "post_final_norm"),
    "embedding_source": ("shared_embed_tokens",),
    "position_rule": ("hidden_pos", "input_token_pos", "predicted_pos"),
    "lm_head_input": ("mtp_norm", "raw_block"),
}

# Options the checkpoint itself rules out (never swept):
# a dedicated MTP embedding would need a vocab-sized mtp.* tensor (none
# exists among the 22) and config mtp_use_dedicated_embeddings=false.
PRUNED_WIRING: tuple[dict[str, str], ...] = (
    {"axis": "embedding_source", "option": "dedicated_mtp_embedding",
     "reason": "no vocab-sized mtp.* tensor exists and "
               "mtp_use_dedicated_embeddings=false; only the shared "
               "embed_tokens table is available"},
)


def enumerate_wiring_candidates() -> list[dict[str, str]]:
    """All live candidate wirings: the cartesian product of WIRING_AXES.

    2 (fc order) x 2 (norm pairing) x 2 (hidden source) x 3 (position)
    x 2 (lm_head input) = 48 candidates, each a dict axis -> option.
    Pruned options (PRUNED_WIRING) are excluded by construction.
    """
    import itertools
    axes = list(WIRING_AXES)
    return [dict(zip(axes, combo))
            for combo in itertools.product(*(WIRING_AXES[a] for a in axes))]


# Leading candidate from benchmarks/mtp/wiring_sweep.json (2026-10-18):
# the sweep is UNRESOLVED (top rows tie at 0.8721, margin_over_second=0),
# so this is not a proven wiring -- it is the first-ranked candidate,
# used as the draft modules' default behind the wiring= parameter.
# fc=embedding_first is decisive (0.665-0.872 vs 0.005-0.016); the tied
# axes (hidden_source, position_rule, lm_head_input) follow the top row.
LEADING_WIRING: dict[str, str] = {
    "fc_concat_order": "embedding_first",
    "norm_pairing": "named",
    "hidden_source": "post_final_norm",
    "embedding_source": "shared_embed_tokens",
    "position_rule": "hidden_pos",
    "lm_head_input": "raw_block",
}

_POSITION_OFFSET = {"hidden_pos": 0, "input_token_pos": 1, "predicted_pos": 2}


def resolve_wiring(wiring: Mapping[str, str] | None = None) -> dict[str, str]:
    """Validate/complete a wiring dict against :data:`WIRING_AXES`.

    ``None`` selects :data:`LEADING_WIRING` (the sweep's first-ranked,
    unresolved candidate). Unknown axes/options raise ``ValueError``;
    the pruned dedicated-embedding option is rejected by construction.
    """
    out = dict(LEADING_WIRING if wiring is None else wiring)
    for axis, option in out.items():
        if axis not in WIRING_AXES:
            raise ValueError(f"unknown wiring axis {axis!r}; axes are {sorted(WIRING_AXES)}")
        if option not in WIRING_AXES[axis]:
            raise ValueError(f"wiring[{axis!r}] must be one of {WIRING_AXES[axis]}, got {option!r}")
    for axis in WIRING_AXES:
        out.setdefault(axis, LEADING_WIRING[axis])
    return out


def clone_mtp_state(state: Any) -> Any:
    """Deep-copy an MTP draft state (tensors cloned; structure kept)."""
    if state is None or isinstance(state, (int, float, str, bool)):
        return state
    if isinstance(state, torch.Tensor):
        return state.clone()
    if isinstance(state, list):
        return [clone_mtp_state(s) for s in state]
    if isinstance(state, tuple):
        return tuple(clone_mtp_state(s) for s in state)
    if isinstance(state, dict):
        return {k: clone_mtp_state(v) for k, v in state.items()}
    clone = getattr(state, "clone", None)
    if callable(clone):
        return clone()
    raise TypeError(f"cannot clone MTP state entry of type {type(state).__name__}")


def mtp_state_length(state: Any) -> int:
    """Positions held in an MTP draft state (0 for a fresh state).

    Dense states are per-layer ``(k, v)`` pairs / ``None``; TP states
    are per-layer local ``(k, v)`` pairs / ``None``. The length is the
    KV time dim of the first non-empty layer.
    """
    if state is None:
        return 0
    entries = state if isinstance(state, (list, tuple)) else [state]
    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], torch.Tensor):
            return int(entry[0].shape[-2])
        if isinstance(entry, torch.Tensor):
            return int(entry.shape[-2])
    return 0


def teacher_forced_draft_indices(seq_len: int) -> list[tuple[int, int, int]]:
    """Teacher-forced MTP index arithmetic for one trajectory.

    For a token sequence ``S[0..L-1]`` with target hidden states
    ``h[t]`` aligned to ``S[t]``, the MTP draft at step ``t`` fuses
    ``h[t]`` with ``embed(S[t+1])`` and predicts ``S[t+2]`` (DeepSeek-
    style MTP semantics, docs/24). Returns one
    ``(hidden_index, embedding_token_index, target_index)`` triple per
    valid step, i.e. ``t`` in ``[0, L-3]``; empty when ``L < 3``.
    Example: ``teacher_forced_draft_indices(5)`` over ``[10,11,12,13,14]``
    gives steps predicting 12 from (h0, embed 11), 13 from (h1, embed
    12) and 14 from (h2, embed 13).
    """
    if int(seq_len) < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    return [(t, t + 1, t + 2) for t in range(int(seq_len) - 2)]


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

    def __init__(self, config: Any = None, *,
                 fc_concat_order: str = "hidden_first",
                 norm_pairing: str = "named",
                 lm_head_input: str = "mtp_norm"):
        super().__init__()
        if fc_concat_order not in WIRING_AXES["fc_concat_order"]:
            raise ValueError(f"fc_concat_order must be one of {WIRING_AXES['fc_concat_order']}")
        if norm_pairing not in WIRING_AXES["norm_pairing"]:
            raise ValueError(f"norm_pairing must be one of {WIRING_AXES['norm_pairing']}")
        if lm_head_input not in WIRING_AXES["lm_head_input"]:
            raise ValueError(f"lm_head_input must be one of {WIRING_AXES['lm_head_input']}")
        # Wiring options (defaults are the inferred name-ordered reading;
        # the acceptance sweep overrides them per candidate).
        self.fc_concat_order = fc_concat_order
        self.norm_pairing = norm_pairing
        self.lm_head_input = lm_head_input
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

        if self.norm_pairing == "named":
            h_normed = self.pre_fc_norm_hidden(h3)
            e_normed = self.pre_fc_norm_embedding(e3)
        else:  # swapped: each branch through the other branch's norm
            h_normed = self.pre_fc_norm_embedding(h3)
            e_normed = self.pre_fc_norm_hidden(e3)
        branches = ([h_normed, e_normed] if self.fc_concat_order == "hidden_first"
                    else [e_normed, h_normed])
        fused = self.fc(torch.cat(branches, dim=-1))

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

        draft_hidden = self.norm(x) if self.lm_head_input == "mtp_norm" else x
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


# ---------------------------------------------------------------------------
# MTPDraftModule: reusable single-step drafter (dense + tensor-parallel).
#
# State semantics (the MTP block is a proven full-attention layer, so its
# state is KV, not recurrent): draft_token NEVER mutates its input state.
# The returned state is TENTATIVE -- it holds the KV entry for the token
# just consumed. The caller commits it only when the target accepts the
# associated token run, and simply keeps the old state on rejection
# (functional commit-on-accept; snapshot_state/restore_state give the
# explicit deep-copy form). The state therefore advances only on
# accepted tokens, which is exactly the speculative-decoding contract:
# speculative greedy output must equal target greedy token-for-token.
# If a future MTP layer were linear-attention, the same functional
# contract would carry its (S, conv) pair instead of KV.
# ---------------------------------------------------------------------------


def _as_token_ids(token_id_t1: Any, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(token_id_t1, int):
        return torch.full((batch,), int(token_id_t1), dtype=torch.long, device=device)
    ids = torch.as_tensor(token_id_t1, device=device)
    if ids.dim() == 0:
        return ids.to(torch.long).expand(batch).reshape(batch)
    ids = ids.reshape(-1).to(torch.long)
    if ids.shape[0] == 1 and batch > 1:
        return ids.expand(batch)
    if ids.shape[0] != batch:
        raise ValueError(f"token_id_t1 must be a scalar or [{batch}], got {tuple(ids.shape)}")
    return ids


class MTPDraftModule(nn.Module):
    """Reusable one-step MTP drafter over a dense :class:`MTPHead`.

    ``draft_token(hidden_t, token_id_t1, state, position)`` fuses the
    target's final hidden at ``t`` (which norm side is given by
    ``wiring["hidden_source"]``; the leading candidate reads the
    post-final-norm hidden) with the embedding of the target's chosen
    token for ``t+1`` and returns ``(draft_token_id, draft_logits,
    tentative_state)`` predicting ``t+2``. With ``top_k=k`` the second
    value is ``(top_values, top_indices)`` instead of full logits.
    ``embedding`` / ``lm_head`` are the target's shared modules (the
    head owns neither); pass them here or per call. ``wiring`` selects
    a candidate from :func:`enumerate_wiring_candidates`; the default
    is :data:`LEADING_WIRING` because the sweep is unresolved -- no
    wiring is claimed proven.
    """

    def __init__(self, config: Any = None, *,
                 embedding: Any = None, lm_head: Any = None,
                 wiring: Mapping[str, str] | None = None,
                 head: MTPHead | None = None):
        super().__init__()
        self.wiring = resolve_wiring(wiring)
        if head is not None:
            self.head = head
            self.wiring.update({
                "fc_concat_order": head.fc_concat_order,
                "norm_pairing": head.norm_pairing,
                "lm_head_input": head.lm_head_input})
            self.config = head.config
        else:
            self.config = MTPConfig.from_config(config) if config is not None else MTPConfig()
            self.head = MTPHead(
                self.config,
                fc_concat_order=self.wiring["fc_concat_order"],
                norm_pairing=self.wiring["norm_pairing"],
                lm_head_input=self.wiring["lm_head_input"])
        # Shared target modules: referenced, never owned/registered.
        object.__setattr__(self, "_embedding", embedding)
        object.__setattr__(self, "_lm_head", lm_head)

    def initial_state(self) -> list[Any]:
        """Fresh draft state (one empty KV entry per MTP layer)."""
        return self.head.initial_state()

    @staticmethod
    def snapshot_state(state: Any) -> Any:
        """Deep-copy a draft state for later restore/branching."""
        return clone_mtp_state(state)

    @staticmethod
    def restore_state(snapshot: Any) -> Any:
        """Clone a snapshot into an independently continuable state."""
        return clone_mtp_state(snapshot)

    def draft_token(self, hidden_t: torch.Tensor, token_id_t1: Any,
                    state: Any = None, position: int = 0, *,
                    top_k: int | None = None,
                    embedding: Any = None, lm_head: Any = None):
        """One draft step: ``(draft_token_id, logits_or_topk, state')``.

        ``hidden_t`` is ``[H]`` or ``[B, H]``; ``token_id_t1`` a scalar
        or ``[B]`` token id tensor. ``position`` is the hidden's own
        absolute position ``t``; the wiring's ``position_rule`` offset
        is applied internally. Input ``state`` is never mutated: the
        returned state is tentative until the caller commits it on
        acceptance (see the section comment above).
        """
        emb = embedding if embedding is not None else self._embedding
        head_fn = lm_head if lm_head is not None else self._lm_head
        if emb is None:
            raise ValueError("no embedding: pass embedding= or construct with one "
                             "(mtp_use_dedicated_embeddings=false, shared table)")
        if head_fn is None:
            raise ValueError("no lm_head: pass lm_head= or construct with one "
                             "(the MTP head owns no vocab projection)")
        if not isinstance(hidden_t, torch.Tensor) or hidden_t.dim() not in (1, 2):
            raise ValueError(f"hidden_t must be [H] or [B, H], got "
                             f"{tuple(hidden_t.shape) if isinstance(hidden_t, torch.Tensor) else type(hidden_t).__name__}")
        single = hidden_t.dim() == 1
        h2 = hidden_t.unsqueeze(0) if single else hidden_t
        ids = _as_token_ids(token_id_t1, int(h2.shape[0]), h2.device)
        abs_pos = int(position) + _POSITION_OFFSET[self.wiring["position_rule"]]
        _, logits, new_state = self.head(
            h2, token_ids=ids, embedding=emb, lm_head=head_fn,
            start_pos=abs_pos, state=state)
        assert logits is not None
        draft_id = torch.argmax(logits, dim=-1)
        second: Any = logits
        if top_k is not None:
            if int(top_k) <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            second = torch.topk(logits, int(top_k), dim=-1)
            second = (second.values, second.indices)
        if single:
            draft_id = draft_id[0]
            second = (second[0][0], second[1][0]) if isinstance(second, tuple) else second[0]
        return draft_id, second, new_state

    forward = draft_token


class TPMTPDraftModule(nn.Module):
    """Tensor-parallel MTP drafter: the MTP block sharded like a block.

    ``fc`` and the pre-fc / final norms are replicated on every rank
    (docs/23: ``mtp.fc`` replicated so both fused branches stay local);
    the MTP layer itself is a :class:`DistributedHybridBlock`, so its
    collectives are exactly a normal full-attention block's (one
    all-reduce in attention's ``o_proj``, one in the MLP ``down_proj``).
    State is this rank's local KV shard per MTP layer, with the same
    commit-on-accept semantics as :class:`MTPDraftModule`.
    """

    def __init__(self, ctx: Any, dense_head: MTPHead, *,
                 embedding: Any = None, lm_head: Any = None,
                 wiring: Mapping[str, str] | None = None):
        super().__init__()
        import copy as _copy
        from .tp_distributed import DistributedHybridBlock
        self.ctx = ctx
        self.wiring = resolve_wiring(wiring)
        self.wiring.update({
            "fc_concat_order": dense_head.fc_concat_order,
            "norm_pairing": dense_head.norm_pairing,
            "lm_head_input": dense_head.lm_head_input})
        self.config = dense_head.config
        dev = ctx.device
        self.pre_fc_norm_embedding = _copy.deepcopy(dense_head.pre_fc_norm_embedding).to(dev)
        self.pre_fc_norm_hidden = _copy.deepcopy(dense_head.pre_fc_norm_hidden).to(dev)
        self.fc = nn.Linear(dense_head.fc.in_features, dense_head.fc.out_features, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(dense_head.fc.weight.data.to(dev))
        self.fc = self.fc.to(dev)
        hybrid_cfg = self.config.to_hybrid_config()
        self.layers = nn.ModuleList([
            DistributedHybridBlock(ctx, block, config=hybrid_cfg)
            for block in dense_head.layers])
        self.norm = _copy.deepcopy(dense_head.norm).to(dev)
        object.__setattr__(self, "_embedding", embedding)
        object.__setattr__(self, "_lm_head", lm_head)

    @classmethod
    def from_dense(cls, ctx: Any, dense_head: MTPHead, **kwargs: Any) -> "TPMTPDraftModule":
        return cls(ctx, dense_head, **kwargs)

    @classmethod
    def empty(cls, ctx: Any, config: Any = None, *, fp8: bool = False) -> "TPMTPDraftModule":
        """Shard-shaped empty TP draft module on meta (loader fills it)."""
        from .hybrid import QwenRMSNorm
        from .tp_distributed import DistributedHybridBlock
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.ctx = ctx
        obj.wiring = resolve_wiring(None)
        obj.config = MTPConfig.from_config(config) if config is not None else MTPConfig()
        cfg = obj.config
        meta = torch.device("meta")
        obj.pre_fc_norm_embedding = QwenRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(meta)
        obj.pre_fc_norm_hidden = QwenRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(meta)
        obj.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        obj.fc.weight = nn.Parameter(torch.empty(
            (cfg.hidden_size, 2 * cfg.hidden_size), device=meta))
        hybrid_cfg = cfg.to_hybrid_config()
        obj.layers = nn.ModuleList([
            DistributedHybridBlock.empty(ctx, hybrid_cfg, j, fp8=fp8)
            for j in range(cfg.mtp_num_hidden_layers)])
        obj.norm = QwenRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(meta)
        object.__setattr__(obj, "_embedding", None)
        object.__setattr__(obj, "_lm_head", None)
        return obj

    def set_ctx(self, ctx: Any) -> "TPMTPDraftModule":
        self.ctx = ctx
        for module in self.modules():
            if "ctx" in module.__dict__:
                module.ctx = ctx
        return self

    def initial_state(self) -> list[Any]:
        """Fresh per-rank draft state (empty local KV per MTP layer)."""
        return [None] * len(self.layers)

    @staticmethod
    def snapshot_state(state: Any) -> Any:
        return clone_mtp_state(state)

    @staticmethod
    def restore_state(snapshot: Any) -> Any:
        return clone_mtp_state(snapshot)

    def draft_token(self, hidden_t: torch.Tensor, token_id_t1: Any,
                    state: Any = None, position: int = 0, *,
                    top_k: int | None = None,
                    embedding: Any = None, lm_head: Any = None):
        """TP draft step; same contract as :meth:`MTPDraftModule.draft_token`.

        ``hidden_t`` / token ids live on ``ctx.device`` (replicated
        inputs); the returned logits are the full replicated logits
        when ``lm_head`` is a replicated head (e.g. the target LM's
        ``lm_head``), so every rank returns the same draft id.
        """
        emb = embedding if embedding is not None else self._embedding
        head_fn = lm_head if lm_head is not None else self._lm_head
        if emb is None:
            raise ValueError("no embedding: pass embedding= or construct with one")
        if head_fn is None:
            raise ValueError("no lm_head: pass lm_head= or construct with one")
        if not isinstance(hidden_t, torch.Tensor) or hidden_t.dim() not in (1, 2):
            raise ValueError("hidden_t must be [H] or [B, H]")
        single = hidden_t.dim() == 1
        h2 = hidden_t.unsqueeze(0) if single else hidden_t
        ids = _as_token_ids(token_id_t1, int(h2.shape[0]), h2.device)
        e2 = emb(ids) if callable(emb) else emb
        if isinstance(e2, torch.Tensor) and e2.dim() == 2 and h2.dim() == 2:
            pass
        else:
            e2 = torch.as_tensor(e2)
        if self.wiring["norm_pairing"] == "named":
            h_n, e_n = self.pre_fc_norm_hidden(h2), self.pre_fc_norm_embedding(e2)
        else:
            h_n, e_n = self.pre_fc_norm_embedding(h2), self.pre_fc_norm_hidden(e2)
        branches = ([h_n, e_n] if self.wiring["fc_concat_order"] == "hidden_first"
                    else [e_n, h_n])
        x = self.fc(torch.cat(branches, dim=-1)).unsqueeze(1)  # [B, 1, H]
        layer_states = list(state) if state is not None else self.initial_state()
        if len(layer_states) != len(self.layers):
            raise ValueError(f"state must hold {len(self.layers)} per-layer entries")
        abs_pos = int(position) + _POSITION_OFFSET[self.wiring["position_rule"]]
        new_state: list[Any] = []
        for block, layer_state in zip(self.layers, layer_states):
            x, new_layer_state = block.step(x, layer_state=layer_state, start_pos=abs_pos)
            new_state.append(new_layer_state)
        draft_hidden = self.norm(x[:, 0, :]) if self.wiring["lm_head_input"] == "mtp_norm" else x[:, 0, :]
        logits = head_fn(draft_hidden)
        draft_id = torch.argmax(logits, dim=-1)
        second: Any = logits
        if top_k is not None:
            if int(top_k) <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            top = torch.topk(logits, int(top_k), dim=-1)
            second = (top.values, top.indices)
        if single:
            draft_id = draft_id[0]
            second = (second[0][0], second[1][0]) if isinstance(second, tuple) else second[0]
        return draft_id, second, new_state

    def initial_static_state(self, batch_size: int = 1,
                             max_seq_len: int = 128,
                             dtype: torch.dtype | None = None):
        """Preallocated static KV state for :meth:`draft_token_static`.

        A :class:`~inference_from_scratch.tp_distributed.TPStaticState`
        over this head's own hybrid config (all full-attention layers):
        KV slots are sequential (one per committed draft) and
        ``state.position`` counts committed drafts only.
        """
        from .tp_distributed import TPStaticState
        if dtype is None:
            dtype = self.fc.weight.dtype
        return TPStaticState(
            self.config.to_hybrid_config(), batch_size=int(batch_size),
            max_seq_len=int(max_seq_len), device=self.ctx.device,
            dtype=dtype, ctx=self.ctx)

    def draft_token_static(self, hidden_t: torch.Tensor, token_id_t1: Any,
                           state: Any, *, rope_position: Any = None,
                           embedding: Any = None, lm_head: Any = None,
                           out: torch.Tensor | None = None):
        """Static twin of :meth:`draft_token` over a TPStaticState.

        Same maths and wiring as :meth:`draft_token`, but the MTP
        block runs through ``step_static``: K/V land in place at
        ``state.position`` (the sequential committed-draft slot) and
        RoPE uses ``rope_position`` (0-dim tensor or int; default
        ``state.position``) -- the target's absolute hidden position,
        which advances by 2 per accepted cycle and so cannot double
        as the KV slot. ``state.position`` is NOT advanced here: the
        caller advances it only when the draft is committed (accepted),
        which is the static form of commit-on-accept. Returns
        ``(draft_token_id, logits)``; ``out`` optionally receives the
        logits by copy (the CUDA-graph form).
        """
        emb = embedding if embedding is not None else self._embedding
        head_fn = lm_head if lm_head is not None else self._lm_head
        if emb is None:
            raise ValueError("no embedding: pass embedding= or construct with one")
        if head_fn is None:
            raise ValueError("no lm_head: pass lm_head= or construct with one")
        if not isinstance(hidden_t, torch.Tensor) or hidden_t.dim() not in (1, 2):
            raise ValueError("hidden_t must be [H] or [B, H]")
        single = hidden_t.dim() == 1
        h2 = hidden_t.unsqueeze(0) if single else hidden_t
        ids = _as_token_ids(token_id_t1, int(h2.shape[0]), h2.device)
        e2 = emb(ids) if callable(emb) else emb
        e2 = torch.as_tensor(e2)
        if self.wiring["norm_pairing"] == "named":
            h_n, e_n = self.pre_fc_norm_hidden(h2), self.pre_fc_norm_embedding(e2)
        else:
            h_n, e_n = self.pre_fc_norm_embedding(h2), self.pre_fc_norm_hidden(e2)
        branches = ([h_n, e_n] if self.wiring["fc_concat_order"] == "hidden_first"
                    else [e_n, h_n])
        x = self.fc(torch.cat(branches, dim=-1)).unsqueeze(1)  # [B, 1, H]
        if rope_position is None:
            rope_t = None
        elif isinstance(rope_position, torch.Tensor):
            rope_t = rope_position
        else:
            rope_t = torch.tensor(int(rope_position), dtype=torch.long,
                                  device=h2.device)
        for block in self.layers:
            x = block.step_static(x, state, rope_position=rope_t)
        draft_hidden = self.norm(x[:, 0, :]) if self.wiring["lm_head_input"] == "mtp_norm" else x[:, 0, :]
        logits = head_fn(draft_hidden)
        if out is not None:
            out.copy_(logits)
            logits = out
        draft_id = torch.argmax(logits, dim=-1)
        if single:
            draft_id = draft_id[0]
            logits = logits[0]
        return draft_id, logits

    forward = draft_token
