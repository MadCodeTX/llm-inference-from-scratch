"""Ch 06 -- MLP: SwiGLU (Llama/Qwen) and GELU MLP (GPT-2).

Why this exists in inference
------------------------------
The MLP is where most parameters (and most prefill FLOPs) live: each
token independently goes ``d_model -> d_ff -> d_model``. "Independently"
is the point -- unlike attention there is no cross-token maths here, so
the MLP is identical in prefill and decode apart from how many rows
(``B * T``) are pushed through it.

How production differs
-----------------------
Fused kernels merge gate+up (and the SiLU-multiply epilogue) into one
pass; the three separate Linears below are the dissected version. No
dropout and no biases (Llama/Qwen style) in either variant: inference
is deterministic, and bias terms are a negligible win at this scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU gated MLP: ``down( silu(gate(x)) * up(x) )``.

    Three bias-free Linears::

        gate_proj: [d_model -> d_ff]
        up_proj:   [d_model -> d_ff]
        down_proj: [d_ff    -> d_model]

    The gate branch passes through SiLU (a.k.a. Swish:
    ``silu(z) = z * sigmoid(z)``) and then *multiplies* the up branch
    elementwise. That data-dependent product is the "gate": the up
    branch proposes features, the gate branch decides per-feature how
    much of each survives. It consistently beats a plain GELU MLP at
    equal parameter count (Shazeer 2020), which is why Llama / Qwen
    use it -- at the cost of a third matrix, so ``d_ff`` is usually
    ~2/3 of what a two-matrix MLP would use for the same budget.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # x [B, T, d_model] -> gate [B, T, d_ff]
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        # x [B, T, d_model] -> up [B, T, d_ff]
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # hidden [B, T, d_ff] -> y [B, T, d_model]
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        gate = self.gate_proj(x)  # [B, T, d_ff]
        up = self.up_proj(x)  # [B, T, d_ff]
        # silu(gate) [B, T, d_ff] * up [B, T, d_ff] -> hidden [B, T, d_ff]
        hidden = F.silu(gate) * up  # [B, T, d_ff]
        # [B, T, d_ff] -> [B, T, d_model]
        return self.down_proj(hidden)  # [B, T, d_model]


class GeluMLP(nn.Module):
    """Classic two-layer GELU MLP (GPT-2 style): ``down( gelu(up(x)) )``.

    Two bias-free Linears::

        up_proj:   [d_model -> d_ff]
        down_proj: [d_ff    -> d_model]

    Kept alongside :class:`SwiGLU` so ``ModelConfig.mlp_type`` can pick
    either and the tests can contrast them: one non-linearity applied
    to a single projection, no gating product. ``F.gelu`` defaults to
    the exact erf form (not the tanh approximation).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # x [B, T, d_model] -> hidden [B, T, d_ff]
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # hidden [B, T, d_ff] -> y [B, T, d_model]
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        hidden = self.up_proj(x)  # [B, T, d_ff]
        hidden = F.gelu(hidden)  # [B, T, d_ff] (exact erf GELU)
        # [B, T, d_ff] -> [B, T, d_model]
        return self.down_proj(hidden)  # [B, T, d_model]
