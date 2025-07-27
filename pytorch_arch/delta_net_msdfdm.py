# -*- coding: utf-8 -*-
"""
DeltaNet – Multi-Scale Dual FIR + Delta Memory (MS-DFDM)
======================================================
This evolutionary DeltaNet variant adds **explicit multi-scale local memory**
paths to address the precision drop observed in earlier hybrids that relied on
just a *single* long-kernel FIR convolution.  Concretely we introduce:

1. **Two causal depth-wise FIR branches**
   • *Short-kernel* path (k≈7) captures very local lexical / syntactic cues.
   • *Long-kernel* path (k≈64 – identical to previous HMGM variant) captures
     mid-range patterns that benefit tasks like BoolQ and Lambada.

2. **Quad-path Adaptive Fusion**
   Outputs from *(short-FIR, long-FIR, delta-rule, direct-value)* paths are
   fused using a *per-token, per-head* softmax gate produced by a lightweight
   two-layer MLP.  The gate biases are initialised such that **direct value
   path dominates at the start of training**, preventing early over-smoothing
   – a weakness identified in HMGM experiments.

3. All other mechanics (chunk-wise delta recurrence, short convolutions in the
   projection stack, optional gated RMSNorm) are retained from the strongest
   prior variant to preserve its proven benefits.

The implementation respects every technical constraint:
• O(N) runtime & memory  –   all additional ops are depth-wise 1-D convs.
• Strict causality        –   FIR branches are left-padded, delta kernel is
                               unchanged.
• Batch / sequence agnostic – dynamic shapes via einops.rearrange.
• Public interface & signatures unchanged.

"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

__all__ = ["DeltaNet"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (ELU+1) used by several DeltaNet variants."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so that values along the last dim sum to one."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# ---------------------------------------------------------------------------
# Core chunk-wise delta rule (identical to HMGM baseline)
# ---------------------------------------------------------------------------
@torch.compile
def _delta_rule_chunkwise(
    q: torch.Tensor,  # [B H L D_k]
    k: torch.Tensor,  # [B H L D_k]
    v: torch.Tensor,  # [B H L D_v]
    beta: torch.Tensor,  # [B H L]
    *,
    chunk_size: int = 32,
):
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_spec = (0, 0, 0, pad_len)
        q = F.pad(q, pad_spec)
        k = F.pad(k, pad_spec)
        v = F.pad(v, pad_spec)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape to chunk view
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )

    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, device=q.device)
    attn_inv = attn_inv.to(torch.bfloat16)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)

    strict_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S.detach()

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (per-head, per-channel)
# ---------------------------------------------------------------------------
class _DepthwiseFIR1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # (H, D, K)
        self.filters = nn.Parameter(
            torch.randn(num_heads, head_dim, kernel_size) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, H, D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")  # groups = h*d
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal padding
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        y = rearrange(y, "b (h d) l -> b l h d", h=h)
        return y

# ---------------------------------------------------------------------------
# Type hints for cache (only used for static check / doc)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack  # pragma: no cover
    from fla.models.utils import Cache

# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet with explicit *multi-scale* FIR paths and adaptive quad-fusion."""

    def __init__(
        self,
        *,
        mode: str = "ms-dfdm",
        d_model: int | None = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int | None = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # --- new hyper-params --- #
        fir_short_kernel: int = 7,
        fir_long_kernel: int = 64,
        fusion_hidden_mult: int = 2,
        gate_bias_init: float = 2.0,  # favour direct value at init
        **kwargs: "Unpack[Dict]",
    ) -> None:
        super().__init__()

        # ---------------- Parameter bookkeeping ----------------
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert qk_activation in ["silu", "relu", "elu", "identity"]
        assert qk_norm in ["l2", "sum"]

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        # ---------------- Derived dims -------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        # ---------------- Projections --------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- Short conv enhancements --------------
        if use_short_conv:
            activation_name = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=activation_name)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=activation_name)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for this DeltaNet variant.")

        # ---------------- Multi-scale FIR paths ----------------
        self.fir_short = _DepthwiseFIR1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = _DepthwiseFIR1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ---------------- Fusion gate MLP ----------------------
        # 4 streams: short-fir, long-fir, delta, direct-v
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 4, bias=True),
        )
        # Bias initialisation: favour direct value path initially
        with torch.no_grad():
            self.fusion_gate[-1].bias.view(num_heads, 4)[:] = 0.0
            self.fusion_gate[-1].bias.view(num_heads, 4)[:, 3] = gate_bias_init  # direct value

        # ---------------- Output norm / projection ------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        # ----------- attention mask sanity -------------------
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be [batch, seq_len] padding mask"

        batch_size, seq_len, _ = hidden_states.shape

        # ----------- retrieve cached state -------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Unpad variable-length batch for efficiency ----------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # -------------- Q K V projections + short conv -------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state", None) is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # -------------- split heads --------------------------
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # -------------- activations & norms ------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            # identity -> no-op
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # -------------- beta computation ---------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones((*hidden_states.shape[:2], self.num_heads), device=hidden_states.device, dtype=q.dtype)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------------- delta path (global) ------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")

        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # -------------- FIR local paths ----------------------
        v_direct = v  # identity path
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # -------------- Fusion gating ------------------------
        fusion_logits = self.fusion_gate(hidden_states)  # [B, L, H*4]
        fusion_logits = rearrange(fusion_logits, "b l (h c) -> b l h c", h=self.num_heads, c=4)
        fusion_weights = torch.softmax(fusion_logits, dim=-1)  # convex combination

        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # -------------- cache update -------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # -------------- output norm/proj ---------------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # re-pad if unpadded earlier
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
