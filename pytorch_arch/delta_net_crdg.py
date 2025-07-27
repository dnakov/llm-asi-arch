# -*- coding: utf-8 -*-
"""
DeltaNet – Convolutional-Residual Dropout Gating (CRDG)
======================================================
Identifier: delta_net_crdg

Motivation
----------
This evolution tackles the *conv–path starvation* and *over-reliance on
individual memory branches* problems identified in earlier experiments.
Two complementary mechanisms are introduced (enabled **by default**):

1. **Residual Convolutional Paths**
   A small learnable residual connection from the *short* and *long* FIR
   convolutional outputs is added **in parallel** to the softmax-gated
   fusion.  This guarantees a persistent gradient signal for the local
   convolutional memories, protecting them from being completely shut
   out during the early training phase when the gate is strongly biased
   towards the Value/Δ branches.  The residual scales are *per-path
   scalars* initialised to `0.1`, allowing the optimiser to freely
   increase or decrease their influence.

2. **Path Dropout (Stochastic Router)**
   During *training* a lightweight *token-wise, per-head* dropout is
   applied to the gate weights.  Each path is dropped with probability
   `p=0.1` **independently per token & head**; the remaining weights are
   re-normalised to sum to one.  This simple stochastic router forces
   all paths to be used throughout training, mitigating gate collapse
   without introducing any extra trainable parameters or inference-time
   overhead (disabled during `.eval()`).

Both additions preserve the original O(N) complexity, maintain strict
causality, and are fully batch-agnostic.  Interface, constructor
signature, and the public class name **DeltaNet** remain unchanged, so
checkpoints and higher-level code continue to work without modification.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ============================================================================
# Utility helpers
# ============================================================================

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU so the output is strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise final dimension to sum to one."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# ============================================================================
# Depth-wise causal FIR convolution (unchanged numerics)
# ============================================================================

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filters = nn.Parameter(torch.randn(num_heads, head_dim, kernel_size) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, _, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ============================================================================
# Core chunk-wise Δ-rule kernel (identical numerics)
# ============================================================================

@torch.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficient O(N) associative Δ-rule using fixed-size chunks."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise & scale ------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks --------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = q.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones_like(tri_mask), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# ============================================================================
# Main DeltaNet – Convolutional-Residual Dropout Gating
# ============================================================================

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore


class DeltaNet(nn.Module):
    """DeltaNet layer with residual convolutional paths & stochastic gate dropout."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        # ---- baseline args ------------------------------------------------
        mode: str = "crdg",
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
        # ---- FIR kernel sizes -------------------------------------------
        fir_kernel_short: int = 5,
        fir_kernel_long: int = 64,
        # ---- Gating network ---------------------------------------------
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),  # τ≈0.7 softplus-param.
        # ---- New CRDG parameters ----------------------------------------
        path_dropout: float = 0.1,
        residual_conv_init: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---- Basic bookkeeping -----------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.path_dropout = float(path_dropout)

        # ---- Derived dimensions ----------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Key/Value dims must divide num_heads"

        # ---- Linear projections ----------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- Short convolutions ----------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- FIR convolutions -----------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---- Content-aware gating MLP ----------------------------------
        # Stats per path: 4 metrics → 16 scalars total
        self._stat_dim = 16
        gate_in_dim = hidden_size + self._stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        with torch.no_grad():
            self.fusion_gate[-1].bias.copy_(torch.tensor(gate_bias_init))

        # Learnable temperature for gate logits -------------------------
        self.logit_temperature = nn.Parameter(torch.full((1,), gate_logit_init))

        # ---- Residual convolutional path scales -----------------------
        self.res_scale_short = nn.Parameter(torch.full((1,), residual_conv_init))
        self.res_scale_long = nn.Parameter(torch.full((1,), residual_conv_init))

        # ---- Output processing ----------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Helper – per-head statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)  # (B,L,H,4)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"

        B_orig, L_orig, _ = hidden_states.shape

        # --------------------------------------------------------------
        # Retrieve previous layer state (if any)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # Optional unpadding for variable sequences --------------------
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # --------------------------------------------------------------
        # Q/K/V projections + causal short conv
        # --------------------------------------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # Head reshape --------------------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # Activations ---------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # Beta scaling --------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global pathway ----------------------------------------
        delta_out_d, recurrent_state_new = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # Local FIR paths ---------------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # --------------------------------------------------------------
        # Content-aware gate logits
        # --------------------------------------------------------------
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = torch.cat([stats_short, stats_long, stats_delta, stats_value], dim=-1)  # (B,L,H,16)

        hs_expanded = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,D)
        gate_in = torch.cat([hs_expanded, stats_vec], dim=-1)  # (B,L,H,D+16)
        gate_logits = self.fusion_gate(rearrange(gate_in, "b l h d -> (b l h) d"))
        gate_logits = rearrange(gate_logits, "(b l h) p -> b l h p", b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads)

        # Temperature scaling -----------------------------------------
        temperature = F.softplus(self.logit_temperature) + 1e-4
        gate_logits = gate_logits / temperature
        fusion_weights = torch.softmax(gate_logits, dim=-1)  # (B,L,H,4)

        # --------------------------------------------------------------
        # Path Dropout (training only)
        # --------------------------------------------------------------
        if self.training and self.path_dropout > 0.0:
            drop_mask = torch.rand_like(fusion_weights).le(self.path_dropout).to(fusion_weights)  # 1 if drop
            keep_weights = fusion_weights.masked_fill(drop_mask.bool(), 0.0)
            # Renormalise—avoid division by zero by clamping the sum
            denom = keep_weights.sum(-1, keepdim=True).clamp(min=1e-6)
            fusion_weights = keep_weights / denom

        # --------------------------------------------------------------
        # Fuse paths + residual convolutional contribution
        # --------------------------------------------------------------
        fused = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        residual = self.res_scale_short * local_short + self.res_scale_long * local_long
        o = fused + residual

        # --------------------------------------------------------------
        # Cache update -------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state_new,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_orig,
            )

        # --------------------------------------------------------------
        # Output norm / projection
        # --------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # Re-pad if we unpadded earlier --------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B_orig, L_orig)

        return o, None, past_key_values
