# -*- coding: utf-8 -*-
"""
DeltaNet – Path-Aware Head-Gated Fusion (delta_net_pathgated)
============================================================
Identifier: **delta_net_pathgated**

This variant builds upon the previously successful *Head-Gated* design but
solves its key short-comings – indiscriminate suppression / amplification of
heads due to a *blind* gate – by conditioning the head-gate **on the fused
multi-path statistics**.

Key Innovations
---------------
1. Path-Aware Head Gate (PA-HG)
   •  The per-head, per-token output gate is now computed from a feature vector
      containing **(a) the pre-layer hidden state**, **(b) statistics of the
      fused head output** (mean, variance, abs-mean, L2), **(c) the softmax
      fusion weights of the four paths**.  This lets the gate learn when a
      head is mainly global/Δ-rule vs. local/FIR and act accordingly.
   •  Implementation: a lightweight MLP shared across heads maps the
      feature vector `(D + 8)` → `1`, followed by a *scaled* sigmoid producing
      gates in the range **(0 , 4)**.  The bias is set such that the initial
      gate value is **1.0**, preserving the identity function at init.

2. Wider Dynamic Range
   •  The gate can now *amplify* up to ×4 or dampen to almost zero, giving the
      model freedom to boost critical heads for global reasoning (ARC,
      HellaSwag) while still attenuating noisy heads for ultra-local tasks.

All other components – probability-floor fusion, dynamic residual conv,
chunk-wise Δ-rule, causal FIR convolutions, cache logic – remain unchanged.
The architecture keeps **O(N)** complexity, strict causality and universal
batch-shape robustness.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU – strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalisation along the last dimension."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identical maths to previous variants)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left-padding (O(N))."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            # Identity-like kernel: impulse at t = 0 (right-most index after padding)
            weight[..., -1] = 1.0
            weight.add_(0.02 * torch.randn_like(weight))
        self.filters = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, H, D)
        b, l, h, d = x.shape
        w = rearrange(self.filters, "h d k -> (h d) 1 k")  # (H*D,1,K)
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged)
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: torch.Tensor,  # (B H L D_k)
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,  # (B H L)
    *,
    chunk_size: int = 32,
):
    """Efficient associative Δ-rule with strict causality and O(N) complexity."""

    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into (chunks, chunk_size)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones_like(tri), 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Typing helper for cache
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

# -----------------------------------------------------------------------------
# Main DeltaNet implementation (Path-Aware Head-Gated)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – name must remain DeltaNet
    """DeltaNet layer with probability-floor fusion **and** path-aware head gating."""

    # pylint: disable=too-many-instance-attributes, too-many-branches, too-many-statements
    def __init__(
        self,
        *,
        mode: str = "pathgated",
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
        # FIR kernel sizes
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # Fusion gate params
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),
        prob_floor: float = 0.02,
        # Residual conv path
        conv_residual_init: float = -1.0,  # logit space (≈0.27 after sigmoid)
        # Path-aware head gate params
        out_gate_hidden_mult: float = 0.5,  # hidden dim multiplier relative to hidden_size
        out_gate_init_bias: float = -1.0986122886681098,  # logit(0.25) so gate ~1.0 (4*σ)
        **kwargs,
    ) -> None:
        super().__init__()

        # -------------------- bookkeeping --------------------------
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.prob_floor = float(prob_floor)

        # -------------------- dimensions ---------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # -------------------- projections --------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # -------------------- short convs --------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # -------------------- multi-scale FIR convs ----------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)

        # -------------------- fusion softmax gate ------------------
        self.stat_dim = 16  # (4 stats × 4 branches)
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias[:] = torch.tensor(gate_bias_init)

        self.logit_temperature = nn.Parameter(torch.full((1,), gate_logit_init))

        # -------------------- residual conv scaling ---------------
        self.conv_residual_logit = nn.Parameter(torch.full((num_heads,), conv_residual_init))
        self.res_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        with torch.no_grad():
            self.res_gate_proj.bias.fill_(conv_residual_init)

        # -------------------- path-aware head gate -----------------
        out_gate_in_dim = hidden_size + 8  # hidden + fused stats (4) + fusion weights (4)
        out_gate_hidden = int(hidden_size * out_gate_hidden_mult)
        self.out_gate_mlp = nn.Sequential(
            nn.Linear(out_gate_in_dim, out_gate_hidden, bias=True),
            nn.GELU(),
            nn.Linear(out_gate_hidden, 1, bias=True),
        )
        with torch.no_grad():
            self.out_gate_mlp[-1].bias.fill_(out_gate_init_bias)

        # -------------------- output norm / proj -------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Statistic helper
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D) → (B,L,H,4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compatibility
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, L_full, _ = hidden_states.shape

        # -------- optional unpadding for packed sequences ----------
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_full:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # -------- retrieve cached conv states ----------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # -------- projections + short conv -------------------------
        q_in, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_in, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_in, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # -------- head reshape ------------------------------------
        q = rearrange(q_in, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_in, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_in, "b l (h d) -> b l h d", d=self.head_v_dim)

        # -------- activations / norms on Q,K -----------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # -------- β projection for Δ-rule --------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------- Δ-rule pathway ----------------------------------
        delta_out_t, recurrent_state = _delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_t, "b h l d -> b l h d")

        # -------- local FIR paths ---------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # -------- fusion softmax ----------------------------------
        stats_vec = torch.cat([
            self._per_head_stats(local_short),
            self._per_head_stats(local_long),
            self._per_head_stats(delta_out),
            self._per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H,16)
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,D)
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)
        gate_in_flat = rearrange(gate_in, "b l h d -> (b l h) d")
        fusion_logits_flat = self.fusion_gate_mlp(gate_in_flat)
        # temperature scaling
        temperature = F.softplus(self.logit_temperature) + 1e-4
        fusion_logits_flat = fusion_logits_flat / temperature
        fusion_logits = rearrange(
            fusion_logits_flat,
            "(b l h) c -> b l h c",
            b=gate_in.shape[0],
            l=gate_in.shape[1],
            h=self.num_heads,
        )
        fusion_weights = torch.softmax(fusion_logits, dim=-1)
        if self.prob_floor > 0.0:
            floor_vec = torch.tensor([self.prob_floor, self.prob_floor, 0.0, 0.0], dtype=fusion_weights.dtype, device=fusion_weights.device)
            fusion_weights = torch.clamp(fusion_weights, min=floor_vec)
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)

        # -------- weighted fusion ---------------------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # -------- residual conv injection -------------------------
        res_gate_dyn = torch.sigmoid(self.res_gate_proj(hidden_states))  # (B,L,H)
        static_gamma = torch.sigmoid(self.conv_residual_logit).view(1, 1, self.num_heads, 1)
        o = o + (static_gamma * res_gate_dyn.unsqueeze(-1)) * local_short

        # ------------------------------------------------------------------
        # NEW: Path-Aware Head Gate ----------------------------------------
        # Features: hidden state, fused output stats (4), fusion weights (4)
        fused_stats = self._per_head_stats(o)  # (B,L,H,4)
        gate_feat = torch.cat([hs_exp, fused_stats, fusion_weights], dim=-1)  # (B,L,H,D+8)
        gate_feat_flat = rearrange(gate_feat, "b l h d -> (b l h) d")
        head_gate_logits = self.out_gate_mlp(gate_feat_flat)  # (B*L*H,1)
        head_gate = 4.0 * torch.sigmoid(head_gate_logits)  # (0,4) range
        head_gate = rearrange(head_gate, "(b l h) 1 -> b l h", b=gate_feat.shape[0], l=gate_feat.shape[1], h=self.num_heads)
        o = o * head_gate.unsqueeze(-1)

        # -------- cache update -----------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_full,
            )

        # -------- output norm / projection -----------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # -------- re-pad if we unpadded ---------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B_orig, L_full)

        return o, None, past_key_values
