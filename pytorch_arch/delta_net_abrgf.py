# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Bias & Residual Gated Fusion (ABRGF)
=======================================================
This evolution synthesises the strongest elements of earlier DeltaNet
variants while fixing their respective weaknesses:

1.  **Dirac-initialised multi-scale FIR memory**
    • Identity-preserving initialisation of depth-wise FIR kernels avoids early
      signal degradation and accelerates optimisation.
2.  **Learnable path-specific bias (per-head)**
    • Replaces fixed logits bias with a trainable parameter tensor allowing the
      model to *adaptively* balance global vs. local pathways over training.
3.  **Residual convolutional bypass**
    • Lightweight, learnable residual scalars (one per FIR path) guarantee that
      local-detail signals always propagate, preventing gradient starvation
      even when the gate down-weights conv branches.
4.  **Path-dropout regularisation**
    • A small dropout on fusion logits (token, head, path level) encourages
      exploration and mitigates premature path collapse.

All changes preserve: O(N) complexity, strict causality, batch-agnostic
operation, original API signatures, and @torch.compile acceleration of the
core Δ-rule kernel.
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

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU ensuring strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise last dim to sum-to-one (avoids divide-by-zero)."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac-initialised)
# -----------------------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left-padding.

    Kernels are initialised as *Dirac* (identity): filter[..., -1] = 1.
    Optionally small Gaussian noise (std=0.02) encourages early exploration.
    """

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 2e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # Dirac (identity for causal conv)
            if noise_std > 0:
                weight.add_(torch.randn_like(weight) * noise_std)
        self.filters = nn.Parameter(weight)  # (H, D, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")  # (H*D,1,K)
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        y = rearrange(y, "b (h d) l -> b l h d", h=h)
        return y

# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule kernel (unchanged    proven baseline)
# -----------------------------------------------------------------------------
@torch.compile  # keeps linear complexity
def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
):
    """Efficient chunk-wise associative Δ-rule with O(N) cost."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape: (B H N C D)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    # In-chunk inverse (I − tril(K β Kᵀ))⁻¹
    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Main DeltaNet implementation (ABRGF)
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

class DeltaNet(nn.Module):
    """DeltaNet layer with Adaptive Bias & Residual Gated Fusion (ABRGF)."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        mode: str = "abrgf",
        d_model: int | None = None,
        hidden_size: int = 1024,
        *,
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
        # ---- FIR kernel sizes ----
        fir_kernel_size_short: int = 3,
        fir_kernel_size_long: int = 63,
        # ---- gating & regularisation ----
        fusion_hidden_mult: int = 2,
        fusion_logit_dropout: float = 0.05,
        # learnable bias init (short, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 0.5, 1.5),
        # residual scalar initial value for conv paths (short, long)
        residual_init: Tuple[float, float] = (0.05, 0.05),
        **kwargs,
    ) -> None:
        super().__init__()

        # ---- bookkeeping ----
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
        assert qk_activation in ("silu", "relu", "elu", "identity")
        assert qk_norm in ("l2", "sum")
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---- dimensions ----
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Key/Value dims must divide num_heads"

        # ---- linear projections ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # ---- beta projection for Δ-rule ----
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- mandatory short convs ----
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- multi-scale FIR convs ----
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)

        # ---- learnable residual scalars (broadcast over heads) ----
        self.residual_short = nn.Parameter(torch.full((1, 1, 1, 1), residual_init[0]))
        self.residual_long = nn.Parameter(torch.full((1, 1, 1, 1), residual_init[1]))

        # ---- content-aware gating ----
        # stats per branch (mean, var, abs-mean, l2) => 4
        self.stat_dim = 4 * 3  # stats for short, long, delta (value branch stats omitted)
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=False),  # logits per path (shared across heads)
        )
        # per-head learnable bias added post-MLP
        bias_tensor = torch.tensor(gate_bias_init).repeat(num_heads, 1)  # (H,4)
        self.gate_bias = nn.Parameter(bias_tensor)  # (H,4)

        # temperature per head (start ~0.7 -> init param log(expm1(0.7)))
        self.logit_temperature = nn.Parameter(torch.full((num_heads, 1), math.log(math.expm1(0.7))))

        self.fusion_logit_dropout = fusion_logit_dropout

        # ---- output normalisation / projection ----
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # statistic helper
    # ------------------------------------------------------------------
    @staticmethod
    def _stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mean, var, abs_mean, l2

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compatibility
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"

        batch_size, seq_len, _ = hidden_states.shape

        # ---- retrieve cache ----
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # ---- optional unpadding ----
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---- Q/K/V projections + short conv ----
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---- head reshape ----
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ---- activation & norm on Q/K ----
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- beta for Δ-rule ----
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global pathway ----
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---- local FIR paths ----
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---- gather stats for gating (per-head) ----
        stats = []
        for branch in (local_short, local_long, delta_out):
            stats.extend(self._stats(branch))
        # stats list length = 4*3, each tensor (B,L,H,1)
        stats_vec = torch.cat(stats, dim=-1)  # (B,L,H,12)

        # broadcast hidden_states to heads & build gate input
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,C)
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)  # (B,L,H,C+stats)
        gate_in_flat = rearrange(gate_in, "b l h d -> (b l h) d")
        logits_flat = self.fusion_gate_mlp(gate_in_flat)  # (B*L*H,4)

        logits = rearrange(logits_flat, "(b l h) p -> b l h p", b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads, p=4)
        # add learnable per-head bias
        logits = logits + self.gate_bias.unsqueeze(0).unsqueeze(0)  # (B,L,H,4)

        # optional dropout on logits for regularisation
        if self.training and self.fusion_logit_dropout > 0.0:
            logits = F.dropout(logits, p=self.fusion_logit_dropout, inplace=False)

        # temperature scaling per head
        temp = F.softplus(self.logit_temperature) + 1e-4  # (H,1)
        logits = logits / temp.unsqueeze(0).unsqueeze(0)

        fusion_weights = torch.softmax(logits, dim=-1)

        # ---- weighted fusion + residual bypass ----
        o_gated = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        o = o_gated + self.residual_short * local_short + self.residual_long * local_long

        # ---- cache update ----
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ---- output normalisation / projection ----
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---- re-pad if we unpadded earlier ----
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
