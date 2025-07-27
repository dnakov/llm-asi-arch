# -*- coding: utf-8 -*-
"""
DeltaNet – Residual-Guaranteed Gated Fusion (RGGF)
=================================================
This innovation combines key breakthroughs from the state-of-the-art experimental portfolio:

- A *fixed, non-learnable residual (min-leak) connection* is injected on the local (short-FIR) path, guaranteeing signal and gradient flow for local feature extraction, in line with the best results from CAGF-RC/BCMF research. This ensures robust local detail even when the gate collapses elsewhere.
- The gating MLP retains rich per-branch statistics and per-head structure, ensuring dynamic specialization for global/contextual integration, with an expressive two-layer MLP.
- We replace the learnable temperature with per-head, per-path learnable temperatures, providing maximum flexibility for blended/hard selection. This allows per-task adaptation (as in HTNG).
- All computations remain strictly causal, sub-quadratic (O(N)), chunked, and batch-size agnostic (einops).
- The minimal, fixed residual fraction (default 5%) is injected post-gating, tuning local/global performance trade-off. All interface and signature constraints are preserved.

This design is directly motivated by empirical proof that *hard* guarantees—not only learnable or scheduled—are required for robust local information retention and optimal cognitive reasoning.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, TYPE_CHECKING, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)
def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity init for local preservation)
# -----------------------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left-padding and identity init."""
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        filt = torch.zeros(num_heads, head_dim, kernel_size)
        with torch.no_grad():
            filt[..., -1] = 1.0
        self.filters = nn.Parameter(filt)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule kernel (identical, @torch.compile)
# -----------------------------------------------------------------------------
@torch.compile
def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )
    mask_tri = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0
    )
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)
    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1
    )
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# DeltaNet – Residual-Guaranteed Gated Fusion
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
    from fla.models.utils import Cache

class DeltaNet(nn.Module):
    """
    DeltaNet layer with per-head per-path learnable temperature, per-head statistics,
    content-aware gating MLP, and a *fixed (non-learnable) minimal-leak residual* on local FIR path.
    """
    def __init__(
        self,
        mode: str = "rggf",
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
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        min_local_leak: float = 0.05,  # e.g., 5% fixed minimal residual (not learnable)
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert (
            self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        ), "Key/Value dims must divide num_heads"

        # -- Projections --
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.use_beta = use_beta
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # -- Short conv enhancement (mandatory for stability) --
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")
        
        # -- Multi-scale local FIR --
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_short
        )
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_long
        )

        # -- Gating network --
        # 4 statistics per branch × 4 branches (short, long, delta, value)
        self.stat_dim = 16
        gate_in_dim = hidden_size + self.stat_dim
        gate_hidden_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 4, bias=True),
        )
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias.zero_()
            # Encourage direct value moderately, delta weakly
            self.fusion_gate_mlp[-1].bias[2] = 0.7
            self.fusion_gate_mlp[-1].bias[3] = 2.5

        # -- Per-head/per-path learnable temperatures --
        self.logit_temperature = nn.Parameter(torch.zeros(num_heads, 4))
        # -- Minimal-leak residual, non-learnable (default 5%) --
        self.min_local_leak = float(min_local_leak)

        # -- Output norm/proj --
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

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
        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if (
            past_key_values is not None
            and self.layer_idx is not None
            and len(past_key_values) > self.layer_idx
        ):
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).unsqueeze(0)
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state", None) is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = torch.cat([
            stats_short, stats_long, stats_delta, stats_value
        ], dim=-1)  # (B, L, H, 16)
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)  # (B,L,H,C+16)
        gate_in_flat = rearrange(gate_in, "b l h d -> (b l h) d")
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)
        # Per-head/path temperature scaling (learned)
        temp = torch.exp(self.logit_temperature).clamp(0.05, 10.0)
        temp = rearrange(temp, "h p -> 1 1 h p")
        fusion_logits = rearrange(
            gate_logits_flat,
            "(b l h) c -> b l h c",
            b=gate_in.shape[0],
            l=gate_in.shape[1],
            h=self.num_heads,
        ) / temp
        fusion_soft = torch.softmax(fusion_logits, dim=-1)
        # Add fixed 5% min-leak as (1-alpha) residual on local_short
        alpha = 1.0 - self.min_local_leak
        # Main fusion (excluding guaranteed min-leak local)
        o_main = (
            fusion_soft[..., 0:1] * local_short
            + fusion_soft[..., 1:2] * local_long
            + fusion_soft[..., 2:3] * delta_out
            + fusion_soft[..., 3:4] * v_direct
        )
        o = alpha * o_main + self.min_local_leak * local_short
        # -- Cache update --
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)
        return o, None, past_key_values
