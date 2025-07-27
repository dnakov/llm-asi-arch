# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Multi-Scale Fusion with Dynamic Per-Path Gating and Entropy-Regularized Routing (DeltaNet-AMF)
===============================================================================================================
Innovation highlights:
  1. **Adaptive Multi-Scale Local Memory**: FIR block now offers deeper multi-scale diversity
     with learnable kernel set (1, 3, 7, 15, 31): includes true identity (k=1) for ultra-local cues.
     Kernels are identity- and noise-initialized for gradient flow and branch uniqueness.

  2. **Dynamic Per-Path Gating**: The fusion gate is upgraded to accept both input token embedding
     and compressed branch statistics (L2-norm/mean of each path), producing path logits per token, per head.
     A learnable per-head temperature regulates softmax sharpness.

  3. **Entropy Regularization**: Gate entropy is computed in forward; if the module is in training mode,
     -λ·entropy penalty is returned with the output, encouraging mixture diversity and preventing collapse.
     λ=0.03 by default (ablation-based default).

  4. **Adaptive Path Floor**: Rather than a static ε floor, the minimum path allocation is annealed as a learnable parameter per path: enables model to safely allocate required capacity to critical branches while not limiting global context at depth.

  5. **Fully Batch-agnostic / Chunked**: All operations use einops for reshaping and chunked implementations for memory efficiency and O(N) time.

  6. **Robust Causal Information Flow**: Causal masking, O(N) complexity and strict interface compatibility preserved.

Implements deep research insights:
  - Multi-path + adaptive routing per Hyena/GLA/TransNormer advances
  - Annealed path floors (dynamic, learnable) to resolve local/global capacity trade-off
  - Entropy regularization for robust mixture (from MoE, SSM, Gated Attention, etc.)
  - Path statistics facilitate adaptive, information-rich routing without excess MLP overhead
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ========================================================================
# Utility functions (no @torch.compile for helpers)
# ========================================================================
def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)

def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)

# ========================================================================
# Chunk-wise O(N) delta kernel (unchanged from baseline, batch-size-agnostic)
# ========================================================================
@torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# ========================================================================
# Adaptive Multi-Scale Depthwise FIR block (includes k=1 for identity)
# ========================================================================
class DepthwiseAdaptiveMultiScaleFIR(nn.Module):
    """Parallel depth-wise causal convolutions (kernels 1,3,7,15,31). Identity+noise init."""
    def __init__(self, num_heads: int, head_dim: int, kernel_sizes: Tuple[int, ...] = (1,3,7,15,31)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_channels = num_heads * head_dim

        self.filters: nn.ParameterList = nn.ParameterList()
        for k in kernel_sizes:
            filt = nn.Parameter(torch.zeros(self.total_channels, 1, k))
            # Identity init: last position is 1 if k>1, else all-ones (for k=1)
            with torch.no_grad():
                if k == 1:
                    filt[:, 0, 0] = 1.0
                else:
                    filt[:, 0, -1] = 1.0
                filt.add_(0.02 * torch.randn_like(filt))
            self.filters.append(filt)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # x: [B,L,H,D]
        b, L, h, d = x.shape
        x_ch = rearrange(x, "b l h d -> b (h d) l")
        outs: List[torch.Tensor] = []
        for filt, k in zip(self.filters, self.kernel_sizes):
            x_pad = F.pad(x_ch, (k-1, 0))
            y = F.conv1d(x_pad, weight=filt, groups=self.total_channels)
            y = rearrange(y, "b (h d) l -> b l h d", h=h)
            outs.append(y)
        return outs

# ========================================================================
# Main DeltaNet-AMF block (Adaptive Multi-Scale Fusion with Per-Path Routing & Entropy Reg)
# ========================================================================
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

class DeltaNet(nn.Module):
    """DeltaNet-AMF: Adaptive multi-scale routing, per-path annealing, entropy reg."""
    def __init__(
        self,
        *,
        mode: str = "amf_routing",
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
        ms_kernel_sizes: Tuple[int,...] = (1,3,7,15,31),
        fusion_hidden_mult: int = 2,
        routing_entropy_weight: float = 0.03,
        min_floor_init: float = 0.03,
        **kwargs: "Unpack[Dict]",
    ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
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
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.ms_kernel_sizes = ms_kernel_sizes
        self.routing_entropy_weight = routing_entropy_weight

        # Core dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Short convolutional (mandatory)
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet-AMF.")

        # --- Adaptive Multi-Scale FIR block (with k=1) ---
        self.local_fir = DepthwiseAdaptiveMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)
        self.num_streams = self.num_scales + 2  # (all FIRs, delta, value)

        # --- Dynamic gating: fuse token, path stats; learnable temperature, dynamic/annealed floor ---
        compressed_stat_dim = self.num_streams * self.num_heads
        mlp_in_dim = hidden_size + compressed_stat_dim
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_size * fusion_hidden_mult),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * self.num_streams)
        )
        # Per-head temperature parameter
        self.gate_log_temp = nn.Parameter(torch.zeros(self.num_heads) + math.log(1.0))
        # Per-path, per-head minimum allocation floor (learnable, clamped)
        self.min_floor = nn.Parameter(torch.full((self.num_heads, self.num_streams), min_floor_init))

        # Early bias: identity/value gets slight advantage
        with torch.no_grad():
            bias = self.fusion_gate_mlp[-1].bias
            bias.zero_()
            bias.view(self.num_heads, self.num_streams)[:, -1] += 0.15  # value path
            bias.view(self.num_heads, self.num_streams)[:, -2] += 0.05  # delta path

        # Output norm/projection
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ----------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: "Unpack[Dict]",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:
        # (1) Optional unpadding
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # (2) Projections + Short conv
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv:
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # (3) Head split & activation
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # (4) Beta for delta path
        beta = self.b_proj(hidden_states).sigmoid() if self.use_beta else torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # (5) Delta-rule O(N) global memory
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # (6) Multi-scale FIR local paths (identity+local/mid/long)
        conv_branches = self.local_fir(v)  # list length = num_scales
        # All streams: FIR branches, delta, direct-value
        streams: List[torch.Tensor] = conv_branches + [delta_out, v]
        # Stack for routing, [B,L,H,num_streams,D]
        streams_stack = torch.stack(streams, dim=-2)

        # (7) Branch statistics for dynamic routing
        # [L2-norm per token, head, branch]
        stats = [s.norm(dim=-1) for s in streams]  # list of [B,L,H]
        stats_tensor = torch.stack(stats, dim=-1)  # [B,L,H,S]
        # Flatten stats per sample as [B,L,H*S] then concat per heads
        stat_feat = rearrange(stats_tensor, "b l h s -> b l (h s)")
        fusion_in = torch.cat([hidden_states, stat_feat], dim=-1)  # [B,L, hidden + H*S]
        fusion_logits = self.fusion_gate_mlp(fusion_in)  # [B,L,H*S]
        fusion_logits = rearrange(fusion_logits, "b l (h s) -> b l h s", h=self.num_heads, s=self.num_streams)
        temp = torch.exp(self.gate_log_temp).clamp(min=0.1, max=8.0).view(1,1,-1,1)  # [1,1,H,1]
        fusion_logits = fusion_logits / temp

        # Adaptive/learnable min-floor per head/branch: sigmoid [0,1], scaled to [0,0.2]
        floor = torch.sigmoid(self.min_floor).clamp(0.0, 1.0) * 0.2
        floor = floor.view(1,1,self.num_heads,self.num_streams)  # broadcast

        raw_weights = torch.softmax(fusion_logits, dim=-1)
        weights = raw_weights * (1 - floor.sum(-1, keepdim=True)) + floor
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Entropy penalty for auxiliary gate reg
        entropy = -(weights * (weights+1e-8).log()).sum(-1).mean()

        # (8) Route & fuse
        o = (streams_stack * weights.unsqueeze(-1)).sum(dim=-2)  # [B,L,H,D]

        # (9) Cache update (if requested)
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # (10) Output norm/projection
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)
        # Return entropy regularizer in training mode (for loss addend)
        if self.training:
            return o, -self.routing_entropy_weight * entropy, past_key_values
        return o, None, past_key_values
