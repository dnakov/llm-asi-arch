# -*- coding: utf-8 -*-
"""
DeltaNet – Headwise Mixed Gating with Additive Parallel Fusion and Adaptive Residual (DeltaNet-HMGAPF)
=======================================================================================
Identifier: *delta_net_hmgapf*

(See original header for full motivation and design notes.)
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# =============================================================================
# Utility helpers
# =============================================================================

def _elu_p1(x: torch.Tensor) -> torch.Tensor:  # Shifted ELU (>0)
    """Shifted ELU activation that is strictly positive (as used in S4)."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise *within the last dimension* so the elements sum to one."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# =============================================================================
# Depth-wise FIR convolution (local) – remains O(N)
# =============================================================================

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 0.02):
        super().__init__()
        self.kernel_size = kernel_size
        # Each (head, dim) pair gets its own 1-D kernel
        f = torch.zeros(num_heads, head_dim, kernel_size)
        with torch.no_grad():
            f[..., -1] = 1.0  # delta-initialisation
            f += noise_std * torch.randn_like(f)
        self.filters = nn.Parameter(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")       # groups = h*d
        x_f = rearrange(x, "b l h d -> b (h d) l")                   # (B, H*D, L)
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))                 # causal padding left side
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# =============================================================================
# Chunk-wise delta-rule path (causal, O(N))
# =============================================================================

@torch.compile()
def _delta_rule_chunkwise(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          beta: torch.Tensor,
                          chunk_size: int = 32):
    """Chunk-wise, strictly causal delta-rule kernel.

    Shapes
    -------
    q, k, v : (B, H, L, D)
    beta    : (B, H, L)
    """
    b, h, L, d_k = q.shape

    # ------------------------------------------------------------------
    # 1) Pad length so it is an exact multiple of the chunk size.
    # ------------------------------------------------------------------
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_seq = (0, 0, 0, pad_len)  # pad on sequence dimension (second from last)
        q, k, v = (F.pad(t, pad_seq) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # ------------------------------------------------------------------
    # 2) Pre-normalisation and weighting
    # ------------------------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # ------------------------------------------------------------------
    # 3) Chunk into (B, H, N_chunks, C, D)
    # ------------------------------------------------------------------
    reshape5 = lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size)
    q, k, v, k_beta = map(reshape5, (q, k, v, k_beta))

    # ------------------------------------------------------------------
    # 4) Pre-compute intra-chunk matrices (causal masked)
    # ------------------------------------------------------------------
    # mask for future positions inside a chunk (upper-tri incl. diag)
    mask_ut = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_ut, 0)

    # triangular recursion (block exclusive prefix sums)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    # u = A * V,  w = A * (K·β)
    u = attn @ v
    w = attn @ k_beta

    # ------------------------------------------------------------------
    # 5) Scan over chunks recurrently (causal)
    # ------------------------------------------------------------------
    S = k.new_zeros(b, h, d_k, v.shape[-1])  # carry state
    o = torch.zeros_like(v)                  # output placeholder
    excl = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)  # strictly future positions

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]                    # (B, H, C, D)
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(excl, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S                    # (B, H, C, Dv)
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    # ------------------------------------------------------------------
    # 6) Reshape back and remove padding if any
    # ------------------------------------------------------------------
    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache

# =============================================================================
# Main module – DeltaNet-HMGAPF
# =============================================================================

class DeltaNet(nn.Module):
    """DeltaNet-HMGAPF: additive parallel fusion with adaptive residuals."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self,
                 mode: str = "hmgapf",
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
                 fusion_hidden_mult: float = 0.75,
                 prob_floor: float = 0.01,
                 res_dyn_bias: float = 0.5,
                 res_static_init: float = 0.5,
                 **kwargs):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model  # keep backward compat naming

        # Save config
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
        self.prob_floor = float(prob_floor)
        self.fusion_hidden_mult = fusion_hidden_mult

        # ------------------------------------------------------------------
        # Dimensions
        # ------------------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Head dims must divide evenly."

        # ------------------------------------------------------------------
        # Projections
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------------------------------------------------------------
        # Optional shallow convolutional mixing (causal)
        # ------------------------------------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            # Dummy identities for functional uniformity (always returns tuple like (x, None))
            self.q_conv1d = lambda x, **_: (x, None)
            self.k_conv1d = lambda x, **_: (x, None)
            self.v_conv1d = lambda x, **_: (x, None)

        # ------------------------------------------------------------------
        # Local FIR convolutions (depth-wise)
        # ------------------------------------------------------------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)

        # ------------------------------------------------------------------
        # Per-token, per-head content-aware gating
        # ------------------------------------------------------------------
        self.stat_dim = 16  # 4 paths × 4 statistics
        gate_in_dim = hidden_size + self.stat_dim
        gate_hidden_dim = max(8, int(gate_in_dim * fusion_hidden_mult))
        self.fusion_gate_fc1 = nn.Linear(gate_in_dim, gate_hidden_dim, bias=True)
        self.fusion_gate_fc2 = nn.Linear(gate_hidden_dim, 4, bias=True)
        nn.init.zeros_(self.fusion_gate_fc1.bias)
        nn.init.zeros_(self.fusion_gate_fc2.weight)
        self.fusion_gate_fc2.bias.data.copy_(torch.tensor([0., 0., 0., 0.], dtype=self.fusion_gate_fc2.bias.dtype))

        # Learnable per-head temperature τ
        self.log_tau = nn.Parameter(torch.zeros(num_heads))

        # ------------------------------------------------------------------
        # Additive per-path residuals (static + dynamic)
        # ------------------------------------------------------------------
        self.res_alpha = nn.Parameter(torch.full((num_heads, 4), res_static_init))  # static per head/path
        self.res_dyn_proj = nn.Linear(hidden_size, num_heads * 4, bias=True)        # dynamic per token/head/path
        nn.init.constant_(self.res_dyn_proj.bias, res_dyn_bias)

        # ------------------------------------------------------------------
        # Output layer
        # ------------------------------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Helper: statistics per head
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:
        """Return four statistics (mean, var, abs-mean, ℓ2-norm) per (B, L, H)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        absmean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, absmean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional["Cache"] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
                **kwargs):
        """Forward pass with optional unpadding / caching."""
        # ------------------------------------------------------------------
        # 1) Optional unpadding (for Flash-like kernels)
        # ------------------------------------------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (B, S)"
        B0, L0, _ = hidden_states.shape  # original batch / seq length (needed for repadding)

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None  # for repadding later
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L0:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ------------------------------------------------------------------
        # 2) Linear projections (+ optional depth-wise convolution)
        # ------------------------------------------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # (B, L, H, D)
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ------------------------------------------------------------------
        # 3) Activation / normalisation on q,k
        # ------------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError(f"Unknown qk_activation {self.qk_activation}")
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ------------------------------------------------------------------
        # 4) β-gating (optionally allow negative eigen-values)
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # 5) Delta-rule path (chunk-wise, causal)
        # ------------------------------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)  # (B, H, L, Dv)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ------------------------------------------------------------------
        # 6) Other memory paths
        # ------------------------------------------------------------------
        v_direct = v  # direct value path
        local_fir_short = self.local_fir_short(v_direct)  # short FIR
        local_fir_long = self.local_fir_long(v_direct)    # long FIR

        # ------------------------------------------------------------------
        # 7) Per-head statistics for content-aware router
        # ------------------------------------------------------------------
        stats_short = self._per_head_stats(local_fir_short)
        stats_long = self._per_head_stats(local_fir_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = torch.cat([stats_short, stats_long, stats_delta, stats_value], dim=-1)  # (B, L, H, 16)

        # ------------------------------------------------------------------
        # 8) Gating network (token/head-wise) – softmax with floor ε
        # ------------------------------------------------------------------
        hidden_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B, L, H, C)
        gate_in = torch.cat([hidden_exp, stats_vec], dim=-1)                         # (B, L, H, C+16)
        B, L, H, _ = gate_in.shape

        gate_flat = rearrange(gate_in, "b l h f -> (b l h) f")
        gate_fc1 = F.gelu(self.fusion_gate_fc1(gate_flat))
        gate_logits = self.fusion_gate_fc2(gate_fc1)
        gate_logits = rearrange(gate_logits, "(b l h) p -> b l h p", b=B, l=L, h=H)

        tau = F.softplus(self.log_tau) + 1e-3  # positive temperature, shape (H,)
        gate_logits = gate_logits / tau.view(1, 1, H, 1)
        weights = torch.softmax(gate_logits, dim=-1)

        # Floor ε and re-normalise
        if self.prob_floor > 0.0:
            weights = torch.clamp(weights, min=self.prob_floor)
            weights = weights / weights.sum(-1, keepdim=True)

        # ------------------------------------------------------------------
        # 9) Weighted fusion (dynamic) + additive residuals
        # ------------------------------------------------------------------
        fused = (
            weights[..., 0:1] * local_fir_short +
            weights[..., 1:2] * local_fir_long +
            weights[..., 2:3] * delta_out       +
            weights[..., 3:4] * v_direct
        )

        # Dynamic per-token sigmoid gate
        dyn_gate_logits = self.res_dyn_proj(hidden_states)             # (B, L, H*4)
        dyn_gate = torch.sigmoid(rearrange(dyn_gate_logits, "b l (h p) -> b l h p", h=self.num_heads, p=4))

        # Static per-head/path scale (sigmoid constrained to (0,1))
        static_res_scale = torch.sigmoid(self.res_alpha)[None, None, :, :]  # (1, 1, H, 4)

        add_residuals = (
            static_res_scale[..., 0:1] * dyn_gate[..., 0:1] * local_fir_short +
            static_res_scale[..., 1:2] * dyn_gate[..., 1:2] * local_fir_long  +
            static_res_scale[..., 2:3] * dyn_gate[..., 2:3] * delta_out       +
            static_res_scale[..., 3:4] * dyn_gate[..., 3:4] * v_direct
        )

        o = fused + add_residuals  # (B, L, H, Dv)

        # ------------------------------------------------------------------
        # 10) Cache update (if requested)
        # ------------------------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L0,
            )

        # ------------------------------------------------------------------
        # 11) Output normalisation / projection back to model dim
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)

        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ------------------------------------------------------------------
        # 12) Re-pad if input was un-padded earlier
        # ------------------------------------------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B0, L0)

        return o, None, past_key_values
