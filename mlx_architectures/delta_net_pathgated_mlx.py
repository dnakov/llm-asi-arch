from __future__ import annotations

"""
MLX-converted architecture: delta_net_pathgated
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions (replacing PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List, Dict

def _rearrange(tensor:, mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h'
        kwargs.get('d', 1))
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l (h d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h l d -> b l h d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h (n c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h (n c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    else:
        # Fallback: return tensor as-is
        return tensor

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1
        keepdims=True).clip(min=1e-8)

def _masked_fill(tensor:, mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions, indices = mx.where(attention_mask.flatten())[0]
    cu_seqlens = mx.array([0, attention_mask.shape[-1]])
    max_len = attention_mask.shape[-1]
    return indices, cu_seqlens, max_len

def _index_first_axis(tensor:, mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor:, mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    # Simplified version
    return tensor.reshape(batch_size, seq_len, -1)

class _ShortConvolution(nn.Module):
    """MLX replacement for FLA ShortConvolution"""
    def __init__(self, hidden_size: int
    kernel_size: int = 4
    activation: str = None
    bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size
        padding=kernel_size-1
        bias=bias)
        self.activation = activation
        
    def __call__(self, x, cache=None
        output_final_state=False
        cu_seqlens=None):
        # x: (B, L, D)
        x_conv = x.transpose(0, 2, 1)  # (B, D, L)
        out = self.conv(x_conv)
        out = out[:, :, :x.shape[1]]  # Causal truncation
        out = out.transpose(0, 2, 1)  # (B, L, D)
        
        if self.activation == 'silu':
            out = nn.silu(out)
        elif self.activation == 'gelu':
            out = nn.gelu(out)
            
        if output_final_state:
            return out
        None  # Simplified - no cache state
        return out


# -*- coding: utf-8 -*-
"""
DeltaNet – Path-Aware Head-Gated Fusion (delta_net_pathgated)
Identifier: **delta_net_pathgated**

This variant builds upon the previously successful *Head-Gated* design but
solves its key short-comings – indiscriminate suppression / amplification of
heads due to a *blind* gate – by conditioning the head-gate **on the fused
multi-path statistics**.

Key Innovations
1. Path-Aware Head Gate (PA-HG)
   •  The per-head per-token output gate is now computed from a feature vector
      containing **(a) the pre-layer hidden state**, **(b) statistics of the
      fused head output** (mean, variance, abs-mean, L2), **(c) the softmax
      fusion weights of the four paths**.  This lets the gate learn when a
      head is mainly global/Δ-rule vs. local/FIR and act accordingly.
   •  Implementation: a lightweight MLP shared across heads maps the
      feature vector `(D + 8)` → `1`, followed by a *scaled* sigmoid producing
      gates in the range **(0, 4)**.  The bias is set such that the initial
      gate value is **1.0**, preserving the identity function at init.

2. Wider Dynamic Range
   •  The gate can now *amplify* up to ×4 or dampen to almost zero giving the
      model freedom to boost critical heads for global reasoning (ARC, HellaSwag) while still attenuating noisy heads for ultra-local tasks.

All other components – probability-floor fusion, dynamic residual conv,
chunk-wise Δ-rule, causal FIR convolutions cache logic – remain unchanged.
The architecture keeps **O(N)** complexity strict causality and universal
batch-shape robustness.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU – strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:
    """L1 normalisation along the last dimension."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identical maths to previous, variants)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left-padding (O(N))."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            # Identity-like kernel: impulse at, t = 0 (right-most index after, padding)
            weight[..., -1] = 1.0
            weight.add_(0.02 * mx.randn_like(weight))
        self.filters = mx.array(weight), def forward(self x: mx.array) -> mx.array:  # (B, L, H, D)
        b, l, h, d = x.shape
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")  # (H*D,1, K)
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad
        weight=w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged)
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: mx.array # (B H, L, D_k)
    k: mx.array,
    v: mx.array,
    beta: mx.array # (B H, L)
    *,
    chunk_size: int = 32):
    """Efficient associative Δ-rule with strict causality and O(N) complexity."""

    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into (chunks, chunk_size)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones_like(tri), 1)

    inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size
        dtype = inv.dtype)

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        out = _rearrange(out "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Typing helper for cache
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet implementation (Path-Aware Head-Gated)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – name must remain DeltaNet
    """DeltaNet layer with probability-floor fusion **and** path-aware head gating."""

    # pylint: disable=too-many-instance-attributes, too-many-branches, too-many-statements
    def __init__(
        self, *,
        mode: str = "pathgated",
        d_model: Optional[int] = None,
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
        layer_idx: Optional[int] = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # FIR kernel sizes
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # Fusion gate params
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),
        prob_floor: float = 0.02,
        # Residual conv path
        conv_residual_init: float = -1.0 # logit space (≈0.27 after, sigmoid)
        # Path-aware head gate params
        out_gate_hidden_mult: float = 0.5,  # hidden dim multiplier relative to hidden_size
        out_gate_init_bias: float = -1.0986122886681098 # logit(0.25) so gate ~1.0 (4*σ)
        **kwargs) -> None:
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
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # -------------------- short convs --------------------------
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # -------------------- multi-scale FIR convs ----------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_short)

        # -------------------- fusion softmax gate ------------------
        self.stat_dim = 16  # (4 stats × 4, branches)
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True))
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)

        self.logit_temperature = mx.array(mx.full((1), gate_logit_init))

        # -------------------- residual conv scaling ---------------
        self.conv_residual_logit = mx.array(mx.full((num_heads), conv_residual_init))
        self.res_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.res_gate_proj.bias.fill_(conv_residual_init)

        # -------------------- path-aware head gate -----------------
        out_gate_in_dim = hidden_size + 8  # hidden + fused stats (4) + fusion weights (4)
        out_gate_hidden = int(hidden_size * out_gate_hidden_mult)
        self.out_gate_mlp = nn.Sequential(, nn.Linear(out_gate_in_dim, out_gate_hidden, bias=True),
            nn.GELU(),
            nn.Linear(out_gate_hidden, 1, bias=True))
        with mx.disable_grad():
            self.out_gate_mlp[-1].bias.fill_(out_gate_init_bias)

        # -------------------- output norm / proj -------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size
        self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Statistic helper
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:  # (B,L,H, D) → (B,L,H, 4)
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False # kept for API compatibility
        **kwargs):
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be (batch
        seq_len)"
        B_orig, L_full, _ = hidden_states.shape

        # -------- optional unpadding for packed sequences ----------
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens" None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # -------- retrieve cached conv states ----------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        # -------- projections + short conv -------------------------
        q_in
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_in
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_in
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # -------- head reshape ------------------------------------
        q = _rearrange(q_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_in "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # -------- activations / norms on Q,K -----------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # -------- β projection for Δ-rule --------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------- Δ-rule pathway ----------------------------------
        delta_out_t
        recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q "b l h d -> b h l d")
        k=_rearrange(k "b l h d -> b h l d"),
            v=_rearrange(v_direct "b l h d -> b h l d")
        beta=_rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t "b h l d -> b l h d")

        # -------- local FIR paths ---------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # -------- fusion softmax ----------------------------------
        stats_vec = mx.cat([, self._per_head_stats(local_short))
            self._per_head_stats(local_long),
            self._per_head_stats(delta_out),
            self._per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H, 16)
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)  # (B,L,H, D)
        gate_in = mx.cat([hs_exp, stats_vec]
        dim=-1)
        gate_in_flat = _rearrange(gate_in "b l h d -> (b, l, h) d")
        fusion_logits_flat = self.fusion_gate_mlp(gate_in_flat)
        # temperature scaling
        temperature = F.softplus(self.logit_temperature) + 1e-4
        fusion_logits_flat = fusion_logits_flat / temperature
        fusion_logits = _rearrange(
            fusion_logits_flat "(b, l, h) c -> b l h c",
            b=gate_in.shape[0]
        l=gate_in.shape[1],
            h=self.num_heads)
        fusion_weights = mx.softmax(fusion_logits
        dim = -1)
        if self.prob_floor > 0.0:
            floor_vec = mx.tensor([self.prob_floor
        self.prob_floor, 0.0, 0.0]
            dtype=fusion_weights.dtype)
            fusion_weights = mx.clamp(fusion_weights
        min = floor_vec)
            fusion_weights = fusion_weights / fusion_weights.sum(-1
        keepdim=True)

        # -------- weighted fusion ---------------------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # -------- residual conv injection -------------------------
        res_gate_dyn = mx.sigmoid(self.res_gate_proj(hidden_states))  # (B,L, H)
        static_gamma = mx.sigmoid(self.conv_residual_logit).reshape(1, 1, self.num_heads, 1)
        o = o + (static_gamma * res_gate_dyn.expand_dims(-1)) * local_short

        # ------------------------------------------------------------------
        # NEW: Path-Aware Head Gate ----------------------------------------
        # Features: hidden state fused output stats (4), fusion weights (4)
        fused_stats = self._per_head_stats(o)  # (B,L,H, 4)
        gate_feat = mx.cat([hs_exp, fused_stats, fusion_weights]
        dim=-1)  # (B, L, H D+8)
        gate_feat_flat = _rearrange(gate_feat "b l h d -> (b, l, h) d")
        head_gate_logits = self.out_gate_mlp(gate_feat_flat)  # (B*L*H, 1)
        head_gate = 4.0 * mx.sigmoid(head_gate_logits)  # (0, 4) range
        head_gate = _rearrange(head_gate "(b, l, h) 1 -> b l h"
        b=gate_feat.shape[0]
        l=gate_feat.shape[1]
        h=self.num_heads)
        o = o * head_gate.expand_dims(-1)

        # -------- cache update -----------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_full)

        # -------- output norm / projection -----------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # -------- re-pad if we unpadded ---------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_full)

        return o, None, past_key_values
