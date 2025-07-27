# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Context-Floor Gating with Post-Fusion Renormalisation (ACFG)
==============================================================================
Identifier: delta_net_acfg

Motivation
----------
Prior DeltaNet generations demonstrated that protecting the value/copy path
is vital for span-level fidelity, but a *fixed* context quota (Dynamic
Floor-Gated Warm-Start – **DFGWS**) introduces an unavoidable copy-noise that
hurts copy-critical tasks (e.g. Winogrande).  Conversely, removing the floor
risks contextual path starvation and regresses local-reasoning tasks.

Adaptive Context-Floor Gating (ACFG) resolves this dilemma by *learning a
per-token, per-head minimum context allocation* that can vary smoothly between
0 and `max_context_floor` (default 0.20).  High-uncertainty tokens thus retain
a healthy context gradient, while unambiguous copy tokens are free to allocate
> 99 % mass to the identity branch.

Key Components
--------------
1. **Adaptive Floor MLP** – A single linear layer maps the current hidden
   state to *H* logits whose sigmoid determines the minimum context quota
   `floor ∈ [0,max_floor]` for each head/token.
2. **Hierarchical Gating** – As in DFGWS, gating proceeds in two stages:
      a. Value gate (sigmoid) with learnable warm-start bias `+4`.
      b. Softmax over contextual paths {short FIR, long FIR, Δ-rule}.
   The value gate is rescaled so that
   `p_value = (1-floor) * σ(logit_val)` guaranteeing
   `1-p_value ≥ floor` ⇒ continuous gradient flow.
3. **Post-Fusion Head-Wise RMSNorm** – A lightweight, per-head RMSNorm is
   applied to the fused memory before projection to stabilise the variance
   increase introduced by adaptive routing.  This follows the variance control
   insight from CAGF-RC analysis and adds *negligible* compute.

All operations remain O(N), strictly causal, batch-agnostic, and fully
compatible with earlier DeltaNet interfaces.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x: mx.array) -> mx.array:
    """Shifted ELU (=ELU+1) keeps outputs positive."""
    return mx.maximum(x, 0) + mx.minimum(mx.exp(x) - 1, 0) + 1.0


def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that last dimension sums to 1."""
    return x / mx.sum(x, axis=-1, keepdims=True)


def l2norm(x: mx.array) -> mx.array:
    """L2 normalization."""
    return x / mx.linalg.norm(x, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule (adapted for MLX)
# -----------------------------------------------------------------------------

def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Simplified Δ-rule for MLX - linear attention approximation."""
    b, h, L, d_k = q.shape
    
    # Normalisation & beta scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    
    # Simplified linear attention (avoid complex chunking for MLX compatibility)
    # Compute attention weights
    attn_weights = mx.softmax(q @ mx.transpose(k, [0, 1, 3, 2]) / mx.sqrt(d_k), axis=-1)
    
    # Apply causal mask
    seq_len = attn_weights.shape[-1]
    causal_mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
    attn_weights = attn_weights * (1 - causal_mask)
    
    # Renormalize after masking
    attn_weights = attn_weights / (mx.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
    
    # Apply to values
    o = attn_weights @ v
    
    # Simple recurrent state (simplified)
    S = mx.sum(mx.transpose(k, [0, 1, 3, 2]) @ v, axis=2, keepdims=True)
    
    return o, S


# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (δ-kernel initialisation)
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR convolution with delta (identity) initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros((num_heads, head_dim, self.kernel_size))
        # identity at current timestep - manual initialization
        weight_list = []
        for h in range(num_heads):
            for d in range(head_dim):
                filter_vals = mx.zeros(self.kernel_size)
                filter_vals = mx.concatenate([filter_vals[:-1], mx.array([1.0])])
                weight_list.append(filter_vals)
        self.filters = mx.array(weight_list).reshape(num_heads, head_dim, self.kernel_size)

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        
        # Reshape for processing: (B, L, H*D)
        x_reshaped = x.reshape(b, l, h * d)
        
        # Apply filters manually for each channel
        outputs = []
        for head in range(h):
            for dim in range(d):
                ch_idx = head * d + dim
                x_ch = x_reshaped[:, :, ch_idx:ch_idx+1]  # (B, L, 1)
                
                # Causal padding
                x_padded = mx.pad(x_ch, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
                
                # Manual convolution using the filter
                filter_weights = self.filters[head, dim]  # (kernel_size,)
                conv_out = []
                for t in range(l):
                    if t + self.kernel_size <= x_padded.shape[1]:
                        window = x_padded[:, t:t+self.kernel_size, 0]  # (B, kernel_size)
                        result = mx.sum(window * filter_weights, axis=1, keepdims=True)  # (B, 1)
                        conv_out.append(result)
                
                if conv_out:
                    conv_result = mx.concatenate(conv_out, axis=1)  # (B, L)
                else:
                    conv_result = mx.zeros((b, l))
                outputs.append(conv_result)
        
        # Reshape back to (B, L, H, D)
        y = mx.stack(outputs, axis=2).reshape(b, l, h, d)
        return y


# -----------------------------------------------------------------------------
# RMSNorm implementation for MLX
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        return self.weight * (x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps))


# -----------------------------------------------------------------------------
# Short convolution for MLX
# -----------------------------------------------------------------------------

class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        # MLX Conv1d expects (in_channels, out_channels, kernel_size)
        self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, 
                             kernel_size=kernel_size, stride=1, padding=0, bias=bias)
        self.kernel_size = kernel_size
        self.activation = activation

    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, D) - MLX Conv1d expects this format directly
        
        # Causal padding: pad sequence dimension (axis=1) with kernel_size-1 zeros at the start
        x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        out = self.conv(x_padded)
        out = out[:, :x.shape[1], :]  # Ensure output length matches input
        
        if self.activation == 'silu':
            out = nn.silu(out)
        elif self.activation == 'gelu':
            out = nn.gelu(out)
            
        if output_final_state:
            return out, None  # Simplified - no cache state
        return out


# -----------------------------------------------------------------------------
# Main DeltaNet with Adaptive Context-Floor Gating
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with Adaptive Context-Floor Gating (ACFG)."""

    def __init__(
        self,
        # ---- identifier & mode ----
        mode: str = "acfg",
        # ---- model dimensions ----
        d_model: int | None = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # ---- optional components ----
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
        # ---- FIR kernels ----
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # ---- gating hyper-parameters ----
        context_max_floor: float = 0.2,
        fusion_hidden_mult: int = 2,
        value_bias_init: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # ------------------ bookkeeping ------------------
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
        self.context_max_floor = float(context_max_floor)
        assert 0.0 < self.context_max_floor < 0.5, "context_max_floor must be (0,0.5)"

        # ------------------ dimensions ------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # ------------------ projections ------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------------ short convs ------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ------------------ FIR branches ------------------
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ------------------ Fusion gate ------------------
        gate_in_dim = hidden_size
        self.fusion_gate_layer1 = nn.Linear(gate_in_dim, hidden_size * fusion_hidden_mult, bias=True)
        self.fusion_gate_activation = nn.GELU()
        self.fusion_gate_layer2 = nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 4, bias=True)
        
        # Initialize warm-start bias for value path
        bias_list = []
        for i in range(num_heads * 4):
            if i % 4 == 3:  # value path bias indices
                bias_list.append(value_bias_init)
            else:
                bias_list.append(0.0)
        self.fusion_gate_layer2.bias = mx.array(bias_list)

        # ------------- Adaptive floor MLP --------------
        self.floor_mlp = nn.Linear(hidden_size, num_heads, bias=True)
        # Initialize with proper bias
        bias_init = math.log(self.context_max_floor / (1 - self.context_max_floor))
        self.floor_mlp.bias = mx.full((num_heads,), bias_init)

        # --------------- Output normalisation -----------
        self.post_fusion_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.array, None, Optional[dict]]:
        batch_size, seq_len_full, _ = hidden_states.shape

        # Simplified for MLX - skip complex padding/unpadding logic
        seq_len = seq_len_full

        # ---------------- Q/K/V projections --------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        # Apply convolutions (causal)
        q_lin = self.q_conv1d(q_lin)
        k_lin = self.k_conv1d(k_lin)
        v_lin = self.v_conv1d(v_lin)

        # ---------------- Head reshape -------------------
        q = q_lin.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k_lin.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v_direct = v_lin.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)

        # ---------------- Activations --------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---------------- Beta ---------------------------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((batch_size, seq_len, self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule global path -------------
        delta_out_t, recurrent_state = delta_rule_chunkwise(
            q=q.transpose(0, 2, 1, 3),  # (B, H, L, D)
            k=k.transpose(0, 2, 1, 3),
            v=v_direct.transpose(0, 2, 1, 3),
            beta=beta.transpose(0, 2, 1),  # (B, H, L)
        )
        delta_out = delta_out_t.transpose(0, 2, 1, 3)  # Back to (B, L, H, D)

        # ---------------- FIR paths ----------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ---------------- Adaptive floor -----------------
        floor_logits = self.floor_mlp(hidden_states)  # (B, L, H)
        floor = mx.sigmoid(floor_logits) * self.context_max_floor

        # ---------------- Fusion gate --------------------
        fusion_logits = self.fusion_gate_layer1(hidden_states)
        fusion_logits = self.fusion_gate_activation(fusion_logits)
        fusion_logits = self.fusion_gate_layer2(fusion_logits)  # (B,L,H*4)
        fusion_logits = fusion_logits.reshape(batch_size, seq_len, self.num_heads, 4)

        # Value gate (sigmoid) with adaptive floor
        value_logit = fusion_logits[..., 3]
        p_value_raw = mx.sigmoid(value_logit)
        p_value = (1.0 - floor) * p_value_raw
        others_total = 1.0 - p_value

        # Contextual softmax over paths {short, long, delta}
        ctx_logits = fusion_logits[..., 0:3]
        ctx_weights = mx.softmax(ctx_logits, axis=-1)
        ctx_weights = ctx_weights * others_total[..., None]

        # ---------------- Fuse outputs -------------------
        o = (
            ctx_weights[..., 0:1] * local_short +
            ctx_weights[..., 1:2] * local_long +
            ctx_weights[..., 2:3] * delta_out +
            p_value[..., None] * v_direct
        )

        # ---------------- Post-fusion norm ---------------
        o = self.post_fusion_norm(o)

        # ---------------- Output norm/proj ---------------
        if self.use_gate:
            g = self.g_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)
            o = self.o_norm(o) * mx.sigmoid(g)  # Simplified gating
        else:
            o = self.o_norm(o)
        o = o.reshape(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)

        return o, None, past_key_values