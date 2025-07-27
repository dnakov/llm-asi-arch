# LLM-Generated Architecture: delta_net_cagf_rc_pf_mlx
# Parent: None
# Performance: 0.3063
# Testing existing MLX architecture: delta_net_cagf_rc_pf_mlx
# LLM Analysis: ```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
```

## OUTPUT FORMAT
``...

# -*- coding: utf-8 -*-
"""
DeltaNet – Content-Aware Gated Fusion with **Dynamic Residual Convolution** and
**Probability-Floor Normalised Mixture** (CAGF-RC-PF) - MLX Version
==========================================================================
Key architectural innovations (enabled by default):

1.  Probability-floor gated fusion
    •  A small, fixed ε-floor (default = 2 %) is applied **after** the softmax
      over the four memory paths (short-FIR, long-FIR, Δ-rule, value).
    •  This guarantees a *strictly positive* gradient signal for *every* path
      while keeping the final mixture **exactly normalized** (sums to 1).  It
      combines the stability of floor-gated routing (DFGWS) with the strict
      variance control of softmax fusion (CAGF), fixing the variance inflation
      issue observed in *delta_net_cagf_rc*.

2.  Dynamic, context-aware residual convolutional injection
    •  The static per-head gate γₕ from *cagf_rc* is replaced by the product of
      a *learnable per-head scalar* **and** a *per-token, per-head* dynamic gate
      computed from the current hidden representation.  Formally:

          γ̂[b,t,h] = σ(γ_h) · σ(W_res · x[b,t] + b_res)_h

      where `σ` is the logistic sigmoid.  This preserves the guaranteed gradient
      flow to the convolutional filters while allowing the network to suppress
      the residual when global context is more important – directly addressing
      the BoolQ / Lambada regression identified in prior experiments.

3.  Post-fusion RMS normalisation (RMSNorm)
    •  The original implementation already applied an RMSNorm after the residual
      path via `self.o_norm`.  This variant keeps the same projection pipeline
      – the probability-floor ensures the variance seen by `o_norm` is well-
      behaved.

The design keeps *all* proven strengths of DeltaNet – O(N) chunked Δ-rule,
causal depth-wise FIR, batch-agnostic shape handling, and compile optimization on the
heavy kernel – while eliminating the variance spike and adding context-sensitive
control of the residual convolution.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ================================================================
# Utility helpers
# ================================================================

def _elu_p1(x: mx.array) -> mx.array:  # Shifted ELU (>0)
    return mx.where(x > 0, x + 1.0, mx.exp(x))

def _sum_norm(x: mx.array) -> mx.array:  # L1 normalisation
    return x / mx.sum(x, axis=-1, keepdims=True)

def _l2norm(x: mx.array, axis: int = -1, eps: float = 1e-5) -> mx.array:
    """L2 normalization along specified axis."""
    return x / (mx.linalg.norm(x, axis=axis, keepdims=True) + eps)

# ================================================================
# Depth-wise causal FIR convolution
# ================================================================
class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal padding: inputs (B, L, H, D)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Identity (Dirac) initialisation with small noise for stability
        # Initialize with zeros and manually set the last element
        filt = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Create a one-hot vector for the last position
        one_hot = mx.zeros(self.kernel_size)
        one_hot = mx.concatenate([mx.zeros(self.kernel_size - 1), mx.ones(1)])
        # Broadcast to all heads and dims
        dirac_init = mx.broadcast_to(one_hot[None, None, :], (num_heads, head_dim, self.kernel_size))
        filt = filt + dirac_init
        filt = filt + 0.02 * mx.random.normal(filt.shape)
        self.filters = filt

    def __call__(self, x: mx.array) -> mx.array:  # (B,L,H,D)
        b, l, h, d = x.shape
        # Reshape for convolution
        x_reshaped = x.reshape(b, l, h * d)  # (B, L, H*D)
        
        # Causal padding
        pad_width = [(0, 0), (self.kernel_size - 1, 0), (0, 0)]
        x_pad = mx.pad(x_reshaped, pad_width)  # (B, L+K-1, H*D)
        
        # Simplified convolution using vectorized operations
        output_list = []
        for i in range(l):
            output_i = mx.zeros((b, h, d))
            for j in range(self.kernel_size):
                if i + j < x_pad.shape[1]:
                    x_slice = x_pad[:, i + j, :].reshape(b, h, d)  # (B, H, D)
                    filter_slice = self.filters[:, :, self.kernel_size - 1 - j]  # (H, D)
                    conv_result = x_slice * filter_slice[None, :, :]  # (B, H, D)
                    output_i = output_i + conv_result
            output_list.append(output_i)
        
        output = mx.stack(output_list, axis=1)  # (B, L, H, D)
        return output

# ================================================================
# Chunk-wise Δ-rule kernel
# ================================================================
def _delta_rule_chunkwise(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    chunk_size: int = 32,
):
    """Efficient causal associative Δ-rule with O(N) complexity."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    
    if pad_len:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    
    L_pad = L + pad_len
    
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    # Reshape into chunks
    n_chunks = L_pad // chunk_size
    q = q.reshape(b, h, n_chunks, chunk_size, d_k)
    k = k.reshape(b, h, n_chunks, chunk_size, d_k)
    v = v.reshape(b, h, n_chunks, chunk_size, -1)
    k_beta = k_beta.reshape(b, h, n_chunks, chunk_size, d_k)
    
    # Create causal mask
    tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0)
    
    # Process chunks
    S = mx.zeros((b, h, d_k, v.shape[-1]))
    out = mx.zeros_like(v)
    
    for idx in range(n_chunks):
        q_i = q[:, :, idx]  # (b, h, chunk_size, d_k)
        k_i = k[:, :, idx]  # (b, h, chunk_size, d_k)
        v_i = v[:, :, idx]  # (b, h, chunk_size, d_v)
        k_beta_i = k_beta[:, :, idx]  # (b, h, chunk_size, d_k)
        
        # Attention within chunk
        attn = -(k_beta_i @ mx.transpose(k_i, [0, 1, 3, 2])) * (1 - tri)[None, None, :, :]
        
        # Simplified cumulative attention computation
        for i in range(1, chunk_size):
            attn_i = attn[:, :, i:i+1, :i]  # (b, h, 1, i)
            attn_prev = attn[:, :, :i, :i]  # (b, h, i, i)
            update = (attn_i @ attn_prev).sum(axis=-2)  # (b, h, i)
            # Create new attention matrix with the update
            attn_row_orig = attn[:, :, i, :]  # (b, h, chunk_size)
            attn_row_prefix = attn_row_orig[:, :, :i] + update  # (b, h, i) + (b, h, i)
            attn_row_suffix = attn_row_orig[:, :, i:]  # (b, h, remaining)
            attn_new_row = mx.concatenate([attn_row_prefix, attn_row_suffix], axis=-1)
            # Replace row i with updated row
            if i == chunk_size - 1:
                attn = mx.concatenate([attn[:, :, :i, :], attn_new_row[:, :, None, :]], axis=2)
            else:
                attn = mx.concatenate([attn[:, :, :i, :], attn_new_row[:, :, None, :], attn[:, :, i+1:, :]], axis=2)
        
        attn = attn + mx.eye(chunk_size)[None, None, :, :]
        
        u_i = attn @ v_i
        w_i = attn @ k_beta_i
        
        # Cross-chunk attention
        cross_attn = q_i @ S  # (b, h, chunk_size, d_v)
        
        # Local attention (strictly lower triangular)
        tri_strict = mx.triu(mx.ones((chunk_size, chunk_size)), k=1)
        local_attn = (q_i @ mx.transpose(k_i, [0, 1, 3, 2])) * (1 - tri_strict)[None, None, :, :]
        local_out = local_attn @ u_i
        
        # Update state from cross-chunk interaction
        update_term = w_i @ S
        u_i_corrected = u_i - update_term
        
        # Update output manually
        chunk_result = cross_attn + local_out
        out_parts = [
            out[:, :, :idx, :],
            chunk_result[:, :, None, :] if idx < n_chunks - 1 else chunk_result[:, :, None, :],
            out[:, :, idx+1:, :] if idx < n_chunks - 1 else mx.zeros((b, h, 0, chunk_result.shape[-1]))
        ]
        out = mx.concatenate([part for part in out_parts if part.shape[2] > 0], axis=2)
        
        # Update recurrent state
        S = S + mx.transpose(k_i, [0, 1, 3, 2]) @ u_i_corrected
    
    # Reshape back
    out = out.reshape(b, h, L_pad, -1)
    if pad_len:
        out = out[:, :, :L]
    
    return out, S

# ================================================================
# RMSNorm implementation
# ================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / norm

class FusedRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * gate * x / norm

# ================================================================
# Short convolution
# ================================================================
class ShortConvolution(nn.Module):
    def __init__(self, dim: int, kernel_size: int, activation: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = mx.random.normal((dim, kernel_size)) * 0.02
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array, cache=None, output_final_state: bool = False, cu_seqlens=None):
        b, l, d = x.shape
        
        # Causal padding
        x_pad = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        # Manual convolution
        output_list = []
        for i in range(l):
            output_i = mx.zeros((b, d))
            for j in range(self.kernel_size):
                if i + j < x_pad.shape[1]:
                    conv_contrib = x_pad[:, i + j, :] * self.weight[:, self.kernel_size - 1 - j]
                    output_i = output_i + conv_contrib
            output_list.append(output_i)
        
        output = mx.stack(output_list, axis=1)
        
        output = output + self.bias
        
        if self.activation == "silu":
            output = output * mx.sigmoid(output)
        elif self.activation == "relu":
            output = mx.maximum(output, 0)
        
        final_state = None
        if output_final_state:
            final_state = x[:, -self.kernel_size + 1:, :] if l >= self.kernel_size - 1 else x
        
        return output, final_state

# ================================================================
# Main DeltaNet Layer
# ================================================================
class DeltaNet(nn.Module):
    """DeltaNet layer with probability-floor fusion and dynamic residual conv."""

    def __init__(
        self,
        *,
        mode: str = "cagf_rc_pf",
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
        # ─── Multi-scale FIR kernel sizes ─────────────────────────
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # Fusion network params
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),  # τ≈0.7
        # Probability floor (ε)
        prob_floor: float = 0.02,
        # Dynamic residual conv path
        conv_residual_init: float = -2.0,
        **kwargs,
    ):
        super().__init__()

        # ---- Book-keeping & dims ------------------------------------
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

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---- Linear projections -------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- Short convolution enhancements -------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- Multi-scale FIR convolutions ---------------------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)

        # ---- Content-aware gating network ---------------------------
        self.stat_dim = 16
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        
        self.fusion_gate_linear1 = nn.Linear(gate_in_dim, hidden_gate_dim, bias=True)
        self.fusion_gate_linear2 = nn.Linear(hidden_gate_dim, 4, bias=True)
        
        # Initialize bias
        self.fusion_gate_linear2.bias = mx.array(gate_bias_init)

        self.logit_temperature = mx.array([gate_logit_init])

        # ---- Dynamic residual convolution scaling ------------------
        self.conv_residual_logit = mx.full((num_heads,), conv_residual_init)  # per-head scalar
        self.res_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.res_gate_proj.bias = mx.full((num_heads,), -2.0)  # start with small gate

        # ---- Output normalisation / projection ---------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Statistic helpers (per-head)
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        abs_mean = mx.mean(mx.abs(x), axis=-1, keepdims=True)
        l2 = mx.linalg.norm(x, axis=-1, keepdims=True)
        return mx.concatenate([mean, var, abs_mean, l2], axis=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, L_full, _ = hidden_states.shape

        # Simple implementation without complex padding/unpadding for MLX
        L = L_full

        # ---------------- Q/K/V projections + short conv --------------
        conv_q = conv_k = conv_v = None
        
        q_in, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache)
        k_in, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache)
        v_in, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache)

        # ---------------- Head reshape ---------------------------------
        q = q_in.reshape(q_in.shape[0], q_in.shape[1], self.num_heads, self.head_k_dim)
        k = k_in.reshape(k_in.shape[0], k_in.shape[1], self.num_heads, self.head_k_dim)
        v_direct = v_in.reshape(v_in.shape[0], v_in.shape[1], self.num_heads, self.head_v_dim)

        # ---------------- Activation on Q/K ---------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = mx.maximum(q, 0), mx.maximum(k, 0)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---------------- Beta for Δ-rule -----------------------------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones(q.shape[:-1])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule global pathway -----------------------
        delta_out_t, recurrent_state = _delta_rule_chunkwise(
            q=q.transpose(0, 2, 1, 3),  # (b, h, l, d)
            k=k.transpose(0, 2, 1, 3),
            v=v_direct.transpose(0, 2, 1, 3),
            beta=beta.transpose(0, 2, 1),  # (b, h, l)
        )
        delta_out = delta_out_t.transpose(0, 2, 1, 3)  # (b, l, h, d)

        # ---------------- Local FIR paths ----------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---------------- Per-head statistics for gating -------------
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = mx.concatenate([stats_short, stats_long, stats_delta, stats_value], axis=-1)  # (B,L,H,16)

        # ---------------- Build gating input -------------------------
        hs_exp = mx.expand_dims(hidden_states, -2)
        hs_exp = mx.broadcast_to(hs_exp, (hs_exp.shape[0], hs_exp.shape[1], self.num_heads, hs_exp.shape[-1]))
        gate_in = mx.concatenate([hs_exp, stats_vec], axis=-1)  # (B,L,H,D+16)
        gate_in_flat = gate_in.reshape(-1, gate_in.shape[-1])
        
        # Apply fusion gate MLP
        fusion_logits_flat = self.fusion_gate_linear1(gate_in_flat)
        fusion_logits_flat = mx.where(fusion_logits_flat > 0, fusion_logits_flat, 
                                    fusion_logits_flat * 0.01)  # GELU approximation
        fusion_logits_flat = self.fusion_gate_linear2(fusion_logits_flat)  # (B*L*H,4)

        # Temperature scaling & reshape
        temperature = mx.log(1 + mx.exp(self.logit_temperature)) + 1e-4  # softplus
        fusion_logits_flat = fusion_logits_flat / temperature
        fusion_logits = fusion_logits_flat.reshape(gate_in.shape[0], gate_in.shape[1], self.num_heads, 4)

        # ---------------- Softmax + ε-floor ---------------------------
        fusion_weights = mx.softmax(fusion_logits, axis=-1)  # (B,L,H,4)
        if self.prob_floor > 0.0:
            fusion_weights = mx.maximum(fusion_weights, self.prob_floor)
            # Prevent division by zero in renormalisation
            fusion_weights_sum = mx.sum(fusion_weights, axis=-1, keepdims=True)
            # Clamp fusion_weights_sum higher (prevent 1e-6/0.02 ~ 0.05 losses): stability fix
            fusion_weights_sum = mx.maximum(fusion_weights_sum, 4 * self.prob_floor + 1e-6)
            fusion_weights = fusion_weights / fusion_weights_sum

        # ---------------- Weighted fusion ----------------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # ---------------- Dynamic residual conv path -----------------
        res_gate = mx.sigmoid(self.res_gate_proj(hidden_states))  # (B,L,H)
        # Clamp res_gate to avoid saturation or underflow
        res_gate = mx.clip(res_gate, 1e-4, 1 - 1e-4)
        static_scale = mx.sigmoid(self.conv_residual_logit)[None, None, :, None]  # (1,1,H,1)
        conv_res_scale = static_scale * res_gate[..., None]  # (B,L,H,1)
        o = o + conv_res_scale * local_short

        # ---------------- Normalisation / projection -----------------
        if self.use_gate:
            g_vec = self.g_proj(hidden_states).reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = o.reshape(o.shape[0], o.shape[1], -1)
        o = self.o_proj(o)

        return o, None, past_key_values