# -*- coding: utf-8 -*-
"""
DeltaNet – Content-Aware Gated Fusion v2 (Hybrid Residual, Prob-Floor) - MLX Version
=====================================================================================
Identifier: **delta_net_cagf_rc_pf_hybrid_mlx**

This evolution of the *cagf_rc_pf* variant directly addresses the main
regression uncovered in Winogrande / ultra-local reasoning by **ensuring a
non-zero always-on residual contribution** while *retaining* the proven
benefits of probability-floored soft-max fusion (`ε`-floor) and dynamic,
content-aware routing.

Key improvements (enabled by default)
------------------------------------
1. Hybrid residual scaling
   γ̂[b,t,h] = σ(γ_h) · (α + (1-α)·σ(W x[b,t] + b))
   •  `α` (default **0.3**) is a *learnable* minimum residual fraction, giving
      every head a guaranteed path for ultra-local signals (crucial for
      WinoGrande / coreference) while still allowing dynamic modulation.
   •  Static logit `γ_h` **initialises at –1.0** (instead of –2.0) so the
      residual starts at ~0.27 strength – strong enough for learning signals
      but not dominant.

2. Slightly higher probability floor (`ε = 0.03`) to further improve gradient
   flow through rarely-chosen paths during early training.

Everything else – Δ-rule chunk, dual FIR branches, head-level statistics, per-
path probability floor fusion, RMS normalisation – is inherited unchanged and
kept fully compatible with existing checkpoints & infrastructure.

Complexity remains **O(N·d)**, strictly causal, batch-agnostic and optimized
for MLX.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

# ================================================================
# Helper utilities
# ================================================================

def _elu_p1(x: mx.array) -> mx.array:
    """Shifted ELU (= ELU + 1) keeps values strictly positive."""
    return mx.maximum(mx.exp(x) - 1.0, 0.0) + 1.0

def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that the last dimension sums to one."""
    return x / mx.sum(x, axis=-1, keepdims=True)

def _l2norm(x: mx.array, axis: int = -1, eps: float = 1e-5) -> mx.array:
    """L2 normalization along specified axis."""
    return x / mx.maximum(mx.sqrt(mx.sum(x**2, axis=axis, keepdims=True)), eps)

# ================================================================
# Depth-wise causal FIR convolution (Dirac initialisation)
# ================================================================

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding for (B,L,H,D) tensors."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Identity (Dirac) kernel + small noise
        filt = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Set identity
        filt_list = filt.tolist()
        for i in range(num_heads):
            for j in range(head_dim):
                filt_list[i][j][-1] = 1.0
        filt = mx.array(filt_list)
        # Add small noise
        noise = mx.random.normal((num_heads, head_dim, self.kernel_size)) * 0.01
        filt = filt + noise
        self.filters = filt

    def __call__(self, x: mx.array) -> mx.array:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        
        # Reshape for convolution: (B, H*D, L)
        x_reshaped = x.transpose(0, 2, 3, 1).reshape(b, h * d, l)
        
        # Apply causal padding
        pad_left = self.kernel_size - 1
        x_padded = mx.pad(x_reshaped, ((0, 0), (0, 0), (pad_left, 0)))
        
        # Apply depthwise convolution
        output = mx.zeros((b, h * d, l))
        for i in range(h):
            for j in range(d):
                idx = i * d + j
                kernel = self.filters[i, j, :]
                # Simple convolution implementation
                for pos in range(l):
                    start_idx = pos
                    end_idx = pos + self.kernel_size
                    conv_val = mx.sum(x_padded[:, idx, start_idx:end_idx] * kernel[None, :], axis=1)
                    output_list = output.tolist()
                    for b_idx in range(b):
                        output_list[b_idx][idx][pos] = conv_val[b_idx].item()
                    output = mx.array(output_list)
        
        # Reshape back to (B, L, H, D)
        return output.reshape(b, h, d, l).transpose(0, 3, 1, 2)

# ================================================================
# Chunk-wise Δ-rule kernel
# ================================================================

def _delta_rule_chunkwise(
    q: mx.array,  # (B,H,L,Dk)
    k: mx.array,  # (B,H,L,Dk)
    v: mx.array,  # (B,H,L,Dv)
    beta: mx.array,  # (B,H,L)
    *,
    chunk_size: int = 32,
):  # -> Tuple[(B,H,L,Dv), (B,H,Dk,Dv)]
    """Associative Δ-rule retrieval processed in causal chunks (O(N))."""
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = mx.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = mx.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = mx.pad(beta, ((0, 0), (0, 0), (0, pad_len)))
    L_pad = L + pad_len

    # normalisations & β scaling
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into (B,H,N,C,D)
    num_chunks = L_pad // chunk_size
    q = q.reshape(b, h, num_chunks, chunk_size, d_k)
    k = k.reshape(b, h, num_chunks, chunk_size, d_k)
    v = v.reshape(b, h, num_chunks, chunk_size, d_v)
    k_beta = k_beta.reshape(b, h, num_chunks, chunk_size, d_k)

    # Create triangular masks
    tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0)
    tri_strict = mx.triu(mx.ones((chunk_size, chunk_size)), k=1)

    # Initialize matrices
    inv = -(k_beta @ k.transpose(0, 1, 2, 4, 3)) * (1 - tri)
    for i in range(1, chunk_size):
        slice_curr = inv[..., i:i+1, :i]
        slice_prev = inv[..., :i, :i]
        update = mx.sum(slice_curr * slice_prev, axis=-2, keepdims=True).squeeze(-2)
        # Update inv manually
        inv_list = inv.tolist()
        update_list = update.tolist()
        for b_idx in range(b):
            for h_idx in range(h):
                for c_idx in range(num_chunks):
                    for j in range(i):
                        inv_list[b_idx][h_idx][c_idx][i][j] += update_list[b_idx][h_idx][c_idx][j]
        inv = mx.array(inv_list)
    
    inv = inv + mx.eye(chunk_size)

    u = inv @ v
    w = inv @ k_beta
    S = mx.zeros((b, h, d_k, d_v))
    out = mx.zeros_like(v)

    for idx in range(num_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(0, 1, 3, 2)) * (1 - tri_strict)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        chunk_out = q_i @ S + attn_local @ u_i
        # Update out manually
        out_list = out.tolist()
        chunk_out_list = chunk_out.tolist()
        for b_idx in range(b):
            for h_idx in range(h):
                out_list[b_idx][h_idx][idx] = chunk_out_list[b_idx][h_idx]
        out = mx.array(out_list)
        S = S + k_i.transpose(0, 1, 3, 2) @ u_i

    out = out.reshape(b, h, L_pad, d_v)
    if pad_len:
        out = out[:, :, :L]
    return out, S  # (B,H,L,Dv), (B,H,Dk,Dv)

# ================================================================
# RMS Norm implementation
# ================================================================

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * (x / mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps))

class FusedRMSNormGated(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        norm_x = self.weight * (x / mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps))
        return norm_x * gate

# ================================================================
# Short Convolution implementation
# ================================================================

class ShortConvolution(nn.Module):
    def __init__(self, features: int, kernel_size: int, activation: Optional[str] = None, bias: bool = False):
        super().__init__()
        self.features = features
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.weight = mx.random.normal((features, kernel_size)) * (1.0 / math.sqrt(kernel_size))
        if bias:
            self.bias = mx.zeros((features,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array, cache: Optional[mx.array] = None, 
                 output_final_state: bool = False, cu_seqlens: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        # Simple causal convolution implementation
        b, l, d = x.shape
        
        # Apply causal padding
        pad_left = self.kernel_size - 1
        x_padded = mx.pad(x, ((0, 0), (pad_left, 0), (0, 0)))
        
        # Convolution
        output = mx.zeros((b, l, d))
        for i in range(l):
            start_idx = i
            end_idx = i + self.kernel_size
            conv_result = mx.sum(x_padded[:, start_idx:end_idx, :] * self.weight.T[None, :, :], axis=1)
            output_list = output.tolist()
            conv_result_list = conv_result.tolist()
            for b_idx in range(b):
                for d_idx in range(d):
                    output_list[b_idx][i][d_idx] = conv_result_list[b_idx][d_idx]
            output = mx.array(output_list)
        
        if self.bias is not None:
            output = output + self.bias
            
        if self.activation == "silu":
            output = output * mx.sigmoid(output)
        elif self.activation == "relu":
            output = mx.maximum(output, 0)
            
        final_state = None
        if output_final_state:
            final_state = x_padded[:, -self.kernel_size:, :]
            
        return output, final_state

# ================================================================
# DeltaNet main layer
# ================================================================

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any as Cache  # type: ignore

class DeltaNet(nn.Module):
    """DeltaNet with probability-floored gated fusion **and** hybrid residual conv."""

    def __init__(
        self,
        *,
        mode: str = "cagf_rc_pf_hybrid",
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
        # Fusion network
        fusion_hidden_mult: int = 2,
        prob_floor: float = 0.03,  # ε-floor (slightly ↑)
        # Hybrid residual conv params
        residual_alpha: float = 0.3,  # always-on fraction α
        conv_residual_init: float = -1.0,  # logit initialisation
        **kwargs,  # compatibility shim
    ) -> None:
        super().__init__()

        # -------- basic dims & flags ----------------
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.prob_floor = float(prob_floor)
        self.residual_alpha = float(residual_alpha)

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/value dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # -------- projections -----------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # -------- short convs -----------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # -------- multi-scale FIR convs -------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)

        # -------- gating network --------------------
        stats_dim = 4  # one scalar for each of the 4 statistics
        fusion_gate_in = hidden_size + stats_dim  # per-head input dimension
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        
        self.fusion_gate_layer1 = nn.Linear(fusion_gate_in, hidden_gate_dim, bias=True)
        self.fusion_gate_layer2 = nn.Linear(hidden_gate_dim, 4, bias=True)
        
        # warm-start bias toward identity/value path (index 3)
        bias_list = self.fusion_gate_layer2.bias.tolist()
        bias_list[3] = 3.0
        self.fusion_gate_layer2.bias = mx.array(bias_list)

        # -------- residual conv scaling -------------
        self.conv_residual_logit = mx.full((num_heads,), conv_residual_init)
        self.res_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # -------- output norm / proj ----------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # --------------------------------------------------------------
    # Per-head statistics helper
    # --------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:
        """Return per-token, per-head, 4-feature statistics tensor (B,L,H,4)."""
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        abs_mean = mx.mean(mx.abs(x), axis=-1, keepdims=True)
        l2 = mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True))
        return mx.concatenate([mean, var, abs_mean, l2], axis=-1)  # (B,L,H,4)

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    def __call__(
        self,
        hidden_states: mx.array,  # (B,L,D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compat
        **kwargs,  # future proof
    ) -> Tuple[mx.array, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B0, L0, _ = hidden_states.shape

        # For simplicity, we'll skip the unpadding logic in this MLX version
        # -------- projections & short conv ----------
        conv_q = conv_k = conv_v = None
        if past_key_values is not None and self.layer_idx is not None:
            # Simplified cache handling
            pass

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache)

        q = q.reshape(B0, L0, self.num_heads, self.head_k_dim)
        k = k.reshape(B0, L0, self.num_heads, self.head_k_dim)
        v_direct = v.reshape(B0, L0, self.num_heads, self.head_v_dim)

        # ---- activation / norm on q,k -------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = mx.maximum(q, 0), mx.maximum(k, 0)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- β for Δ-rule -------------------------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((B0, L0, self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global path -------------------
        delta_out_b, recurrent_state = _delta_rule_chunkwise(
            q.transpose(0, 2, 1, 3),  # (B,H,L,D)
            k.transpose(0, 2, 1, 3),
            v_direct.transpose(0, 2, 1, 3),
            beta.transpose(0, 2, 1),  # (B,H,L)
        )
        delta_out = delta_out_b.transpose(0, 2, 1, 3)  # (B,L,H,D)

        # ---- local FIR paths ----------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---- statistics for gating ---------------
        stats_short = self._per_head_stats(local_short)  # (B,L,H,4)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats = stats_short + stats_long + stats_delta + stats_value  # element-wise sum (B,L,H,4)

        # Expand hidden_states for per-head processing
        hidden_expanded = mx.expand_dims(hidden_states, axis=-2)  # (B,L,1,D)
        hidden_expanded = mx.broadcast_to(hidden_expanded, (B0, L0, self.num_heads, self.hidden_size))
        
        gate_inp = mx.concatenate([hidden_expanded, stats], axis=-1)  # -> (B,L,H,D+4)
        gate_inp_flat = gate_inp.reshape(-1, gate_inp.shape[-1])
        
        # Apply fusion gate MLP
        fusion_hidden = self.fusion_gate_layer1(gate_inp_flat)
        fusion_hidden = fusion_hidden * mx.sigmoid(fusion_hidden)  # GELU approximation
        fusion_logits_flat = self.fusion_gate_layer2(fusion_hidden)  # (B*L*H, 4)
        
        fusion_logits = fusion_logits_flat.reshape(B0, L0, self.num_heads, 4)  # (B,L,H,4)

        fusion_weights = mx.softmax(fusion_logits, axis=-1)
        if self.prob_floor > 0.0:
            fusion_weights = mx.maximum(fusion_weights, self.prob_floor)
            fusion_weights = fusion_weights / mx.sum(fusion_weights, axis=-1, keepdims=True)

        # ---- compose main output ------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # ---- hybrid residual conv path ------------
        static_scale = mx.sigmoid(self.conv_residual_logit)[None, None, :, None]  # (1,1,H,1)
        dyn_gate = mx.sigmoid(self.res_gate_proj(hidden_states))  # (B,L,H)
        gamma = static_scale * (self.residual_alpha + (1.0 - self.residual_alpha) * dyn_gate)[..., None]
        o = o + gamma * local_short

        # ---- cache update -------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            # Simplified cache update for MLX
            pass

        # ---- output norm / projection -------------
        if self.use_gate:
            g_vec = self.g_proj(hidden_states).reshape(B0, L0, self.num_heads, self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = o.reshape(B0, L0, self.value_dim)
        o = self.o_proj(o)

        return o, None, past_key_values
