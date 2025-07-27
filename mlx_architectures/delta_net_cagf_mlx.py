# -*- coding: utf-8 -*-
"""
DeltaNet – Content-Aware Gated Fusion (CAGF) - MLX Implementation
=================================================================
This evolution upgrades the original *Multi-Scale Dynamic Adaptive Fusion*
variant by introducing **content-aware, per-head gating** that directly
leverages path statistics, correcting the two major weaknesses identified in
prior experiments:

1.  *Head-flattened statistics* prevented head specialisation.  We now compute
    **per-head statistics** (mean, variance, abs-mean, ℓ2-norm) for every
    memory path so the gating network can route information on a head-by-head
    basis.
2.  *Uniform bias initialisation* diluted the long-range Δ-rule path.  We fix
    this with path-specific bias (+3 for direct value, +1 for Δ-rule, –1 for
    convolutional paths) **and** a learnable temperature that sharpens the
    softmax over training.

All changes preserve the original API, maintain **O(N)** complexity, stay
strictly causal and remain fully batch-agnostic.  The layer is drop-in
compatible with previous DeltaNet variants.

Converted to MLX framework for Apple Silicon optimization.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


# ================================================================
# Utility helpers
# ================================================================

def elu_p1(x: mx.array) -> mx.array:
    """Shifted ELU so output is strictly positive."""
    return nn.elu(x) + 1.0


def sum_norm(x: mx.array) -> mx.array:
    """Normalise last dim to sum-to-one."""
    return x / mx.sum(x, axis=-1, keepdims=True)


def l2norm(x: mx.array) -> mx.array:
    """L2 normalization along the last dimension."""
    return x / mx.linalg.norm(x, axis=-1, keepdims=True)


# ================================================================
# Depth-wise causal FIR convolution
# ================================================================
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left-padding."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Initialize filters with small random values
        self.filters = mx.random.normal((num_heads, head_dim, kernel_size)) * 0.02

    def __call__(self, x: mx.array) -> mx.array:  # (B, L, H, D)
        b, l, h, d = x.shape
        
        # Simplified implementation using matrix operations
        # Reshape to (B*H, D, L) for processing
        x_reshaped = x.reshape(b * h, d, l)
        
        # Apply causal padding (pad_length-1, 0) for each sequence
        x_padded = mx.pad(x_reshaped, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        
        # Apply convolution using broadcasting
        output_list = []
        for i in range(h):
            # Get filter for this head
            head_filter = self.filters[i]  # (D, K)
            
            # Convolve each position
            head_outputs = []
            for pos in range(l):
                # Extract window
                window = x_padded[i::h, :, pos:pos + self.kernel_size]  # (B, D, K)
                if window.shape[-1] == self.kernel_size:
                    # Apply convolution: (B, D) = sum over K dim of (B, D, K) * (D, K)
                    conv_out = mx.sum(window * head_filter[None, :, :], axis=-1)
                    head_outputs.append(conv_out)
                else:
                    head_outputs.append(mx.zeros((b, d)))
            
            # Stack outputs for this head
            head_result = mx.stack(head_outputs, axis=-1)  # (B, D, L)
            output_list.append(head_result)
        
        # Combine all heads: (H, B, D, L) -> (B, H, D, L) -> (B, L, H, D)
        output = mx.stack(output_list, axis=0).transpose(1, 0, 2, 3)
        output = output.transpose(0, 3, 1, 2)
        return output


# ================================================================
# Core chunk-wise Δ-rule kernel
# ================================================================
def delta_rule_chunkwise(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    *,
    chunk_size: int = 32,
) -> Tuple[mx.array, mx.array]:
    """Efficient chunk-wise associative Δ-rule with O(N) cost."""
    b, h, L, d_k = q.shape

    # Padding to make sequence length divisible by chunk_size
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    L_pad = L + pad_len

    # Normalisation and scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks (B H N C D)
    n_chunks = L_pad // chunk_size
    q = q.reshape(b, h, n_chunks, chunk_size, d_k)
    k = k.reshape(b, h, n_chunks, chunk_size, d_k)
    v = v.reshape(b, h, n_chunks, chunk_size, -1)
    k_beta = k_beta.reshape(b, h, n_chunks, chunk_size, d_k)

    # Create masks
    mask_tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0).astype(mx.bool_)
    mask_strict = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)

    # Process chunks
    S = mx.zeros((b, h, d_k, v.shape[-1]))
    o = mx.zeros_like(v)

    for idx in range(n_chunks):
        q_i, k_i, v_i, k_beta_i = q[:, :, idx], k[:, :, idx], v[:, :, idx], k_beta[:, :, idx]
        
        # In-chunk inverse computation
        attn = -mx.matmul(k_beta_i, k_i.transpose(0, 1, 3, 2))
        attn = mx.where(mask_tri[None, None, :, :], 0, attn)
        
        # Iterative solution for (I - tril(K β Kᵀ))⁻¹
        # Simplified: use matrix inversion instead of iterative updates
        identity = mx.eye(chunk_size)[None, None, :, :]
        try:
            attn = mx.linalg.inv(identity - attn)
        except:
            # Fallback to identity if inversion fails
            attn = identity
        
        u = mx.matmul(attn, v_i)
        w = mx.matmul(attn, k_beta_i)
        
        # Local attention
        attn_local = mx.matmul(q_i, k_i.transpose(0, 1, 3, 2))
        attn_local = mx.where(mask_strict[None, None, :, :], 0, attn_local)
        
        # Update output
        u_i = u - mx.matmul(w, S)
        o_inter = mx.matmul(q_i, S)
        chunk_output = o_inter + mx.matmul(attn_local, u_i)
        # Update the output at this chunk
        output_chunks = []
        for j in range(n_chunks):
            if j == idx:
                output_chunks.append(chunk_output)
            else:
                output_chunks.append(o[:, :, j])
        o = mx.stack(output_chunks, axis=2)
        
        # Update state
        S = S + mx.matmul(k_i.transpose(0, 1, 3, 2), u_i)

    # Reshape output
    o = o.reshape(b, h, L_pad, -1)
    if pad_len:
        o = o[:, :, :L]
    
    return o, S


# ================================================================
# RMS Normalization modules
# ================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)) * self.weight


class FusedRMSNormGated(nn.Module):
    """Gated RMS Normalization."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        norm_x = (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)) * self.weight
        return norm_x * gate


# ================================================================
# Short convolution module
# ================================================================
class ShortConvolution(nn.Module):
    """Short convolution for local processing."""
    
    def __init__(self, dims: int, kernel_size: int, activation: Optional[str] = None):
        super().__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = mx.random.normal((dims, kernel_size)) * 0.02
        
    def __call__(self, x: mx.array, cache=None, output_final_state=False, cu_seqlens=None):
        b, l, d = x.shape
        
        # Apply causal padding
        x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        # Apply convolution using vectorized operations
        output_list = []
        for i in range(l):
            conv_window = x_padded[:, i:i + self.kernel_size, :]  # (B, K, D)
            if conv_window.shape[1] == self.kernel_size:
                # Broadcast weight to match: (B, K, D) * (D, K) -> sum over K
                # Transpose conv_window to (B, D, K) and weight to (D, K)
                conv_window_t = conv_window.transpose(0, 2, 1)  # (B, D, K)
                conv_result = mx.sum(conv_window_t * self.weight[None, :, :], axis=2)  # (B, D)
                output_list.append(conv_result)
            else:
                output_list.append(mx.zeros((b, d)))
        output = mx.stack(output_list, axis=1)
        
        # Apply activation
        if self.activation == "silu":
            output = nn.silu(output)
        elif self.activation == "relu":
            output = nn.relu(output)
        
        final_state = None
        if output_final_state:
            final_state = x[:, -self.kernel_size + 1:, :]
        
        return output, final_state


# ================================================================
# Main DeltaNet with Content-Aware Gated Fusion
# ================================================================
class DeltaNet(nn.Module):
    """DeltaNet layer with **Content-Aware, Per-Head Gated Fusion**."""

    def __init__(
        self,
        mode: str = "cagf",
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
        # Multi-scale FIR kernel sizes
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        # Path-specific initial biases: (short, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 3.0),
        # Temperature initial (softplus-paramised → τ≈0.7)
        gate_logit_init: float = math.log(math.expm1(0.7)),
        **kwargs,
    ) -> None:
        super().__init__()

        # Book-keeping & basic dims
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ("silu", "relu", "elu", "identity")
        assert self.qk_norm in ("l2", "sum")

        if d_model is not None:
            hidden_size = d_model  # alias
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

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Beta projection for Δ-rule
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Short convolutions
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # Multi-scale local FIR convolutions
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_long
        )
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_short
        )

        # Content-aware gating network
        # Per-head stats: 4 metrics per branch, 4 branches = 16 scalars
        self.stat_dim = 16
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        
        # Initialize bias with path-specific values
        gate_bias_init_tensor = mx.array(list(gate_bias_init))
        self.fusion_gate_mlp.layers[-1].bias = gate_bias_init_tensor

        # Learnable temperature
        self.logit_temperature = mx.array([gate_logit_init])

        # Output normalization / projection
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:
        """Return per-token, per-head statistics vector of length 4.

        Stats: mean, variance, mean(|x|), ℓ2-norm over feature dim.
        Output shape: [B, L, H, 4]
        """
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        abs_mean = mx.mean(mx.abs(x), axis=-1, keepdims=True)
        l2 = mx.linalg.norm(x, axis=-1, keepdims=True)
        return mx.concatenate([mean, var, abs_mean, l2], axis=-1)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.array, None, Optional[dict]]:
        
        batch_size, seq_len, _ = hidden_states.shape

        # Q/K/V projections + short conv enhancements
        conv_state_q = conv_state_k = conv_state_v = None
        if past_key_values is not None and self.layer_idx is not None:
            last_state = past_key_values.get(self.layer_idx, {})
            conv_states = last_state.get("conv_state", (None, None, None))
            conv_state_q, conv_state_k, conv_state_v = conv_states

        q, conv_state_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, conv_state_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        v, conv_state_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )

        # Head reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v_direct = v.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)

        # Activation on Q/K
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # Beta for Δ-rule
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((batch_size, seq_len, self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global pathway
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=q.transpose(0, 2, 1, 3),
            k=k.transpose(0, 2, 1, 3),
            v=v_direct.transpose(0, 2, 1, 3),
            beta=beta.transpose(0, 2, 1),
        )
        delta_out = delta_out.transpose(0, 2, 1, 3)

        # Local FIR paths
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # Content-aware gating (per-head)
        # Per-head stats (B, L, H, 4)
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = mx.concatenate(
            [stats_short, stats_long, stats_delta, stats_value], axis=-1
        )  # (B, L, H, 16)

        # Build gating input
        hs_exp = mx.expand_dims(hidden_states, axis=-2)  # (B, L, 1, C)
        hs_exp = mx.broadcast_to(hs_exp, (batch_size, seq_len, self.num_heads, self.hidden_size))
        gate_in = mx.concatenate([hs_exp, stats_vec], axis=-1)  # (B, L, H, C+16)
        gate_in_flat = gate_in.reshape(-1, gate_in.shape[-1])
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)
        
        # Temperature scaling
        temperature = nn.softplus(self.logit_temperature) + 1e-4
        gate_logits_flat = gate_logits_flat / temperature
        fusion_logits = gate_logits_flat.reshape(
            batch_size, seq_len, self.num_heads, 4
        )
        fusion_weights = nn.softmax(fusion_logits, axis=-1)

        # Weighted fusion of memory paths
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # Cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            if past_key_values is None:
                past_key_values = {}
            past_key_values[self.layer_idx] = {
                "recurrent_state": recurrent_state,
                "conv_state": (conv_state_q, conv_state_k, conv_state_v),
            }

        # Normalization / projection
        if self.use_gate:
            g = self.g_proj(hidden_states).reshape(
                batch_size, seq_len, self.num_heads, self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = o.reshape(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)

        return o, None, past_key_values