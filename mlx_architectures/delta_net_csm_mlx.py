# -*- coding: utf-8 -*-
"""
DeltaNet – Hybrid Content-Sharp Multi-Scale Memory (CSM) - MLX Implementation
=======================================================
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict
import mlx.core as mx
import mlx.nn as nn

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """MLX implementation of einops rearrange"""
    if pattern == "b l h d -> b (h d) l":
        b, l, h, d = tensor.shape
        return tensor.transpose(0, 2, 3, 1).reshape(b, h * d, l)
    elif pattern == "h d k -> (h d) 1 k":
        h, d, k = tensor.shape
        return tensor.reshape(h * d, 1, k)
    elif pattern == "b (h d) l -> b l h d":
        b, hd, l = tensor.shape
        h = kwargs.get('h', hd // kwargs.get('d', 1))
        d = hd // h
        return tensor.reshape(b, h, d, l).transpose(0, 3, 1, 2)
    elif pattern == "... (h d) -> ... h d":
        *dims, hd = tensor.shape
        d = kwargs.get('d')
        h = hd // d
        return tensor.reshape(*dims, h, d)
    elif pattern == "b s d -> (b s) d":
        b, s, d = tensor.shape
        return tensor.reshape(b * s, d)
    elif pattern == "b l h d -> b h l d":
        return tensor.transpose(0, 2, 1, 3)
    elif pattern == "b h l d -> b l h d":
        return tensor.transpose(0, 2, 1, 3)
    elif pattern == "b l h d -> b l (h d)":
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif pattern == "b h (n c) d -> b h n c d":
        b, h, nc, d = tensor.shape
        c = kwargs.get('c')
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif pattern == "b h n c d -> b h (n c) d":
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1, keepdims=True)

def elu_p1(x: mx.array) -> mx.array:
    """ELU + 1 (always positive)"""
    return nn.elu(x) + 1.0

def sum_norm(x: mx.array) -> mx.array:
    """Normalise vectors so that they sum to 1 along the last dim"""
    return x / mx.sum(x, axis=-1, keepdims=True)

def linear_decay(step: int, start: int, end: int) -> float:
    """Linear decay from 1.0 at start to 0.0 at end (clamped)"""
    if step < start:
        return 1.0
    if step > end:
        return 0.0
    return 1.0 - (step - start) / float(end - start)

def _get_unpad_data(attention_mask: mx.array):
    """Get unpadding data from attention mask"""
    seqlens = mx.sum(attention_mask, axis=1)
    indices = mx.arange(attention_mask.shape[0] * attention_mask.shape[1])
    cu_seqlens = mx.concatenate([mx.array([0]), mx.cumsum(seqlens)])
    return indices, cu_seqlens, seqlens.max()

def _index_first_axis(tensor: mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor: mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    return tensor.reshape(batch_size, seq_len, -1)

@mx.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Causal chunk-wise Δ-rule (unchanged, proven implementation)"""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    
    L_pad = L + pad_len
    
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * mx.expand_dims(beta, -1)
    k_beta = k * mx.expand_dims(beta, -1)
    
    q = _rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size)
    k = _rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = _rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    k_beta = _rearrange(k_beta, "b h (n c) d -> b h n c d", c=chunk_size)
    
    mask_tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)
    
    att_inv = mx.eye(chunk_size) - (k_beta @ mx.transpose(k, [0, 1, 2, 4, 3]))
    att_inv = mx.where(mask_tri, 0, att_inv)
    
    u = att_inv @ v
    w = att_inv @ k_beta
    
    S = mx.zeros((b, h, d_k, v.shape[-1]))
    o = mx.zeros_like(v)
    
    for idx in range(L_pad // chunk_size):
        q_i = q[:, :, idx]
        k_i = k[:, :, idx]
        
        attn_local = q_i @ mx.transpose(k_i, [0, 1, 3, 2])
        attn_local = mx.where(mask_tri, 0, attn_local)
        
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o = o.at[:, :, idx].set(q_i @ S + attn_local @ u_i)
        S = S + mx.transpose(k_i, [0, 1, 3, 2]) @ u_i
    
    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len > 0:
        o = o[:, :, :L]
    
    return o, S

class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution"""
    
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        filters = mx.random.normal((num_heads, head_dim, self.kernel_size)) * 0.02
        self.filters = filters

    def __call__(self, x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        w = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        x_pad = mx.pad(x_f, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        
        # Simplified depthwise convolution
        y = mx.zeros((b, h * d, l))
        for i in range(h * d):
            for j in range(l):
                start_idx = j
                end_idx = j + self.kernel_size
                y = y.at[..., i, j].set(
                    mx.sum(x_pad[..., i, start_idx:end_idx] * w[i, 0, :])
                )
        
        return _rearrange(y, "b (h d) l -> b l h d", h=h)

class ContentSharpGate(nn.Module):
    """Per-Head Temperature-Sharpened Content Gate"""
    
    def __init__(self, hidden_size: int, num_heads: int, fusion_hidden_mult: int = 2):
        super().__init__()
        self.num_heads = num_heads
        
        # Input: hidden_states + 8 statistics per head (mean/std for 4 paths)
        gate_in_dim = hidden_size + num_heads * 8
        fusion_hidden_dim = fusion_hidden_mult * num_heads * 4
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, num_heads * 4, bias=True),
        )
        
        # Per-head temperature parameters (log-space)
        self.log_temp = mx.zeros(num_heads)

    def __call__(self, hidden_states: mx.array, path_stats: mx.array) -> mx.array:
        # path_stats: (B, L, H, 8) - mean/std for each of 4 paths
        b, l, h, _ = path_stats.shape
        
        # Flatten path stats
        path_stats_flat = path_stats.reshape(b, l, h * 8)
        
        # Concatenate hidden states with path statistics
        gate_in = mx.concatenate([hidden_states, path_stats_flat], axis=-1)
        
        # Get logits
        fusion_logits = self.gate_mlp(gate_in)
        fusion_logits = fusion_logits.reshape(b, l, h, 4)
        
        # Apply per-head temperature
        temp = mx.exp(self.log_temp) + 0.1  # Minimum temperature
        fusion_logits = fusion_logits / temp.reshape(1, 1, -1, 1)
        
        # Softmax
        fusion_weights = nn.softmax(fusion_logits, axis=-1)
        
        return fusion_weights

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x / mx.sqrt(variance + self.eps)
        return self.weight * x

class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x / mx.sqrt(variance + self.eps)
        return self.weight * x * gate

class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)

    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        x_conv = x.transpose(0, 2, 1)
        y = self.conv(x_conv)
        y = y[:, :, :x.shape[1]]
        y = y.transpose(0, 2, 1)
        
        if self.activation == "silu":
            y = nn.silu(y)
        
        final_state = None if not output_final_state else y[:, -self.kernel_size+1:]
        return y, final_state

class DeltaNet(nn.Module):
    """DeltaNet with Hybrid Content-Sharp Multi-Scale Memory"""
    
    def __init__(
        self,
        mode: str = "csm",
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
        fir_kernel_size_long: int = 31,
        fir_kernel_size_short: int = 3,
        fusion_hidden_mult: int = 2,
        entropy_reg_alpha: float = 0.02,
        entropy_reg_start: int = 0,
        entropy_reg_end: int = 30000,
        **kwargs,
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
        self.fir_kernel_size_short = fir_kernel_size_short
        self.fir_kernel_size_long = fir_kernel_size_long
        self.fusion_hidden_mult = fusion_hidden_mult
        self.entropy_reg_alpha = entropy_reg_alpha
        self.entropy_reg_start = entropy_reg_start
        self.entropy_reg_end = entropy_reg_end
        
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # Short convolutions
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        
        # FIR filters
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)
        
        # Content-sharp gating network
        self.content_sharp_gate = ContentSharpGate(hidden_size, num_heads, fusion_hidden_mult)
        
        # Output layers
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        # Training step counter for entropy regularization
        self.training_step = 0

    def _compute_path_stats(self, x: mx.array) -> mx.array:
        """Compute dual-statistic path features (mean, std) for each head"""
        # x: (B, L, H, D)
        mean = mx.mean(x, axis=-1, keepdims=True)  # (B, L, H, 1)
        var = mx.var(x, axis=-1, keepdims=True)    # (B, L, H, 1)
        std = mx.sqrt(var + 1e-8)                  # (B, L, H, 1)
        
        return mx.concatenate([mean, std], axis=-1)  # (B, L, H, 2)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> mx.array:
        
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        
        batch_size, seq_len, _ = hidden_states.shape
        
        last_state = None
        if past_key_values is not None and self.layer_idx is not None:
            last_state = past_key_values.get(self.layer_idx)
        
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).reshape(1, -1, hidden_states.shape[-1])
        
        # Convolutions
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        
        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        
        # Split into heads
        q = _rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = _rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = _rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        
        # Activations and normalization
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)
        elif self.qk_norm == "l2":
            q, k = _l2norm(q), _l2norm(k)
        
        v_direct = v
        
        # Beta gate
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones_like(q[..., 0])
        
        if self.allow_neg_eigval:
            beta = beta * 2.0
        
        # Delta rule path
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")
        
        # FIR paths
        fir_short = self.local_fir_short(v_direct)
        fir_long = self.local_fir_long(v_direct)
        
        # Compute path statistics
        fir_short_stats = self._compute_path_stats(fir_short)
        fir_long_stats = self._compute_path_stats(fir_long)
        delta_stats = self._compute_path_stats(delta_out)
        direct_stats = self._compute_path_stats(v_direct)
        
        # Concatenate all path statistics
        path_stats = mx.concatenate([
            fir_short_stats, fir_long_stats, delta_stats, direct_stats
        ], axis=-1)  # (B, L, H, 8)
        
        # Content-sharp gating
        fusion_weights = self.content_sharp_gate(hidden_states, path_stats)
        
        # Path fusion
        o = (
            mx.expand_dims(fusion_weights[..., 0], -1) * fir_short +
            mx.expand_dims(fusion_weights[..., 1], -1) * fir_long +
            mx.expand_dims(fusion_weights[..., 2], -1) * delta_out +
            mx.expand_dims(fusion_weights[..., 3], -1) * v_direct
        )
        
        # Output projection
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        
        # Re-pad if needed
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, seq_len)
        
        return o