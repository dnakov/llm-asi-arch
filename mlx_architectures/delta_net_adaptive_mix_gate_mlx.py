# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import Optional, Dict, Tuple
import mlx.core as mx
import mlx.nn as nn


# -----------------------------------------------------------------------------
# NOTE:
# This file contains an evolved version of DeltaNet.  The main innovation is an
# "adaptive mixing gate" that learns, for every token and head, how much of the
# newly-computed delta-rule output should be trusted versus the freshly computed
# value vector coming from the current time-step.  Empirically, such per-token
# adaptive residual connections have been shown to improve length generalisation
# and stabilise optimisation, while incurring negligible computation overhead.
# -----------------------------------------------------------------------------

# MLX Utility Functions
def rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if pattern == "b l (h d) -> b l h d":
        b, l, hd = tensor.shape
        h = kwargs.get('h', hd // kwargs.get('d', 1))
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif pattern == "b l h d -> b l (h d)":
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif pattern == "b l h d -> b h l d":
        return tensor.transpose(0, 2, 1, 3)
    elif pattern == "b h l d -> b l h d":
        return tensor.transpose(0, 2, 1, 3)
    elif pattern == "b l h -> b h l":
        return tensor.transpose(0, 2, 1)
    elif pattern == "b h (n c) d -> b h n c d":
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif pattern == "b h n c d -> b h (n c) d":
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    elif pattern == "b l h -> b l h 1":
        return mx.expand_dims(tensor, axis=-1)
    elif pattern == "b t h d -> b t (h d)":
        b, t, h, d = tensor.shape
        return tensor.reshape(b, t, h * d)
    elif pattern == "... (h d) -> ... h d":
        *leading_dims, hd = tensor.shape
        h = kwargs.get('h', hd // kwargs.get('d', 1))
        d = kwargs.get('d', hd // h)
        return tensor.reshape(*leading_dims, h, d)
    else:
        return tensor


def l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    norm = mx.maximum(norm, 1e-8)
    return x / norm


def get_unpad_data(attention_mask):
    """Simple unpad data extraction"""
    # Simplified version - just return all indices (no actual unpadding)
    # In a real implementation, this would extract valid sequence positions
    batch_size, seq_len = attention_mask.shape
    total_elements = batch_size * seq_len
    indices = mx.arange(total_elements)
    cu_seqlens = mx.array([0, seq_len])
    max_len = seq_len
    return indices, cu_seqlens, max_len


def index_first_axis(tensor: mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]


def pad_input(tensor: mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    return tensor.reshape(batch_size, seq_len, -1)


class ShortConvolution(nn.Module):
    """MLX replacement for FLA ShortConvolution"""
    
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        # MLX Conv1d: in_channels, out_channels, kernel_size
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)
        self.activation = activation
        self.kernel_size = kernel_size
        
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, D) - MLX Conv1d expects (N, L, C_in) format, so no transpose needed
        out = self.conv(x)
        # Causal truncation - remove future positions from the end
        out = out[:, :x.shape[1], :]
        
        if self.activation == 'silu':
            out = nn.silu(out)
        elif self.activation == 'gelu':
            out = nn.gelu(out)
            
        if output_final_state:
            return out, None  # Simplified - no cache state
        return out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states / mx.sqrt(variance + self.eps)
        return self.weight * hidden_states.astype(input_dtype)


def softmax(x):
    return mx.softmax(x, axis=-1)


def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Delta rule implementation identical to the original version.

    Args:
        q, k, v: (...) Same semantics as previously – see the original paper.
        beta:     (...)
        chunk_size (int): controls the window size of the parallel algorithm.
    Returns:
        o: Output tensor with identical shape to *v*.
        S: Recurrent state to be passed to the next forward call.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    # ------------------------------------------------------------------
    # Padding to an integer multiple of *chunk_size*
    # ------------------------------------------------------------------
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])

    padded_len = l + pad_len

    # ------------------------------------------------------------------
    # Normalisation & parameter preparation
    # ------------------------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    # beta shape: (b, h, l) -> (b, h, l, 1) for broadcasting with v: (b, h, l, d)
    beta_expanded = mx.expand_dims(beta, axis=-1)
    v = v * beta_expanded
    k_beta = k * beta_expanded

    # ------------------------------------------------------------------
    # Compute (I - tri(diag(beta) K K^T))^{-1}
    # ------------------------------------------------------------------
    mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=0)
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    k_beta = rearrange(k_beta, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    attn = -(k_beta @ k.transpose(0, 1, 2, 4, 3))
    attn = mx.where(mask, 0, attn)
    
    for i in range(1, chunk_size):
        attn_slice = attn[..., i, :i] + (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        # Create a new tensor with the updated slice
        attn_list = []
        for j in range(chunk_size):
            if j == i:
                # Replace row i with updated values for columns :i, keep rest unchanged
                row = mx.concatenate([attn_slice, attn[..., i, i:]], axis=-1)
                attn_list.append(row)
            else:
                attn_list.append(attn[..., j, :])
        attn = mx.stack(attn_list, axis=-2)
    
    attn = attn + mx.eye(chunk_size)

    u = attn @ v
    w = attn @ k_beta
    S = mx.zeros((b, h, d_k, d_v))
    o = mx.zeros_like(v)
    mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=1)
    
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn_i = q_i @ k_i.transpose(0, 1, 3, 2)
        attn_i = mx.where(mask, 0, attn_i)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        # MLX doesn't have JAX-style .at[].set(), use list-based reconstruction
        o_new = o_inter + attn_i @ u_i
        o_list = []
        for chunk_idx in range(o.shape[2]):
            if chunk_idx == i:
                o_list.append(o_new)
            else:
                o_list.append(o[:, :, chunk_idx])
        o = mx.stack(o_list, axis=2)
        S = S + k_i.transpose(0, 1, 3, 2) @ u_i

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S


def elu_p1(x):
    return nn.elu(x, alpha=1.0) + 1.0


def sum_norm(x):
    return x / mx.sum(x, axis=-1, keepdims=True)


class DeltaNet(nn.Module):
    """DeltaNet with Adaptive Mixing Gate (AMG).

    The adaptive gate decides, per-token and per-head, whether to rely on the
    newly computed *delta-rule* output or to fall back to the instantaneous
    value vector.  This improves length generalisation by letting the network
    skip recurrent accumulation when it is detrimental (e.g. on very long
    contexts) while keeping the strong associative recall abilities when
    beneficial.
    """

    def __init__(
        self,
        mode: str = 'chunk1',
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
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        use_mix_gate: bool = True,  # NEW: adaptive mixing gate enabled by default
        **kwargs,
    ):
        super().__init__()
        self.mode = mode

        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_mix_gate = use_mix_gate

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        assert self.key_dim % num_heads == 0, (
            f"key dim must be divisible by num_heads of {num_heads}")
        assert self.value_dim % num_heads == 0, (
            f"value dim must be divisible by num_heads of {num_heads}")

        # ------------------------------------------------------------------
        # Projection layers
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Adaptive mixing gate projection (per-token, per-head scalar in [0,1])
        if self.use_mix_gate:
            self.mix_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # ------------------------------------------------------------------
        # Beta projection (forget gate from the original DeltaNet paper)
        # ------------------------------------------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # ------------------------------------------------------------------
        # Convolutional enhancement for local patterns
        # ------------------------------------------------------------------
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu',
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.")

        # ------------------------------------------------------------------
        # Output normalisation / gating
        # ------------------------------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Dict]]:
        # ------------------------------------------------------------------
        # 1. Input validation & unpadding
        # ------------------------------------------------------------------
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed.")

        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        # Simplified attention mask handling for MLX
        if attention_mask is not None:
            # For now, just apply mask as multiplicative factor rather than unpadding
            # This is a simplification - full unpadding would require more complex logic
            mask_expanded = mx.expand_dims(attention_mask, axis=-1)
            hidden_states = hidden_states * mask_expanded

        # ------------------------------------------------------------------
        # 2. Projections + optional short convolution
        # ------------------------------------------------------------------
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = (None, None, None)
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state.get('conv_state', (None, None, None))

            q_output = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            if use_cache:
                q, conv_state_q = q_output
            else:
                q = q_output

            k_output = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            if use_cache:
                k, conv_state_k = k_output
            else:
                k = k_output

            v_output = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            if use_cache:
                v, conv_state_v = v_output
            else:
                v = v_output
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = nn.silu(q), nn.silu(k)
            v = nn.silu(self.v_proj(hidden_states))

        # Save *token-local* value representation for gating later (b, l, h, d)
        v_token = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # ------------------------------------------------------------------
        # 3. Activation + normalisation for q/k, plus reshape to heads
        # ------------------------------------------------------------------
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        
        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            else:
                raise NotImplementedError

        if self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)

        # ------------------------------------------------------------------
        # 4. Beta gate preparation
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # 5. Delta-rule core computation (chunk-wise, causal)
        # ------------------------------------------------------------------
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v_for_delta = rearrange(v_token, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')

        recurrent_state = last_state.get('recurrent_state') if last_state is not None else None
        # Note: recurrent_state is returned but not used inside delta_rule_chunkwise;
        # preserved for API compatibility.
        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v_for_delta, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        # ------------------------------------------------------------------
        # 6. NEW: Adaptive mixing between delta output and instantaneous value
        # ------------------------------------------------------------------
        if self.use_mix_gate:
            mix_gate = mx.sigmoid(self.mix_proj(hidden_states))  # shape: (b, l, h)
            mix_gate = rearrange(mix_gate, 'b l h -> b l h 1')
            # Blend outputs – keep shapes identical
            o = mix_gate * o + (1.0 - mix_gate) * v_token

        # ------------------------------------------------------------------
        # 7. Update cache (if any)
        # ------------------------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None:
            if self.layer_idx >= len(past_key_values):
                past_key_values.extend([{}] * (self.layer_idx - len(past_key_values) + 1))
            
            past_key_values[self.layer_idx] = {
                'recurrent_state': recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                'offset': seq_len,
            }

        # ------------------------------------------------------------------
        # 8. Optional gating + normalisation
        # ------------------------------------------------------------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            # Simplified gating for MLX - just apply element-wise multiplication
            o = self.o_norm(o * g)
        else:
            o = self.o_norm(o)

        # ------------------------------------------------------------------
        # 9. Final projection back to model dimension
        # ------------------------------------------------------------------
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        # ------------------------------------------------------------------
        # 10. Apply attention mask to output (simplified approach)
        # ------------------------------------------------------------------
        if attention_mask is not None:
            mask_expanded = mx.expand_dims(attention_mask, axis=-1)
            o = o * mask_expanded

        return o, None, past_key_values