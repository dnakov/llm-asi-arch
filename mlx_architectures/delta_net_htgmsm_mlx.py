# -*- coding: utf-8 -*-
"""
DeltaNet – Hierarchical Two-Stage Gated Multi-Scale Memory (HTG-MSM)
===================================================================
Identifier: delta_net_htgmsm

MLX Implementation of the hierarchical two-stage gated multi-scale memory architecture.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict

import mlx.core as mx
import mlx.nn as nn


################################################################################
# Helper functions                                                             #
################################################################################

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h', 1)
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
    elif "b l h -> b h l" in pattern:
        return tensor.transpose(0, 2, 1)
    elif "b h l -> b l h" in pattern:
        return tensor.transpose(0, 2, 1)
    elif "b h (n c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h (n c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    elif "b l (h k) -> b l h k" in pattern:
        h = kwargs.get('h', 1)
        k = kwargs.get('k', 1)
        b, l, hk = tensor.shape
        return tensor.reshape(b, l, h, k)
    elif "b s d -> (b s) d" in pattern:
        b, s, d = tensor.shape
        return tensor.reshape(b * s, d)
    else:
        # Fallback: return tensor as-is
        return tensor

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    return x / mx.maximum(norm, mx.array(1e-8))

def _masked_fill(tensor: mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
    cu_seqlens = mx.array([0, attention_mask.shape[-1]])
    max_len = attention_mask.shape[-1]
    return indices, cu_seqlens, max_len

def _index_first_axis(tensor: mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor: mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    # Simplified version
    return tensor.reshape(batch_size, seq_len, -1)

class _ShortConvolution(nn.Module):
    """MLX replacement for FLA ShortConvolution using depthwise convolution"""
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        
        # Use a simple linear transformation as a simplified convolution
        # In practice, this is often sufficient for short convolutions
        self.weight = mx.random.normal((hidden_size, kernel_size)) * 0.02
        if bias:
            self.bias = mx.zeros((hidden_size,))
        else:
            self.bias = None
        
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Simple causal 1D convolution implementation
        # Pad input for causal convolution
        x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        # Manual convolution
        output_list = []
        for i in range(L):
            # Extract window
            window = x_padded[:, i:i+self.kernel_size, :]  # (B, kernel_size, D)
            # Apply convolution (sum over kernel dimension)
            conv_out = mx.sum(window * self.weight.T[None, :, :], axis=1)  # (B, D)
            if self.bias is not None:
                conv_out = conv_out + self.bias
            output_list.append(mx.expand_dims(conv_out, 1))
        
        out = mx.concatenate(output_list, axis=1)  # (B, L, D)
        
        if self.activation == 'silu':
            out = nn.silu(out)
        elif self.activation == 'gelu':
            out = nn.gelu(out)
            
        if output_final_state:
            return out, None  # Simplified - no cache state
        return out

def _elu_p1(x: mx.array) -> mx.array:
    """ELU+1 (RetNet / Hyena convention – keeps positives)."""
    return nn.elu(x) + 1.0

def _sum_norm(x: mx.array) -> mx.array:
    """L1 normalise along last dim (used as optional q/k normalisation)."""
    return x / mx.sum(x, axis=-1, keepdims=True)

################################################################################
# O(N) chunk-wise kernels (Δ-rule & EMA)                                       #
################################################################################

def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Simplified delta rule implementation for MLX."""
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # Normalize q and k
    q = _l2norm(q)
    k = _l2norm(k)
    
    # Initialize state
    S = mx.zeros((b, h, d_k, d_v))
    output_list = []
    
    # Process each time step
    for i in range(l):
        q_i = q[:, :, i]  # (b, h, d_k)
        k_i = k[:, :, i]  # (b, h, d_k)
        v_i = v[:, :, i]  # (b, h, d_v)
        beta_i = beta[:, :, i]  # (b, h)
        
        # Compute output for this step
        o_i = mx.sum(q_i[:, :, :, None] * S, axis=2)  # (b, h, d_v)
        output_list.append(mx.expand_dims(o_i, 2))
        
        # Update state with beta scaling
        k_scaled = k_i * mx.expand_dims(beta_i, -1)  # (b, h, d_k)
        v_scaled = v_i * mx.expand_dims(beta_i, -1)  # (b, h, d_v)
        S = S + mx.expand_dims(k_scaled, -1) * mx.expand_dims(v_scaled, 2)
    
    output = mx.concatenate(output_list, axis=2)  # (b, h, l, d_v)
    return output, S

def ema_rule_chunkwise(v: mx.array,  # (b h l d_v)
                        gamma: mx.array,  # (b h l)
                        init_state: Optional[mx.array] = None):
    """Chunk-wise causal EMA (stateful) – O(N d)."""
    b, h, l, d_v = v.shape
    ema_out_list = []
    state = mx.zeros((b, h, d_v)) if init_state is None else init_state
    for t in range(l):
        g_t = mx.expand_dims(gamma[:, :, t], -1)
        state = g_t * state + (1.0 - g_t) * v[:, :, t]
        ema_out_list.append(mx.expand_dims(state, 2))
    ema_out = mx.concatenate(ema_out_list, axis=2)
    return ema_out, state

################################################################################
# Hierarchical two-stage gate                                                  #
################################################################################

class HierarchicalGate(nn.Module):
    """Per-token, per-head hierarchical gate producing weights for 4 paths.

    Stage-1 (coarse): Local vs Global  → probabilities p_L, p_G.
    Stage-2 (fine)  : within each group (2 paths each) producing q_V, q_Es
                      and r_Δ, r_El respectively.
    Final weights   : [V, Es, Δ, El] = [p_L*q_V, p_L*q_Es, p_G*r_Δ, p_G*r_El]
    """

    def __init__(self, hidden_size: int, num_heads: int, temp_init: float = 1.0):
        super().__init__()
        self.num_heads = num_heads

        # Shared trunk MLP (lightweight)
        hid = max(8, hidden_size // 2)
        self.trunk_linear = nn.Linear(hidden_size, hid)
        # Output projections
        self.coarse_proj = nn.Linear(hid, num_heads * 2)   # Local / Global
        self.local_proj  = nn.Linear(hid, num_heads * 2)    # V / Es
        self.global_proj = nn.Linear(hid, num_heads * 2)    # Δ / El

        # Bias initialisation following curriculum insights
        # Coarse gate bias to favor Local initially
        self.coarse_proj.bias = mx.ones((num_heads * 2,))
        
        # local fine-gate bias: favour V
        bias_local = mx.zeros((num_heads * 2,))
        bias_local_new = []
        for i in range(num_heads * 2):
            if i % 2 == 0:  # path-0 (V) has +1
                bias_local_new.append(1.0)
            else:
                bias_local_new.append(0.0)
        self.local_proj.bias = mx.array(bias_local_new)
        
        # global fine-gate bias: favour Δ
        bias_global = mx.zeros((num_heads * 2,))
        bias_global_new = []
        for i in range(num_heads * 2):
            if i % 2 == 0:  # path-0 (Δ) has +1
                bias_global_new.append(1.0)
            else:
                bias_global_new.append(0.0)
        self.global_proj.bias = mx.array(bias_global_new)

        # Learnable per-head temperature (>0) for both stages
        self.log_temp_coarse = mx.log(mx.array(temp_init)) * mx.ones((num_heads,))
        self.log_temp_fine   = mx.log(mx.array(temp_init)) * mx.ones((num_heads,))

    def _softmax_h(self, logits: mx.array, temp: mx.array):
        # logits: (b l h k), temp:(h,) – broadcast along (b,l)
        logits = logits / temp.reshape(1, 1, -1, 1)
        return mx.softmax(logits, axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        """Return gate weights with shape (b, l, h, 4) in order [V, Es, Δ, El]."""
        b, l, _ = x.shape
        h = self.num_heads
        z = nn.silu(self.trunk_linear(x))  # (b, l, hid)

        # ---- Stage-1: coarse Local/Global ----
        coarse_logits = _rearrange(self.coarse_proj(z), "b l (h k) -> b l h k", h=h, k=2)
        temp_c = nn.softplus(self.log_temp_coarse) + 1e-4
        pg = self._softmax_h(coarse_logits, temp_c)  # (b l h 2)
        p_local, p_global = pg[..., 0:1], pg[..., 1:2]  # keep last dim size=1 for broadcasting

        # ---- Stage-2: fine gates ----
        local_logits = _rearrange(self.local_proj(z), "b l (h k) -> b l h k", h=h, k=2)
        global_logits = _rearrange(self.global_proj(z), "b l (h k) -> b l h k", h=h, k=2)
        temp_f = nn.softplus(self.log_temp_fine) + 1e-4
        q = self._softmax_h(local_logits, temp_f)  # (b l h 2)
        r = self._softmax_h(global_logits, temp_f)  # (b l h 2)

        # Combine hierarchically
        w_v = p_local * q[..., 0:1]   # (b l h 1)
        w_es = p_local * q[..., 1:2]
        w_delta = p_global * r[..., 0:1]
        w_el   = p_global * r[..., 1:2]

        weights = mx.concatenate([w_v, w_es, w_delta, w_el], axis=-1)  # (b l h 4)
        return weights  # Already sums to 1 per token/head

################################################################################
# Main DeltaNet class                                                          #
################################################################################

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet layer with Hierarchical Two-Stage Gated Multi-Scale Memory."""

    def __init__(
        self,
        *,
        mode: str = "htgmsm",
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
        gate_temp_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---------------- Book-keeping ----------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.mode = mode
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

        # --------------- Dimensions ---------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # --------------- Linear projections ---------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # EMA decay projections – two distinct scales
        # NOTE: bias=True is REQUIRED here because we set biases to specific
        # values (≈logit of 0.05 / 0.95). Setting bias=False would have caused
        # an AttributeError when trying to access `.bias` and, more critically,
        # would remove the intended warm-start behaviour.
        self.dec_proj_short = nn.Linear(hidden_size, num_heads, bias=True)
        self.dec_proj_long  = nn.Linear(hidden_size, num_heads, bias=True)
        # Bias init: sigmoid(bias) ≈ γ ; want γ_s≈0.05 , γ_l≈0.95
        self.dec_proj_short.bias = mx.full((num_heads,), -2.9444)  # sigmoid ≈ 0.05
        self.dec_proj_long.bias = mx.full((num_heads,), 2.9444)    # sigmoid ≈ 0.95

        # Hierarchical gate
        self.h_gate = HierarchicalGate(hidden_size, num_heads, temp_init=gate_temp_init)

        # Short convolution (mandatory as per requirements)
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act)
            self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act)
            self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is crucial; do not disable it.")

        # Output normalisation & projection
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(
        self,
        hidden_states: mx.array,  # (B, L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.array, None, Optional[Dict]]:
        # -------- Input unpadding (optional) --------
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "attention_mask must be (batch, seq_len)."
        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = mx.expand_dims(_index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices), 0)

        # -------- Linear projections + optional conv --------
        if self.use_short_conv:
            cs_q = cs_k = cs_v = None
            if last_state is not None and last_state.get("conv_state") is not None:
                cs_q, cs_k, cs_v = last_state["conv_state"]
            
            if use_cache:
                q, cs_q = self.q_conv1d(self.q_proj(hidden_states), cache=cs_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
                k, cs_k = self.k_conv1d(self.k_proj(hidden_states), cache=cs_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
                v, cs_v = self.v_conv1d(self.v_proj(hidden_states), cache=cs_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            else:
                q = self.q_conv1d(self.q_proj(hidden_states), cache=cs_q, output_final_state=False, cu_seqlens=cu_seqlens)
                k = self.k_conv1d(self.k_proj(hidden_states), cache=cs_k, output_final_state=False, cu_seqlens=cu_seqlens)
                v = self.v_conv1d(self.v_proj(hidden_states), cache=cs_v, output_final_state=False, cu_seqlens=cu_seqlens)
                cs_q = cs_k = cs_v = None
        else:  # never reached per design
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = nn.silu(q), nn.silu(k)
            v = nn.silu(self.v_proj(hidden_states))

        # -------- Head reshape + activations --------
        q = _rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = _rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = _rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # -------- β scaling for Δ path --------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))  # (b, l, h)
        else:
            beta = mx.ones((batch_size, seq_len, self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------- Compute Δ path (chunk-wise) --------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        rec_prev = last_state.get("recurrent_state") if last_state else None
        delta_out_d, rec_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # -------- EMA paths --------
        # Short EMA
        gamma_short = mx.sigmoid(self.dec_proj_short(hidden_states))  # (b l h)
        gamma_s_d = _rearrange(gamma_short, "b l h -> b h l")
        ema_s_prev = last_state.get("ema_state_s") if last_state else None
        ema_s_d, ema_s_state = ema_rule_chunkwise(v_d, gamma_s_d, ema_s_prev)
        ema_s = _rearrange(ema_s_d, "b h l d -> b l h d")
        # Long EMA
        gamma_long = mx.sigmoid(self.dec_proj_long(hidden_states))
        gamma_l_d = _rearrange(gamma_long, "b l h -> b h l")
        ema_l_prev = last_state.get("ema_state_l") if last_state else None
        ema_l_d, ema_l_state = ema_rule_chunkwise(v_d, gamma_l_d, ema_l_prev)
        ema_l = _rearrange(ema_l_d, "b h l d -> b l h d")

        # -------- Hierarchical gating --------
        weights = self.h_gate(hidden_states)  # (b l h 4)
        w_v = mx.expand_dims(weights[..., 0], -1)
        w_es = mx.expand_dims(weights[..., 1], -1)
        w_delta = mx.expand_dims(weights[..., 2], -1)
        w_el = mx.expand_dims(weights[..., 3], -1)

        o = w_v * v + w_es * ema_s + w_delta * delta_out + w_el * ema_l  # (b l h d)

        # -------- Cache update --------
        if past_key_values is not None and use_cache:
            layer_state = {
                "recurrent_state": rec_state,
                "conv_state": (cs_q, cs_k, cs_v) if self.use_short_conv else None,
                "ema_state_s": ema_s_state,
                "ema_state_l": ema_l_state,
                "layer_idx": self.layer_idx,
                "offset": seq_len,
            }
            if hasattr(past_key_values, "__setitem__") and self.layer_idx is not None:
                past_key_values[self.layer_idx] = layer_state
            else:
                past_key_values.update(layer_state)

        # -------- Output norm / projection --------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o * g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # -------- Re-padding if needed --------
        if attention_mask is not None:
            o = _pad_input(mx.squeeze(o, 0), indices, batch_size, seq_len)

        return o, None, past_key_values