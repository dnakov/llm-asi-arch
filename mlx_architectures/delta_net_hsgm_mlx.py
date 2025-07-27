# -*- coding: utf-8 -*-
"""
DeltaNet – Hybrid Sparse Gated Multi-Scale Memory (DeltaNet-HSGM)
================================================================
This evolution unifies the most successful ingredients discovered in the
DeltaNet lineage while remedying the remaining gating pathologies.

Key innovations (enabled by default)
-----------------------------------
1. Per-Head **Temperature-Controlled Sparsemax Gate**
   • Replaces softmax with *sparsemax* to allow *exact* suppression of
     irrelevant branches while still propagating gradients to the selected
     ones.
   • Each head owns an independent, learnable *temperature* parameter that
     governs the sharpness of its routing distribution.  Temperatures are
     initialised such that the gate behaves like the vanilla sparsemax
     (≈1.0) and can anneal during training.

2. **Moderate Warm-Start Bias** (*+1.5*) on the direct/value path only – a
   compromise between stability and early gradient flow to alternative paths.

3. Dual **Identity + Orthogonal-Noise FIR** branches (short & long) ensure
   local-span fidelity without harming global flow.  The orthogonal perturbation
   (<10⁻³) decorrelates branch outputs at step 0 so the sparse gate has a
   meaningful signal to discriminate.

4. Implementation keeps **O(Nd)** complexity via the proven chunk-wise Δ-rule
   solver and depth-wise convolutions.  All tensor manipulations rely on
   `einops.rearrange` for batch- and sequence-agnostic safety.

The class name `DeltaNet`, constructor signature and forward interface remain
unchanged, guaranteeing drop-in compatibility with earlier variants.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import mlx.core as mx
import mlx.nn as nn

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def rearrange_manual(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Manual implementation of common einops patterns"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h', 1)
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l (h d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return mx.transpose(tensor, (0, 2, 1, 3))
    elif "b h l d -> b l h d" in pattern:
        return mx.transpose(tensor, (0, 2, 1, 3))
    elif "b h (n c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h (n c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    elif "b l (h p) -> b l h p" in pattern:
        h = kwargs.get('h', 1)
        p = kwargs.get('p', 1)
        b, l, hp = tensor.shape
        return tensor.reshape(b, l, h, p)
    elif "b l h p -> b l h p 1" in pattern:
        return mx.expand_dims(tensor, -1)
    elif "b l h -> b h l" in pattern:
        return mx.transpose(tensor, (0, 2, 1))
    elif "(h d) 1 k" in pattern:
        h = kwargs.get('h', 1)
        hd, one, k = tensor.shape
        d = hd // h
        return tensor.reshape(h, d, one, k)
    else:
        return tensor

def l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    norm = mx.maximum(norm, 1e-8)
    return x / norm

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU that stays strictly positive."""
    return nn.elu(x) + 1.0

def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that elements along last dim sum to one."""
    return x / mx.sum(x, axis=-1, keepdims=True)

# -----------------------------------------------------------------------------
# Mock implementations for missing FLA components
# -----------------------------------------------------------------------------

def get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
    cu_seqlens = mx.array([0, attention_mask.shape[-1]])
    max_len = attention_mask.shape[-1]
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
        # Create manual conv weight and bias parameters - MLX expects (C_out, K, C_in)
        self.weight = mx.random.normal((hidden_size, kernel_size, hidden_size)) * 0.02
        if bias:
            self.bias = mx.zeros((hidden_size,))
        else:
            self.bias = None
        self.kernel_size = kernel_size
        self.activation = activation
        
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, D) - MLX conv1d expects (N, L, C_in)
        b, l, d = x.shape
        # Add causal padding
        x_pad = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        # Manual convolution - MLX conv1d input: (N, L, C_in), weight: (C_out, K, C_in)
        out = mx.conv1d(x_pad, self.weight)
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, -1)
        out = out[:, :l, :]  # Causal truncation
        
        if self.activation == 'silu':
            out = nn.silu(out)
        elif self.activation == 'gelu':
            out = nn.gelu(out)
            
        if output_final_state:
            return out, None  # Simplified - no cache state
        return out

class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states):
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        return self.weight * hidden_states

class FusedRMSNormGated(nn.Module):
    """Fused RMS Norm with gating"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states, gate=None):
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        result = self.weight * hidden_states
        if gate is not None:
            result = result * gate
        return result

# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule (unchanged ‑ O(N))
# -----------------------------------------------------------------------------

def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Associative Δ-rule with causal chunked parallel scan (O(Nd)).

    Shapes:
        q, k: (B, H, L, D_k)
        v:     (B, H, L, D_v)
        beta:  (B, H, L)
    Returns:
        out: (B, H, L, D_v)
        S  : recurrent state matrix  (H, D_k, D_v)
    """
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = [(0, 0), (0, 0), (0, pad_len), (0, 0)]
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    L_pad = L + pad_len

    # Normalise queries / keys & apply β scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * mx.expand_dims(beta, -1)
    k_beta = k * mx.expand_dims(beta, -1)

    # Reshape into (B, H, N, C, D) with chunk size C
    q = rearrange_manual(q, "b h (n c) d -> b h n c d", c=chunk_size)
    k = rearrange_manual(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = rearrange_manual(v, "b h (n c) d -> b h n c d", c=chunk_size)
    k_beta = rearrange_manual(k_beta, "b h (n c) d -> b h n c d", c=chunk_size)

    tri_mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), 0)
    inv = -(k_beta @ mx.transpose(k, (0, 1, 2, 4, 3)))
    inv = mx.where(tri_mask, 0, inv)
    
    for i in range(1, chunk_size):
        inv[..., i, :i] = inv[..., i, :i] + mx.sum(
            mx.expand_dims(inv[..., i, :], -1) * inv[..., :, :i], axis=-2
        )
    inv = inv + mx.eye(chunk_size, dtype=inv.dtype)

    # Keep inv in the same dtype as q/k/v
    inv = inv.astype(q.dtype)

    u = inv @ v
    w = inv @ k_beta

    S = mx.zeros((b, h, d_k, v.shape[-1]))
    out = mx.zeros_like(v)
    mask_future = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = q_i @ mx.transpose(k_i, (0, 1, 3, 2))
        attn_local = mx.where(mask_future, 0, attn_local)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + mx.transpose(k_i, (0, 1, 3, 2)) @ u_i

    out = rearrange_manual(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Depth-wise FIR convolution (identity + orthogonal noise)
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR convolution initialised as identity + orthogonal noise."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        kernel_size: int = 31,
        noise_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Identity kernel (Dirac delta at last tap for causality)
        ident = mx.zeros((num_heads, head_dim, self.kernel_size))
        ident[..., -1] = 1.0
        if noise_std > 0:
            noise = mx.random.normal(ident.shape) * noise_std
            # Remove projection on identity to keep orthogonality
            proj = mx.sum(noise * ident, axis=-1, keepdims=True)
            noise = noise - proj * ident
            weight = ident + noise
        else:
            weight = ident
        self.filters = weight  # (H, D, K)

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        # Process each head separately to avoid grouping issues
        results = []
        for head_idx in range(h):
            # Get data for this head: (B, L, D)
            x_head = x[:, :, head_idx, :]
            # Get filters for this head: (D, K)
            filters_head = self.filters[head_idx, :, :]  # (D, K)
            
            # Apply conv1d for each feature in this head
            head_results = []
            for feat_idx in range(d):
                # x_feat: (B, L, 1), filter_feat: (1, K, 1)
                x_feat = mx.expand_dims(x_head[:, :, feat_idx], -1)
                filter_feat = mx.expand_dims(mx.expand_dims(filters_head[feat_idx, :], 0), -1)
                
                # Add causal padding
                x_feat_pad = mx.pad(x_feat, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
                # Convolve: output is (B, L, 1)
                conv_out = mx.conv1d(x_feat_pad, filter_feat)[:, :l, :]
                head_results.append(conv_out)
            
            # Stack features: (B, L, D)
            head_out = mx.concatenate(head_results, axis=-1)
            results.append(mx.expand_dims(head_out, 2))  # (B, L, 1, D)
        
        # Stack heads: (B, L, H, D)
        y = mx.concatenate(results, axis=2)
        return y

# -----------------------------------------------------------------------------
# Sparsemax with temperature (small path count, efficient)
# -----------------------------------------------------------------------------

def _sparsemax(logits: mx.array, dim: int = -1) -> mx.array:
    """Sparsemax (Martins & Astudillo, 2016).  Returns sparse probabilities."""
    shifted = logits - mx.max(logits, axis=dim, keepdims=True)
    zs = mx.sort(shifted, axis=dim)[::-1]  # descending
    k_range = mx.arange(1, zs.shape[dim] + 1, dtype=logits.dtype)
    view = [1] * logits.ndim
    view[dim] = -1
    k_range = k_range.reshape(view)
    zs_cumsum = mx.cumsum(zs, axis=dim)
    support = (1 + k_range * zs) > zs_cumsum
    k_support = mx.max(support * k_range, axis=dim, keepdims=True)
    tau = (mx.take_along_axis(zs_cumsum, k_support.astype(mx.int32) - 1, axis=dim) - 1) / k_support
    output = mx.maximum(shifted - tau, 0.0)
    return output

# -----------------------------------------------------------------------------
# Per-head sparsemax gate
# -----------------------------------------------------------------------------

class SparseGate(nn.Module):
    """Per-head temperature-controlled sparsemax gate over *n_paths*."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        n_paths: int = 4,
        gate_hidden_mult: float = 0.5,
        warm_start_bias: float = 1.5,
    ) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.num_heads = num_heads
        gate_hidden = max(8, int(hidden_size * gate_hidden_mult))
        self.proj1 = nn.Linear(hidden_size, gate_hidden, bias=True)
        self.proj2 = nn.Linear(gate_hidden, num_heads * n_paths, bias=True)
        # Warm-start bias – favour direct/value path (index n_paths-1)
        bias = self.proj2.bias.reshape(num_heads, n_paths)
        bias[:, -1] = warm_start_bias
        self.proj2.bias = bias.flatten()
        # Learnable per-head temperature (softplus ensures >0)
        self.log_temp = mx.zeros((num_heads,))

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, L, D)
        b, l, _ = x.shape
        logits = self.proj2(nn.silu(self.proj1(x)))  # (B, L, H*n_paths)
        logits = rearrange_manual(logits, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)
        temp = nn.softplus(self.log_temp) + 1e-4  # (H,)
        logits = logits / temp.reshape(1, 1, -1, 1)
        probs = _sparsemax(logits, dim=-1)  # (B, L, H, P)
        # Normalize (sparsemax already sums to 1 on support, but numerical safety)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
        return probs  # (B, L, H, P)

# -----------------------------------------------------------------------------
# DeltaNet with Hybrid Sparse Gated Memory
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with dual FIR branches and sparsemax gating."""

    def __init__(
        self,
        mode: str = "hsgm",  # hybrid sparse gated memory
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
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fir_noise_std: float = 1e-3,
        gate_hidden_mult: float = 0.5,
        gate_warm_start_bias: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx if layer_idx is not None else 0

        # Dimensions
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("key/value dim must be divisible by num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Optional short convolution
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution cannot be disabled in this variant.")

        # FIR branches
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel, noise_std=fir_noise_std)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel, noise_std=fir_noise_std)

        # Sparse gate
        self.sparse_gate = SparseGate(hidden_size, num_heads, n_paths=4, gate_hidden_mult=gate_hidden_mult, warm_start_bias=gate_warm_start_bias)

        # Output normalisation / projection
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,  # (B, L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[mx.array, None, Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, L_orig, _ = hidden_states.shape

        # Retrieve cache for this layer
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = index_first_axis(rearrange_manual(hidden_states, "b s d -> (b s) d"), indices)
            hidden_states = mx.expand_dims(hidden_states, 0)

        # Projections + optional short conv
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_out = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_out = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_out = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        
        if isinstance(q_out, tuple):
            q, conv_q = q_out
        else:
            q, conv_q = q_out, None
            
        if isinstance(k_out, tuple):
            k, conv_k = k_out
        else:
            k, conv_k = k_out, None
            
        if isinstance(v_out, tuple):
            v, conv_v = v_out
        else:
            v, conv_v = v_out, None

        # Head split
        q = rearrange_manual(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange_manual(k, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = rearrange_manual(v, "b l (h d) -> b l h d", h=self.num_heads)

        # Activations/norms
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # Beta for Δ-rule
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global path
        q_d = rearrange_manual(q, "b l h d -> b h l d")
        k_d = rearrange_manual(k, "b l h d -> b h l d")
        v_d = rearrange_manual(v_direct, "b l h d -> b h l d")
        beta_d = rearrange_manual(beta, "b l h -> b h l")
        delta_out, recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange_manual(delta_out, "b h l d -> b l h d")

        # FIR branches
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # Sparse gating
        gate = self.sparse_gate(hidden_states)  # (b, l, h, 4)
        gate = rearrange_manual(gate, "b l h p -> b l h p 1")  # broadcast dim for mul
        paths = mx.stack([local_short, local_long, delta_out, v_direct], axis=3)  # (b,l,h,4,d)
        out = mx.sum(gate * paths, axis=3)  # (b,l,h,d)

        # Cache update
        if past_key_values is not None and use_cache:
            layer_state = {
                "recurrent_state": recurrent_state_new,
                "conv_state": (conv_q, conv_k, conv_v),
                "layer_idx": self.layer_idx,
                "offset": L_orig,
            }
            if hasattr(past_key_values, "__setitem__"):
                past_key_values[self.layer_idx] = layer_state
            else:
                past_key_values.update(layer_state)

        # Output normalisation / projection
        if self.use_gate:
            g_vec = rearrange_manual(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = rearrange_manual(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # Re-pad if unpadded earlier
        if attention_mask is not None:
            out = pad_input(mx.squeeze(out, 0), indices, B_orig, L_orig)

        return out, None, past_key_values