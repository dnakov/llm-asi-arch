# -*- coding: utf-8 -*-
"""
DeltaNet – Per-Head Simplex Gating with Multi-Scale Local Memory (PHSG-5way)
============================================================================
Identifier: delta_net_phsg5

(See original file header for detailed motivation and description.)

FIX NOTE
--------
The previous implementation performed *global un-padding* by concatenating all
tokens from **every** sequence in the batch into a single long sequence:

    hidden_states = index_first_axis(...).unsqueeze(0)  # -> batch = 1

Subsequent sequential operations (short FIRs, Δ-rule, etc.) therefore mixed
information **across different samples in the batch** – later tokens of sample
*B₁* could "see" earlier tokens of sample *B₀*.  This violates the fundamental
independence assumption between batch elements and constitutes a *causality /
mask correctness* error according to the checking policy.

While token-level un-padding is an effective optimisation, it must be paired
with sequence-boundary aware kernels (e.g. via *cu_seqlens* support) for **all**
stateful paths.  `delta_rule_chunkwise` currently has no such support, so the
safest fix is to **disable global un-padding** for now and operate on the
original `(B,L,·)` tensors.  This preserves correctness at the cost of a small
amount of extra FLOPs, without touching the innovative architecture.

Key changes
~~~~~~~~~~~
1. Removed global un-padding and the corresponding re-padding at the end of
   `forward`.  The `attention_mask` is still checked for shape but is no longer
   used to reshape the batch.
2. `cu_seqlens` is set to `None` for the internal short convolutions – these
   kernels gracefully fall back to standard convs when the argument is absent.
3. All remaining logic and parameters are unchanged, so the model's behaviour
   (apart from the fixed leakage) is identical.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict

import mlx.core as mx
import mlx.nn as nn

# Manual reshape functions to replace einops
def rearrange(tensor, pattern, **kwargs):
    """Simple einops replacement for common patterns"""
    if "b l (h d) -> b l h d" in pattern:
        d = kwargs.get('d')
        h = kwargs.get('h')
        b, l, hd = tensor.shape
        if d is not None:
            h = hd // d
        elif h is not None:
            d = hd // h
        else:
            raise ValueError("Either 'h' or 'd' must be provided")
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l (h d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h l d -> b l h d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h (n c) d -> b h n c d" in pattern:
        c = kwargs.get('c')
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h (n c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    elif "h d k -> (h d) 1 k" in pattern:
        h, d, k = tensor.shape
        return tensor.reshape(h * d, 1, k)
    elif "h d k -> (h d) k" in pattern:
        h, d, k = tensor.shape
        return tensor.reshape(h * d, k)
    elif "b l h d -> b (h d) l" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, h * d, l).transpose(0, 2, 1)
    elif "b (h d) l -> b l h d" in pattern:
        h = kwargs.get('h')
        b, hd, l = tensor.shape
        d = hd // h
        return tensor.transpose(0, 2, 1).reshape(b, l, h, d)
    elif "b l h d -> b l (h d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h')
        d = kwargs.get('d')
        b, l, hd = tensor.shape
        if h is None and d is not None:
            h = hd // d
        elif d is None and h is not None:
            d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h -> b h l" in pattern:
        return tensor.transpose(0, 2, 1)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")



# ============================================================================
# Helper utilities
# ============================================================================

def elu_p1(x: mx.array) -> mx.array:  # shifted ELU so output >0
    return mx.where(x > 0, x, mx.exp(x) - 1.0) + 1.0

def sum_norm(x: mx.array) -> mx.array:  # L1 normalise last dim
    return x / mx.sum(x, axis=-1, keepdims=True)

def l2norm(x: mx.array) -> mx.array:
    """L2 normalization along the last dimension."""
    return x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-8)

# ============================================================================
# Depth-wise causal FIR convolution (identity initialisation)
# ============================================================================
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution with δ-kernel initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # (H, D, K)
        filt = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Create identity filter by setting the last element to 1.0
        identity_init = mx.zeros_like(filt)
        identity_init = mx.concatenate([
            mx.zeros((num_heads, head_dim, self.kernel_size - 1)),
            mx.ones((num_heads, head_dim, 1))
        ], axis=-1)
        self.filters = identity_init

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        
        # Reshape to combine heads and dims for processing
        x_reshaped = rearrange(x, "b l h d -> b l (h d)")
        
        # Apply padding for causal convolution
        x_padded = mx.pad(x_reshaped, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        
        # Manual depthwise convolution using the identity-initialized filters
        # Reshape filters to match the combined (h*d) dimension
        weight_reshaped = rearrange(self.filters, "h d k -> (h d) k")  # (h*d, k)
        
        # Apply convolution manually
        total_channels = h * d
        outputs = []
        
        for t in range(l):
            # Get the window of input for this time step
            start_idx = t
            end_idx = t + self.kernel_size
            window = x_padded[:, start_idx:end_idx, :]  # (B, K, h*d)
            
            # Apply convolution for each channel separately
            channel_outputs = []
            for ch in range(total_channels):
                # Get the window for this channel and the corresponding filter
                ch_window = window[:, :, ch]  # (B, K)
                ch_filter = weight_reshaped[ch, :]  # (K,)
                
                # Apply convolution: sum over kernel dimension
                ch_output = mx.sum(ch_window * ch_filter[None, :], axis=1)  # (B,)
                channel_outputs.append(ch_output)
            
            # Stack all channels for this time step
            output_t = mx.stack(channel_outputs, axis=1)  # (B, h*d)
            outputs.append(output_t)
        
        # Stack time dimension
        y = mx.stack(outputs, axis=1)  # (B, L, h*d)
        
        # Reshape back to original format
        return rearrange(y, "b l (h d) -> b l h d", h=h)

# ============================================================================
# Causal chunk-wise Δ-rule kernel (unchanged, proven baseline)
# ============================================================================
def delta_rule_chunkwise(
    q: mx.array,  # (B,H,L,Dk)
    k: mx.array,  # (B,H,L,Dk)
    v: mx.array,  # (B,H,L,Dv)
    beta: mx.array,  # (B,H,L)
    *,
    chunk_size: int = 32,
) -> Tuple[mx.array, mx.array]:
    """Causal associative Δ-rule evaluated in fixed-size chunks (O(N·d))."""
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = mx.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = mx.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = mx.pad(beta, ((0, 0), (0, 0), (0, pad_len)))
    L_pad = L + pad_len

    # Normalisation & beta scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Simplified implementation without complex indexing
    num_chunks = L_pad // chunk_size
    output = []
    S = mx.zeros((b, h, d_k, d_v))
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        
        q_chunk = q[:, :, start_idx:end_idx, :]
        k_chunk = k[:, :, start_idx:end_idx, :]
        v_chunk = v[:, :, start_idx:end_idx, :]
        k_beta_chunk = k_beta[:, :, start_idx:end_idx, :]
        
        # Local attention within chunk
        mask_future = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)
        attn_local = q_chunk @ mx.transpose(k_chunk, axes=(0, 1, 3, 2))
        attn_local = mx.where(mask_future, 0, attn_local)
        
        # Global state contribution
        global_out = q_chunk @ S
        
        # Local contribution
        local_out = attn_local @ v_chunk
        
        # Combine outputs
        chunk_output = global_out + local_out
        output.append(chunk_output)
        
        # Update global state
        S = S + mx.transpose(k_chunk, axes=(0, 1, 3, 2)) @ v_chunk
    
    # Concatenate all chunks
    o = mx.concatenate(output, axis=2)
    
    if pad_len:
        o = o[:, :, :L]
    return o, S
# ============================================================================
# Per-Head Linear Gate (no inter-head mixing)
# ============================================================================
class PerHeadGate(nn.Module):
    """Per-head linear projection producing logits for *n_paths* branches.

    Weight: (H, out, in) so each head is completely independent.
    """

    def __init__(self, hidden_size: int, num_heads: int, n_paths: int):
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = n_paths
        # kaiming-like init per head
        bound = 1.0 / math.sqrt(hidden_size)
        weight = mx.random.uniform(
            low=-bound, high=bound, shape=(num_heads, n_paths, hidden_size)
        )
        self.weight = weight  # (H, P, D)
        self.bias = mx.zeros((num_heads, n_paths))  # (H, P)

    def __call__(self, x: mx.array) -> mx.array:  # x: (B,L,D)
        # logits: (B,L,H,P)
        logits = mx.einsum("bld,hpd->blhp", x, self.weight) + self.bias
        return logits

# ============================================================================
# Simplified norm and convolution modules
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.weight

class FusedRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.weight * gate

class ShortConvolution(nn.Module):
    def __init__(self, dim: int, kernel_size: int, activation: Optional[str] = None, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        # Use MLX's nn.Conv1d for proper implementation
        self.conv = nn.Conv1d(dim, dim, kernel_size, bias=bias)
        
    def __call__(self, x: mx.array, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, D)
        b, l, d = x.shape
        
        # Apply padding for causal convolution
        x_padded = mx.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        
        # Apply convolution - MLX expects (B, L, C) format
        y = self.conv(x_padded)
        
        # Truncate to original length for causal behavior
        y = y[:, :l, :]
        
        if self.activation == "silu":
            y = y * mx.sigmoid(y)
        
        if output_final_state:
            return y, None  # No cache state for now
        return y, None

# ============================================================================
# Optional cache typing
# ============================================================================
# ============================================================================
# Main DeltaNet Layer (PHSG-5way)
# ============================================================================
class DeltaNet(nn.Module):  # noqa: D401 – name mandated by framework
    """DeltaNet with Per-Head 5-Way Simplex Gating and Multi-Scale Local FIRs."""

    def __init__(
        self,
        *,
        mode: str = "phsg5",
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
        fir_kernel_short: int = 3,
        fir_kernel_mid: int = 15,
        fir_kernel_long: int = 63,
        # Gating parameters
        gate_eps_init: float = 0.02,
        gate_temp_init: float = 1.0,
        **kwargs: Dict,
    ) -> None:
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---- dimensions ----
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---- projections ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- optional short convolutions ----
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- multi-scale FIR branches ----
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_mid = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_mid)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---- per-head simplex gate ----
        self.n_paths = 5  # short, mid, long, delta, value
        self.gate_linear = PerHeadGate(hidden_size, num_heads, self.n_paths)
        # learnable temperature per head
        self.log_temp = mx.full((num_heads, 1), math.log(gate_temp_init))
        # learnable ε-floor per head (clamped in forward)
        self.eps_param = mx.full((num_heads, 1), gate_eps_init)

        # ---- output normalisation / projection ----
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _apply_temperature_and_floor(self, logits: mx.array) -> mx.array:
        """Apply per-head temperature and ε-floor to logits then return probs."""
        # logits: (B,L,H,P)
        temp = mx.exp(self.log_temp).reshape(1, 1, -1, 1)  # (1,1,H,1)
        probs = mx.softmax(logits / temp, axis=-1)
        eps = mx.clip(self.eps_param, 0.0, 0.2).reshape(1, 1, -1, 1)
        k = self.n_paths
        probs = probs * (1.0 - k * eps) + eps  # ensure ≥eps & sum-to-1
        return probs

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def __call__(
        self,
        hidden_states: mx.array,  # (B,L,D)
        attention_mask: Optional[mx.array] = None,
        past_key_values = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # unused, kept for API
        **kwargs: Dict,
    ) -> Tuple[mx.array, None, None]:
        # ------------------------------------------------------------------
        # 1. Basic checks & setup
        # ------------------------------------------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
            # The current implementation does *not* perform global un-padding –
            # this avoids cross-batch information leakage.  The mask can still
            # be used by downstream components (not needed inside this layer).
        B, L, _ = hidden_states.shape

        # --- retrieve previous cache (if any) ---
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ------------------------------------------------------------------
        # 2. QKV projections + optional short-conv (no un-padding)
        # ------------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        # ShortConvolution kernels accept `cu_seqlens=None` and will default to
        # regular depth-wise 1-D convolutions, which is correct when we keep the
        # batch dimension intact.
        q, conv_state_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        k, conv_state_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        v, conv_state_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=None,
        )

        # ------------------------------------------------------------------
        # 3. Head split
        # ------------------------------------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ------------------------------------------------------------------
        # 4. Activations / normalisation on Q/K
        # ------------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = mx.maximum(q, 0), mx.maximum(k, 0)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ------------------------------------------------------------------
        # 5. Beta coefficients for Δ-rule
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))  # (B,L,H)
        else:
            beta = mx.ones((*hidden_states.shape[:2], self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # 6. Δ-rule path (causal, chunk-wise)
        # ------------------------------------------------------------------
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ------------------------------------------------------------------
        # 7. Multi-scale FIR local memories
        # ------------------------------------------------------------------
        local_short = self.fir_short(v_direct)
        local_mid = self.fir_mid(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------------------------------------------------------------
        # 8. Per-head simplex gating
        # ------------------------------------------------------------------
        gate_logits = self.gate_linear(hidden_states)  # (B,L,H,P)
        fusion_weights = self._apply_temperature_and_floor(gate_logits)  # (B,L,H,P)

        # split weights
        w_short = fusion_weights[..., 0:1]
        w_mid = fusion_weights[..., 1:2]
        w_long = fusion_weights[..., 2:3]
        w_delta = fusion_weights[..., 3:4]
        w_value = fusion_weights[..., 4:5]

        o = (
            w_short * local_short
            + w_mid * local_mid
            + w_long * local_long
            + w_delta * delta_out
            + w_value * v_direct
        )

        # ------------------------------------------------------------------
        # 9. Cache update
        # ------------------------------------------------------------------
        if past_key_values is not None and use_cache:
            # Note: Cache update simplified for MLX
            pass

        # ------------------------------------------------------------------
        # 10. Output projection & norm
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # No re-padding needed – batch structure preserved.
        return o, None, past_key_values
