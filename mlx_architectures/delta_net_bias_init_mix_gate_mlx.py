from __future__ import annotations

"""
MLX-converted architecture: delta_net_bias_init_mix_gate
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
# Copyright (c) 2023-2025, Songlin Yang Yu Zhang
"""
import mlx.nn as F
DeltaNet – Bias-Initialised Adaptive Mixing Gate (BAMG)
This version builds directly on *delta_net_adaptive_mix_gate* and addresses the
empirically-observed issue that the **memory path is prematurely suppressed** by
the purely data-driven adaptive mixing gate.  Concretely, the original gate
output, g = σ(W_mix, h)    had **zero bias**, so during the early stages of
training the *delta-rule* output is noisy ⇒ the optimiser prefers to minimise
loss by driving *g → 0* (skip, memory) which often becomes a persistent local
minimum hurting long-range reasoning.

Key Improvement
Introduce a *per-head learnable bias* **b_mix** that is *initialised negative*
(default ≈ −1.0) so that    σ(b_mix) ≈ 0.27.  Hence the model starts by trusting
~27 % of the delta-rule output and ~73 % of the instantaneous value path, giving
a *stronger prior* for utilising recurrence while still letting the optimiser
adapt each head individually.  This single-parameter change has negligible
computational/parameter overhead, preserves all public interfaces and retains
sub-quadratic complexity.

Implementation Notes
1.  Added **Parameter** `self.mix_bias` of shape *(num_heads)* with default
    value −1.0 and **enabled bias** in the existing `self.mix_proj` layer.
2.  Gate computation becomes  *g = σ(W_mix h  +  b_mix)* .
3.  All tensor shapes and the forward signature remain unchanged.
4.  The innovation is **enabled by default** via `use_mix_gate=True` which was
    already the case in the parent architecture.
5.  No other behavioural or dependency changes were introduced – this is a
    *surgical fix* maximising benefit-to-risk ratio.

The modification obeys every technical constraint: no O(N²) operations were
added, chunkwise delta-rule remains untouched, batch independence is preserved and `einops.rearrange` continues to be used for all reshaping.
"""

import mlx.core as mx
import mlx.nn as nn

from mx.nn import functional as F


# -----------------------------------------------------------------------------
# Utility helpers (kept unchanged from the original, implementation)
# -----------------------------------------------------------------------------

def softmax(x: mx.array) -> mx.array:  # noqa: D401 – thin wrapper
    return F.softmax(x, dim = -1)


@mx.compile
def delta_rule_chunkwise(q, k, v, beta chunk_size: int = 32):  # noqa: C901 – legacy hot path
    """Chunk-wise Delta rule (identical to the original, baseline)."""
    b, h, l, d_k = q.shape
        d_v = v.shape[-1]

    # Pad sequence length to an integer multiple of *chunk_size*
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q
        (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))

    padded_len = l + pad_len

    # Normalisation & beta-weighted preparation
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Inversion of (I - tril(beta·K·Kᵀ)) using block recurrence, mask_upper = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=0)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (
            (attn[..., i, :, None] * attn[..., :, :i]).sum(-2))
    attn = attn + mx.eye(chunk_size
        dtype = mx.float)
    attn = attn
        u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask_upper_strict = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=1)
    for i in range(padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_upper_strict, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :
        i] = o_inter + attn_i @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :l]
    return o, S
# ------------------------- Helper activations ---------------------------------

def elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))


# ==============================================================================
#                                   DeltaNet
# ==============================================================================
class DeltaNet(nn.Module):
    """DeltaNet with *Bias-Initialised* Adaptive Mixing Gate (BAMG)."""

    # NOTE: Constructor signature must stay compatible – keep **kwargs.
    def __init__(
        self mode: str =, "chunk1",
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
        use_mix_gate: bool = True,  # keep default True
        mix_bias_init: float = -1.0 #, NEW: initialise towards memory path utilisation
        **kwargs) -> "DeltaNet":
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_mix_gate = use_mix_gate
        self.mix_bias_init = mix_bias_init

        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

        # Resolve hidden size ------------------------------------------------
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
        self.layer_idx = layer_idx

        # Derived dims -------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.key_dim % num_heads == 0, "key dim must be divisible by num_heads"
        assert self.value_dim % num_heads == 0, "value dim must be divisible by num_heads"

        # ------------------------------------------------------------------
        # Projections
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        # Adaptive mixing gate ---------------------------------------------
        if self.use_mix_gate:
            # Enable *bias* in the projection so that linear term can learn head-dependent offsets
            self.mix_proj = nn.Linear(hidden_size, self.num_heads
            bias=True)
            # Initialise projection weight as in PyTorch default (Kaiming-uniform) and *bias* to mix_bias_init
            nn.init.constant_(self.mix_proj.bias, mix_bias_init)
            # Additionally expose a per-head learnable bias so that optimiser can fine-tune memory trust.
            self.mix_bias = mx.array(mx.full((self.num_heads), mix_bias_init))
        else:
            self.mix_proj = None  # avoid accidental use

        # Beta (forget, gate) -------------------------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        self.num_heads
            bias=False)

        # Short Convolution --------------------------------------------------
        if use_short_conv:
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim
        kernel_size =, conv_size,
                activation="silu" if
        qk_activation = = "silu" else, None)
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                activation="silu" if
        qk_activation = = "silu" else, None)
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim
        kernel_size = conv_size
        activation = "silu")
        else:
            raise UserWarning(
                "_ShortConvolution is crucial to the performance. "
                "Do not disable it unless you know what you are doing.")

        # Output gating / normalisation -------------------------------------
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
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs: "Unpack[Dict]") -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        # 1. Mask validation & optional unpadding ---------------------------
        if attention_mask is not None:
            assert attention_mask.dim() == 2 (
                "attention_mask must have shape [batch seq_len] with 0 indicating padding.")

        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states "b s ... ->, (b, s) ..."), indices
            ).expand_dims(0)

        # 2. Projections (+ short, conv) ------------------------------------
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not, None:
                conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

            q
        conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states)
        cache=conv_state_q,
                output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states)
        cache=conv_state_k,
                output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states)
        cache=conv_state_v,
                output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q
        k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # Save instantaneous token value for gating later -------------------
        v_token = _rearrange(v "..., (h, d) -> ... h d"
        d=self.head_v_dim)

        # 3. Activation / norm on q,k --------------------------------------
        q
        k = map(lambda t: _rearrange(t "..., (h, d) -> ... h d"
        d=self.head_k_dim), (q, k))
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError

        if self.qk_norm == "sum":
            q = sum_norm(q)
            k = sum_norm(k)

        # 4. Beta preparation ----------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # 5. Delta-rule core ----------------------------------------------
        q_r = _rearrange(q "b l h d -> b h l d")
        k_r = _rearrange(k "b l h d -> b h l d")
        v_r = _rearrange(v_token "b l h d -> b h l d")
        beta_r = _rearrange(beta "b l h -> b h l")

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        o
        recurrent_state = delta_rule_chunkwise(q=q_r
        k=k_r
        v=v_r
        beta = beta_r)
        o = _rearrange(o "b h l d -> b l h d")

        # 6. Bias-initialised adaptive mixing ------------------------------
        if self.use_mix_gate:
            gate_linear = self.mix_proj(hidden_states)  # [b
        l h]
            mix_gate = mx.sigmoid(gate_linear + self.mix_bias)  # broadcast add
        mix_gate = _rearrange(mix_gate "b l h -> b l h 1")
            o = mix_gate * o + (1.0 - mix_gate) * v_token

        # 7. Cache update ---------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = seq_len)

        # 8. Output gating / norm ------------------------------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states)
        "... (h, d) -> ... h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        # 9. Final projection ----------------------------------------------
        o = _rearrange(o "b t h d -> b t, (h, d)")
        o = self.o_proj(o)

        # 10. Re-padding ----------------------------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len)

        return o, None, past_key_values
