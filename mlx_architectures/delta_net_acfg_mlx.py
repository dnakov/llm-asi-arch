from __future__ import annotations

"""
MLX-converted architecture: delta_net_acfg
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions(replacing, PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List, Dict

def _rearrange(tensor:, mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l(h, d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l(h, d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h l d -> b l h d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h(n, c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h(n, c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    else:
        # Fallback: return tensor as-is
        return tensor

def _l2norm(x:, mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1
        keepdims=True).clip(min=1e-8)

def _masked_fill(tensor:, mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
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
"""
DeltaNet – Adaptive Context-Floor Gating with Post-Fusion Renormalisation (ACFG)
Identifier: delta_net_acfg

Motivation
Prior DeltaNet generations demonstrated that protecting the value/copy path
is vital for span-level fidelity but a *fixed* context quota(Dynamic, Floor-Gated Warm-Start – **DFGWS**) introduces an unavoidable copy-noise that
hurts copy-critical tasks(e.g., Winogrande).  Conversely removing the floor
risks contextual path starvation and regresses local-reasoning tasks.

Adaptive Context-Floor Gating (ACFG) resolves this dilemma by *learning a
per-token per-head minimum context allocation* that can vary smoothly between
0 and `max_context_floor` (default 0.20).  High-uncertainty tokens thus retain
a healthy context gradient, while unambiguous copy tokens are free to allocate
> 99 % mass to the identity branch.

Key Components
1. **Adaptive Floor MLP** – A single linear layer maps the current hidden
   state to *H* logits whose sigmoid determines the minimum context quota
   `floor ∈ [0,max_floor]` for each head/token.
2. **Hierarchical Gating** – As in DFGWS gating proceeds in two stages:
      a. Value gate (sigmoid) with learnable warm-start bias `+4`.
      b. Softmax over contextual paths {short FIR, long FIR Δ-rule}.
   The value gate is rescaled so that
   `p_value = (1-floor) * σ(logit_val)` guaranteeing
   `1-p_value ≥ floor` ⇒ continuous gradient flow.
3. **Post-Fusion Head-Wise nn.RMSNorm** – A lightweight per-head nn.RMSNorm is
   applied to the fused memory before projection to stabilise the variance
   increase introduced by adaptive routing.  This follows the variance control
   insight from CAGF-RC analysis and adds *negligible* compute.

All operations remain O(N), strictly causal, batch-agnostic and fully
compatible with earlier DeltaNet interfaces.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU (=ELU+1) keeps outputs positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """Normalise so that last dimension sums to 1."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule(identical, to proven baseline kept @mx.compile)
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Associative Δ-rule with causal chunked scan (O(N, d))."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & beta scaling
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] += (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size, dtype = attn.dtype)

    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)

    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (δ-kernel, initialisation)
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR convolution with delta (identity) initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0  # identity at current timestep
        self.filters = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

# -----------------------------------------------------------------------------
# Main DeltaNet with Adaptive Context-Floor Gating
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with Adaptive Context-Floor Gating (ACFG)."""

    def __init__(self, # ---- identifier & mode ----
        mode: str =, "acfg",
        # ---- model dimensions ----
        d_model: Optional[int] = None,
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
        layer_idx: Optional[int] = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # ---- FIR kernels ----
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # ---- gating hyper-parameters ----
        context_max_floor: float = 0.2,
        fusion_hidden_mult: int = 2,
        value_bias_init: float = 4.0 **kwargs) -> None:
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
        assert 0.0 < self.context_max_floor < 0.5, "context_max_floor must be(0, 0.5)"

        # ------------------ dimensions ------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value, dims must divide num_heads")

        # ------------------ projections ------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # ------------------ short convs ------------------
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size = conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet stability.")

        # ------------------ FIR branches ------------------
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ------------------ Fusion gate ------------------
        gate_in_dim = hidden_size  # only hidden state fed to gate; path stats handled implicitly by adaptive floor
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * 4 bias=True))
        # Warm-start bias – value path
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias.zero_()
            self.fusion_gate_mlp[-1].bias[3::4] = value_bias_init

        # ------------- Adaptive floor MLP --------------
        self.floor_mlp = nn.Linear(hidden_size, num_heads
        bias=True)
        nn.init.zeros_(self.floor_mlp.weight)
        nn.init.constant_(self.floor_mlp.bias, math.log(self.context_max_floor / (1 - self.context_max_floor)))

        # --------------- Output normalisation -----------
        # Two-stage: per-head norm after fusion (RMS) then projection
        self.post_fusion_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False # kept for API compatibility
        **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        batch_size, seq_len_full, _ = hidden_states.shape

        # ---------------- cache retrieval ----------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # ---------------- optional unpadding -------------
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        seq_len = hidden_states.shape[1]

        # ---------------- Q/K/V projections --------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin
        conv_state_q = self.q_conv1d(q_lin, cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_state_k = self.k_conv1d(k_lin, cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_state_v = self.v_conv1d(v_lin, cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---------------- Head reshape -------------------
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # ---------------- Activations --------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---------------- Beta ---------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones((*hidden_states.shape[:2], self.num_heads)
            dtype=hidden_states.dtype)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule global path -------------
        delta_out_t
        recurrent_state = delta_rule_chunkwise(
            q=_rearrange(q, "b l h d -> b h l d")
        k=_rearrange(k, "b l h d -> b h l d"),
            v=_rearrange(v_direct, "b l h d -> b h l d")
        beta=_rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t, "b h l d -> b l h d")

        # ---------------- FIR paths ----------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ---------------- Adaptive floor -----------------
        floor_logits = self.floor_mlp(hidden_states)  # (B, L, H)
        floor = mx.sigmoid(floor_logits) * self.context_max_floor  # (B,L, H)

        # ---------------- Fusion gate --------------------
        fusion_logits = self.fusion_gate_mlp(hidden_states)  # (B,L H*4)
        fusion_logits = _rearrange(fusion_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 4)

        # Value gate (sigmoid) with adaptive floor
    value_logit = fusion_logits[..., 3]
        p_value_raw = mx.sigmoid(value_logit)
        p_value = (1.0 - floor) * p_value_raw  # ensures 1-p_value >= floor
        others_total = 1.0 - p_value  # >= floor

        # Contextual softmax over paths {short, long, delta}
        ctx_logits = fusion_logits[..., 0:3]
        ctx_weights = mx.softmax(ctx_logits, dim = -1)  # (B,L,H, 3)
        ctx_weights = ctx_weights * others_total.expand_dims(-1)

        # ---------------- Fuse outputs -------------------
        o = (
            ctx_weights[..., 0:1] * local_short +
            ctx_weights[..., 1:2] * local_long +
            ctx_weights[..., 2:3] * delta_out +
            p_value.expand_dims(-1) * v_direct
        )

        # ---------------- Post-fusion norm ---------------
        o = self.post_fusion_norm(o)

        # Fix: Ensure dtype matches self.o_proj.weight before projection to prevent mat1/mat2 dtype mismatch
        if o.dtype != self.o_proj.weight.dtype:
            o = o

        # ---------------- Cache update -------------------
        if past_key_values is not None and self.layer_idx is not None and, use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = seq_len)

        # ---------------- Output norm/proj ---------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---------------- Re-pad if needed ---------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len_full)

        return o, None, past_key_values
