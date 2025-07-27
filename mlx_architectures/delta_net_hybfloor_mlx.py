from __future__ import annotations

"""
MLX-converted architecture: delta_net_hybfloor
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
    return x / mx.linalg.norm(x, axis=-1,
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
    def __init__(self, hidden_size: int,
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
DeltaNet – Hybrid Floor & Identity Residual Fusion (delta_net_hybfloor)
Identifier: delta_net_hybfloor

Motivation
This variant merges the most effective components discovered in prior
experiments to simultaneously preserve **local lexical fidelity** and
**global reasoning capacity** without re-introducing the local–global
trade-off:

1. Per-Head / Per-Path Temperature
   • Each head owns an independent temperature **τ₍h p₎** (learnable, positive) allowing some heads to specialise in *sharp* routing
     while others remain *soft* for evidence fusion.

2. Hard Hybrid Floor (dual, floor)
   •   A **constant hard minimum probability** εₛ (short-FIR) and
       εᵥ (value/identity) is reserved before the softmax allocation.
       This guarantees that *local convolutional* and *direct identity*
       branches never vanish – fixing the extraction / Winogrande
       regressions seen when the floor decays to zero.
   •   The remaining (1-εₛ-εᵥ) mass is distributed by the gate between
       *long-FIR* and *Δ-rule* as well as any additional share for the
       already floored paths.

3. Identity Residual (outside, gate)
   •   A parallel additive residual from a learned **per-head scalar
       αᵢd** times an identity projection is added after fusion ensuring
       undistorted token copying irrespective of the gate state.

4. Shared-Context Statistics
   •   The gate receives not only per-head branch statistics but also a
       light *shared context vector* (mean statistics across, heads),
       improving cross-head coordination for passage-level tasks(e.g., BoolQ).

All other proven elements – **chunk-wise Δ-rule** (O(N)), **dual FIR
convolutions**, mandatory **_ShortConvolution** enhancement, and optional
**cache** interface – are inherited unchanged.  Complexity stays strictly
linear in sequence length.

Default hard-floor values εₛ=εᵥ=0.02 were chosen from ablations: small
enough to avoid over-biasing, large enough to protect gradient flow.

The class name **DeltaNet** and forward signature are preserved making
this variant a drop-in replacement.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_plus_one(x:, mx.array) -> mx.array:  # noqa: D401
    """Shifted ELU ensures strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:  # noqa: D401
    """L1-normalise the last dimension to sum to one."""
    return(x, / x.sum(dim=-1
        keepdim=True))

# -----------------------------------------------------------------------------
# Causal depth-wise FIR convolution (identical math identity, init)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding (O(N))."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0  # start as identity (Dirac)
        self.filters = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal
        y = F.conv1d(x_pad, w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise associative Δ-rule kernel(unchanged, still @mx.compile)
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(q:, mx.array,  # [B,H,L,D]
    k: mx.array,
    v: mx.array,
    beta: mx.array,  # [B,H,L]
    *,
    chunk_size: int = 32):
    """Efficient O(N) Δ-rule scan with strict causality."""
    b, h, L, d_k = q.shape

    # Pad so that L is divisible by chunk_size
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))
    tri_full = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad, // chunk_size):
        q_i = q[:, :, idx]
        k_i = k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Typing helper
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet – Hybrid Floor variant
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – required name
    """DeltaNet layer with hybrid hard-floor and identity residual."""

    def __init__(
        self, *,
        mode: str = "hybfloor",
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
        # FIR kernel sizes
        fir_kernel_short: int = 5,
        fir_kernel_long: int = 64,
        # Gate hyper-params
        gate_hidden_mult: int = 2,
        floor_short: float = 0.02,
        floor_value: float = 0.02,
        temp_init: float = 1.0,
        # Identity residual
        identity_scale_init: float = 0.5,
        **kwargs: Dict # compatibility
    ) -> None:
        super().__init__()

        # ---------------- bookkeeping ----------------
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---------------- dimensions -----------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key, / Value dims must divide num_heads")

        # ---------------- projections ----------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # Identity projection(for, residual, path)
        self.id_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        self.alpha_identity = mx.array(identity_scale_init, *, mx.ones(num_heads))

        # ---------------- short conv enhancers --------
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet stability.")

        # ---------------- FIR convolutions -----------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---------------- gating network -------------
        per_head_stat_dim = 16  # 4 stats × 4 branches
        shared_stat_dim = 16   # same size for shared context
        gate_in_dim = hidden_size + per_head_stat_dim + shared_stat_dim
        gate_hidden_dim = hidden_size * gate_hidden_mult // 2

        # Shared MLP applied per head for parameter efficiency
        self.gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 4, bias=True),  # 4, paths)

        # Bias initialisation – favour delta & value lightly
        with mx.disable_grad():
            self.gate_mlp[-1].bias.zero_()
            self.gate_mlp[-1].bias[2] = 0.5  # delta
            self.gate_mlp[-1].bias[3] = 1.0  # value

        # Per-head / per-path temperature
        self.log_temp = mx.array(mx.log(mx.ones(num_heads, 4) * temp_init))

        # ---------------- hard floors ---------------
        self.floor_short = float(floor_short)
        self.floor_value = float(floor_value)
        if floor_short + floor_value >= 1.0:
            raise ValueError("Sum, of hard floors must be < 1")

        # ---------------- output processing ---------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Statistic helper (per-head)
    # ------------------------------------------------------------------
    @staticmethod
    def _stats4(t:, mx.array) -> mx.array:  # [B,L,H,D] -> [B,L,H,4]
        mean = t.mean(dim=-1, keepdim=True)
        var = t.var(dim=-1, unbiased=False
        keepdim = True)
        abs_mean = t.abs().mean(dim=-1, keepdim=True)
        l2 = t.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False # kept for api parity
        **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape

        # ------------- optional unpadding for seq-var batches --------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ------------- retrieve cached conv state -------------------
        conv_q = conv_k = conv_v = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            cache_layer = past_key_values[self.layer_idx]
            if cache_layer is not None and cache_layer.get("conv_state") is not None:
                conv_q
        conv_k, conv_v = cache_layer["conv_state"]
        # ------------- projections + short conv ---------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin
        conv_q = self.q_conv1d(q_lin, cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_k = self.k_conv1d(k_lin, cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_v = self.v_conv1d(v_lin, cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # head reshape
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # activation / normalisation
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # beta for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global path
        delta_out_d
        rec_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # FIR local paths
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # ------------- gating --------------------------------------
        # statistics per head
        stats = mx.cat([, self._stats4(fir_short))
            self._stats4(fir_long),
            self._stats4(delta_out),
            self._stats4(v_direct),
        ], dim=-1)  # [B,L,H,16]
        shared_stats = stats.mean(dim=2, keepdim=True).expand(-1, -1, self.num_heads -1)
        gate_in = mx.cat([, hidden_states.expand_dims(2).expand(-1, -1, self.num_heads -1))
            stats,
            shared_stats,
        ], dim=-1)  # [B,L,H D+16+16]

        gate_logits = self.gate_mlp(gate_in)  # [B,L,H,4]

        # temperature scaling
        temp = mx.exp(self.log_temp).clamp(min=1e-3, max=10.0)  # [H 4]
        gate_logits = gate_logits / temp.expand_dims(0).expand_dims(0)

        soft = mx.softmax(gate_logits, dim = -1)  # [B,L,H,4]

        # apply hard hybrid floor: indices (0 short-FIR 3, value)
        floor_vec = mx.tensor([self.floor_short, 0.0, 0.0, self.floor_value]
        dtype=soft.dtype)
        floor_vec = floor_vec.reshape(1, 1, 1, 4)
        residual_mass = 1.0 - floor_vec.sum(-1, keepdim=True)
        gate_w = floor_vec + residual_mass * soft

        # ------------- fuse branches --------------------------------
        o_mix = (
            gate_w[..., 0:1] * fir_short +
            gate_w[..., 1:2] * fir_long +
            gate_w[..., 2:3] * delta_out +
            gate_w[..., 3:4] * v_direct
        )

        # identity residual (outside, gate)
        id_val = _rearrange(self.id_proj(hidden_states), "b l (h, d) -> b l h d"
        h=self.num_heads)
        alpha = self.alpha_identity.reshape(1, 1, -1, 1)
        o = o_mix + alpha * id_val

        # ------------- cache update ---------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # ------------- output norm / projection ---------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ------------- re-pad if needed -----------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)

        return o, None, past_key_values
