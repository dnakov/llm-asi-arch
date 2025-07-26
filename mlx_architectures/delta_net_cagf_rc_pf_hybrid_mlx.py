from __future__ import annotations

"""
MLX-converted architecture: delta_net_cagf_rc_pf_hybrid
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
DeltaNet – Content-Aware Gated Fusion v2(Hybrid, Residual Prob-Floor)
Identifier: **delta_net_cagf_rc_pf_hybrid**

This evolution of the *cagf_rc_pf* variant directly addresses the main
regression uncovered in Winogrande / ultra-local reasoning by **ensuring a
non-zero always-on residual contribution** while *retaining* the proven
benefits of probability-floored soft-max fusion (`ε`-floor) and dynamic content-aware routing.

Key improvements(enabled, by, default)
1. Hybrid residual scaling
   γ̂[b,t h] = σ(γ_h) · (α + (1-α)·σ(W, x[b t] + b))
   •  `α` (default **0.3**) is a *learnable* minimum residual fraction giving
      every head a guaranteed path for ultra-local signals(crucial, for
      WinoGrande / coreference) while still allowing dynamic modulation.
   •  Static logit `γ_h` **initialises at –1.0** (instead of –2.0) so the
      residual starts at ~0.27 strength – strong enough for learning signals
      but not dominant.

2. Slightly higher probability floor(`ε, = 0.03`) to further improve gradient
   flow through rarely-chosen paths during early training.

Everything else – Δ-rule chunk, dual FIR branches, head-level statistics, per-
path probability floor fusion RMS normalisation – is inherited unchanged and
kept fully compatible with existing checkpoints & infrastructure.

Complexity remains **O(N·d)**, strictly causal batch-agnostic and @mx.compile
optimised on the heavy Δ-rule kernel.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ================================================================
# Helper utilities
# ================================================================

def _elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU(=, ELU + 1) keeps values strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """Normalise so that the last dimension sums to one."""
    return (x / x.sum(-1, keepdim=True))

# ================================================================
# Depth-wise causal FIR convolution (Dirac, initialisation)
# ================================================================

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding for(B, L,H, D) tensors."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int) -> None:  # noqa: D401
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Identity (Dirac) kernel + small noise
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0
            filt.add_(0.01, * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, l, h, d = x.shape
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")  # (H*D,1, k)
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal
        y = F.conv1d(x_pad, weight=w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# ================================================================
# Chunk-wise Δ-rule kernel(identical, maths still @mx.compile)
# ================================================================

@mx.compile  # type: ignore[misc]
# pylint: disable=too-many-statements,too-many-locals

def _delta_rule_chunkwise(q:, mx.array,  # (B,H,L, Dk)
    k: mx.array,  # (B,H,L, Dk)
    v: mx.array,  # (B,H,L, Dv)
    beta: mx.array,  # (B,H, L)
    *,
    chunk_size: int = 32):  # -> Tuple[(B,H,L, Dv), (B,H,Dk, Dv)]
    """Associative Δ-rule retrieval processed in causal chunks (O(N))."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalisations & β scaling
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into (B,H,N,C, D)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones_like(tri), 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S  # (B,H,L, Dv), (B H,Dk, Dv)

# ================================================================
# DeltaNet main layer
# ================================================================

class DeltaNet(nn.Module):
    """DeltaNet with probability-floored gated fusion **and** hybrid residual conv."""

    def __init__(
        self, *,
        mode: str = "cagf_rc_pf_hybrid",
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
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # Fusion network
        fusion_hidden_mult: int = 2,
        prob_floor: float = 0.03 # ε-floor(slightly, ↑)
        # Hybrid residual conv params
        residual_alpha: float = 0.3,  # always-on fraction α
        conv_residual_init: float = -1.0,  # logit initialisation
        **kwargs # noqa: ANN001 – compatibility, shim) -> None:
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

        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/value, dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # -------- projections -----------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # -------- short convs -----------------------
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

        # -------- multi-scale FIR convs -------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_short)

        # -------- gating network --------------------
        # We operate **per head**, therefore the statistics dimensionality is 4
        # (mean, var, abs, L2).  Hidden state features (hidden_size) are shared
        # across heads through broadcasting, but for the MLP each head receives
        # its own copy, so the final input feature size is `hidden_size + 4`.
        stats_dim = 4  # one scalar for each of the 4 statistics
        fusion_gate_in = hidden_size + stats_dim  # per-head input dimension
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(fusion_gate_in, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # 4 fusion coefficients per, head)
        # warm-start bias toward identity/value path (index, 3)
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias.zero_()
            self.fusion_gate_mlp[-1].bias[3] = 3.0

        # -------- residual conv scaling -------------
        self.conv_residual_logit = mx.array(mx.full((num_heads), conv_residual_init))
        self.res_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.res_gate_proj.bias.fill_(0.0)  # neutral start

        # -------- output norm / proj ----------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias = False)

    # --------------------------------------------------------------
    # Per-head statistics helper
    # --------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x:, mx.array) -> mx.array:
        """Return per-token, per-head, 4-feature statistics tensor(B, L,H, 4)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)  # (B,L,H, 4)

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compat
        **kwargs # noqa: ANN401 – future, proof) -> Tuple[mx.array, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be(batch, seq_len)"
        B0, L0, _ = hidden_states.shape

        # -------- unpadding (optional) --------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L0:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # -------- cache retrieval -------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # -------- projections & short conv ----------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---- activation / norm on q,k -------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---- β for Δ-rule -------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global path -------------------
        delta_out_b
        recurrent_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_b, "b h l d -> b l h d")

        # ---- local FIR paths ----------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---- statistics for gating ---------------
        # We aggregate statistics across branches by **addition** to keep the
        # final dimensionality at 4 while still conveying relative magnitudes.
        stats_short = self._per_head_stats(local_short)  # (B,L,H, 4)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats = stats_short + stats_long + stats_delta + stats_value  # element-wise sum (B,L,H, 4)

        gate_inp = mx.cat([, hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1),  # (B,L,H, D)
            stats,  # (B,L,H, 4)
        ], dim=-1)  # -> (B,L,H D+4)
        gate_inp_flat = _rearrange(gate_inp, "b l h f -> (b, l, h) f")
        fusion_logits_flat = self.fusion_gate_mlp(gate_inp_flat)  # (B*L*H, 4)
        fusion_logits = _rearrange(fusion_logits_flat, "(b, l, h) c -> b l h c",
            b=gate_inp.shape[0]
        l=gate_inp.shape[1],
            h=self.num_heads)  # (B,L,H, 4)

        fusion_weights = mx.softmax(fusion_logits, dim = -1)
        if self.prob_floor > 0.0:
            fusion_weights = mx.clamp(fusion_weights, min = self.prob_floor)
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)

        # ---- compose main output ------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # ---- hybrid residual conv path ------------
        static_scale = mx.sigmoid(self.conv_residual_logit)[None, None, :, None]  # (1,1,H, 1)
        dyn_gate = mx.sigmoid(self.res_gate_proj(hidden_states))  # (B,L, H)
        gamma = static_scale * (self.residual_alpha + (1.0 - self.residual_alpha) * dyn_gate).expand_dims(-1)
        o = o + gamma * local_short

        # ---- cache update -------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L0)

        # ---- output norm / projection -------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---- re-pad if unpadded earlier -----------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B0, L0)

        return o, None, past_key_values
