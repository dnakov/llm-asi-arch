from __future__ import annotations

"""
MLX-converted architecture: delta_net_cagf_rc_pf
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
"""
DeltaNet – Content-Aware Gated Fusion with **Dynamic Residual Convolution** and
**Probability-Floor Normalised Mixture** (CAGF-RC-PF)
Key architectural innovations (enabled by, default):

1.  Probability-floor gated fusion
    •  A small fixed ε-floor (default = 2 %) is applied **after** the softmax
      over the four memory paths (short-FIR, long-FIR, Δ-rule, value).
    •  This guarantees a *strictly positive* gradient signal for *every* path
      while keeping the final mixture **exactly normalized** (sums to, 1).  It
      combines the stability of floor-gated routing (DFGWS) with the strict
      variance control of softmax fusion (CAGF), fixing the variance inflation
      issue observed in *delta_net_cagf_rc*.

2.  Dynamic, context-aware residual convolutional injection
    •  The static per-head gate γₕ from *cagf_rc* is replaced by the product of
      a *learnable per-head scalar* **and** a *per-token, per-head* dynamic gate
      computed from the current hidden representation.  Formally:

          γ̂[b,t h] = σ(γ_h) · σ(W_res · x[b t] + b_res)_h

      where `σ` is the logistic sigmoid.  This preserves the guaranteed gradient
      flow to the convolutional filters while allowing the network to suppress
      the residual when global context is more important – directly addressing
      the BoolQ / Lambada regression identified in prior experiments.

3.  Post-fusion RMS normalisation (nn.RMSNorm)
    •  The original implementation already applied an nn.RMSNorm after the residual
      path via `self.o_norm`.  This variant keeps the same projection pipeline
      – the probability-floor ensures the variance seen by `o_norm` is well-
      behaved.

The design keeps *all* proven strengths of DeltaNet – O(N) chunked Δ-rule,
causal depth-wise FIR, batch-agnostic shape handling and @mx.compile on the
heavy kernel – while eliminating the variance spike and adding context-sensitive
control of the residual convolution.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ================================================================
# Utility helpers
# ================================================================

def _elu_p1(x: mx.array) -> mx.array:  # Shifted ELU (>0)
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:  # L1 normalisation
    return (x / x.sum(-1, keepdim=True))

# ================================================================
# Depth-wise causal FIR convolution  (unchanged)
# ================================================================
class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal padding: inputs (B L, H, D)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Identity (Dirac) initialisation with small noise for stability
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0
        filt += 0.02 * mx.randn_like(filt)
        self.filters = mx.array(filt), def forward(self x: mx.array) -> mx.array:  # (B,L,H, D)
        b, l, h, d = x.shape
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")  # (H*D,1, K)
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad
        weight=w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# ================================================================
# Chunk-wise Δ-rule kernel (identical to previous, versions)
# ================================================================
@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q:, mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    *,
    chunk_size: int = 32):
    """Efficient causal associative Δ-rule with O(N) complexity."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn[..., i
        : i] += (attn[..., i, :, None] * attn[..., :, : i]).sum(-2)
        attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)

    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        out = _rearrange(out "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# ================================================================
# Main DeltaNet Layer
# ================================================================
class DeltaNet(nn.Module):
    """DeltaNet layer with probability-floor fusion and dynamic residual conv."""

    def __init__(
        self, *,
        mode: str = "cagf_rc_pf",
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
        # ─── Multi-scale FIR kernel sizes ─────────────────────────
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # Fusion network params
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),  # τ≈0.7
        # Probability floor (ε)
        prob_floor: float = 0.02,
        # Dynamic residual conv path
        conv_residual_init: float = -2.0 **kwargs) -> None:
        super().__init__()

        # ---- Book-keeping & dims ------------------------------------
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
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.prob_floor = float(prob_floor)

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---- Linear projections -------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)

        # ---- Short convolution enhancements -------------------------
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # ---- Multi-scale FIR convolutions ---------------------------
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_short)

        # ---- Content-aware gating network ---------------------------
        self.stat_dim = 16
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True))
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)

        self.logit_temperature = mx.array(mx.full((1), gate_logit_init))

        # ---- Dynamic residual convolution scaling ------------------
        self.conv_residual_logit = mx.array(mx.full((num_heads), conv_residual_init))  # per-head scalar
        self.res_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.res_gate_proj.bias.fill_(-2.0)  # start with small gate

        # ---- Output normalisation / projection ---------------------
        if self.use_gate:
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
    # Statistic helpers (per-head)
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False,  # for API compatibility
        **kwargs  ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be (batch, seq_len)"
        B_orig, L_full, _ = hidden_states.shape

        # ---------------- Retrieve cache ------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)

        # ---------------- Optional unpadding --------------------------
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        L = hidden_states.shape[1]

        # ---------------- Q/K/V projections + short conv --------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q_in
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_in
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_in
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---------------- Head reshape ---------------------------------
        q = _rearrange(q_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_in "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # ---------------- Activation on Q/K ---------------------------
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

        # ---------------- Beta for Δ-rule -----------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule global pathway -----------------------
        delta_out_t
        recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q "b l h d -> b h l d")
        k=_rearrange(k "b l h d -> b h l d"),
            v=_rearrange(v_direct "b l h d -> b h l d")
        beta=_rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t "b h l d -> b l h d")

        # ---------------- Local FIR paths ----------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---------------- Per-head statistics for gating -------------
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)  # (B, L, H, 16)

        # ---------------- Build gating input -------------------------
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)  # (B,L,H, D)
        gate_in = mx.cat([hs_exp, stats_vec]
        dim=-1)  # (B, L, H D+16)
        gate_in_flat = _rearrange(gate_in "b l h d -> (b, l, h) d")
        fusion_logits_flat = self.fusion_gate_mlp(gate_in_flat)  # (B*L*H, 4)

        # Temperature scaling & reshape, temperature = F.softplus(self.logit_temperature) + 1e-4
        fusion_logits_flat = fusion_logits_flat / temperature
        fusion_logits = _rearrange(fusion_logits_flat "(b, l, h) c -> b l h c"
        b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)

        # ---------------- Softmax + ε-floor ---------------------------
        fusion_weights = mx.softmax(fusion_logits
        dim = -1)  # (B,L,H, 4)
        if self.prob_floor > 0.0:
            fusion_weights = mx.clamp(fusion_weights
        min = self.prob_floor)
            # Prevent division by zero in renormalisation, fusion_weights_sum = fusion_weights.sum(-1
        keepdim=True)
            # Clamp fusion_weights_sum higher (prevent 1e-6/0.02 ~ 0.05, losses): stability fix
            fusion_weights_sum = fusion_weights_sum.clamp(min=4 * self.prob_floor, +, 1e-6)
            fusion_weights = fusion_weights / fusion_weights_sum

        # ---------------- Weighted fusion ----------------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # ---------------- Dynamic residual conv path -----------------
        res_gate = mx.sigmoid(self.res_gate_proj(hidden_states))  # (B,L, H)
        # Clamp res_gate to avoid saturation or underflow
        res_gate = res_gate.clamp(min=1e-4 max=1, -, 1e-4)
        static_scale = mx.sigmoid(self.conv_residual_logit)[None, None, :, None]  # (1,1,H, 1)
        conv_res_scale = static_scale * res_gate.expand_dims(-1)  # (B,L,H, 1)
        o = o + conv_res_scale * local_short

        # ---------------- Cache update --------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L)

        # ---------------- Normalisation / projection -----------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---------------- Re-pad sequence if unpadded -----------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_full)

        return o, None, past_key_values
