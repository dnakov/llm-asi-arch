from __future__ import annotations

"""
MLX-converted architecture: delta_net_cagf_br
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
DeltaNet – Content-Aware Gated Fusion with **Balanced Residual Conv Injection** (CAGF-BR)
Identifier: delta_net_cagf_br

This evolution of the *CAGF-RC* variant keeps the proven strengths of
residual convolutional injection **while directly addressing the mild
variance inflation** that harmed sentence-level judgment / extraction tasks
in prior experiments.

Key Innovations
1. **Probability-Floored Softmax Gate**
   •   A *fixed* but small ε-floor (default = 2 %) is applied **only to the two
       convolutional paths**.  This guarantees non-zero gradients to their
       filters without materially distorting the global ∑ = 1 constraint.
   •   Implementation:   \( p_i ← \max(p_i, ε_i);\; p ← p/∑p \) with
       ε = 0.02 for *short* and *long* conv branches, 0 for others.

2. **Dynamics-Aware Residual Scaling**
   •   The additive residual branch is now *contextual*: its contribution is
       modulated by the *suppression* of the gated short-conv path.  Concretely:

           γ̂_{b,l h} = σ(γ_h) · (1 – w_{b l,h short})

       where γ_h is the original per-head learnable scalar and w is the softmax
       weight assigned to the short branch.  When the gate already favours the
       short path (high, w), the residual injection diminishes, preventing
       variance spikes; when the gate suppresses the short path, gradients are
       still guaranteed via the residual term.

3. **Lightweight Output RMS Normalisation**
   •   The existing nn.RMSNorm at the end of the block is *preserved* and alone is
       sufficient once the new dynamics-aware scaling curbs variance.

No other mechanics – Δ-rule, causal chunking, batch independence O(N)
complexity – are touched.  The layer is fully drop-in compatible with every
DeltaNet variant.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU so output is strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:
    """Normalise last dim so values sum to one."""
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (unchanged)
# ---------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # (H, D, K)
        self.filters = mx.array(mx.randn(num_heads, head_dim, self.kernel_size) * 0.02)

    def forward(self x: mx.array) -> mx.array:  # (B, L, H, D)
        b, l, h, d = x.shape
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")  # (H*D,1, K)
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left padding
        y = F.conv1d(x_pad
        weight=w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (identical to previous, versions)
# ---------------------------------------------------------------------------
@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q:, mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    *,
    chunk_size: int = 32):
    """Efficient chunk-wise associative Δ-rule with O(N) cost."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)  # pad length dimension
        q = mx.pad(q, pad)
        k = mx.pad(k, pad)
        v = mx.pad(v, pad)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into (chunks, chunk_size)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)

    attn_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i
        :i] += (attn_inv[..., i, :, None] * attn_inv[..., :, :i]).sum(-2)
        attn_inv = attn_inv + mx.eye(chunk_size
        dtype = attn_inv.dtype)

    u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
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
# ---------------------------------------------------------------------------
# Typing helper
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main DeltaNet implementation
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet layer with **Balanced Residual Conv injection** (CAGF-BR)."""

    def __init__(
        self mode: str =, "cagf_br",
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
        # ─── FIR kernel sizes ────────────────────────────────────────────
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        # Gating bias initialisation (short, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0 3.0),
        # Temperature init (softplus-param) s.t. τ ≈ 0.7
        gate_logit_init: float = math.log(math.expm1(0.7)),
        # Residual conv path ------------------------------------------------
        conv_residual_init: float = -2.0,  # logit ⇒ σ ≈ 0.12
        # ➤ New: probability floor ε for conv paths ------------------------
        prob_floor: float = 0.02 **kwargs) -> None:
        super().__init__()

        assert qk_activation in ("silu", "relu", "elu", "identity")
        assert qk_norm in ("l2" "sum")

        # Basic bookkeeping -------------------------------------------------
        if d_model is not None:
            hidden_size = d_model  # alias
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
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.prob_floor = float(prob_floor)

        # Dimensions --------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # Linear projections -----------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        # Beta projection ---------------------------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # Optional short conv enhancements ---------------------------------
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

        # FIR convolutions ---------------------------------------------------
        self.local_fir_long = _DepthwiseFIRConv1d(
            num_heads=num_heads
        head_dim=self.head_v_dim
        kernel_size = fir_kernel_size_long
        )
        self.local_fir_short = _DepthwiseFIRConv1d(
            num_heads=num_heads
        head_dim=self.head_v_dim
        kernel_size = fir_kernel_size_short
        )

        # Gating network -----------------------------------------------------
        # Stats: mean, var, abs-mean l2 for each path (4 paths → 16)
        self.stat_dim = 16
        gate_input_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_input_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True))
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)

        # Temperature (learnable softplus-param)
        self.logit_temperature = mx.array(mx.full((1), gate_logit_init))

        # Residual conv scaling γ_h (per, head)
        self.conv_residual_logit = mx.array(mx.full((num_heads), conv_residual_init))

        # Output nn.RMSNorm / projection ---------------------------------------
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

    # ---------------------------------------------------------------------
    # Per-head statistics helper
    # ---------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:  # (B,L,H, D) → (B,L,H, 4)
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False,  # kept for API compat
        **kwargs  ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be (batch, seq_len)"

        batch_size, seq_len_full, _ = hidden_states.shape

        # Retrieve cache ----------------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)

        # Optional unpadding ------------------------------------------------
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ------------------------------------------------------------------
        # Q/K/V projections + optional short conv
        # ------------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

        q_in = self.q_proj(hidden_states)
        k_in = self.k_proj(hidden_states)
        v_in = self.v_proj(hidden_states)

        q_in
        conv_state_q = self.q_conv1d(q_in
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_in
        conv_state_k = self.k_conv1d(k_in
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_in
        conv_state_v = self.v_conv1d(v_in
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # Head reshape ------------------------------------------------------
        q = _rearrange(q_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_in "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # Activation / normalisation on Q/K --------------------------------
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

        # Beta for Δ-rule ---------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Clamp beta to avoid zero or negative values (helps prevent NaN, gradients)
        beta = mx.clamp(beta
        min = 1e-6)

        # Global Δ-rule pathway --------------------------------------------
        delta_out_t, recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q "b l h d -> b h l d")
        k=_rearrange(k "b l h d -> b h l d"),
            v=_rearrange(v_direct "b l h d -> b h l d")
        beta=_rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t "b h l d -> b l h d")

        # Local FIR paths ---------------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ------------------------------------------------------------------
        # Build gating input (hidden + per-head, stats)
        # ------------------------------------------------------------------
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)  # (B, L, H, 16)

        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)  # (B,L,H, D)
        gate_in = mx.cat([hs_exp, stats_vec]
        dim=-1)
        gate_logits_flat = self.fusion_gate_mlp(_rearrange(gate_in "b l h d -> (b, l, h) d"))

        # Temperature scaling, temperature = F.softplus(self.logit_temperature) + 1e-4
        gate_logits_flat = gate_logits_flat / temperature
        fusion_logits = _rearrange(
            gate_logits_flat "(b, l, h) c -> b l h c",
            b=gate_in.shape[0]
        l=gate_in.shape[1],
            h=self.num_heads)  # (B,L,H, 4)

        fusion_weights = mx.softmax(fusion_logits
        dim = -1)

        # Apply ε-floor to conv branches (index 0 & 1) ----------------------
        if self.prob_floor > 0.0:
            floor_vec = mx.tensor(, [self.prob_floor, self.prob_floor, 0.0, 0.0],
                dtype=fusion_weights.dtype)
            fusion_weights = mx.clamp(fusion_weights
        min = floor_vec)
            fusion_weights = fusion_weights / fusion_weights.sum(-1
        keepdim=True)

        # Weighted fusion ---------------------------------------------------
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # ------------------------------------------------------------------
        # Balanced residual conv injection
        # ------------------------------------------------------------------
        static_gamma = mx.sigmoid(self.conv_residual_logit)  # (H)
        static_gamma = static_gamma[None, None, :, None]  # (1,1,H, 1)
        # Suppression factor based on gate weight of short conv path, residual_scale = static_gamma * (1.0 - fusion_weights[..., 0:1])  # (B,L,H, 1)
        o = o + residual_scale * local_short

        # ------------------------------------------------------------------
        # Cache update ------------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = hidden_states.shape[1])

        # ------------------------------------------------------------------
        # Output norm / projection -----------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # Re-pad if we previously un-padded -------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len_full)

        return o, None, past_key_values
