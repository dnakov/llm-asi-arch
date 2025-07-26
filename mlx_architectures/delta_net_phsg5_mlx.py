"""
MLX-converted architecture: delta_net_phsg5
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions (replacing PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
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
    return x / mx.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)

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
    """MLX replacement for FLA ShortConvolution"""
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)
        self.activation = activation
        
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
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
            return out, None  # Simplified - no cache state
        return out

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

    hidden_states = _index_first_axis(...).expand_dims(0)  # -> batch = 1

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
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# ============================================================================
# Helper utilities
# ============================================================================

def elu_p1(x: mx.Tensor) -> mx.Tensor:  # shifted ELU so output >0
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.Tensor) -> mx.Tensor:  # L1 normalise last dim
    return (x / x.sum(-1, keepdim=True))

# ============================================================================
# Depth-wise causal FIR convolution (identity initialisation)
# ============================================================================
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution with δ-kernel initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # (H, D, K)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity at time-step 0 (causal)
        self.filters = mx.array(filt)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k -> (h d) 1 k")  # groups=h*d
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # left pad – causal
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

# ============================================================================
# Causal chunk-wise Δ-rule kernel (unchanged, proven baseline)
# ============================================================================
@mx.compile  # type: ignore[misc]
def delta_rule_chunkwise(
    q: mx.Tensor,  # (B,H,L,Dk)
    k: mx.Tensor,  # (B,H,L,Dk)
    v: mx.Tensor,  # (B,H,L,Dv)
    beta: mx.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
) -> Tuple[mx.Tensor, mx.Tensor]:
    """Causal associative Δ-rule evaluated in fixed-size chunks (O(N·d))."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & beta scaling
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk view -> (B,H,N,C,D)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + mx.eye(chunk_size, dtype=attn_inv.dtype, q.device)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)
    mask_future = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
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
        weight = mx.zeros(num_heads, n_paths, hidden_size)
        # kaiming-like init per head
        bound = 1.0 / math.sqrt(hidden_size)
        weight.uniform_(-bound, bound)
        self.weight = mx.array(weight)  # (H, P, D)
        self.bias = mx.array(mx.zeros(num_heads, n_paths))  # (H, P)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B,L,D)
        # logits: (B,L,H,P)
        logits = mx.einsum("b l d, h p d -> b l h p", x, self.weight) + self.bias
        return logits

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
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # ---- multi-scale FIR branches ----
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_mid = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_mid)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---- per-head simplex gate ----
        self.n_paths = 5  # short, mid, long, delta, value
        self.gate_linear = PerHeadGate(hidden_size, num_heads, self.n_paths)
        # learnable temperature per head
        self.log_temp = mx.array(mx.full((num_heads, 1), math.log(gate_temp_init)))
        # learnable ε-floor per head (clamped in forward)
        self.eps_param = mx.array(mx.full((num_heads, 1), gate_eps_init))

        # ---- output normalisation / projection ----
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _apply_temperature_and_floor(self, logits: mx.Tensor) -> mx.Tensor:
        """Apply per-head temperature and ε-floor to logits then return probs."""
        # logits: (B,L,H,P)
        temp = mx.exp(self.log_temp).reshape(1, 1, -1, 1)  # (1,1,H,1)
        probs = mx.softmax(logits / temp, dim=-1)
        eps = mx.clamp(self.eps_param, 0.0, 0.2).reshape(1, 1, -1, 1)
        k = self.n_paths
        probs = probs * (1.0 - k * eps) + eps  # ensure ≥eps & sum-to-1
        return probs

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # unused, kept for API
        **kwargs: Dict,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:
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
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ------------------------------------------------------------------
        # 2. QKV projections + optional short-conv (no un-padding)
        # ------------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        # _ShortConvolution kernels accept `cu_seqlens=None` and will default to
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
        q = _rearrange(q, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k, "b l (h d) -> b l h d"d=self.head_k_dim)
        v_direct = _rearrange(v, "b l (h d) -> b l h d"d=self.head_v_dim)

        # ------------------------------------------------------------------
        # 4. Activations / normalisation on Q/K
        # ------------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
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
            beta = self.b_proj(hidden_states).sigmoid()  # (B,L,H)
        else:
            beta = mx.ones((*hidden_states.shape[:2], self.num_heads), dtype=q.dtype, q.device)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # 6. Δ-rule path (causal, chunk-wise)
        # ------------------------------------------------------------------
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=_rearrange(q, "b l h d -> b h l d"),
            k=_rearrange(k, "b l h d -> b h l d"),
            v=_rearrange(v_direct, "b l h d -> b h l d"),
            beta=_rearrange(beta, "b l h -> b h l"),
        )
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

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
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=L,
            )

        # ------------------------------------------------------------------
        # 10. Output projection & norm
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # No re-padding needed – batch structure preserved.
        return o, None, past_key_values
