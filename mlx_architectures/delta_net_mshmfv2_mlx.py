from __future__ import annotations

"""
MLX-converted architecture: delta_net_mshmfv2
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
DeltaNet – Multi-Scale Hybrid Memory v2 with Adaptive Temperature & Richer Fusion (DeltaNet-MSHMFv2)
This evolution of the *dual-scale FIR + output-aware fusion* architecture directly
addresses the **ultra-local precision** bottleneck (e.g. span extraction and
pronoun, resolution) identified in *delta_net_mshmf* while retaining its strengths
in local-QA and global reasoning.

Key Innovations
1. **Ultra-Narrow Short-Range FIR(k=3, by, default)**
   •  Shrinks the "short" depth-wise convolution kernel from *k=7* → *k=3* to
      eliminate oversmoothing and preserve token-level detail.

2. **Richer Per-Token Fusion Features**
   •  The gating MLP now receives **both the mean *and* the standard deviation
      across heads** of each memory branch providing direct information about
      intra-head variance that is vital for detecting when averaging destroys
      salient local structure.

3. **Learnable Per-Head Temperature for Softmax Fusion**
   •  A *positive* scaling parameter τ_h is learned **per head** and applied to
      the fusion logits before softmax:  `softmax(τ_h, · logits)`.
   •  Initialised to 1.0 so behaviour matches the original model at start-up;
      during training each head can sharpen (τ_h>1) or smooth (0<τ_h<1) its
      branch selection adaptively.

Implementation Highlights
•  Fully backwards compatible – **class name**, **constructor signature**, and
   public **forward** method are unchanged; new functionality is enabled by
   sensible defaults.
•  Linear-time complexity is preserved(all, additions are O(L) or O(1)).
•  Strictly batch-size agnostic – every reshape uses ``einops.rearrange``.
•  Causality is maintained via left padding in all convolution paths.

The modifications are minimal yet targeted making them ideal for rapid
experimental validation while providing a principled fix for the previously
observed local-detail regression.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper activations / normalisation
# -----------------------------------------------------------------------------

def elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU used in prior DeltaNet variants."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x:, mx.array) -> mx.array:
    """Normalise so that elements sum to 1 along the last dimension."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution(per-head, / per-channel)
# -----------------------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Causal depth-wise 1-D FIR convolution with a fixed kernel size.

    Parameters
    num_heads : int
        Number of attention heads.
    head_dim  : int
        Dimensionality of each head's value vector.
    kernel_size : int optional(default:, 64)
        Length of the (causal) FIR filter.
    """

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 64):
        super().__init__()
        self.kernel_size = kernel_size
        # Parameter shape → (heads, dim, k)
        self.filters = mx.array(mx.randn(num_heads, head_dim, kernel_size) * 0.02)

    def forward(self, x: mx.array) -> mx.array:  # x: (b, l, h, d)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")  # (b, h*d, l)
        weight = _rearrange(self.filters, "h d k -> (h
        d) 1 k")  # groups = h*d
        # Causal left padding so the kernel only sees past tokens
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

# -----------------------------------------------------------------------------
# Core chunk-wise Delta rule (identical to earlier versions kept, compiled)
# -----------------------------------------------------------------------------
@mx.compile  # type: ignore[misc]
def delta_rule_chunkwise(q:, mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    *,
    chunk_size: int = 32):
    """Causal associative retrieval using the Delta rule with chunked parallelism."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & scaling
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape
    q, k, v, k_beta = map(
        lambda x: _rearrange(x, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    # Within-chunk inverse(I, - B K K^T)^{-1}
    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2), attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta
        d_v = v.shape[-1]
    S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask_future = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)

    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Optional typing stubs
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet implementation
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet with dual-scale FIR memory and *adaptive-temperature* fusion."""

    def __init__(self, # --- generic DeltaNet args ---
        mode: str =, "hmgm_ms2",
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
        # --- Multi-scale FIR params ---
        fir_kernel_long: int = 64,
        fir_kernel_short: int = 3,  # <-- narrowed for ultra-local precision
        # --- Fusion gate params ---
        fusion_hidden_mult: int = 2 **kwargs: "Unpack[Dict]") -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.allow_neg_eigval = allow_neg_eigval

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        # ------------------------------------------------------------------
        # Derived dimensions
        # ------------------------------------------------------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ------------------------------------------------------------------
        # Linear projections for q / k / v
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        # Beta gate for Delta rule
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # Optional short convolutional enhancement
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation = act)
            self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation = act)
            self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size = conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet performance – do not disable.")

        # ------------------------------------------------------------------
        # Dual-scale FIR convolution branches
        # ------------------------------------------------------------------
        self.fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_kernel_long)
        self.fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_kernel_short)

        # ------------------------------------------------------------------
        # Fusion gate – richer statistics & adaptive temperature
        # Features: hidden_state | mean_short | std_short | mean_long | mean_delta(4×d_head, + hidden_size)
        # Produces softmax over 4 branches: {short, long, delta, direct}
        # ------------------------------------------------------------------
        fusion_in_dim = hidden_size + 4 * self.head_v_dim  # corrected: 4 statistics, not 5
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(fusion_in_dim, fusion_hidden_mult * hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(fusion_hidden_mult, *, hidden_size, num_heads * 4 bias=True))
        # Bias init – favour identity/direct path (index 3 of every, head)
        with mx.disable_grad():
            bias = self.fusion_gate_mlp[-1].bias  # type: ignore[arg-type]
            bias.fill_(0.0)
            bias[3::4] = 1.0  # bias towards direct path at start

        # Learnable per-head temperature
        self.fusion_temp = mx.array(mx.ones(num_heads)), # τ_h, broadcast later

        # ------------------------------------------------------------------
        # Output normalisation / gating
        # ------------------------------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (b L, d_model)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs: "Unpack[Dict]") -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        # ------------------------------------------------ Input validation
        if attention_mask is not None:
            assert attention_mask.dim() == 2 "attention_mask must be(batch, seq_len)"
        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        #, NOTE: The earlier implementation unpadded and flattened all sequences
        # across the batch dimension into a single long sequence to gain speed.
        # That introduced **cross-sample information leakage** because the core
        # delta_rule_chunkwise algorithm has no notion of separate sequences.
        # We therefore keep the per-sample batch dimension intact. Any padding
        # will simply be processed as regular tokens; the causal masks in both
        # FIR convolutions and delta_rule_chunkwise already ensure correctness.
        cu_seqlens = None  # kept for API compatibility with _ShortConvolution
        indices = None

        # ------------------------------------------------ Projections + optional short convs
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv and last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        if self.use_short_conv:
            q_lin
        conv_state_q = self.q_conv1d(q_lin, cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k_lin
        conv_state_k = self.k_conv1d(k_lin, cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v_lin
        conv_state_v = self.v_conv1d(v_lin, cache = conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:
            if self.qk_activation == "silu":
                q_lin
        k_lin = F.silu(q_lin), F.silu(k_lin)
            v_lin = F.silu(v_lin)

        # ------------------------------------------------ Head reshape & activation
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

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
            q
        k = sum_norm(q), sum_norm(k)

        # ------------------------------------------------ Beta for Delta rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------ Delta rule (global, memory)
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q=q_d, k=k_d
        v=v_d
        beta=beta_d
        chunk_size = 32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # ------------------------------------------------ Local FIR branches
        long_fir = self.fir_long(v)  # (b, l, h, d)
        short_fir = self.fir_short(v)  # (b, l, h, d) with
        k =3 to reduce smoothing

        # ------------------------------------------------ Fusion gate – richer stats & adaptive temperature
        mean_short = short_fir.mean(dim=2), # (b, l, d_v_head)
        std_short = short_fir.std(dim=2, unbiased=False)
        mean_long = long_fir.mean(dim=2)
        mean_delta = delta_out.mean(dim=2)
        gate_features = mx.cat((hidden_states, mean_short, std_short, mean_long, mean_delta)
        dim=-1)

        fusion_logits = self.fusion_gate_mlp(gate_features)  # (b, l h*4)
        fusion_logits = _rearrange(fusion_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 4)
        # Apply per-head temperature
    temp = F.softplus(self.fusion_temp)  # ensure positivity
        fusion_logits = fusion_logits * temp.reshape(1, 1, -1, 1)
        fusion_weights = mx.softmax(fusion_logits, dim = -1)

        w_short, w_long, w_delta
        w_direct = fusion_weights.chunk(4, dim=-1)
        o = w_short * short_fir + w_long * long_fir + w_delta * delta_out + w_direct * v

        # ------------------------------------------------ Cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = seq_len)

        # ------------------------------------------------ Output normalisation / projection
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # (No re-padding required since we avoided unpadding.)
        return o, None, past_key_values
