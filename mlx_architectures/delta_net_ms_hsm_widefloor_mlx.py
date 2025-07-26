"""
MLX-converted architecture: delta_net_ms_hsm_widefloor
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
DeltaNet – Wide Multi-Scale Convolution + HSM with ε-Floor Gating
================================================================
Identifier: delta_net_ms_hsm_widefloor

This evolutionary variant merges the most successful components discovered so
far (multi-scale depth-wise convolution, hierarchical segment memory, per-head
temperature gating) **and** directly tackles the two weaknesses repeatedly
observed in earlier experiments:

1. **Missing Mid-Range Locality (16-64 tokens)**
   Previous *ms_hsm_tempgate* limited convolutional kernels to ≤15 and relied on
   HSM for longer context.  Benchmarks that require mid-range span extraction
   (SQuAD/SWDE) regressed.  We fix this by including a *wide* k = 31 causal
   kernel in the depth-wise convolution stack.  This adds negligible cost while
   reinstating deterministic receptive fields up to 31 tokens.

2. **Branch Starvation & Instability**
   Softmax gates can drive some paths to near-zero probability, starving them of
   gradients (observed for FIR / local conv in earlier runs).  We impose a small
   ε-floor (default 0.02) on **all** branch weights *after* softmax then
   renormalise – guaranteeing each path receives ≥ε share of the signal and
   gradients.

Architecture Overview
---------------------
Paths fused per-token & per-head (4-way gate):
  • Conv – multi-scale depth-wise conv  (k = 3,7,15,31)
  • Delta – global associative memory (chunk-wise Δ-rule)
  • HSM  – hierarchical segment averages  (scales = 1,2,4,8,16,32)
  • Id   – identity shortcut of the value projection (v_direct)

A lightweight MLP produces per-head logits which are temperature-scaled &
biased.  We then apply ε-floor renormalised softmax.

All operations remain O(N) or O(N log N) (HSM) and strictly causal.
Interfaces are fully backward compatible – no config changes required; new
features are active by default with sensible hyper-parameters.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def elu_p1(x: mx.Tensor) -> mx.Tensor:  # shifted ELU(+1)
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.Tensor) -> mx.Tensor:
    return (x / x.sum(dim=-1, keepdim=True))

# -----------------------------------------------------------------------------
# Chunk-wise Delta Rule (unchanged numerics – O(N))
# -----------------------------------------------------------------------------

@mx.compile  # noqa: D401
def delta_rule_chunkwise(
    q: mx.Tensor,  # (B,H,L,Dk)
    k: mx.Tensor,  # (B,H,L,Dk)
    v: mx.Tensor,  # (B,H,L,Dv)
    beta: mx.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Associative Δ-rule scan with causal chunking (linear time)."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # unit-norm feature map ----------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into chunks of length *chunk_size* ------------------------------
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_full = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0
    )

    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + mx.eye(chunk_size, dtype=attn_inv.dtype, q.device)
    attn_inv = attn_inv

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)
    tri_strict = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S  # recurrent state (gradient detached by caller if needed)

# -----------------------------------------------------------------------------
# Hierarchical Segment Memory (HSM) utilities – O(N log N)
# -----------------------------------------------------------------------------

@mx.compile  # noqa: D401
def _hierarchical_context(
    v: mx.Tensor,          # (B,H,L,Dv)
    gates: mx.Tensor,      # (B,H,L,S)
    scales: List[int],
) -> mx.Tensor:           # (B,H,L,Dv)
    """Content-gated causal average pooling over a pyramid of scales."""
    b, h, L, d = v.shape
    out = mx.zeros_like(v)
    v_flat = _rearrange(v, "b h l d -> (b h) d l")  # group heads for conv

    for idx, win in enumerate(scales):
        if win == 1:
            pooled = v_flat  # identity
        else:
            pad = win - 1
            pooled = F.avg_pool1d(mx.pad(v_flat, (pad, 0)), kernel_size=win, stride=1)
        pooled = _rearrange(pooled, "(b h) d l -> b h l d"b=b, h=h)
        gate = gates[..., idx].expand_dims(-1)  # (B,H,L,1)
        out = out + pooled * gate
    return out


def _get_scales(max_len: int, max_scales: int) -> List[int]:
    """Exponentially increasing window sizes <= max_len (always includes 1)."""
    scales: List[int] = [1]
    w = 2
    while len(scales) < max_scales and w <= max_len:
        scales.append(w)
        w <<= 1
    return scales

# -----------------------------------------------------------------------------
# Multi-scale depth-wise causal conv + channel mix
# -----------------------------------------------------------------------------

class MultiScaleDepthwiseConv1d(nn.Module):
    """Depth-wise causal conv at multiple kernel sizes + point-wise channel mix."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: Tuple[int, ...] | List[int] = (3, 7, 15, 31),
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        channels = num_heads * head_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=k,
                    groups=channels,  # depth-wise
                    bias=False,
                )
                for k in self.kernel_sizes
            ]
        )
        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.02)

        # Point-wise mixing across channels
        self.channel_mix = nn.Linear(head_dim * len(self.kernel_sizes), head_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_flat = _rearrange(x, "b l h d -> b (h d) l")  # group heads as channels
        outs = []
        for k_size, conv in zip(self.kernel_sizes, self.convs):
            pad = k_size - 1
            out = conv(mx.pad(x_flat, (pad, 0)))  # causal left pad
            outs.append(out)
        y = mx.cat(outs, dim=1)  # (B, H*D*|K|, L)
        y = _rearrange(y, "b (h d_mult) l -> b l h d_mult"h=h)
        y = self.channel_mix(y)  # reduce back to head_dim
        return y  # (B,L,H,D)

# -----------------------------------------------------------------------------
# Main DeltaNet layer
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 required name
    """DeltaNet with wide multi-scale conv, HSM and ε-floor gated fusion."""

    def __init__(
        self,
        *,
        mode: str = "ms_hsm_widefloor",
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
        # --- new hyper-parameters -----------------------------------
        ms_kernel_sizes: Tuple[int, ...] | List[int] = (3, 7, 15, 31),
        hsm_max_scales: int = 6,
        fusion_hidden_mult: int = 2,
        gate_floor: float = 0.02,  # ε-floor on branch weights
        # -------------------------------------------------------------
        **kwargs,  # absorb & ignore for fw-compat
    ) -> None:
        super().__init__()

        # basic bookkeeping -----------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value dim must be divisible by num_heads")

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.hsm_max_scales = hsm_max_scales
        self.gate_floor = float(gate_floor)

        # projections ------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # optional short conv on Q/K/V --------------------------------
        if self.use_short_conv:
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
                bias=conv_bias,
            )
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
                bias=conv_bias,
            )
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation="silu",
                bias=conv_bias,
            )
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet performance.")

        # multi-scale conv path --------------------------------------
        self.local_conv = MultiScaleDepthwiseConv1d(
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_sizes=ms_kernel_sizes,
        )

        # HSM gate for scale selection -------------------------------
        self.hsm_scale_gate = nn.Linear(self.head_k_dim, hsm_max_scales, bias=False)

        # fusion gate MLP (token-wise) -------------------------------
        gate_in = hidden_size + self.num_heads * 4  # hidden + 4 path norms
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 4, bias=True),
        )

        # per-head temperature & bias --------------------------------
        self.gate_log_temp = mx.array(mx.zeros(num_heads))
        self.gate_bias = mx.array(mx.zeros(num_heads, 4))

        # output normalisation --------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # unused – kept for HF API
        **kwargs: Dict,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B, L_in, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # 1. projections + optional short conv ------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # 2. head split & activations --------------------------------
        q = _rearrange(q, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k, "b l (h d) -> b l h d"d=self.head_k_dim)
        v = _rearrange(v, "b l (h d) -> b l h d"d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # 3. beta scaling factor -------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # 4. delta path ----------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # 5. local conv path -----------------------------------------
        conv_out = self.local_conv(v)

        # 6. HSM path -------------------------------------------------
        scales = _get_scales(L_in, self.hsm_max_scales)
        hsm_gate_logits = self.hsm_scale_gate(q)  # (B,L,H,S)
        hsm_gate_logits = hsm_gate_logits[..., : len(scales)]
        hsm_gates = F.softmax(_rearrange(hsm_gate_logits, "b l h s -> b h l s"), dim=-1)
        hsm_out = _hierarchical_context(v_d, hsm_gates, scales)  # (B,H,L,D)
        hsm_out = _rearrange(hsm_out, "b h l d -> b l h d")

        # 7. identity (value) path -----------------------------------
        v_direct = v  # already (B,L,H,D)

        # 8. Fusion gate ---------------------------------------------
        def _norm(t: mx.Tensor) -> mx.Tensor:
            return t.abs().mean(dim=-1)  # (B,L,H)

        fusion_feat = mx.cat([
            hidden_states,
            _rearrange(_norm(conv_out), "b l h -> b l (h)"),
            _rearrange(_norm(delta_out), "b l h -> b l (h)"),
            _rearrange(_norm(hsm_out), "b l h -> b l (h)"),
            _rearrange(_norm(v_direct), "b l h -> b l (h)"),
        ], dim=-1)

        gate_logits = self.fusion_gate_mlp(fusion_feat)  # (B,L,H*4)
        gate_logits = _rearrange(gate_logits, "b l (h c) -> b l h c"h=self.num_heads, c=4)

        temp = F.softplus(self.gate_log_temp) + 1e-3  # ensure >0
        gate_logits = gate_logits * temp.reshape(1, 1, self.num_heads, 1) + self.gate_bias.reshape(1, 1, self.num_heads, 4)

        gate_weights = F.softmax(gate_logits, dim=-1)  # (B,L,H,4)

        # ε-floor -----------------------------------------------------
        if self.gate_floor > 0.0:
            eps = self.gate_floor
            gate_weights = gate_weights * (1.0 - eps * 4) + eps  # keep sum ==1
        # no renorm needed – linear transform keeps sum to 1

        # 9. fuse paths ----------------------------------------------
        out = (
            gate_weights[..., 0:1] * conv_out
            + gate_weights[..., 1:2] * delta_out
            + gate_weights[..., 2:3] * hsm_out
            + gate_weights[..., 3:4] * v_direct
        )

        # 10. cache update -------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # 11. output normalisation & projection -----------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # 12. re-pad if we un-padded ----------------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0), indices, B, L_in)

        return out, None, past_key_values
