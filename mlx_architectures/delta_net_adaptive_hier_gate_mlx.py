"""
MLX-converted architecture: delta_net_adaptive_hier_gate
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
DeltaNet – Adaptive Hierarchical Gating with Learnable Floor (AHG)
=================================================================
Identifier: delta_net_adaptive_hier_gate

Key Innovations
---------------
1. Adaptive ε-Floor Gating
   • A *learnable* per-head parameter controls the minimum share (ε_h ∈ [0, ε_max])
     that each memory path receives.  This preserves gradient flow early in
     training yet allows the network to anneal the floor towards zero when a
     head benefits from sharper, more selective routing.
   • Combined with a learnable per-head **temperature** (τ_h) the gate can
     smoothly interpolate between uniform blending and near hard selection –
     recovering the best of both ε-stable and sharp-temperature variants.

2. Identity-Initialised Wide Depth-wise Convolution
   • The multi-scale local path now includes kernels (3, 7, 15, 31) whose
     *central/last* weight is initialised to 1.0 (identity FIR).  The very wide
     k=31 kernel particularly benefits mid-range span tasks while avoiding
     early signal wash-out.

3. Expanded Kernel Spectrum (+k=1 Passthrough)
   • A k=1 depth-wise convolution branch (effectively an extra linear path)
     is added, giving the gate another fine-grained local alternative that can
     be mixed independently of the direct value path.

4. Output-Aware Gate Features
   • The gate MLP receives branch L1 norms (‖·‖₁) in addition to hidden state
     embeddings, enabling *output-aware* routing without expensive extra
     projections.

All operations preserve O(L) complexity and strict causality.  The class name
and public interface remain unchanged – this is a drop-in replacement.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def elu_p1(x: mx.Tensor) -> mx.Tensor:
    """Shifted ELU (+1) keeps outputs positive like SILU but cheaper."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.Tensor) -> mx.Tensor:
    """Normalise so that values sum to 1 along the last dimension."""
    return (x / x.sum(dim=-1, keepdim=True))

# -----------------------------------------------------------------------------
# Chunk-wise Delta rule (unchanged numerics – linear time)
# -----------------------------------------------------------------------------

@mx.compile
def _delta_rule_chunkwise(
    q: mx.Tensor,  # (B,H,L,Dk)
    k: mx.Tensor,  # (B,H,L,Dk)
    v: mx.Tensor,  # (B,H,L,Dv)
    beta: mx.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Associative Δ-rule scan with causal chunking (O(L))."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Unit-norm feature map ----------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks of length *chunk_size* ------------------------------
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_full = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0
    )
    tri_strict = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1
    )

    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + mx.eye(chunk_size, dtype=attn_inv.dtype, q.device)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Multi-Scale Depth-wise Convolution (identity initialised)
# -----------------------------------------------------------------------------

class _DepthwiseMultiScaleConv(nn.Module):
    """Parallel depth-wise causal convolutions with identity initialisation."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15, 31),
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        channels = num_heads * head_dim
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv1d(
                channels,
                channels,
                kernel_size=k,
                groups=channels,
                bias=False,
            )
            # Identity init: make the last weight 1 so the path starts as passthrough
            with mx.no_grad():
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1.0
            self.convs.append(conv)

        # Point-wise mix to fuse different kernel outputs
        self.channel_mix = nn.Linear(head_dim * len(kernel_sizes), head_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_flat = _rearrange(x, "b l h d -> b (h d) l")
        outs: List[mx.Tensor] = []
        for k_size, conv in zip(self.kernel_sizes, self.convs):
            pad = k_size - 1  # causal left pad
            y = conv(mx.pad(x_flat, (pad, 0)))
            outs.append(y)
        y_cat = mx.cat(outs, dim=1)  # (B, H*D*|K|, L)
        y = _rearrange(y_cat, "b (h d_mult) l -> b l h d_mult"h=h)
        y = self.channel_mix(y)
        return y  # (B,L,H,D)

# -----------------------------------------------------------------------------
# Optional typing stubs
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# DeltaNet – Adaptive Hierarchical Gate variant
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401
    """DeltaNet with Adaptive ε-Floor & Temperature Gating over Local/Global/Value paths."""

    def __init__(
        self,
        *,
        mode: str = "adaptive_hier_gate",
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
        # --- new hyper-parameters -----------------------------------------
        ms_kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15, 31),
        gate_hidden_mult: int = 2,
        gate_eps_max: float = 0.05,  # upper bound for ε
        gate_eps_init: float = 0.02,  # initial ε value
        # -------------------------------------------------------------------
        **kwargs: "Unpack[Dict]",  # noqa: F722 type comment
    ) -> None:
        super().__init__()

        # Store / validate basic parameters
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
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
        self.layer_idx = layer_idx or 0
        self.gate_eps_max = float(gate_eps_max)

        # Linear projections --------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Optional short convolution pre-processing ---------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias
            )
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias
            )
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias
            )
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet variants.")

        # Multi-scale local convolution path ----------------------------------
        self.local_conv = _DepthwiseMultiScaleConv(
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_sizes=ms_kernel_sizes,
        )

        # ------------- Adaptive fusion gate ----------------------------------
        self.num_streams = 3  # conv, delta, value
        gate_in_dim = hidden_size + num_heads * self.num_streams  # hidden + norms
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_size * gate_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * gate_hidden_mult, num_heads * self.num_streams, bias=True),
        )

        # Per-head temperature (sharpness)
        self.gate_log_temp = mx.array(mx.zeros(num_heads))
        # Per-head learnable ε floor (initialised to gate_eps_init)
        init_eps_val = math.log(gate_eps_init / (gate_eps_max - gate_eps_init + 1e-6))
        self.gate_logit_eps = mx.array(mx.full((num_heads,), init_eps_val))
        # Per-head bias to favour value path early (like DMGHM)
        self.gate_bias = mx.array(mx.zeros(num_heads, self.num_streams))
        with mx.no_grad():
            self.gate_bias[:, -1] += 0.1  # slight bias to identity/value path

        # Output norm & projection -------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B, L_in, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).expand_dims(0)

        # ---------------- projections + optional short conv ----------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # ---------------- split heads --------------------------------------
        q = _rearrange(q, "b l (h d) -> b l h d"h=self.num_heads)
        k = _rearrange(k, "b l (h d) -> b l h d"h=self.num_heads)
        v = _rearrange(v, "b l (h d) -> b l h d"h=self.num_heads)

        # ---------------- activation & normalisation -----------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q = sum_norm(q)
            k = sum_norm(k)

        # ---------------- beta scaling ------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Delta path --------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")

        delta_out_d, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---------------- Local convolution path --------------------------
        conv_out = self.local_conv(v)  # (B,L,H,D)

        # ---------------- Identity/value path -----------------------------
        value_out = v  # (B,L,H,D)

        # ---------------- Build features for gate -------------------------
        def _norm(t: mx.Tensor) -> mx.Tensor:  # (B,L,H)
            return t.abs().mean(dim=-1)

        gate_feat = mx.cat(
            [
                hidden_states,
                _rearrange(_norm(conv_out), "b l h -> b l (h)"),
                _rearrange(_norm(delta_out), "b l h -> b l (h)"),
                _rearrange(_norm(value_out), "b l h -> b l (h)"),
            ],
            dim=-1,
        )

        gate_logits = self.fusion_gate_mlp(gate_feat)  # (B,L,H*streams)
        gate_logits = rearrange(
            gate_logits, "b l (h s) -> b l h s", h=self.num_heads, s=self.num_streams
        )

        # Temperature & bias -------------------------------------------------
        temp = F.softplus(self.gate_log_temp) + 1e-3  # ensure >0
        gate_logits = gate_logits * temp.reshape(1, 1, self.num_heads, 1)
        gate_logits = gate_logits + self.gate_bias.reshape(1, 1, self.num_heads, self.num_streams)

        gate_soft = F.softmax(gate_logits, dim=-1)  # (B,L,H,S)

        # Adaptive ε floor ----------------------------------------------------
        eps_head = mx.sigmoid(self.gate_logit_eps) * self.gate_eps_max  # (H)
        eps_head = eps_head.reshape(1, 1, self.num_heads, 1)  # broadcast
        gate_weights = gate_soft * (1.0 - self.num_streams * eps_head) + eps_head
        # No re-normalisation needed – linear transform keeps sum to 1

        # ---------------- Fuse paths ---------------------------------------
        out = (
            gate_weights[..., 0:1] * conv_out
            + gate_weights[..., 1:2] * delta_out
            + gate_weights[..., 2:3] * value_out
        )

        # ---------------- Cache update -------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ---------------- Output norm / projection -------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # ---------------- Re-pad if unpadded -------------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0), indices, B, L_in)

        return out, None, past_key_values
