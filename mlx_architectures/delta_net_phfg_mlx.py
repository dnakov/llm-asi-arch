from __future__ import annotations

"""
MLX-converted architecture: delta_net_phfg
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
DeltaNet – Parallel–Hierarchical Fusion Gate (DeltaNet-PHFG)
Identifier: delta_net_phfg

This evolution integrates the *most successful* ideas from previous
experiments while directly addressing their limitations:

    •   It keeps the proven ingredients
            –  Dirac-initialised depth-wise FIR filters
            –  Correct warm-start bias on the direct value path
            –  Head-wise routing for per-head specialisation
            –  ε-floors to avoid gradient starvation
    •   It resolves the **local ↔ global trade-off** introduced by hard
        competitive gating by switching to a *parallel–hierarchical* gate:
            1. A *sigmoid*  **group gate** decides the proportion of
               probability mass that flows to the **Local** (short & long, FIR) versus **Global** (Δ-rule & value) group.  Because it is a
               sigmoid(not, a, softmax) the two groups are *independent* –
               increasing one does **not** strictly decrease the other.
            2. Inside each group a per-head *softmax* distributes that
               group’s mass between its two paths(short, ↔ long or
               delta ↔ value).
            3. A small **ε-floor** (default 0.02) is mixed into every path
               *before* normalisation ensuring non-zero gradients.

This design retains the stabilising effect of an identity-biased value path
while guaranteeing that *all* branches retain trainable signal throughout
training.  The group-level sigmoid gate removes the destructive
zero-sum competition that plagued previous hierarchical variants and is
inspired by recent successes of parallel SSM/attention hybrids such as
Block-State Transformers.

All operations remain **O(N)** with strict causal masking, and the public
API(`DeltaNet`, class name constructor and `forward` signature) is fully
preserved making this a drop-in upgrade.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_plus_one(x:, mx.array) -> mx.array:
    """Shifted ELU (ELU+1) that stays positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """Normalise last dim to sum to 1."""
    return(x, / x.sum(dim=-1
        keepdim=True))

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule(identical, to baseline – kept in a separate @mx.compile)
# -----------------------------------------------------------------------------

@mx.compile  # noqa: D401 – core hot-path kernel
# pylint: disable=too-many-locals,too-many-statements
def _delta_rule_chunkwise(q:, mx.array,  # (B,H,L, D_k)
    k: mx.array,  # (B,H,L, D_k)
    v: mx.array,  # (B,H,L, D_v)
    beta: mx.array,  # (B H, L)
    *,
    chunk_size: int = 32):
    """Causal O(N) Δ-rule evaluated in fixed-size chunks."""
    b, h, L, d_k = q.shape  # noqa: F841 – d_k used implicitly later
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise q/k and apply β-scaling to v & k
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into (block, chunk) views
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=0)
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (
            inv[..., i, :, None] * inv[..., :, :i]
        ).sum(-2), inv = inv + mx.eye(chunk_size, dtype = inv.dtype)
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    tri_future = mx.triu(tri_mask, diagonal = 1)

    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S  # (B H,L, D_v), recurrent state

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac, initialised)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise 1-D convolution with Dirac initialisation."""

    def __init__(, self,
        num_heads: int,
        head_dim: int,
        *,
        kernel_size: int = 31
        init_std: float = 0.02) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0  # causal identity
            if init_std > 0:
                weight.add_(mx.randn_like(weight) * init_std)
        self.filters = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Optional external type imports
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet class(Parallel–Hierarchical, Fusion, Gate)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – name must stay exactly "DeltaNet"
    """DeltaNet layer with parallel–hierarchical fusion gating."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, # ---- baseline args -------------------------------------------
        mode: str =, "phfg",
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
        # ---- FIR kernels ---------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # ---- gating params -------------------------------------------
        gate_eps_floor: float = 0.02,
        gate_group_bias: float = 2.0,  # favour global group initially
        gate_value_bias: float = 4.0,  # favour identity path inside global
        gate_hidden_mult: int = 2,
        gate_dropout: float = 0.0 **kwargs: Dict) -> None:
        super().__init__()
        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        # ---- resolve hidden_size param --------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.gate_eps_floor = gate_eps_floor

        # ---- dimensions ----------------------------------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value, dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---- projections ---------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # ---- short convolutions --------------------------------------
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
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet variants.")

        # ---- FIR branches --------------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)

        # ---- gating modules ------------------------------------------
        gate_in_dim = hidden_size + num_heads * 4  # hidden + 4 per-head norm summaries
        hidden_dim = hidden_size * gate_hidden_mult

        self.gate_backbone = nn.Sequential(, nn.Linear(gate_in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(gate_dropout) if gate_dropout > 0.0 else nn.Identity())
        # Per-head projections ----------------------------------------
        self.group_gate_proj = nn.Linear(hidden_dim, num_heads
        bias=True)  # sigmoid – one logit per head
        self.local_sub_proj = nn.Linear(hidden_dim, num_heads * 2
        bias=True)  # softmax over short/long
        self.global_sub_proj = nn.Linear(hidden_dim, num_heads * 2
        bias=True)  # softmax over delta/value

        # ---- bias initialisation ------------------------------------
        with mx.disable_grad():
            self.group_gate_proj.bias.fill_(gate_group_bias)  # push mass to global early
            # local sub-gate: no bias (equal, start)
            # global sub-gate: bias towards value path
    glob_bias_view = self.global_sub_proj.bias.reshape(num_heads, 2)
            glob_bias_view[:, 1] = gate_value_bias  # index 1 -> value path

        # Temperature parameter(one, per, head) for stability ------------
        self.logit_scale = mx.array(mx.zeros(num_heads)), # starts at 1.0 after exp

        # ---- output norm / projection -------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-locals,too-many-statements,too-branches
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False # kept for API compat
        **kwargs: Dict) -> Tuple[mx.array, None Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape

        # ---- optional unpadding -------------------------------------
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---- retrieve cached conv state -----------------------------
        conv_q = conv_k = conv_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state is not None and self.use_short_conv:
                conv_q
        conv_k, conv_v = last_state.get("conv_state", (None None, None))

        # ---- projections + short conv -------------------------------
        q, conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---- head split --------------------------------------------
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---- activations & normalisation ---------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---- β coefficients for Δ-rule -----------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global memory ----------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---- FIR local branches ------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ---- Gate feature construction -----------------------------
        def _norm(t:, mx.array) -> mx.array:  # (B,L,H, D) -> (B,L, H)
            return t.abs().mean(dim=-1)
        gate_feat = mx.cat([, hidden_states _rearrange(_norm(local_short), "b l h -> b l (h)"),
                _rearrange(_norm(local_long), "b l h -> b l (h)"),
                _rearrange(_norm(delta_out), "b l h -> b l (h)"),
                _rearrange(_norm(v_direct), "b l h -> b l (h)"),
            ],
            dim=-1)

        backbone_out = self.gate_backbone(gate_feat)
        scale = mx.exp(self.logit_scale).reshape(1, 1, self.num_heads)

        # ---- Group gate (sigmoid) -----------------------------------
        group_logits = _rearrange(self.group_gate_proj(backbone_out), "b l h -> b l h 1") / scale.expand_dims(-1)
        group_prob_global = mx.sigmoid(group_logits)  # (B L,H, 1)
        group_prob_local = 1.0 - group_prob_global

        # ---- Sub-gates inside each group ----------------------------
        local_logits = _rearrange(
            self.local_sub_proj(backbone_out), "b l (h, c) -> b l h c", h=self.num_heads
        c = 2
        ) / scale.expand_dims(-1)
        local_weights = mx.softmax(local_logits, dim = -1)  # short/long
        global_logits = _rearrange(
            self.global_sub_proj(backbone_out), "b l (h, c) -> b l h c", h=self.num_heads
        c = 2
        ) / scale.expand_dims(-1)
        global_weights = mx.softmax(global_logits, dim = -1)  # delta/value

        # ---- Compose final path weights -----------------------------
        w_short = group_prob_local * local_weights[..., 0:1]
        w_long = group_prob_local * local_weights[..., 1:2]
        w_delta = group_prob_global * global_weights[..., 0:1]
        w_value = group_prob_global * global_weights[..., 1:2]

        weights = mx.cat([w_short, w_long, w_delta, w_value]
        dim=-1)

        # ε-floor ------------------------------------------------------
        eps = self.gate_eps_floor
        if eps > 0.0:
            weights = weights * (1.0 - 4 * eps) + eps
        weights = weights / weights.sum(dim=-1, keepdim=True)  # re-normalise

        # ---- Fuse branches -----------------------------------------
        out = (
            weights[..., 0:1] * local_short
            + weights[..., 1:2] * local_long
            + weights[..., 2:3] * delta_out
            + weights[..., 3:4] * v_direct
        )

        # ---- Cache update ------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)

        # ---- Output normalisation / projection ---------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l, (h, d)")
        out = self.o_proj(out)

        # ---- Re-pad if we unpadded earlier -------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0)
        indices, B_orig, L_in)

        return out, None, past_key_values
