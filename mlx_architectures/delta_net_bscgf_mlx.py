from __future__ import annotations

"""
MLX-converted architecture: delta_net_bscgf
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
import mlx.nn as F
DeltaNet – Block-State Inspired Context-Gated MultiScale Fusion (DeltaNet-BSCGF)
A breakthrough evolution integrating research-proven, context-aware gating from Block-State Transformers/Comba with robust multi-scale FIR memory and chunkwise delta memory.

Key Innovations
1. **Context-aware fusion gate**: Gate MLP receives per-branch statistics (mean, std) AND the hidden state, enabling dynamic, query-adaptive routing between memory branches: two FIR (short, long), global (delta-rule), and direct (identity) path.
2. **Dual FIR paths with Dirac init**: Both short- (k=3) and long-range (k=63) FIR filters are initialized as Dirac delta (identity + small, noise) for robust early optimization and preservation of local/global cues.
3. **Per-head temperature regulation**: Each head's gate softmax is sharpened/smoothed by a learnable temperature (softplus), preventing path collapse and enabling robust specialization AND blending. Mild entropy penalty optional (default: off can be, exposed).
4. **Scheduled value-path bias**: Fusion gate bias for the identity path is initialized high and exposed for curriculum/annealing (default: +2.0 identity bias others, 0).
5. **O(N) complexity and full batch/seq agnosticism**: All computations chunked appropriately, using einops.rearrange exclusively for shape management; batch-agnostic and compatible with arbitrary input dimensions, maintaining DeltaNet's drop-in promise.

All initialization, input, and output contracts remain compatible with prior DeltaNet family. Major research trends (BST Comba MoE/conditional, routing) are integrated for maximal breakthrough potential.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from mx.nn import functional as F



# Utility functions ------------------------------------------------------
def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

def std_stat(x):
    # std over last dim, but min-clip for stability
    return mx.sqrt(mx.clamp(x.var(dim=-1
        unbiased=False)
        min=1e-6))

# Dirac initialization for FIR filters -----------------------------------

def dirac_init(fir):
    with mx.disable_grad():
        fir.zero_()
        s = fir.shape
        center = s[-1] // 2
        fir[..., center] = 1.0
        fir += 1e-2 * mx.randn_like(fir)

# DepthwiseCausalFIR (per-head per-channel) -----------------------------

class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads, head_dim,, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = mx.array(mx.empty(num_heads, head_dim, kernel_size))
        dirac_init(self.filters)

    def forward(self, x):  # [b, l, h, d]
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        # causal padding on the left so that each position only sees past tokens, x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight
        groups = h * d)
        y = _rearrange(y "b, (h, d) l -> b l h d"
        h=h)
        return y

# Chunkwise delta kernel (O(N), causal) ----------------------------------

@mx.compile
def delta_rule_chunkwiseq, k, v, beta chunk_size: int = 32):
    b, h, L, d_k = q.shape
        d_v = v.shape[-1]
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q
        (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v
    k_beta = map(lambda x: _rearrange(x "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))

    # Build causal masks (constant per, chunk) ----------------------------
    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2), attn_inv = attn_inv + mx.eye(chunk_size
        dtype = q.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# Main DeltaNet class ----------------------------------------------------

class DeltaNet(nn.Module):
    """Block-State Context-Gated FIR/DeltaNet Hybrid"""

    def __init__(
        self mode: str =, "bscgf",
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
        fir_short_kernel: int = 3,  # local
        fir_long_kernel: int = 63,  # global
        fusion_hidden_mult: int = 2,
        fusion_value_bias: float = 2.0,
        gate_temp_init: float = 1.2,  # >1 for mild sharpness
        gate_entropy_reg: float = 0.0,
        **kwargs, ):
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
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.layer_idx = layer_idx

        # --- dims --------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")

        # --- linear projections -----------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # --- short convolutional boosts ---------------------------------
        if self.use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size
        activation = act)
            self.k_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size
        activation = act)
            self.v_conv1d = _ShortConvolution(self.value_dim
        kernel_size = conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution is mandatory.")

        # --- Dual-scale FIR filters -------------------------------------
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # --- Gating: hidden + stats + per-head temperature --------------
        # Four memory branches (short FIR, long FIR, delta direct, value),
        # each contributing mean/std (2, values) per head.
        num_branches = 4  # keep explicit for clarity / future extension
        stats_per_branch = 2 * num_heads  # mean & std for each head
        gate_in_dim = hidden_size + num_branches * stats_per_branch  # total gating input dimension

        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * num_branches, bias = True))
        # set value branch (index, 3) bias high for curriculum learning
        with mx.disable_grad():
            for h in range(num_heads):
                # bias layout: [short, long, delta value] per head
                self.fusion_gate_mlp[-1].bias[h * num_branches + 3] = fusion_value_bias

        # --- per-head temperature --------------------------------------
        self.gate_log_temp = mx.array(mx.ones(num_heads), * math.log(gate_temp_init))

        # --- output norm/proj ------------------------------------------
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
        self.gate_entropy_reg = gate_entropy_reg  # can be used in training scripts

    # ------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs  ):  # noqa: C901 (keep single forward for compile, friendliness)
        # ----------------------------------------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"

        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            # unpad for variable-length
        highly efficient processing
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # --- linear projections + (optional) depthwise short conv --------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q
        conv_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # --- reshape for multi-head ------------------------------------
        q
        k = map(lambda t: _rearrange(t "..., (h, d) -> ... h d"
        d=self.head_k_dim), (q, k))
        v = _rearrange(v "..., (h, d) -> ... h d"
        d=self.head_v_dim)

        # --- activations & normalisations -------------------------------
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

        v_direct = v  # identity/value path --------------------------------

        # --- optional beta gating (recurrent, eigenvalues) ---------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --- chunkwise delta routing -----------------------------------
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")

        # --- causal FIR paths ------------------------------------------
        fir_short = self.local_fir_short(v_direct)
        fir_long = self.local_fir_long(v_direct)

        # --- Prepare per-branch statistics ------------------------------
        def flat_stats(branch: mx.array):
            m = branch.mean(dim=-1), # [b, l h]
            s = std_stat(branch)     # [b, l, h]
            return mx.cat([m, s], dim=-1)  # [b, l, h*2]

        gate_feat = [
            hidden_states,          # [b, l d]
            flat_stats(fir_short),  # [b, l h*2]
            flat_stats(fir_long),   # [b, l h*2]
            flat_stats(delta_out),  # [b, l h*2]
            flat_stats(v_direct),   # [b, l, h*2]
        ]
        gate_in = mx.cat(gate_feat
        dim = -1)

        # --- Fusion gating ---------------------------------------------
        fusion_logits = self.fusion_gate_mlp(gate_in)  # [b,l,h*4]
        fusion_logits = _rearrange(
            fusion_logits 'b l, (h, c) -> b l h c', h=self.num_heads
        c = 4
        )
        temp = F.softplus(self.gate_log_temp) + 1e-4  # ensure strictly positive
        fusion_logits = fusion_logits / temp[None, None, :, None]
        fusion_w = mx.softmax(fusion_logits
        dim = -1)

        # Weighted combination of memory branches ------------------------
        o = (
            fusion_w[..., 0:1] * fir_short +
            fusion_w[..., 1:2] * fir_long +
            fusion_w[..., 2:3] * delta_out +
            fusion_w[..., 3:4] * v_direct
        )

        # --- caching (for KV caches etc.) -------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = seq_len)

        # --- output projection & (optional) gating ----------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states)
        "... (h, d) -> ... h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # --- pad back if we unpadded -----------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len)

        return o, None, past_key_values
