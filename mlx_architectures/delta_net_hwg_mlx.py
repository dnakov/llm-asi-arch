# -*- coding: utf-8 -*-
"""
DeltaNet – Head-Wise Output-Conditioned Multi-Scale Gating (DeltaNet-HWG)
=======================================================================
This evolution of DeltaNet introduces a *head-wise*, output-aware fusion gate
that remedies the gradient-starvation and head-specialisation issues observed
in previous HMSMG variants.

Key innovations (enabled *by default*)
-------------------------------------
1. **Head-Wise Fusion Gate** – Each attention head owns an independent
   lightweight linear classifier that receives **its own** branch outputs plus
   the token's hidden state and produces softmax weights over the four memory
   paths (short-FIR, long-FIR, Δ-rule, direct value).  This preserves
   per-head autonomy and greatly improves path specialisation, a
   well-documented weakness of earlier global-MLP gates.

2. **Moderate Warm-Start Bias** – The direct-value path still receives a
   positive initial bias, but it is reduced to `+2.0` (from `+4.0`) to avoid
   starving the other paths of gradient signal while retaining a safe local
   starting point.

3. **Identity-Initialised FIR Kernels with Diversity Noise** – Depth-wise FIR
   filters are initialised to a causal identity (Dirac delta) plus a small
   Gaussian perturbation (`std=0.02`).  This keeps early optimisation stable
   while providing minimal feature diversity for the new head-wise gate to
   exploit.

All heavy computation remains **O(N)** thanks to chunk-wise Δ-rule kernels and
1-D depth-wise convolutions.  The public class name `DeltaNet`, constructor
signature and forward interface remain unchanged, ensuring drop-in
compatibility with the existing infrastructure.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

################################################################################
# Helper functions                                                              #
################################################################################

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU that stays strictly positive (legacy helper)."""
    return mx.maximum(0.0, x) + 1.0

def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that the last-dim elements sum to one."""
    return x / mx.sum(x, axis=-1, keepdims=True)

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalize the input tensor along the last dimension."""
    return x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)

################################################################################
# Core chunk-wise Δ-rule kernel (unchanged – O(N))                               #
################################################################################

def delta_rule_chunkwise(
    q: mx.array,  # (B, H, L, D_k)
    k: mx.array,  # (B, H, L, D_k)
    v: mx.array,  # (B, H, L, D_v)
    beta: mx.array,  # (B, H, L)
    *,
    chunk_size: int = 32,
):
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        # Pad tensors
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    L_pad = L + pad_len

    # Normalisation & β-scaling ------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into blocks of size *chunk_size* --------------------------------
    q = q.reshape(b, h, L_pad // chunk_size, chunk_size, d_k)
    k = k.reshape(b, h, L_pad // chunk_size, chunk_size, d_k)
    v = v.reshape(b, h, L_pad // chunk_size, chunk_size, -1)
    k_beta = k_beta.reshape(b, h, L_pad // chunk_size, chunk_size, d_k)

    tri_mask = mx.triu(mx.ones((chunk_size, chunk_size)), k=0)
    inv = -(k_beta @ mx.transpose(k, (0, 1, 2, 4, 3))) * (1 - tri_mask)
    
    for i in range(1, chunk_size):
        # Compute updated values
        inv_slice = inv[..., i, :i] + mx.sum(inv[..., i:i+1, :] @ inv[..., :i, :i], axis=-2)
        # Create new array with updated slice
        inv_new = mx.concatenate([
            inv[..., :i, :],
            mx.expand_dims(inv_slice, -2),
            inv[..., i+1:, :]
        ], axis=-2)
        inv = inv_new
    
    inv = inv + mx.eye(chunk_size)

    u = inv @ v
    w = inv @ k_beta

    S = mx.zeros((b, h, d_k, v.shape[-1]))
    out = mx.zeros_like(v)
    excl_mask = mx.triu(mx.ones((chunk_size, chunk_size)), k=1)
    
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ mx.transpose(k_i, (0, 1, 3, 2))) * (1 - excl_mask)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        # Update output for this chunk
        chunk_out = q_i @ S + attn_local @ u_i
        # Create new output array with updated chunk
        out_chunks = []
        for j in range(L_pad // chunk_size):
            if j == idx:
                out_chunks.append(chunk_out)
            else:
                out_chunks.append(out[:, :, j])
        out = mx.stack(out_chunks, axis=2)
        S = S + mx.transpose(k_i, (0, 1, 3, 2)) @ u_i

    out = out.reshape(b, h, L_pad, -1)
    if pad_len:
        out = out[:, :, :L]
    return out, S

################################################################################
# Depth-wise causal FIR convolution -------------------------------------------#
################################################################################

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise 1-D FIR convolution with identity initialisation."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_size: int = 31,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Parameters: (H, D, K)
        filters = mx.zeros((num_heads, head_dim, kernel_size))
        # causal identity – last tap = 1.0
        identity_filters = mx.zeros_like(filters)
        identity_filters = identity_filters.at[:, :, -1].set(1.0)
        noise = mx.random.normal(filters.shape) * init_std
        self.filters = identity_filters + noise

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, L, H, D)
        b, L, h, d = x.shape
        
        # Depthwise convolution simulation
        y = mx.zeros((b, L, h, d))
        x_pad = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0), (0, 0)])
        
        for i in range(L):
            for head in range(h):
                for dim in range(d):
                    x_window = x_pad[:, i:i+self.kernel_size, head, dim]  # (b, kernel_size)
                    filter_weights = self.filters[head, dim, :]  # (kernel_size,)
                    conv_out = mx.sum(x_window * filter_weights, axis=1)  # (b,)
                    y = y.at[:, i, head, dim].set(conv_out)
        
        return y

################################################################################
# Short convolution replacement                                                 #
################################################################################

class ShortConvolution(nn.Module):
    """Simple causal 1D convolution replacement."""
    
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = mx.random.normal((hidden_size, kernel_size)) * 0.02
        if bias:
            self.bias = mx.zeros(hidden_size)
        else:
            self.bias = None
    
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, D)
        b, l, d = x.shape
        
        # Causal padding
        x_pad = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        # Manual convolution
        y = mx.zeros((b, l, d))
        for i in range(l):
            x_window = x_pad[:, i:i+self.kernel_size, :]  # (b, kernel_size, d)
            conv_out = mx.sum(x_window * self.weight.T, axis=1)  # (b, d)
            if self.bias is not None:
                conv_out = conv_out + self.bias
            y = y.at[:, i, :].set(conv_out)
        
        if self.activation == 'silu':
            y = nn.silu(y)
        elif self.activation == 'gelu':
            y = nn.gelu(y)
            
        if output_final_state:
            return y, None
        return y

################################################################################
# Main DeltaNet implementation ------------------------------------------------#
################################################################################

class DeltaNet(nn.Module):
    """DeltaNet with Head-Wise Output-Conditioned Multi-Scale Gating."""

    def __init__(
        self,
        # --- inherited baseline args ---
        mode: str = "hwg",  # head-wise gating identifier
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
        # --- new hyper-parameters ---
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fusion_warm_start_bias: float = 2.0,  # moderate bias
        **kwargs,
    ) -> None:
        super().__init__()
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
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.norm_eps = norm_eps

        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        # -------- dimensions --------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # -------- linear projections --------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # -------- short convolutions --------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise ValueError("ShortConvolution is mandatory for stable optimisation.")

        # -------- FIR branches --------
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_short_kernel
        )
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_long_kernel
        )

        # -------- head-wise fusion gate parameters --------
        # Input per head: hidden_state (D) + 3 * head_v_dim (branch outputs)
        self._gate_in_per_head = hidden_size + 3 * self.head_v_dim
        fusion_weight = mx.zeros((num_heads, self._gate_in_per_head, 4))
        fusion_bias = mx.zeros((num_heads, 4))
        # Warm-start bias – favour direct value path (index 3)
        fusion_bias = fusion_bias.at[:, 3].set(fusion_warm_start_bias)
        self.fusion_weight = fusion_weight
        self.fusion_bias = fusion_bias

        # -------- output normalisation / projection --------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _rms_norm(self, x: mx.array) -> mx.array:
        """RMS normalization."""
        return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,  # (B, L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[mx.array, None, Optional[Dict]]:
        B_orig, L_orig, _ = hidden_states.shape

        # --------------------------------------------------
        # Projections + causal short convolutions
        # --------------------------------------------------
        q = self.q_conv1d(self.q_proj(hidden_states))
        k = self.k_conv1d(self.k_proj(hidden_states))
        v = self.v_conv1d(self.v_proj(hidden_states))

        # Head split ----------------------------------------------------
        q = q.reshape(B_orig, L_orig, self.num_heads, self.head_k_dim)
        k = k.reshape(B_orig, L_orig, self.num_heads, self.head_k_dim)
        v_direct = v.reshape(B_orig, L_orig, self.num_heads, self.head_v_dim)

        # Activations / normalisation ----------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # β for Δ-rule ---------------------------------------------------
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((B_orig, L_orig, self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global path -------------------------------------------
        q_d = q.transpose(0, 2, 1, 3)  # (b, h, l, d)
        k_d = k.transpose(0, 2, 1, 3)
        v_d = v_direct.transpose(0, 2, 1, 3)
        beta_d = beta.transpose(0, 2, 1)  # (b, h, l)
        delta_out, recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = delta_out.transpose(0, 2, 1, 3)  # back to (b, l, h, d)

        # FIR local paths ----------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # --------------------------------------------------
        # Head-wise fusion gate
        # --------------------------------------------------
        # Prepare gate input: [hidden | short | long | delta] per head
        h_exp = mx.expand_dims(hidden_states, 2)  # (b, l, 1, D)
        h_exp = mx.broadcast_to(h_exp, (B_orig, L_orig, self.num_heads, self.hidden_size))  # (b, l, h, D)
        gate_in = mx.concatenate([h_exp, local_short, local_long, delta_out], axis=-1)  # (b, l, h, F)
        
        # Compute logits via per-head weight/bias
        fusion_logits = mx.sum(gate_in[..., None, :] * self.fusion_weight[None, None, :, :, :], axis=-1) + self.fusion_bias  # (b, l, h, 4)
        fusion_weights = mx.softmax(fusion_logits, axis=-1)

        # Compose output ----------------------------------------------
        out = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # --------------------------------------------------
        # Output normalisation / projection
        # --------------------------------------------------
        if self.use_gate:
            g_vec = self.g_proj(hidden_states).reshape(B_orig, L_orig, self.num_heads, self.head_v_dim)
            out = self._rms_norm(out) * g_vec
        else:
            out = self._rms_norm(out)
        
        out = out.reshape(B_orig, L_orig, self.value_dim)
        out = self.o_proj(out)

        return out, None, past_key_values

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



################################################################################
# Helper functions                                                              #
################################################################################

def _elu_plus_one(x:, mx.array) -> mx.array:
    """Shifted ELU that stays strictly positive (legacy, helper)."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """Normalise so that the last-dim elements sum to one."""
    return (x / x.sum(-1, keepdim=True))

################################################################################
# Core chunk-wise Δ-rule kernel(unchanged, – O(N))                               #
################################################################################

@mx.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def delta_rule_chunkwise(q:, mx.array,  # (B, H, L, D_k)
    k: mx.array,  # (B, H, L, D_k)
    v: mx.array,  # (B, H, L, D_v)
    beta: mx.array,  # (B, H, L)
    *,
    chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & β-scaling ------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into blocks of size *chunk_size* --------------------------------
    q, k, v, k_beta = map(
        lambda x: _rearrange(x, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] = inv[..., i, :i] + (
            inv[..., i, :, None] * inv[..., :, :i]
        ).sum(-2), inv = inv + mx.eye(chunk_size, dtype = inv.dtype)
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    excl_mask = mx.triu(mx.ones_like(tri_mask), 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(excl_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
################################################################################
# Depth-wise causal FIR convolution -------------------------------------------#
################################################################################

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise 1-D FIR convolution with identity initialisation."""

    def __init__(, self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_size: int = 31
        init_std: float = 0.02) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # Parameters: (H, D, K)
        self.filters = mx.array(mx.zeros(num_heads, head_dim, kernel_size))
        with mx.disable_grad():
            # causal identity – last
    tap = 1.0
            self.filters[:, :, -1] = 1.0
            self.filters.add_(mx.randn_like(self.filters) * init_std)

    def forward(self, x: mx.array) -> mx.array:  # x: (B, L, H, D)
        b, L, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")  # flatten heads & dims
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal padding
        y = F.conv1d(x_pad, weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

################################################################################
# Optional typing imports -----------------------------------------------------#
################################################################################
################################################################################
# Main DeltaNet implementation ------------------------------------------------#
################################################################################

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with Head-Wise Output-Conditioned Multi-Scale Gating."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, # --- inherited baseline args ---,, mode: str = "hwg",  # head-wise gating identifier
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
        # --- new hyper-parameters ---
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fusion_warm_start_bias: float = 2.0 # moderate bias
        **kwargs) -> None:
        super().__init__()
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
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0

        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        # -------- dimensions --------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value, dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # -------- linear projections --------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # -------- short convolutions --------
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
            raise UserWarning("_ShortConvolution, is mandatory for stable optimisation.")

        # -------- FIR branches --------
        self.local_fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_short_kernel
        )
        self.local_fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_long_kernel
        )

        # -------- head-wise fusion gate parameters --------
        # Input per head: hidden_state (D) + 3 * head_v_dim (branch, outputs)
        self._gate_in_per_head = hidden_size + 3 * self.head_v_dim
        self.fusion_weight = mx.array(mx.zeros(num_heads, self._gate_in_per_head, 4))
        self.fusion_bias = mx.array(mx.zeros(num_heads, 4))
        # Warm-start bias – favour direct value path (index, 3)
        with mx.disable_grad():
            self.fusion_bias[:, 3] = fusion_warm_start_bias

        # -------- output normalisation / projection --------
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
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False # kept for API compatibility
        **kwargs: Dict) -> Tuple[mx.array, None Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2 "attention_mask must be(batch, seq_len)"

        B_orig, L_orig, _ = hidden_states.shape
        # --------------------------------------------------
        # Unpadding(for, variable-length, sequences)
        # --------------------------------------------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
            # After unpadding batch size is 1 (required by CUDA, kernels)

        # --------------------------------------------------
        # Projections + causal short convolutions
        # --------------------------------------------------
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

        # Head split ----------------------------------------------------
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # Activations / normalisation ----------------------------------
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

        # β for Δ-rule ---------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global path -------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # FIR local paths ----------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # --------------------------------------------------
        # Head-wise fusion gate
        # --------------------------------------------------
        # Prepare gate input: [hidden | short | long | delta] per head
        h_exp = hidden_states.expand_dims(2).expand(-1, -1, self.num_heads -1)  # (b,l,h, D)
        gate_in = mx.cat([h_exp, local_short, local_long, delta_out]
        dim=-1)  # (b, l, h, F)
        # Compute logits via per-head weight/bias
        #   logits_{b l h c} = Σ_f x_{b l h f} * W_{h f c} + b_{h c}
        fusion_logits = mx.einsum("blhf,hfc->blhc", gate_in self.fusion_weight) + self.fusion_bias  # (b,l,h, 4)
        fusion_weights = mx.softmax(fusion_logits, dim = -1)

        # Compose output ----------------------------------------------
        out = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # --------------------------------------------------
        # Cache update
        # --------------------------------------------------
        if use_cache and past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state_new
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_orig)

        # --------------------------------------------------
        # Output normalisation / projection
        # --------------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l, (h, d)")
        out = self.o_proj(out)

        # Re-pad if we unpadded earlier --------------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0)
        indices, B_orig, L_orig)

        return out, None, past_key_values
