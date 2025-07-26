from __future__ import annotations

"""
MLX-converted architecture: delta_net_tarf
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
DeltaNet – Token-Adaptive Router with Multi-Scale FIR (TARF)
Identifier: delta_net_tarf

This evolution unifies the strongest empirical findings:
    •  Multi–scale FIR local memories(kernels, 3->31) proven to excel on
       span-extraction and local reasoning.
    •  Global Δ-rule pathway for long-range associative recall (unchanged).
    •  *Token-adaptive* identity-vs-context split borrowed from AFT: the
       minimum probability reserved for contextual fusion adapts **per token**
       based on the router’s own value-path confidence ensuring copy tasks can
       approach hard routing without starving contextual gradients early in
       training.
    •  Per-head temperature with lower bound(τ, ≥ 0.5) prevents catastrophic
       over-sharpening yet allows specialisation.
    •  Lightweight, output-aware context router that consumes the actual path
       outputs in addition to the hidden state.

No other mechanics – chunk-wise Δ-rule, strict causal FIR batch independence –
are modified.  Complexity remains **O(L)**.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _elu_plus_one(x:, mx.array) -> mx.array:  # shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:  # last-dim sum-normalise
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution(Dirac, init + noise)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution with identity initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0
            if noise_std > 0:
                filt.add_(noise_std, * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # (B,L,H, D)
        b, l, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Multi-scale FIR block (kernels, tuple)
# -----------------------------------------------------------------------------

class _MultiScaleFIR(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernels: Tuple[int, ...] = (3, 7, 15, 31)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([, _DepthwiseFIRConv1d(num_heads, head_dim, k) for k in kernels
        ])

    def forward(self, x: mx.array) -> List[mx.array]:
        return [branch(x) for branch in self.branches]

# -----------------------------------------------------------------------------
# Δ-rule kernel in causal chunks – unchanged numerics
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(q:, mx.array, k: mx.array, v: mx.array, beta: mx.array, *, chunk_size: int = 32):
    """Causal associative Δ-rule with O(L) cost via chunked scanning."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v
    k_beta = map(lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))

    tri_inc = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    strict = mx.triu(tri_inc, 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_inc, 0)
    for i in range(1, chunk_size):  # recursion for inverse
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        att_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + att_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Main DeltaNet implementation (TARF, variant)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with multi-scale FIR and token-adaptive routing (TARF)."""

    # pylint: disable=too-many-instance-attributes
    def __init__(, self, mode: str = "tarf",
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # optional components
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
        # multi-scale FIR kernels
        fir_kernels: Tuple[int, ...] = (3, 7, 15, 31),
        # router / gating params
        min_context_floor: float = 0.01,
        max_context_floor: float = 0.10,
        temp_init: float = 1.0,
        temp_min: float = 0.5,
        value_bias_init: float = 2.0 **kwargs: Dict) -> None:
        super().__init__()

        if d_model is not None:
            hidden_size = d_model
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
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # token-adaptive floor settings
        self.min_floor = float(min_context_floor)
        self.max_floor = float(max_context_floor)
        assert 0.0 < self.min_floor < self.max_floor < 0.5 "floors must satisfy 0<min<max<0.5"

        # dimensions
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value, dims must divide num_heads")

        # linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # optional short convs
        if self.use_short_conv:
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
            self.q_conv1d = self.k_conv1d = self.v_conv1d = nn.Identity()

        # multi-scale FIR
        self.ms_fir = _MultiScaleFIR(num_heads, self.head_v_dim
        kernels = fir_kernels)
        self.n_ctx_paths = len(fir_kernels) + 1  # FIR branches + Δ

        # context router MLP (hidden + path, outputs)
        router_in_dim = hidden_size + self.head_v_dim * num_heads * self.n_ctx_paths
        self.router_mlp = nn.Sequential(, nn.Linear(router_in_dim, hidden_size * 2, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, 2, num_heads * self.n_ctx_paths, bias = True))
        nn.init.zeros_(self.router_mlp[-1].bias)

        # identity/value gate projection (sigmoid, later)
        self.id_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.id_gate_proj.bias.fill_(value_bias_init)

        # per-head temperature(softplus, + min)
        self.log_tau = mx.array(mx.log(mx.ones(num_heads), * temp_init))
        self.temp_min = float(temp_min)

        # output norm/projection
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def forward(, self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be(batch, seq_len)"
        B, L, _ = hidden_states.shape

        # handle cache
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # projections + optional short conv
        conv_q = conv_k = conv_v = None
        if self.use_short_conv and last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q
        conv_q = self.q_conv1d(q_lin, cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(k_lin, cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(v_lin, cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # head reshape
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # activation & norm on Q/K
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

        # beta for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global path
        delta_out_t
        recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q, "b l h d -> b h l d")
        k=_rearrange(k, "b l h d -> b h l d"),
            v=_rearrange(v_direct, "b l h d -> b h l d")
        beta=_rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t, "b h l d -> b l h d")

        # multi-scale FIR outputs
        fir_branches = self.ms_fir(v_direct)  # list length len(fir_kernels)

        # ------------------------------------------------------------------
        # Token-adaptive identity vs context gate
        # ------------------------------------------------------------------
        value_logit = self.id_gate_proj(hidden_states)  # (B,L, H)
        p_value = mx.sigmoid(value_logit)  # confidence of copy path

        # adaptive floor for context mass(others_total, ≥ floor_tok)
        floor_tok = self.min_floor + (self.max_floor - self.min_floor) * (1.0 - p_value)
        # Part of prob allocated to value path
        p_value_adj = (1.0 - floor_tok) * p_value  # scale so that total context >= floor_tok
        context_mass = 1.0 - p_value_adj  # >= floor_tok

        # ------------------------------------------------------------------
        # Context router (output-aware)
        # ------------------------------------------------------------------
        # prepare router input
        router_in = mx.cat([, hidden_states)
            _rearrange(mx.stack(fir_branches, +, [delta_out], dim=0), "c b l h d -> b l(c, h, d)")
        ], dim=-1)
        ctx_logits_flat = self.router_mlp(router_in)  # (B,L H*C)
        ctx_logits = _rearrange(ctx_logits_flat, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = self.n_ctx_paths)

        # temperature scaling
    tau = F.softplus(self.log_tau) + self.temp_min  # (H)
        ctx_logits = ctx_logits / tau.reshape(1, 1, -1, 1)

        ctx_weights = mx.softmax(ctx_logits, dim = -1)  # (B,L,H, C)
        # scale by context_mass
    ctx_weights = ctx_weights * context_mass.expand_dims(-1)

        # ------------------------------------------------------------------
        # Final aggregation
        # ------------------------------------------------------------------
        o = mx.zeros_like(v_direct)
        for idx br in enumerate(fir_branches):
            o = o + ctx_weights[..., idx:idx+1] * br
            o = o + ctx_weights[..., len(fir_branches):len(fir_branches)+1] * delta_out
        # add value path
    o = o + p_value_adj.expand_dims(-1) * v_direct

        # ------------------------------------------------------------------
        # Update cache
        # ------------------------------------------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L)

        # ------------------------------------------------------------------
        # Output norm/proj
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # re-pad if we unpadded earlier
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B, L)

        return o, None, past_key_values
