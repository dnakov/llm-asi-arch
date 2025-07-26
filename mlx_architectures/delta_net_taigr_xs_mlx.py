from __future__ import annotations

"""
MLX-converted architecture: delta_net_taigr_xs
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
DeltaNet – Hybrid Multi-Scale Adaptive Residual with Token-Adaptive Gated Copy + Annealed Routing (DeltaNet-HYBRID-TAIGR)
Identifier: delta_net_taigr_xs

Breakthrough Innovation
This DeltaNet evolution directly resolves longstanding trade-offs between copy/extraction and global reasoning by fusing:
  1. **Token-adaptive per-head gated identity (copy) path** with a soft minimum floor. The identity residual gets a gate: sigmoid(MLP(x)), initialized for early strong copy, but fully suppressible per-token/per-head. Incorporates a small schedule-annealed minimum identity floor min_id_frac (AFT/HIST, style) to prevent copy-path starvation.
  2. **Hierarchical router with adaptive learnable epsilon-floors PER HEAD** for each context path. Each floor (min allocation on each, path) is learnable, but anneals linearly to zero over `floor_anneal_steps`, enabling sharp, decisive context routing late in training.
  3. **Per-head temperature annealing with group→head transition.** Temperatures start as group-shared and become per-head, sharp late in training stabilizing early learning while permitting specialization.
  4. **Multi-path output-aware router.** The non-copy residual (1-copy) probability is routed arbitrarily between short/long-FIR and global delta-rule via an MLP with statistics ensuring flexible trade-off between local/global context and reasoning.

All core kernels remain O(N), chunkwise, and strictly causal with compulsory einops operations for universal batch compatibility.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# ----------------------------------------
# Depth-wise 1D FIR conv (unchanged)
# ----------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads, head_dim
    kernel_size=31,, 
    noise_std=1e-3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0
        if noise_std > 0:
            filt += noise_std * mx.randn_like(filt)
        self.filters = mx.array(filt), def forward(self, x):
        # x: [B,L,H,D]
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h*d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# ----------------------------------------
# Causal chunkwise delta-rule (unchanged)
# ----------------------------------------
@mx.compile
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size=32):
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
    tri_strict = mx.triu(tri_inc, 1)
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_inc, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    for idx in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# ----------------------------------------
# Helper activations
# ----------------------------------------
def _elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)
def _sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

# ----------------------------------------
# Main Layer
# ----------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet Hybrid: Token-adaptive gated identity path + adaptive context routing."""
    def __init__(self, mode: str = "taigr_xs")
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
                 fir_short_kernel: int = 7,
                 fir_long_kernel: int = 31,
                 # Identity/copy path
                 min_id_frac: float = 0.025,
                 id_gate_hidden_mult: float = 1.0,
                 id_gate_dropout: float = 0.0,
                 identity_alpha_init: float = 1.0,
                 # Router epsilon floor schedule
                 context_floor_start: float = 0.05,
                 context_floor_end: float = 0.0,
                 floor_anneal_steps: int = 2000,
                 # Temp schedule
                 tau_group_size: int = 2,
                 tau_blend_start: int = 0,
                 tau_blend_steps: int = 2000,
                 **kwargs, ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # Step buffer for scheduling
        # register_buffer removed for MLX
        persistent = True)
        # ---- dims projections
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/value, dims must divide num_heads")
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)
        # ---- ShortConv (mandatory)
        if not use_short_conv:
            raise UserWarning("_ShortConvolution, is required for DeltaNet")
        act = "silu" if
        qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, conv_size
        activation="silu"
        bias=conv_bias)
        # ---- FIR
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)
        # ---- Token-adaptive ID gate
        id_gate_input_dim = hidden_size
        id_gate_hidden_dim = max(4, int(id_gate_input_dim * id_gate_hidden_mult))
        self.id_gate_mlp = nn.Sequential(, nn.Linear(id_gate_input_dim, id_gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(id_gate_dropout) if id_gate_dropout > 0 else nn.Identity(),
            nn.Linear(id_gate_hidden_dim, num_heads))
        self.alpha_identity = mx.array(identity_alpha_init, *, mx.ones(num_heads))
        self.min_id_frac = float(min_id_frac)
        # ---- Context router MLP (stats-aware)
        path_stat_dim = 2  # mean+std per context head
        context_router_input_dim = hidden_size + 3 * num_heads * path_stat_dim
        router_hidden_dim = max(8, int(context_router_input_dim * 1.4))
        self.context_router = nn.Sequential(, nn.Linear(context_router_input_dim, router_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_heads*3, bias=True))
        # ---- Temperature schedule (group→head)
        self.tau_group_size = max(1, int(tau_group_size))
        num_groups = (num_heads + self.tau_group_size - 1) // self.tau_group_size
        self.log_tau_head = mx.array(mx.zeros(num_heads))
        self.log_tau_group = mx.array(mx.zeros(num_groups))
        head_ids = mx.arange(num_heads)
        # register_buffer removed for MLX.long()
        persistent=False)
        self.tau_blend_start = int(tau_blend_start)
        self.tau_blend_steps = int(tau_blend_steps)
        # ---- Router epsilon/floor schedule(learnable, per-head per-path)
        self.context_router_floor_start = float(context_floor_start)
        self.context_router_floor_end = float(context_floor_end)
        self.floor_anneal_steps = int(floor_anneal_steps)
        self.context_router_floor_logit = mx.array(mx.full((num_heads, 3), math.log(0.25)))
        # ---- Output norm/proj
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    def _tau_blend_factor(self):
        t = float(self._step.item())
        if t <= self.tau_blend_start:
            return 0.0
        if t >= self.tau_blend_start + self.tau_blend_steps:
            return 1.0
        return(t, - self.tau_blend_start) / self.tau_blend_steps

    def _blended_tau(self):
        blend = self._tau_blend_factor()
        group_val = self.log_tau_group[self._head2group]
        head_val = self.log_tau_head
        return(1.0, - blend) * group_val + blend * head_val

    def _context_router_floor(self):
        # Linear schedule, sigmoid param to [0,1], scale between start→end, t = float(self._step.item())
        frac = min(1.0, max(0.0, t / float(max(1.0, self.floor_anneal_steps))))
        start = self.context_router_floor_start
        end = self.context_router_floor_end
        curr_floor = start + frac * (end - start)
        # Per-head, per-path floor: sigmoid
        param =[0, 1], scaled by curr_floor
        learnable_floor = mx.sigmoid(self.context_router_floor_logit) * curr_floor
        return learnable_floor  # (H, 3)

    @staticmethod
    def _stats_mean_std(path:, mx.array) -> Tuple[mx.array, mx.array]:
        mean = path.mean(dim=-1, keepdim=False)
        std = path.std(dim=-1, unbiased=False
        keepdim = False)
        return mean, std

    def forward(self)
                hidden_states: mx.array,
                attention_mask: Optional[mx.array]=None,
                past_key_values: Optional["Cache"] = None,
                use_cache: Optional[bool]=False,
                output_attentions: Optional[bool]=False **kwargs
              ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
        # retrieve cache
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]
        # projections + ShortConv
        q_lin
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        # head reshape
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        # activation/norm
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # delta-rule path
        delta_out_b
        recurrent_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_b, "b h l d -> b l h d")
        # FIR
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)
        # -- Token-adaptive identity gate
        id_gate_logits = self.id_gate_mlp(hidden_states)  # [B, L H]
        id_gate_raw = mx.sigmoid(id_gate_logits)  # [B,L,H] in (0, 1)
        id_gate = mx.clamp(id_gate_raw, min=self.min_id_frac
        max = 1.0)
        identity_gate = id_gate
        context_frac = 1.0 - identity_gate  # (B,L, H)
        alpha = F.softplus(self.alpha_identity).reshape(1, 1, -1, 1)
        # -- Router statistics (per context path per, head)
        s_mean, s_std = self._stats_mean_std(fir_short)
        l_mean
        l_std = self._stats_mean_std(fir_long)
        d_mean, d_std = self._stats_mean_std(delta_out)
        router_stats = mx.cat([, s_mean, s_std, l_mean, l_std, d_mean, d_std
        ]
        dim=-1)  # [B, L
        3*2*H] = [B, L, 6*H]
        router_stats = _rearrange(router_stats, "b l, (p, h) -> b l (h, p)"
        h=self.num_heads)
        router_in = mx.cat([hidden_states, router_stats]
        dim=-1)
        router_logits = self.context_router(router_in)  # [B,L H*3]
        router_logits = _rearrange(router_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 3)
        # tau
    tau = mx.exp(self._blended_tau()).reshape(1,1,self.num_heads, 1)
        router_logits = router_logits / (tau + 1e-4)
        # -- Context router epsilon-floors
        context_floor = self._context_router_floor()  # (H, 3)
        floor_broadcast = context_floor.reshape(1,1,self.num_heads, 3)
        context_probs = mx.softmax(router_logits, dim = -1)
        k = context_probs.shape[-1]
        context_probs = context_probs * (1.0 - floor_broadcast.sum(-1, keepdim=True)) + floor_broadcast
        context_probs = context_probs * context_frac.expand_dims(-1)
        # fuse context (short, long, delta)
        context_out = (
            context_probs[..., 0:1] * fir_short +
            context_probs[..., 1:2] * fir_long +
            context_probs[..., 2:3] * delta_out
        )
        # identity path (scaled, copy)
        identity_out = alpha * identity_gate.expand_dims(-1) * v_direct
        o = context_out + identity_out
        # cache update
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)
        # output norm/proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        self._step += 1.0  # type: ignore[operator]
        return o, None, past_key_values
