from __future__ import annotations

"""
MLX-converted architecture: delta_net_aft_dsi
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
DeltaNet – Adaptive Floor Token-fusion with Scheduled Identity Residual and Dynamic Alpha (DeltaNet-AFT-DSI)
Identifier: delta_net_aft_dsi

Key innovations(enabled, by, default):
1. **Token-Adaptive Floor Routing**
   •  Replaces hard identity floor (HIST) with a token/context-adaptive floor to the direct/copy/value path. The minimal copy mass is guaranteed only where the context router is uncertain vanishing when context path is sharply confident.
   •  The floor value min_copy_frac decays linearly (schedule) over training(, AFT, BST), and can be modulated per token: (copy_floor = min_copy_frac * (1-context_confidence)). This guarantees early exploration/copy-fidelity then enables pure contextual routing when capable.
2. **Softplus-bounded Per-Head Identity Alpha**
   •  The learnable identity scaling parameter (alpha) per head is now softplus-bounded and regularized, guaranteeing unbounded growth is avoided and providing stable blending of copy/context routes.
3. **Scheduled Temperature & Epsilon-Floor**
   •  Context router(3-way:, short, long, delta) is softmaxed with a classic annealed epsilon floor and scheduled temperature (group-to-head as in, HIST), ensuring early path diversity and late sharp routing.
4. **Strict O(N) Complexity and Causal Integrity**
   •  All sequence operations use chunked computation, depthwise/causal FIR and batch-agnostic einops patterns.
5. **Batch-size and Sequence-robustness**
   •  All design choices & tensor ops are strictly batch/shape agnostic using einops.

"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _elu_p1(x:, mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x:, mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR conv(unchanged:, O(N) causal)
# -----------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int
    kernel_size: int = 31
    eps: float = 2e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0
            filt.add_(eps, * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel
# -----------------------------------------------------------------------------
@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwiseq, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
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
    k_beta = map(lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i
        :i] += (attn_inv[..., i, :, None] * attn_inv[..., :, :i]).sum(-2)
        attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)
    u = attn_inv @ v
        w = attn_inv @ k_beta
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
# -----------------------------------------------------------------------------
# TYPE CHECKING
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """
    DeltaNet-AFT-DSI: Token-adaptive copy path, scheduled context router, softplus-bounded alpha all O(N), batch robust.
    """
    def __init__(self, mode: str =, "aft_dsi",
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
        fir_kernel_size_long: int = 31,
        fir_kernel_size_short: int = 3,
        fusion_hidden_mult: int = 2,
        min_copy_frac_start: float = 0.08,
        min_copy_frac_end: float = 0.008,
        copy_frac_decay_steps: int = 3000,
        identity_alpha_init: float = 1.0,
        fusion_dropout: float = 0.0,
        group_size: int = 2,
        tau_transition_steps: int = 3000,
        epsilon_start: float = 0.03,
        epsilon_end: float = 0.005,
        epsilon_decay: int = 3000 **kwargs: Dict) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
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
        self.mode = mode
        self.fusion_hidden_mult = fusion_hidden_mult
        # dims
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)
        # conv
        if use_short_conv:
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
        else:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet stability.")
        # FIR convs
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_short)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_long)
        # Identity/copy alpha per head: softplus-bounded
        self.identity_alpha_param = mx.array(mx.ones(num_heads), * identity_alpha_init)  # param later passed through softplus
        # copy min floor schedule
        self.min_copy_frac_start = float(min_copy_frac_start)
        self.min_copy_frac_end = float(min_copy_frac_end)
        self.copy_frac_decay_steps = int(copy_frac_decay_steps)
        # register_buffer removed for MLX
        persistent = False)
        # context router eps schedule
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        # register_buffer removed for MLX
        persistent = False)
        # group-to-head tau
        self.group_size = max(1, int(group_size))
        num_groups = (num_heads + self.group_size - 1) // self.group_size
        # register_buffer removed for MLX // self.group_size
        persistent = False)
        self.log_tau_group = mx.array(mx.zeros(num_groups)), # exp(0) ~1
        self.log_tau_head = mx.array(mx.zeros(num_heads))
        self.tau_transition_steps = int(tau_transition_steps)
        # context router MLP (3-way)
        stat_dim_per_head = 2
        router_in_dim = hidden_size + num_heads * stat_dim_per_head * 3
        router_hidden_dim = max(8, hidden_size * fusion_hidden_mult)
        self.router_mlp = nn.Sequential(, nn.Linear(router_in_dim, router_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(router_hidden_dim, num_heads * 3, bias=True))
        with mx.disable_grad():
            self.router_mlp[-1].bias.fill_(0.0)
        # norm/proj
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    def _current_copy_frac(self):
        t = float(self._copy_step.item())
        if t >= self.copy_frac_decay_steps:
            return self.min_copy_frac_end
        r = t / max(1.0, self.copy_frac_decay_steps)
        return self.min_copy_frac_start + r * (self.min_copy_frac_end - self.min_copy_frac_start)

    def _current_epsilon(self):
        t = float(self._eps_step.item())
        if t >= self.epsilon_decay:
            return self.epsilon_end
        r = t / max(1.0, self.epsilon_decay)
        return self.epsilon_start + r * (self.epsilon_end - self.epsilon_start)

    def _mix_temperature(self):
        t = float(self._copy_step.item())
        mix = 1.0 - min(1.0, t / max(1.0, self.tau_transition_steps))
        tau_g = mx.exp(self.log_tau_group)[self._group_index]
        tau_h = mx.exp(self.log_tau_head)
        tau = mix * tau_g + (1.0 - mix) * tau_h
        return tau  # (H)

    @staticmethod
    def _stats_mean_std(path:, mx.array) -> Tuple[mx.array, mx.array]:
        mean = path.mean(dim=-1, keepdim=False)
        std = path.std(dim=-1, unbiased=False
        keepdim = False)
        return mean, std

    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs  ):
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
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]
        q_in
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_in
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_in
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q_in, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_in, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_in, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        delta_out_d
        recurrent_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)
        mean_s
        std_s = self._stats_mean_std(local_short)
        mean_l, std_l = self._stats_mean_std(local_long)
        mean_d
        std_d = self._stats_mean_std(delta_out)
        stats = mx.stack([mean_s, std_s, mean_l, std_l, mean_d, std_d]
        dim=-1)
        stats_flat = _rearrange(stats, "b l h f -> b l, (h, f)")
        router_in = mx.cat([hidden_states, stats_flat]
        dim=-1)
        router_logits = self.router_mlp(router_in)
        router_logits = _rearrange(router_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 3)
        tau = self._mix_temperature()
        router_logits = router_logits / tau.reshape(1, 1, self.num_heads, 1)
        probs = mx.softmax(router_logits, dim = -1)
        eps = self._current_epsilon()
        probs = probs * (1.0 - 3 * eps) + eps
        # context router: context_conf = max(prob_i) -- confidence of context router for this token/head
        context_conf
        _ = probs.max(-1)
        # adaptive/minimal copy floor (token-specific!): floor = min_copy_frac * (1 - context_conf)
        copy_floor = self._current_copy_frac() * (1.0 - context_conf)
        # re-normalize so context mass is (1-copy_floor)
        probs = probs * (1.0 - copy_floor).expand_dims(-1)
        # --- context path output ---
        context_out = (
            probs[..., 0:1] * local_short +
            probs[..., 1:2] * local_long +
            probs[..., 2:3] * delta_out
        )
        # identity/copy out -- per-head alpha * softplus * adaptively floored min mass
    alpha = mx.nn.softplus(self.identity_alpha_param).reshape(1, 1, -1, 1)
        v_direct_res = alpha * (copy_floor.expand_dims(-1)) * v_direct
        o = context_out + v_direct_res
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        self._copy_step += 1
        self._eps_step += 1
        return o, None, past_key_values
