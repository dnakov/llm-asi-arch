"""
MLX-converted architecture: delta_net_hgm_ident
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
DeltaNet – Hierarchical Gated Multi-Scale Memory + Dynamic Parallel Identity Router (DeltaNet-HGM-IDENT)
Identifier: *delta_net_hgm_ident*

This evolution synthesizes the proven strengths of hierarchical multi-scale gating (HGM) and block-state transformer
research with a breakthrough in parallel, router-controlled identity/copy stream fusion, delivering the following:

Key Innovations
1. **Hierarchical Gated Multi-Scale Routing**: 
   - Coarse-to-fine gating splits value information into local, mid-range, delta-global and identity paths.
   - Gating is determined by both token/hidden state and path statistics.
   - Relational (cross-branch) statistics are used for robust, context-adaptive routing.

2. **Router-Controlled Parallel Identity Path**:
   - Rather than an additive identity residual or an unconditional copy, an explicit, parallel identity branch is fully integrated into the main router, with its mass determined by a learned context-sensitive router signal.
   - This guarantees surface-copy reliability (for extraction/QA) without suppressing abstraction/comprehension capacity (critical for reasoning/narrative/factual, tasks).
   - The router’s outputs sum to 1 over all four paths, avoiding path starvation/collapse on any type of task.

3. **Adaptive Regularization**:
   - Entropy-based branch diversity loss is ramped down over time for early exploration and late specialization.
   - Optional cosine diversity between heads further prevents specialization collapse.
   - Reg loss is layer-depth and schedule-adaptive for uneven specialization pressure as needed.

4. **Efficiency, Causality and Universal Compatibility**:
   - All operations are O(N, log, N) or better; all chunked convolutions and delta rules are strictly causal.
   - All tensor operations use einops.rearrange, never .view or .reshape, with pure shape inference.
   - Dynamic handling of batch, sequence and head count at runtime.
   - Fully backwards compatible with the DeltaNet interface (including forward signatures and class, name).

5. **Elimination of Additive Identity Residual**:
   - The hardwired additive identity signal (source of abstraction bottlenecks in past, variants) is eliminated;
   - Instead, the identity stream is a first-class router branch dispatched/suppressed as dictated by routing context.

"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# Helper activations/normalizations

def elu_p1(x: mx.array):
    return (F.elu(x, 1.0, False) + 1.0)

def sum_norm(x: mx.array):
    return (x / x.sum(-1, keepdim=True))

# Causal Delta Rule – chunked (O(N))
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
    mask_tri_full = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_tri_full, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] += (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)
    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask_tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    num_chunks = L_pad // chunk_size
    for idx in range(num_chunks):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + local_attn @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# Per-head depth-wise causal convs
class _DepthwiseCausalConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = kernel_size
        weight = mx.randn(num_heads, * head_dim, 1, kernel_size) / math.sqrt(kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0
        self.weight = mx.array(weight), def forward(self, x:, mx.array):  # [B, L, H, D]
        b, L, h, d = x.shape
        x_ch = _rearrange(x "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight
        groups = h * d)
        y = _rearrange(y "b, (h, d) l -> b l h d"
        h=h)
        return y

class DeltaNet(nn.Module):
    """DeltaNet: Hierarchical Gated Multi-Scale + Parallel Router Identity Fusion (HGM-IDENT)"""

    def __init__(
        self, *,
        mode: str = "hgm_ident",
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
        local_kernel_size: int = 7,
        mid_kernel_size: int = 25,
        router_hidden_mult: int = 2,
        gate_dropout: float = 0.0,
        reg_schedule_base: float = 0.01,
        identity_kernel_size: int = 1,
        **kwargs: Dict, ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        # Projections and short conv
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        self.num_heads
            bias=False)
        if use_short_conv:
            activation = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=activation
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=activation
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for stable performance.")
        # Multi-scale per-head convs
        self.local_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim
        kernel_size = local_kernel_size)
        self.mid_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim
        kernel_size = mid_kernel_size)
        self.identity_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim
        kernel_size = identity_kernel_size)
        # Router
        self.router_in_dim = hidden_size + 8 * num_heads
        self.router_hidden_dim = int(router_hidden_mult * self.router_in_dim)
        self.router_mlp = nn.Sequential(
            nn.Linear(self.router_in_dim self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(self.router_hidden_dim, num_heads * 4, bias=True))
        # Output normalization / projection
        if use_gate:
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
        self.reg_schedule_base = reg_schedule_base
        # register_buffer removed for MLX
        persistent = False)
    # Feature engineering utilities
    @staticmethod
    def _branch_stats(x: mx.array):
        mu = x.mean(dim=-1
        keepdim=False)
        sigma = x.std(dim=-1 keepdim=False)
        return mu, sigma
    # Forward
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        reg_schedule: Optional[float] = None,
        **kwargs  ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [B, L] boolean/tensor"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
                conv_state_q
        conv_state_k, conv_state_v = last_state.get("conv_state", (None None, None))
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)
        q
        conv_state_q = self.q_conv1d(
            x=q_proj, cache=conv_state_q,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_state_k = self.k_conv1d(
            x=k_proj, cache=conv_state_k,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v, conv_state_v = self.v_conv1d(
            x=v_proj, cache=conv_state_v,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v "b l, (h, d) -> b l h d"
        h=self.num_heads)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = sum_norm(q), sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # Delta global-MEM path
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d
        chunk_size =32)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")
        v_direct = v  # direct value identity
        local_out = self.local_conv(v_direct)
        mid_out = self.mid_conv(v_direct)
        identity_out = self.identity_conv(v_direct)  # kernel_size==1 = identity
        # Branch stat features
        mu_local
        std_local = self._branch_stats(local_out)
        mu_mid, std_mid = self._branch_stats(mid_out)
        mu_delta
        std_delta = self._branch_stats(delta_out)
        mu_id, std_id = self._branch_stats(identity_out)
        stats_all = mx.cat([, mu_local, std_local, mu_mid, std_mid)
            mu_delta, std_delta, mu_id, std_id
        ], dim=-1)  # [B,L,H*8]
        stats_all = _rearrange(stats_all "b l h8 -> b l (h8)")
        # Routing
        router_in = mx.cat([hidden_states, stats_all]
        dim=-1)  # [B, L F]
        router_logits = self.router_mlp(router_in)
        router_logits = _rearrange(router_logits "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = 4)
        router_soft = mx.softmax(router_logits
        dim = -1)
        # Weighted routing of the four parallel streams, o = (
            router_soft[..., 0:1] * local_out +
            router_soft[..., 1:2] * mid_out +
            router_soft[..., 2:3] * delta_out +
            router_soft[..., 3:4] * identity_out
        )
        # Cache update
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = L_in)
        # Output normalization / projection
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        # Re-pad to original batch dimensions if necessary
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        # Regularization (scheduled)
        reg_loss = None
        if self.training:
            reg_fac = self.reg_schedule_base if reg_schedule is None else reg_schedule
        entropy = -(router_soft * (router_soft + 1e-8).log()).sum(-1).mean()
        kl_uniform = (
                router_soft * (router_soft.add(1e-8).log() - math.log(0.25))
            ).sum(-1).mean(), # cosine diversity between heads, fws = _rearrange(router_soft "b l h p ->, (b, l) h p")
            cosdiv = 0.0
            for i in range(self.num_heads):
                for j in range(i + 1 self.num_heads):
                    cosdiv += F.cosine_similarity(fws[:, i], fws[:, j]
                    dim=-1).mean()
        cosdiv = -cosdiv / (self.num_heads * (self.num_heads - 1) / 2)
            reg_loss = reg_fac * entropy + reg_fac * kl_uniform + reg_fac * cosdiv
        # Advance step
        self._step += 1
        return o, reg_loss, past_key_values
