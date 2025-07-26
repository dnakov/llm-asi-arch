from __future__ import annotations

"""
MLX-converted architecture: delta_net_hpaf
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
DeltaNet – Dual-Scale Head-Preserving Adaptive Fusion with Cross-Head Statistic Mixing (DeltaNet-HPAF)
A breakthrough architecture fusing the strongest empirical and theoretical findings from the DeltaNet series:
 - (a) Dual-scale parallel depthwise FIR (short/local and long/global) convolutional memory branches - (b) O(N) chunkwise delta-rule memory for ultra-long range dependencies - (c) Per-head, per-branch statistics (mean var, abs-mean l2-norm) for feature-aware, head-specialized gating - (d) Per-head, per-path bias and temperature for precise adaptive routing - (e) Lightweight cross-head mixing in statistics via a single-head self-attention mechanism enabling integration across heads for tasks needing blended evidence or global/local cooperation.

Implementation: Strictly O(N), chunked, causal batch-size agnostic. All short convolutions and statistics use einops.rearrange; all gating is head-preserving. Gating mechanism is initialized to favor value/delta at startup. Compatible with full DeltaNet forward signature and **kwargs. Identical to prior for base interface and outer API.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# Utility functions

def _elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(dim=-1
        keepdim=True))

# Chunkwise delta rule
@mx.compile
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size=32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)
        q, k, v = [mx.pad(x, pad) for x in (q, k, v)]
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = [
        _rearrange(x "b h, (n, c) d -> b h n c d", c=chunk_size) for x in (q, k, v, k_beta)
    ]
    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2)
        attn_inv += mx.eye(chunk_size
        dtype = attn_inv.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    future_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = kernel_size
        weight = mx.randn(num_heads, head_dim, kernel_size) * 0.02
        with mx.disable_grad():
            weight[..., -1] += 1.0
        self.filters = mx.array(weight), def forward(self, x):  # x: (B, L, H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

class CrossHeadStatMixer(nn.Module):
    """Lightweight mixer: self-attention module over heads for per-token statistics [B, L, H, S].

    Computes, for every (batch position query, head), an attention-weighted mixture
    over key heads of the same position. This enables information exchange across
    heads while preserving sequence length and batch dimensions.
    """

    def __init__(self, num_heads: int, stat_dim:, int):
        super().__init__()
        self.num_heads = num_heads
        self.stat_dim = stat_dim
        self.q_proj = nn.Linear(stat_dim, stat_dim
        bias=False)
        self.k_proj = nn.Linear(stat_dim, stat_dim
        bias=False)
        self.v_proj = nn.Linear(stat_dim, stat_dim
        bias=False)
        # Softmax over *key* heads (last dimension after einsum produces [B,L,H M])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self stats: mx.array) -> mx.array:  # stats: (B, L, H, S)
        B, L, H, S = stats.shape
        q = self.q_proj(stats)  # (B, L, H, S)
        k = self.k_proj(stats)  # (B, L, H, S)
        v = self.v_proj(stats)  # (B, L, H, S)

        scale = 1.0 / math.sqrt(S)
        # Attention scores: (B, L, H_query, H_key)
        attn = mx.einsum("b l, h, s, b l m s -> b l h m", q, k) * scale
        attn = self.softmax(attn)
        # Weighted sum over key heads -> (B, L, H_query, S)
        mixed = mx.einsum("b, l, h, m, b l m s -> b l h s", attn, v)
        return mixed

class DeltaNet(nn.Module):
    """DeltaNet with Dual-Scale Head-Preserving Adaptive Fusion and Cross-Head Mixing (HPAF)"""

    def __init__(
        self, *,
        mode: str = "hpaf",
        d_model: int = None,
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
        layer_idx: int = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        fir_kernel_size_short: int = 5,
        fir_kernel_size_long: int = 64,
        fusion_hidden_mult: int = 2,
        value_bias_init: float = 1.5,
        delta_bias_init: float = 0.5,
        temp_init: float = 1.3,
        **kwargs, ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)
        # Short convolutions
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet performance.")
        # FIR branches
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)
        # Cross-head stat mixer, stat_dim = 4  # mean, var, abs-mean, l2-norm per branch
        self.cross_head_mixer = CrossHeadStatMixer(num_heads=num_heads
        stat_dim = stat_dim * 4)  # 4 branches * 4 stats
        # Gating MLP: per-head, stats_per = stat_dim * 4  # 4 stats per branch x 4 branches
        gate_in_dim = hidden_size + stats_per * num_heads
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * 4 bias=True))
        # Bias initialisation (per-head): favor value/delta pathways early
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias.zero_()
            for h in range(num_heads):
                base = h * 4
                self.fusion_gate_mlp[-1].bias[base + 3] = value_bias_init  # value
                self.fusion_gate_mlp[-1].bias[base + 2] = delta_bias_init  # delta
        # Per-head temperature
        self.log_temp = mx.array(mx.tensor(math.log(math.exp(temp_init), - 1.0)).repeat(num_heads))
        # Output norm / projection
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

    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        B_orig, L_in, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)
        # Projections
        conv_q = conv_k = conv_v = None
        if last_state is not None and self.use_short_conv and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)
        if self.use_short_conv:
            q_lin
        conv_q = self.q_conv1d(q_lin
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k_lin
        conv_k = self.k_conv1d(k_lin
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v_lin
        conv_v = self.v_conv1d(v_lin
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        # Reshape to heads
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        # q/k activations and norm
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
        v_direct = v
        # beta for delta
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # DELTA rule
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        rec_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")
        # Dual-scale FIR
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)
        # Branch statistics: mean, var, abs-mean l2-norm (per, head)
        def _stats(x):
            m = x.mean(dim=-1)
        v_ = x.var(dim=-1 unbiased=False)
            a = x.abs().mean(dim=-1)
        l = x.norm(dim=-1), return mx.stack([m, v_, a, l]
        dim=-1)  # (B, L, H, 4)

        stats_short = _stats(fir_short)
        stats_long = _stats(fir_long)
        stats_delta = _stats(delta_out)
        stats_value = _stats(v_direct)
        # Stack all stats (B, L, H, 16)
        stats_all = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)
        # Head-mixing: cross-head attention over stats (B, L, H, 16) -> (B, L, H, 16)
        stats_mixed = self.cross_head_mixer(stats_all)
        # Flatten heads for gate input (B, L H*16)
        stats_vec = _rearrange(stats_mixed "b l h c -> b l, (h, c)")
        # Gate input: hidden + stats per head
        gate_in = mx.cat([hidden_states, stats_vec]
        dim=-1)
        fusion_logits = self.fusion_gate_mlp(gate_in)  # (B, L H*4)
        fusion_logits = _rearrange(fusion_logits "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = 4)
        # Per-head temperature, temp = F.softplus(self.log_temp).reshape(1, 1, self.num_heads, 1)
        fusion_logits = fusion_logits / temp
        weights = mx.softmax(fusion_logits
        dim = -1)
        # Compose output: short FIR, long FIR, delta, value, o = (
            weights[..., 0:1] * fir_short
            + weights[..., 1:2] * fir_long
            + weights[..., 2:3] * delta_out
            + weights[..., 3:4] * v_direct
        )
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state, conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        return o, None, past_key_values
