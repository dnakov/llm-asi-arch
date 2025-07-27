from __future__ import annotations

"""
MLX-converted architecture: delta_net_msfr_mn
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
    return x / mx.linalg.norm(x, axis=-1,
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
    def __init__(self, hidden_size: int,
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
DeltaNet – Multi-Scale Feedback-Routed MixNorm (MSFR-MN) – delta_net_msfr_mn
Breakthrough integrated multi-path chunked memory/fusion architecture:
- Per-head, feedback-conditioned, multi-scale memory routing (local, mid, delta, identity)
- Cross-path statistics and dot products for relational routing (Block-State/SELM, research)
- Minimal high-gain MixNorm(per-token, per-head nn.RMSNorm) post-fusion for robust variance/stability
- KL/entropy-based path diversity regularization to guarantee identity/local path survival(solves, SWDE/BoolQ, regression)
- All chunked O(N), full einops, batch-agnostic robust @mx.compile kernel for core memory/conv ops
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# ----------------------------------------------------
def _elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

# ----------------------------------------------------
@mx.compile
def delta_rule_chunkwise
    q: mx.array, k: mx.array, v: mx.array, beta: mx.array chunk_size: int = 32
):
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
    k_beta = map(lambda x: _rearrange(x, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    mask_tri_full = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri_full, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] += (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size, dtype = attn.dtype)
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
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
class _DepthwiseCausalConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        weight = mx.randn(num_heads, * head_dim, 1, kernel_size) / math.sqrt(kernel_size)
        weight[..., -1] += 1.0  # causal identity bias at latest step
        self.weight = mx.array(weight), def forward(self, x: mx.array):  # [B, L, H, D]
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

class MixNorm(nn.Module):
    """Minimal per-token, per-head nn.RMSNorm (no bias affine only for variance, control)"""
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.array(mx.ones(num_heads, head_dim))
        self.eps = eps
    def forward(self, x):  # [B, L, H D]
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.weight.reshape(1, 1, *self.weight.shape)

class DeltaNet(nn.Module):
    """DeltaNet with Multi-Scale Feedback-Routed MixNorm (MSFR-MN)"""

    def __init__(self, mode: str =, "msfr_mn",
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
        # Multi-scale conv params
        local_kernel_size: int = 7,
        mid_kernel_size: int = 25,
        # Router params
        router_hidden_mult: int = 2,
        router_kl_coeff: float = 0.03,
        router_floor: float = 0.01,
        # MixNorm
        mixnorm_eps: float = 1e-5,
        **kwargs: Dict, ):
        super().__init__()
        self.mode = mode
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
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
            self.b_proj = nn.Linear(hidden_size, self.num_heads
            bias=False)
        if self.use_short_conv:
            activation = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=activation
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=activation
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution, is mandatory for robust DeltaNet variants.")
        self.local_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim
        kernel_size = local_kernel_size)
        self.mid_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim
        kernel_size = mid_kernel_size)
        # Router features: mean/var dot products (cross-relational), cross-min, per-path, router_feat_dim = (
            hidden_size
            + (8 + 6) * num_heads  # 8 = mean/var * 4; 6 = dot/cross-mean (local*mid, local*delta ...) across 4, pairs)
        router_hidden_dim = router_hidden_mult * router_feat_dim
        router_out_dim = num_heads * 4  # [local, mid, delta, id]
        self.router_mlp = nn.Sequential(, nn.Linear(router_feat_dim, router_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, router_out_dim, bias=True))
        # Path diversity bias(for, identity/survival)
        with mx.disable_grad():
            self.router_mlp[-1].bias.zero_()
            bias_view = self.router_mlp[-1].bias.reshape(num_heads, 4)
            bias_view[:, 2] = 0.7  # favor delta at init
            bias_view[:, 3] = 0.7  # favor id at init
        self.router_kl_coeff = router_kl_coeff
        self.router_floor = router_floor
        self.mixnorm = MixNorm(num_heads, self.head_v_dim
        eps = mixnorm_eps)
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    def _router_stats(self, outs):
        """Compute mean/var and cross-dot stats: outs is list [local, mid, delta, id] [B,L,H D] each"""
        feats = []
        for out in outs:
            feats.append(out.mean(-1)), # (B,L, H)
            feats.append(out.var(-1)), # (B,L, H)
        # Cross dot/cosine mean between each unique branch pair (6, pairs)
        for i in range(len(outs)):
            for j in range(i+1, len(outs)):
                # mean dot product per head
    dot = (outs[i] * outs[j]).mean(-1), # (B,L, H)
                feats.append(dot)
        return feats  # list of 8+6 tensors (B,L, H)

    def forward(self, hidden_states: mx.array,  # [B L D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
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
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
                conv_state_q
        conv_state_k, conv_state_v = last_state.get("conv_state", (None None, None))
        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
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
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d
        chunk_size =32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")
        v_direct = v
        local_out = self.local_conv(v_direct)
        mid_out = self.mid_conv(v_direct)
        id_out = v_direct
        # Router advanced stats (mean, var dot, products)
        stats_feats = self._router_stats([local_out, mid_out, delta_out id_out])
        router_in = mx.cat(
            [hidden_states] + [_rearrange(x, "b l h -> b l (h)") for x in stats_feats]
        dim=-1
        )  # (B, L, feat)
        router_logits = self.router_mlp(router_in)  # [B, L num_heads*4]
        router_logits = _rearrange(router_logits, "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = 4)
        # probability flooring for all paths(paths, must, survive)
        router_weights = F.softmax(router_logits, dim = -1)
        if self.router_floor > 0.:
            router_weights = mx.clamp(router_weights, min = self.router_floor)
            router_weights = router_weights / router_weights.sum(-1, keepdim=True)
        # Diversity regularization (KL to Uniform across all, paths)
        reg_loss = None
        if self.router_kl_coeff > 0. and self.training:
            with mx.disable_grad():
                U = mx.full_like(router_weights, 1.0 / 4)
            kl = (router_weights * ((router_weights+1e-8).log() - U.log())).sum(-1).mean()
        reg_loss = self.router_kl_coeff * kl
        # Multi-path fusion
    o = (
            router_weights[..., 0:1] * local_out +
            router_weights[..., 1:2] * mid_out +
            router_weights[..., 2:3] * delta_out +
            router_weights[..., 3:4] * id_out
        )  # [B, L, H, D]
        # Minimal high-gain MixNorm after fusion
    o = self.mixnorm(o)
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        return o, reg_loss, past_key_values
