"""
MLX-converted architecture: delta_net_cagf_dpaf_eash
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
DeltaNet – Context-Conditioned Adaptive Gated Fusion with Dual-Phase Path Floor and Entropy-Annealed Gate Sharpening
=====================================================================================
# ... rest same as before ...
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F


# ---------------------------------------------
def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads, head_dim, kernel_size: int, noise_std: float = 0.02):
        super().__init__()
        self.kernel_size = kernel_size
        filt = mx.zeros(num_heads, head_dim, kernel_size)
        filt[..., -1] = 1.0
        filt += noise_std * mx.randn_like(filt)
        self.filters = mx.array(filt)
    def forward(self, x):
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_flat = _rearrange(x, "b l h d -> b (h d) l")
        x_pad = mx.pad(x_flat, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

@mx.compile
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    attn = -(k_beta @ k.transpose(-1,-2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + mx.eye(chunk_size, dtype=attn.dtype, q.device)
    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len: o = o[:, :, :L]
    return o, S

class DeltaNet(nn.Module):
    """DeltaNet with adaptive dual-phase path floor, content-adaptive gating, and entropy-annealed regularisation."""
    def __init__(self,
        mode: str = "cagf_dpaf_eash",
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
        # FIR kernel sizes
        fir_kernel_size_short: int = 5,
        fir_kernel_size_long: int = 64,
        fir_noise_std: float = 2e-2,
        # Fusion gate
        fusion_hidden_mult: int = 2,
        fusion_dropout: float = 0.0,
        # Dual-phase floor schedule
        epsilon_init: float = 0.10,
        epsilon_final: float = 0.025,
        epsilon_decay_steps: int = 4000,
        # Entropy annealing
        entropy_reg_init: float = 0.02,
        entropy_reg_final: float = 0.001,
        entropy_decay_steps: int = 12000,
        # Per-head learnable temp
        temp_init: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
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
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short, noise_std=fir_noise_std)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long, noise_std=fir_noise_std)
        # Content adaptive gate: [hidden, per-head local stats, per-branch norm, pairwise branch ||diff||]
        self.stat_dim = 4
        in_dim = hidden_size + self.num_heads * (self.stat_dim * 4 + 4 + 6) # simplified: stats for 4 branches, L2 norm per, pairwise 6 diffs
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, self.num_heads * 4, bias=True)
        )
        with mx.no_grad():
            bias = self.fusion_gate_mlp[-1].bias
            bias.zero_()
            bias[3::4] = 2.0 # value
            bias[2::4] = 1.2 # Δ rule
        self.log_temp = mx.array(mx.full((self.num_heads,1), math.log(temp_init)))
        # Scheduling for path floor (dual-phase) and entropy
        self.epsilon_init = float(epsilon_init)
        self.epsilon_final = float(epsilon_final)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.entropy_reg_init = float(entropy_reg_init)
        self.entropy_reg_final = float(entropy_reg_final)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self, persistent=False)
        # Output norm/proj
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @staticmethod
    def _per_head_stats(x: mx.Tensor):
        # mean, var, abs mean, L2 norm
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, unbiased=False)
        absmean = x.abs().mean(dim=-1)
        l2 = x.norm(dim=-1)
        return mx.stack([mean, var, absmean, l2], dim=-1) # (B,L,H,4)
    def _dual_phase_epsilon(self):
        step = float(self._step.item())
        if step >= self.epsilon_decay_steps:
            return self.epsilon_final
        ratio = step / max(1., float(self.epsilon_decay_steps))
        return self.epsilon_init + (self.epsilon_final - self.epsilon_init) * ratio
    def _entropy_lambda(self):
        step = float(self._step.item())
        if step >= self.entropy_decay_steps:
            return self.entropy_reg_final
        ratio = step / max(1., float(self.entropy_decay_steps))
        return self.entropy_reg_init + (self.entropy_reg_final - self.entropy_reg_init) * ratio
    def forward(self,
        hidden_states: mx.Tensor,
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        batch_size, seq_len_in, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get('conv_state') is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q, k = map(lambda x: _rearrange(x, "b l (h d) -> b l h d"d=self.head_k_dim), (q, k))
        v = _rearrange(v, "b l (h d) -> b l h d"d=self.head_v_dim)
        # Ensure numerical dtype matches Linear weight (prevents float != bf16 error)
        dtype = self.o_proj.weight.dtype
        q = q
        k = k
        v = v
        hidden_states = hidden_states
        # (all further tensors are based on q/k/v and hidden_states, so all match)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)
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
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")
        local_short = self.local_fir_short(v)
        local_long = self.local_fir_long(v)
        # Gate input construction (efficient, essential branch stats, per-head)
        s_short = self._per_head_stats(local_short) # (b,l,h,4)
        s_long = self._per_head_stats(local_long)
        s_delta = self._per_head_stats(delta_out)
        s_val = self._per_head_stats(v)
        # Per-branch L2 norm (for adaptive mixing)
        l2_short = local_short.norm(dim=-1)
        l2_long = local_long.norm(dim=-1)
        l2_delta = delta_out.norm(dim=-1)
        l2_val = v.norm(dim=-1)
        # Pairwise ||diff||
        d1 = (local_short-local_long).norm(dim=-1)
        d2 = (local_short-delta_out).norm(dim=-1)
        d3 = (local_short-v).norm(dim=-1)
        d4 = (local_long-delta_out).norm(dim=-1)
        d5 = (local_long-v).norm(dim=-1)
        d6 = (delta_out-v).norm(dim=-1)
        # gate_in shape: (b,l,num_heads*total)
        gate_in = mx.cat([_rearrange(hidden_states, "b l d -> b l d")] +
            [_rearrange(x, "b l h s -> b l (h s)") for x in [s_short,s_long,s_delta,s_val]] +
            [_rearrange(x.expand_dims(-1), "b l h 1 -> b l (h)") for x in [l2_short,l2_long,l2_delta,l2_val]] +
            [_rearrange(x.expand_dims(-1), "b l h 1 -> b l (h)") for x in [d1, d2, d3, d4, d5, d6]], dim=-1)
        gate_logits_full = self.fusion_gate_mlp(gate_in) # (b,l,num_heads*4)
        gate_logits = _rearrange(gate_logits_full, "b l (h p) -> b l h p"h=self.num_heads, p=4)
        temp = mx.exp(self.log_temp).expand_dims(0).expand_dims(0) # (1,1,h,1)
        gate_logits = gate_logits / temp
        fusion_weights = mx.softmax(gate_logits, dim=-1)
        # Dual-phase epsilon (local/short-FIR min allocation)
        eps = self._dual_phase_epsilon()
        eps_vec = mx.tensor([eps, 0.0, 0.0, 0.0], gate_logits.device, dtype=dtype).reshape(1,1,1,4)
        fusion_weights = mx.max(fusion_weights, eps_vec)
        fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)
        # Mixture
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v
        )
        # Cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=seq_len_in,
            )
        # Output norm/proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, seq_len_in)
        # Entropy penalty
        ent = -(fusion_weights * (fusion_weights+1e-8).log()).sum(-1).mean()
        self._step += 1
        reg_loss = -self._entropy_lambda() * ent if self.training and self._entropy_lambda() > 0 else None
        return o, reg_loss, past_key_values
