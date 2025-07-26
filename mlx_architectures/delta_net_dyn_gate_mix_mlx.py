"""
MLX-converted architecture: delta_net_dyn_gate_mix
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
DeltaNet – Dynamic Per-Head, Per-Token Parallel Memory Gating
============================================================
This evolutionary variant implements a breakthrough dynamic mixture-of-memory pathway that addresses
the principal weaknesses of the EMA-Blend DeltaNet model by:

1. **Dynamic Per-Head, Per-Token Gating**: Instead of a global scalar mix between Delta and EMA memory outputs,
   it uses a learned, *input-dependent* per-head, per-token gate (a linear projection of the hidden state with sigmoid)
   for blending. This enables the model to suppress the smoothed EMA memory adaptively at tokens and heads where
   associative precision is crucial for reasoning (e.g. coreference, multi-hop), while leveraging it where
   long-range, context blending is beneficial (e.g. narrative, factual QA).

2. **Research Inspirations**: This design draws directly from research on Gated Attention, GLA, and head-/token-
dynamic gating found in modern Transformers and mixture-of-memory neural architectures, where adaptive, content-aware
mixtures are proven to deliver both state scalability and high-precision associative recall within a single efficient module.

3. **Efficiency and Causality**: The core chunk-wise Delta and EMA rules remain unchanged, respecting all batch size,
O(N) complexity, and masking requirements. Dynamic gating introduces negligible overhead and is fully differentiable.

4. **Implementation Details**:
   - A new projection (mix_proj) produces gates of shape (batch, seq_len, num_heads), followed by sigmoid.
   - The EMA and Delta outputs are blended *per token, per head* using the computed gate.
   - The remaining pathways (state caching, chunked processing, convolutions, etc.) are not changed from baseline.
   - All code is fully batch-size agnostic, uses einops.rearrange for shape handling, and preserves mx.compile
     on kernels.
   - Gate is always on by default, but can be disabled for ablation by setting use_dynamic_mix_gate=False (rare).

5. **Interface/Signature**: All __init__ and forward args/kwargs are preserved for maximal drop-in compatibility.

"""
from __future__ import annotations

import mlx.core as mx
import mlx.core as mx.nn as nn
from mx.nn import functional as F



############################
# ORIGINAL HELPERS – UNCHANGED
############################
def softmax(x):
    return F.softmax(x, dim=-1)

@mx.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = mx.pad(q, (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))

    padded_len = l + pad_len
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    mask = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), diagonal=0)
    q, k, v, k_beta = map(
        lambda x: _rearrange(x, "b h (n c) d -> b h n c d"c=chunk_size),
        [q, k, v, k_beta],
    )
    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = (
            attn[..., i, :i]
            + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
        )
    attn = attn + mx.eye(chunk_size, dtype=mx.float, q.device)
    attn = attn

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device),
        diagonal=1,
    )
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

@mx.compile
def ema_rule_chunkwise(
    v: mx.Tensor,
    gamma: mx.Tensor,
    init_state: Optional[mx.Tensor] = None,
):
    b, h, l, d_v = v.shape
    ema_out = mx.empty_like(v)
    if init_state is None:
        state = mx.zeros((b, h, d_v), dtype=v.dtype, v.device)
    else:
        state = init_state
    for t in range(l):
        g_t = gamma[:, :, t].expand_dims(-1)
        state = g_t * state + (1.0 - g_t) * v[:, :, t]
        ema_out[:, :, t] = state
    return ema_out, state

def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

class DeltaNet(nn.Module):
    """DeltaNet layer with parallel Delta-rule and EMA memory, fused by dynamic per-head, per-token gate."""

    def __init__(
        self,
        *,
        mode: str = "chunk1",
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
        use_ema: bool = True,
        use_dynamic_mix_gate: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.use_beta = use_beta
        self.use_ema = use_ema
        self.use_dynamic_mix_gate = use_dynamic_mix_gate
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.dec_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # NEW: dynamic mixing gate projection (per-head, per-position)
        if self.use_ema and self.use_dynamic_mix_gate:
            self.mix_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
            nn.init.zeros_(self.mix_proj.bias)  # Start unbiased, so 0.5 sigmoid
        else:
            self

        if use_short_conv:
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation="silu",
            )
        else:
            raise UserWarning(
                "_ShortConvolution is crucial to the performance. Do not turn it off."
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: mx.Tensor,
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.Tensor, Optional[mx.Tensor], Optional[Dict]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).expand_dims(0)

        # -- Projections and convolutions --
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state.get("conv_state", (None, None, None))
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(
            lambda x: _rearrange(x, "... (h d) -> ... h d"d=self.head_k_dim),
            (q, k),
        )
        v = _rearrange(v, "... (h d) -> ... h d"d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q = sum_norm(q)
            k = sum_norm(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Prepare for kernel shapes
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")

        recurrent_state = last_state["recurrent_state"] if last_state is not None and "recurrent_state" in last_state else None
        o_d, recurrent_state = delta_rule_chunkwise(q=q_d, k=k_d, v=v_d, beta=beta_d)
        o_d = _rearrange(o_d, "b h l d -> b l h d")

        if self.use_ema:
            gamma = self.dec_proj(hidden_states).sigmoid()  # (b l h)
            gamma = _rearrange(gamma, "b l h -> b h l")
            ema_state_prev = last_state.get("ema_state", None) if last_state is not None else None
            v_for_ema = _rearrange(v, "b l h d -> b h l d")
            ema_out, ema_state = ema_rule_chunkwise(v_for_ema, gamma, ema_state_prev)
            ema_out = _rearrange(ema_out, "b h l d -> b l h d")

            if self.use_dynamic_mix_gate:
                mix_gate = mx.sigmoid(self.mix_proj(hidden_states))  # (b, l, h)
                mix_gate = _rearrange(mix_gate, "b l h -> b l h 1")  # For broadcast
                o = (1.0 - mix_gate) * o_d + mix_gate * ema_out
            else:
                mix = mx.sigmoid(mx.zeros(1, o_d.device, dtype=o_d.dtype))  # 0.5
                o = (1.0 - mix) * o_d + mix * ema_out
        else:
            ema_state = None
            o = o_d

        # Cache update
        if past_key_values is not None and isinstance(past_key_values, dict):
            past_key_values["recurrent_state"] = recurrent_state
            past_key_values["conv_state"] = (
                conv_state_q,
                conv_state_k,
                conv_state_v,
            ) if self.use_short_conv else None
            if self.use_ema:
                past_key_values["ema_state"] = ema_state
            past_key_values["layer_idx"] = self.layer_idx
            past_key_values["offset"] = q_len
        elif past_key_values is not None:
            if hasattr(past_key_values, 'update'):
                past_key_values.update(
                    recurrent_state=recurrent_state,
                    conv_state=(
                        conv_state_q,
                        conv_state_k,
                        conv_state_v,
                    ) if self.use_short_conv else None,
                    layer_idx=self.layer_idx,
                    offset=q_len,
                )
                if self.use_ema and (hasattr(past_key_values, '__setitem__') or hasattr(past_key_values, 'ema_state')):
                    try:
                        past_key_values["ema_state"] = ema_state
                    except Exception:
                        pass

        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d"d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values
