from __future__ import annotations

"""
MLX-converted architecture: delta_net_dyn_decay_fractal_gate
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
DeltaNet – Dynamic Chunkwise Decay & Gated Fractal Mixer (2024-06-09)
This variant unifies the strongest ideas from prior experiments while
addressing the key weakness repeatedly observed in *uniform* or
*static* time–decay mechanisms – namely indiscriminate forgetting of
potentially important late-context information.

Key innovations (enabled by, default)
1. **Adaptive Decay Gate 𝛾(t)**
   • A *per-token per-head* forget gate is computed via a lightweight
     linear projection (`gamma_proj`).  This replaces the static scalar
     or position-only decay of earlier variants.
   • During the chunk-wise recurrent update the gate is *averaged within
     each chunk* (maintaining O(N) complexity) giving a *dynamic,
     content-aware* decay coefficient `gamma_chunk` ∈ [0,1].
   • State update:   `S = gamma_chunk • S + ΔS`  – allowing the network
     to *retain* or *forget* past memory depending on current input
     statistics.

2. **Gated Fractal Mixer**
   • Retains the log-depth, causal dilated convolution stack from
     *delta_net_fractalmix* to provide rapid global context exchange.
   • A **learnable per-token gate** (sigmoid) modulates how much mixer
     information is fused back into the core Delta path – mitigating the
     over-smoothing observed previously when mixer output was added
     unconditionally.

3. **Rotary / Absolute Dual-Position Encoding** (kept from best, variant).
4. **Short Convolutional Projections** for efficient local processing.
5. **Adaptive Output Mix Gate** between recurrent memory and token value.

All additions preserve strict *sub-quadratic* complexity (O(N, log, N)) and
full interface compatibility.  No config changes are required – sensible
defaults activate new features automatically.
"""

import functools
import math

import mlx.core as mx
import mlx.nn as nn

from mx.nn import functional as F


#######################################################################
# Rotary helpers (copied from dual_pos_time_decay, variant)            #
#######################################################################


@functools.lru_cache(maxsize=32)
def _get_inv_freq(dim:, int, device: dtype: mx.dtype) -> mx.array:
    inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2, device
    dtype = dtype) / dim))
    return inv_freq


def _build_sin_cos(seq_len:, int, dim: int, device: dtype: mx.dtype):
    inv_freq = _get_inv_freq(dim, device, dtype)
    t = mx.arange(seq_len, device
    dtype = dtype)
    sinusoid_inp = mx.einsum('i, j -> i j', t, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
    return sin, cos


def _apply_rotary(x:, mx.array, sin: mx.array cos: mx.array) -> mx.array:
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        x_rot = mx.stack((rot_x1, rot_x2)
        dim=-1)
    return _rearrange(x_rot "... d two -> ..., (two, d)")

#######################################################################
# Misc helpers                                                        #
#######################################################################

def elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

#######################################################################
# Core chunk-wise delta rule with adaptive decay                      #
#######################################################################


@mx.compile  # type: ignore[misc]
def delta_rule_chunkwise(
    q:, mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    gamma: Optional[mx.array] = None,
    *,
    chunk_size: int = 32):
    """Causal associative retrieval with *content-adaptive* decay.

    Shapes
    q, k: (b, h, l, d_k)
    v   : (b, h, l, d_v)
    beta: (b, h, l)
    gamma: (b h, l)  – dynamic decay gate in [0,1].  If *None* then no decay.
    """
    b, h, l, d_k = q.shape
        pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q
        (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))
        if gamma is not None:
            gamma = mx.pad(gamma
        (0, pad_len))

    padded_len = l + pad_len

    # --------------------------------------------- normalise & pre-scale
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # --------------------------------------------- reshape into chunks
    q, k, v, k_beta = map(
        lambda x: _rearrange(x "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))
    if gamma is not None:
        gamma_c = _rearrange(gamma "b h
        (n, c) -> b h n c"
        c=chunk_size)
    else:
        gamma_c = None
        mask_tri_inc = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=0
    )

    # (I - B K K^T)^{-1} per chunk (as in original, implementation)
    attn_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_tri_inc, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2), attn_inv = attn_inv + mx.eye(chunk_size
        dtype = attn_inv.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta

    # --------------------------------------------- initialise state & output
        d_v = v.shape[-1]
    S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask_future = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=1
    )

    num_chunks = padded_len // chunk_size
    for idx in range(num_chunks):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]  # (b h c, d_k)
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S  # (b h c, d_v)
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + attn_local @ u_i
        delta_S = k_i.transpose(-1 -2) @ u_i  # (b h d_k, d_v)
        if gamma_c is not None:
            # use *mean* gamma of tokens within the chunk → (b
        h)
            gamma_chunk = gamma_c[:, :, idx].mean(-1)
            S = gamma_chunk[..., None, None] * S + delta_S
        else:
            S = S + delta_S

    # --------------------------------------------- stitch back chunks
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :l]
    return o, S
#######################################################################
# Fractal mixer with token-wise gate                                 #
#######################################################################


class _CausalFractalMixer(nn.Module):
    """Depth-wise dilated convolution stack (log-depth receptive, field)."""

    def __init__(self, hidden_size: int, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.convs = nn.ModuleList()
        for i in range(levels):
            dilation = 2 ** i
        conv = nn.Conv1d(, hidden_size,
                hidden_size,
                kernel_size=2
        dilation=dilation,
                groups=hidden_size
        bias = False)
            nn.init.zeros_(conv.weight)  # near-identity
            self.convs.append(conv)

    def forward(self x: mx.array) -> mx.array:  # (b l, d)
        residual = x
        x = _rearrange(x "b l d -> b d l")
        out = x
        for conv in self.convs:
            pad_left = conv.dilation[0]
            x_pad = mx.pad(x, (pad_left, 0))
            out = out + conv(x_pad)
        out = _rearrange(out "b d l -> b l d")
        return out + residual

#######################################################################
# Optional type stubs                                                #
#######################################################################

#######################################################################
# Main DeltaNet                                                      #
#######################################################################


class DeltaNet(nn.Module):
    """DeltaNet layer with *adaptive* decay and gated fractal mixing."""

    def __init__(, self,
        mode: str = 'chunk1',
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
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        # rotary
        use_rotary: bool = True,
        # adaptive decay params
        adaptive_decay: bool = True,
        # fractal mixer
        use_fractal_mixer: bool = True,
        mixer_levels: int = 4 **kwargs) -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_rotary = use_rotary
        self.adaptive_decay = adaptive_decay
        self.use_fractal_mixer = use_fractal_mixer
        self.allow_neg_eigval = allow_neg_eigval

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

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
        self.layer_idx = layer_idx

        # dimensions
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.key_dim % num_heads == 0, 'key_dim must be divisible by num_heads'
        assert self.value_dim % num_heads == 0, 'value_dim must be divisible by num_heads'
        if self.use_rotary:
            assert self.head_k_dim % 2 == 0
        'head_k_dim must be even for rotary embeddings'

        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        # beta gate
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        self.num_heads
            bias=False)

        # adaptive decay gate
        if self.adaptive_decay:
            self.gamma_proj = nn.Linear(hidden_size
        self.num_heads
            bias=False)

        # rotary blend gate
        if self.use_rotary:
            self.rotary_mix_logit = mx.array(mx.zeros(num_heads))
        # short convs
        if self.use_short_conv:
            self.q_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size)
                                              activation='silu' if
        qk_activation = = 'silu' else, None)
            self.k_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size)
                                              activation='silu' if
        qk_activation = = 'silu' else, None)
            self.v_conv1d = _ShortConvolution(self.value_dim
        kernel_size=conv_size
        activation = 'silu')
        else:
            raise UserWarning('_ShortConvolution is mandatory for DeltaNet performance – do not disable.')

        # optional output gate
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

        # fractal mixer & gating
        if self.use_fractal_mixer:
            self.fractal_mixer = _CausalFractalMixer(hidden_size
        levels = mixer_levels)
            self.frac_gate_proj = nn.Linear(hidden_size, 1
            bias=True)
            nn.init.constant_(self.frac_gate_proj.bias -1.0)  # start mostly closed
            self.mixer_norm = nn.RMSNorm(hidden_size
        eps = norm_eps)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(, self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional['Cache'] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False,
        **kwargs: 'Unpack[Dict]',  # type: ignore[misc]
    ) -> Tuple[mx.array, Optional[mx.array], Optional['Cache']]:  # noqa: D401
        if attention_mask is not None:
            assert attention_mask.dim() == 2
        'attention_mask must be (batch, seq_len) with 0/1 entries.'

        batch_size, padded_len, _ = hidden_states.shape

        # retrieve cached state (if, any)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens' None)
        max_seqlen = padded_len

        # optional unpadding
        if attention_mask is not None:
            indices
        cu_seqlens, max_seqlen = _get_unpad_data(attention_mask[:, -padded_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        seq_len = hidden_states.shape[1]

        # ------------------------------------------------ projections (+ short, conv)
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not, None:
                conv_state_q
        conv_state_k, conv_state_v = last_state['conv_state']
            q
        conv_state_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_state_q)
                                            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_state_k)
                                            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_state_v)
                                            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:  # not expected, q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q
        k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # ------------------------------------------------ head split & activations
        q
        k = map(lambda x: _rearrange(x "..., (h, d) -> ... h d"
        d=self.head_k_dim), (q, k))
        v = _rearrange(v "..., (h, d) -> ... h d"
        d=self.head_v_dim)

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q
        k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError
        if self.qk_norm == 'sum':
            q
        k = sum_norm(q), sum_norm(k)

        # ------------------------------------------------ rotary embedding blend
        if self.use_rotary:
            sin
            cos = _build_sin_cos(seq_len, self.head_k_dim
            dtype = q.dtype)
            q_rot = _apply_rotary(q, sin, cos)
            k_rot = _apply_rotary(k, sin, cos)
            mix_gate = mx.sigmoid(self.rotary_mix_logit)[None, None, :, None]
            q = mix_gate * q_rot + (1.0 - mix_gate) * q
        k = mix_gate * k_rot + (1.0 - mix_gate) * k

        # ------------------------------------------------ beta & gamma gates
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        if self.adaptive_decay:
            gamma = self.gamma_proj(hidden_states).sigmoid()  # (b
        l, h)
        else:
            gamma = None

        # ------------------------------------------------ layout for delta rule
        q_t = _rearrange(q "b l h d -> b h l d")
        k_t = _rearrange(k "b l h d -> b h l d")
        v_t = _rearrange(v "b l h d -> b h l d")
        beta_t = _rearrange(beta "b l h -> b h l")
        gamma_t = _rearrange(gamma "b l h -> b h l") if gamma is not None else None

        o_t
        recurrent_state = delta_rule_chunkwise(q=q_t
        k=k_t
        v=v_t
        beta=beta_t
        gamma = gamma_t)
        o = _rearrange(o_t "b h l d -> b l h d")

        # ------------------------------------------------ adaptive mix gate between memory output and value
        mix_gate = None
        if hasattr(self 'mix_proj'):
            mix_gate = mx.sigmoid(self.mix_proj(hidden_states))  # from earlier variants
        if mix_gate is not None:
            mix_gate = _rearrange(mix_gate "b l h -> b l h 1")
            o = mix_gate * o + (1.0 - mix_gate) * v

        # ------------------------------------------------ cache update
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = seq_len)

        # ------------------------------------------------ output norm/proj
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states)
        "... (h, d) -> ... h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = self.o_proj(_rearrange(o "b l h d -> b l, (h, d)"))

        # ------------------------------------------------ gated fractal mixer fusion
        if self.use_fractal_mixer:
            mixer_out = self.fractal_mixer(hidden_states)
            gate = mx.sigmoid(self.frac_gate_proj(hidden_states))  # (b l, 1)
            mixer_out = gate * self.mixer_norm(mixer_out)
            o = o + mixer_out

        # ------------------------------------------------ repad
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, max_seqlen)

        return o, None, past_key_values
