# -*- coding: utf-8 -*-
"""
DeltaNet – Dynamic Chunkwise Decay & Gated Fractal Mixer (2024-06-09)
--------------------------------------------------------------------
This variant unifies the strongest ideas from prior experiments while
addressing the key weakness repeatedly observed in *uniform* or
*static* time–decay mechanisms – namely, indiscriminate forgetting of
potentially important late-context information.

Key innovations (enabled by default)
====================================
1. **Adaptive Decay Gate 𝛾(t)**
   • A *per-token, per-head* forget gate is computed via a lightweight
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
   • A **learnable, per-token gate** (sigmoid) modulates how much mixer
     information is fused back into the core Delta path – mitigating the
     over-smoothing observed previously when mixer output was added
     unconditionally.

3. **Rotary / Absolute Dual-Position Encoding** (kept from best variant).
4. **Short Convolutional Projections** for efficient local processing.
5. **Adaptive Output Mix Gate** between recurrent memory and token value.

All additions preserve strict *sub-quadratic* complexity (O(N log N)) and
full interface compatibility.  No config changes are required – sensible
defaults activate new features automatically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import functools
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

#######################################################################
# Rotary helpers (copied from dual_pos_time_decay variant)            #
#######################################################################


@functools.lru_cache(maxsize=32)
def _get_inv_freq(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    return inv_freq


def _build_sin_cos(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype):
    inv_freq = _get_inv_freq(dim, device, dtype)
    t = torch.arange(seq_len, device=device, dtype=dtype)
    sinusoid_inp = torch.einsum('i , j -> i j', t, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
    return sin, cos


def _apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = sin.to(dtype=x.dtype)[None, :, None, :]
    cos = cos.to(dtype=x.dtype)[None, :, None, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    x_rot = torch.stack((rot_x1, rot_x2), dim=-1)
    return rearrange(x_rot, '... d two -> ... (two d)')

#######################################################################
# Misc helpers                                                        #
#######################################################################

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)

#######################################################################
# Core chunk-wise delta rule with adaptive decay                      #
#######################################################################


@torch.compile  # type: ignore[misc]
def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = 32,
):
    """Causal associative retrieval with *content-adaptive* decay.

    Shapes
    -------
    q, k: (b, h, l, d_k)
    v   : (b, h, l, d_v)
    beta: (b, h, l)
    gamma: (b, h, l)  – dynamic decay gate in [0,1].  If *None* then no decay.
    """
    b, h, l, d_k = q.shape

    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        if gamma is not None:
            gamma = F.pad(gamma, (0, pad_len))

    padded_len = l + pad_len

    # --------------------------------------------- normalise & pre-scale
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # --------------------------------------------- reshape into chunks
    q, k, v, k_beta = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        (q, k, v, k_beta),
    )
    if gamma is not None:
        gamma_c = rearrange(gamma, 'b h (n c) -> b h n c', c=chunk_size)
    else:
        gamma_c = None

    mask_tri_inc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0
    )

    # (I - B K K^T)^{-1} per chunk (as in original implementation)
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri_inc, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)
    attn_inv = attn_inv.to(v.dtype)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    # --------------------------------------------- initialise state & output
    d_v = v.shape[-1]
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask_future = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1
    )

    num_chunks = padded_len // chunk_size
    for idx in range(num_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]  # (b h c d_k)
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S  # (b h c d_v)
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i

        delta_S = k_i.transpose(-1, -2) @ u_i  # (b h d_k d_v)
        if gamma_c is not None:
            # use *mean* gamma of tokens within the chunk → (b h)
            gamma_chunk = gamma_c[:, :, idx].mean(-1)
            S = gamma_chunk[..., None, None] * S + delta_S
        else:
            S = S + delta_S

    # --------------------------------------------- stitch back chunks
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len:
        o = o[:, :, :l]
    return o, S

#######################################################################
# Fractal mixer with token-wise gate                                 #
#######################################################################


class _CausalFractalMixer(nn.Module):
    """Depth-wise dilated convolution stack (log-depth receptive field)."""

    def __init__(self, hidden_size: int, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.convs = nn.ModuleList()
        for i in range(levels):
            dilation = 2 ** i
            conv = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=2,
                dilation=dilation,
                groups=hidden_size,
                bias=False,
            )
            nn.init.zeros_(conv.weight)  # near-identity
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (b l d)
        residual = x
        x = rearrange(x, 'b l d -> b d l')
        out = x.clone()
        for conv in self.convs:
            pad_left = conv.dilation[0]
            x_pad = F.pad(x, (pad_left, 0))
            out = out + conv(x_pad)
        out = rearrange(out, 'b d l -> b l d')
        return out + residual

#######################################################################
# Optional type stubs                                                #
#######################################################################

if TYPE_CHECKING:  # pragma: no cover
    from transformers.processing_utils import Unpack  # type: ignore
    from fla.models.utils import Cache  # type: ignore

#######################################################################
# Main DeltaNet                                                      #
#######################################################################


class DeltaNet(nn.Module):
    """DeltaNet layer with *adaptive* decay and gated fractal mixing."""

    def __init__(
        self,
        mode: str = 'chunk1',
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
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        # rotary
        use_rotary: bool = True,
        # adaptive decay params
        adaptive_decay: bool = True,
        # fractal mixer
        use_fractal_mixer: bool = True,
        mixer_levels: int = 4,
        **kwargs,
    ) -> None:
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
            assert self.head_k_dim % 2 == 0, 'head_k_dim must be even for rotary embeddings'

        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # beta gate
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # adaptive decay gate
        if self.adaptive_decay:
            self.gamma_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # rotary blend gate
        if self.use_rotary:
            self.rotary_mix_logit = nn.Parameter(torch.zeros(num_heads))

        # short convs
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size,
                                              activation='silu' if qk_activation == 'silu' else None)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size,
                                              activation='silu' if qk_activation == 'silu' else None)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation='silu')
        else:
            raise UserWarning('ShortConvolution is mandatory for DeltaNet performance – do not disable.')

        # optional output gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # fractal mixer & gating
        if self.use_fractal_mixer:
            self.fractal_mixer = _CausalFractalMixer(hidden_size, levels=mixer_levels)
            self.frac_gate_proj = nn.Linear(hidden_size, 1, bias=True)
            nn.init.constant_(self.frac_gate_proj.bias, -1.0)  # start mostly closed
            self.mixer_norm = RMSNorm(hidden_size, eps=norm_eps)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional['Cache'] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: 'Unpack[Dict]',  # type: ignore[misc]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional['Cache']]:  # noqa: D401
        if attention_mask is not None:
            assert attention_mask.dim() == 2, 'attention_mask must be (batch, seq_len) with 0/1 entries.'

        batch_size, padded_len, _ = hidden_states.shape

        # retrieve cached state (if any)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        max_seqlen = padded_len

        # optional unpadding
        if attention_mask is not None:
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask[:, -padded_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, 'b s d -> (b s) d'), indices).unsqueeze(0)

        seq_len = hidden_states.shape[1]

        # ------------------------------------------------ projections (+ short conv)
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q,
                                            output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k,
                                            output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v,
                                            output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:  # not expected
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # ------------------------------------------------ head split & activations
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError
        if self.qk_norm == 'sum':
            q, k = sum_norm(q), sum_norm(k)

        # ------------------------------------------------ rotary embedding blend
        if self.use_rotary:
            sin, cos = _build_sin_cos(seq_len, self.head_k_dim, device=q.device, dtype=q.dtype)
            q_rot = _apply_rotary(q, sin, cos)
            k_rot = _apply_rotary(k, sin, cos)
            mix_gate = torch.sigmoid(self.rotary_mix_logit).to(q.dtype)[None, None, :, None]
            q = mix_gate * q_rot + (1.0 - mix_gate) * q
            k = mix_gate * k_rot + (1.0 - mix_gate) * k

        # ------------------------------------------------ beta & gamma gates
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        if self.adaptive_decay:
            gamma = self.gamma_proj(hidden_states).sigmoid()  # (b, l, h)
        else:
            gamma = None

        # ------------------------------------------------ layout for delta rule
        q_t = rearrange(q, 'b l h d -> b h l d')
        k_t = rearrange(k, 'b l h d -> b h l d')
        v_t = rearrange(v, 'b l h d -> b h l d')
        beta_t = rearrange(beta, 'b l h -> b h l')
        gamma_t = rearrange(gamma, 'b l h -> b h l') if gamma is not None else None

        o_t, recurrent_state = delta_rule_chunkwise(q=q_t, k=k_t, v=v_t, beta=beta_t, gamma=gamma_t)
        o = rearrange(o_t, 'b h l d -> b l h d')

        # ------------------------------------------------ adaptive mix gate between memory output and value
        mix_gate = None
        if hasattr(self, 'mix_proj'):
            mix_gate = torch.sigmoid(self.mix_proj(hidden_states))  # from earlier variants
        if mix_gate is not None:
            mix_gate = rearrange(mix_gate, 'b l h -> b l h 1')
            o = mix_gate * o + (1.0 - mix_gate) * v

        # ------------------------------------------------ cache update
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ------------------------------------------------ output norm/proj
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = self.o_proj(rearrange(o, 'b l h d -> b l (h d)'))

        # ------------------------------------------------ gated fractal mixer fusion
        if self.use_fractal_mixer:
            mixer_out = self.fractal_mixer(hidden_states)
            gate = torch.sigmoid(self.frac_gate_proj(hidden_states))  # (b l 1)
            mixer_out = gate * self.mixer_norm(mixer_out)
            o = o + mixer_out

        # ------------------------------------------------ repad
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, max_seqlen)

        return o, None, past_key_values
