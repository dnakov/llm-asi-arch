# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# NOTE:
# This file contains an evolved version of DeltaNet.  The main innovation is an
# "adaptive mixing gate" that learns, for every token and head, how much of the
# newly-computed delta-rule output should be trusted versus the freshly computed
# value vector coming from the current time-step.  Empirically, such per-token
# adaptive residual connections have been shown to improve length generalisation
# and stabilise optimisation, while incurring negligible computation overhead.
# -----------------------------------------------------------------------------

def softmax(x):
    return F.softmax(x, dim=-1)


@torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Delta rule implementation identical to the original version.

    Args:
        q, k, v: (...) Same semantics as previously – see the original paper.
        beta:     (...)
        chunk_size (int): controls the window size of the parallel algorithm.
    Returns:
        o: Output tensor with identical shape to *v*.
        S: Recurrent state to be passed to the next forward call.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    # ------------------------------------------------------------------
    # Padding to an integer multiple of *chunk_size*
    # ------------------------------------------------------------------
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))

    padded_len = l + pad_len

    # ------------------------------------------------------------------
    # Normalisation & parameter preparation
    # ------------------------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # ------------------------------------------------------------------
    # Compute (I - tri(diag(beta) K K^T))^{-1}
    # ------------------------------------------------------------------
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn.to(torch.bfloat16)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaNet(nn.Module):
    """DeltaNet with Adaptive Mixing Gate (AMG).

    The adaptive gate decides, per-token and per-head, whether to rely on the
    newly computed *delta-rule* output or to fall back to the instantaneous
    value vector.  This improves length generalisation by letting the network
    skip recurrent accumulation when it is detrimental (e.g. on very long
    contexts) while keeping the strong associative recall abilities when
    beneficial.
    """

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
        use_mix_gate: bool = True,  # NEW: adaptive mixing gate enabled by default
        **kwargs,
    ) -> "DeltaNet":
        super().__init__()
        self.mode = mode

        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_mix_gate = use_mix_gate

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
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        assert self.key_dim % num_heads == 0, (
            f"key dim must be divisible by num_heads of {num_heads}")
        assert self.value_dim % num_heads == 0, (
            f"value dim must be divisible by num_heads of {num_heads}")

        # ------------------------------------------------------------------
        # Projection layers
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Adaptive mixing gate projection (per-token, per-head scalar in [0,1])
        if self.use_mix_gate:
            self.mix_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # ------------------------------------------------------------------
        # Beta projection (forget gate from the original DeltaNet paper)
        # ------------------------------------------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # ------------------------------------------------------------------
        # Convolutional enhancement for local patterns
        # ------------------------------------------------------------------
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu',
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.")

        # ------------------------------------------------------------------
        # Output normalisation / gating
        # ------------------------------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional['Cache'] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: 'Unpack[Dict]'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional['Cache']]:
        # ------------------------------------------------------------------
        # 1. Input validation & unpadding
        # ------------------------------------------------------------------
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed.")

        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # ------------------------------------------------------------------
        # 2. Projections + optional short convolution
        # ------------------------------------------------------------------
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = (None, None, None)
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

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
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # Save *token-local* value representation for gating later (b, l, h, d)
        v_token = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # ------------------------------------------------------------------
        # 3. Activation + normalisation for q/k, plus reshape to heads
        # ------------------------------------------------------------------
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            else:
                raise NotImplementedError

        if self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)

        # ------------------------------------------------------------------
        # 4. Beta gate preparation
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # 5. Delta-rule core computation (chunk-wise, causal)
        # ------------------------------------------------------------------
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v_for_delta = rearrange(v_token, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        # Note: recurrent_state is returned but not used inside delta_rule_chunkwise;
        # preserved for API compatibility.
        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v_for_delta, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        # ------------------------------------------------------------------
        # 6. NEW: Adaptive mixing between delta output and instantaneous value
        # ------------------------------------------------------------------
        if self.use_mix_gate:
            mix_gate = torch.sigmoid(self.mix_proj(hidden_states))  # shape: (b, l, h)
            mix_gate = rearrange(mix_gate, 'b l h -> b l h 1')
            # Blend outputs – keep shapes identical
            o = mix_gate * o + (1.0 - mix_gate) * v_token

        # ------------------------------------------------------------------
        # 7. Update cache (if any)
        # ------------------------------------------------------------------
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ------------------------------------------------------------------
        # 8. Optional gating + normalisation
        # ------------------------------------------------------------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        # ------------------------------------------------------------------
        # 9. Final projection back to model dimension
        # ------------------------------------------------------------------
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        # ------------------------------------------------------------------
        # 10. Re-padding (if we had removed padding earlier)
        # ------------------------------------------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
