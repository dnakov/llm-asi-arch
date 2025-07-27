# -*- coding: utf-8 -*-
"""
DeltaNet – Dual-Scale Local-Global Gated Memory (DLGM)
=====================================================
This evolution unifies the *state-space* delta-rule global memory with **two
causal depth-wise convolutional value paths of different receptive fields**
(short-range *local* & mid-range *context*) and a **token-, head- and
position-dependent softmax router** that decides – *per token* – how much of
each memory stream should contribute to the final representation.

Motivation & Design Highlights
------------------------------
1. **Restore Local Fidelity**  – Prior variants (e.g. HMGM) blurred
   high-frequency features by relying on a single large FIR kernel.  We add a
   *small* (k=7) depth-wise convolution branch that captures fine-grained local
   patterns without sacrificing efficiency (kernel size is constant).
2. **Maintain Mid/Global Context** – Keep the proven delta-rule associative
   memory *and* a mid-range convolution branch (k=31) so the model possesses
   three complementary context ranges.
3. **Dynamic Token-wise Routing** – A lightweight MLP (2×hidden) produces
   per-token, per-head logits over the *four* streams – {local, mid, delta,
   identity}.  Softmax selection preserves scale while allowing specialisation.
4. **Identity-Favoured Initialisation** – Gate bias is initialised such that
   the *identity* (direct value) path starts dominant (≈70%) to avoid early
   oversmoothing – a typical failure mode in previous experiments.
5. **Sub-Quadratic Complexity** – All added operations are causal
   depth-wise 1-D convolutions (O(N·k)) and chunk-wise delta-rule (O(N)).
6. **Batch & Sequence Agnostic** – Every tensor reshape uses *einops*; no
   hard-coded batch/sequence dimensions.

The public interface, class name (`DeltaNet`), forward signature and parameter
schema are **fully preserved**.  New features are on by default and incur
minimal parameter overhead.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Utility activations
# -----------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:  # ELU+1
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Delta-rule path (unchanged, O(N))
# -----------------------------------------------------------------------------
@torch.compile
def delta_rule_chunkwise(
    q: torch.Tensor,  # [B H L D_k]
    k: torch.Tensor,  # [B H L D_k]
    v: torch.Tensor,  # [B H L D_v]
    beta: torch.Tensor,  # [B H L]
    *,
    chunk_size: int = 32,
):
    """Original DeltaNet associative memory evaluated chunk-wise (O(N))."""
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)  # pad sequence dimension (second last)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalise q/k
    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into chunks: [... n c d]
    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri_full = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    mask_tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri_full, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    u = attn @ v  # [b h n c d_v]
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    n_chunks = L_pad // chunk_size
    for idx in range(n_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask_tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Depth-wise causal convolution branches
# -----------------------------------------------------------------------------
class _DepthwiseCausalConv1d(nn.Module):
    """Per-head depth-wise 1-D convolution with *causal* left padding."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        weight = torch.randn(num_heads * head_dim, 1, kernel_size) / math.sqrt(kernel_size)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x : [B L H D]
        b, L, h, d = x.shape
        x_ch = rearrange(x, "b l h d -> b (h d) l")
        w = self.weight  # [(h*d) 1 k]
        x_pad = F.pad(x_ch, (self.kernel_size - 1, 0))  # left pad for causality
        y = F.conv1d(x_pad, w, groups=h * d)
        y = rearrange(y, "b (h d) l -> b l h d", h=h)
        return y

# -----------------------------------------------------------------------------
# Optional type hints for external cache utils
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

# -----------------------------------------------------------------------------
#                                DeltaNet
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with Dual-Scale Local-Global Gated Memory (DLGM)."""

    def __init__(
        self,
        mode: str = "dlgm",
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
        # new params --------------------------------------------------
        local_kernel_size: int = 7,
        mid_kernel_size: int = 31,
        router_hidden_mult: int = 2,
        router_init_identity_bias: float = 1.5,  # ≈70% identity path at init
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        # -------- basic setup --------
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
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0  # default to 0 if None for safety
        self.use_short_conv = use_short_conv

        # -------- dimensions ---------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # -------- projections --------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # optional local *short* convs (for q/k/v) --------------------
        if use_short_conv:
            activation = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=activation, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=activation, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for stable performance.")

        # -------- depth-wise conv branches (value space) -------------
        self.local_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, kernel_size=local_kernel_size)
        self.mid_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, kernel_size=mid_kernel_size)

        # -------- router MLP over 4 paths -----------------------------
        # order: local, mid, delta, identity
        router_out_dim = num_heads * 4
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * router_hidden_mult, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * router_hidden_mult, router_out_dim, bias=True),
        )
        # bias init – favour identity path (index 3)
        with torch.no_grad():
            self.router_mlp[-1].bias.data.zero_()
            # reshape to [heads, 4]
            bias_view = self.router_mlp[-1].bias.data.view(num_heads, 4)
            bias_view[:, 3] = router_init_identity_bias  # positive bias to identity

        # -------- output normalisation/projection --------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B L D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape

        # --------------- unpadding for speed -------------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # --------------- projections (+ optional short conv) ---------
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
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

        # --------------- head reshape -------------------------------
        q, k = map(lambda x: rearrange(x, "b l (h d) -> b l h d", h=self.num_heads), (q, k))
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # --------------- activations / norms ------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # --------------- beta gate ----------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------- delta-rule path ----------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d, chunk_size=32)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # --------------- convolutional value paths ------------------
        v_direct = v  # identity path
        local_out = self.local_conv(v_direct)  # fine-grained local
        mid_out = self.mid_conv(v_direct)      # mid-range context

        # --------------- router -------------------------------------
        router_logits = self.router_mlp(hidden_states)  # [B L H*4]
        router_logits = rearrange(router_logits, "b l (h p) -> b l h p", h=self.num_heads, p=4)
        router_weights = F.softmax(router_logits, dim=-1)  # [B L H 4]

        # combine in order: local, mid, delta, identity
        o = (
            router_weights[..., 0:1] * local_out +
            router_weights[..., 1:2] * mid_out +
            router_weights[..., 2:3] * delta_out +
            router_weights[..., 3:4] * v_direct
        )

        # --------------- cache update --------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # --------------- output norm / proj --------------------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # --------------- re-pad if needed ----------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B_orig, L_in)

        return o, None, past_key_values
