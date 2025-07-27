# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang; Evolution: OpenAI
"""
DeltaNet – Sparsemax Multi-Scale Gating with Learnable Temperature (DeltaNet-SMG)
================================================================================
This evolution of the *Breakthrough Multi-Scale Gated Memory* (BMG) variant
addresses the **gate over-smoothing bottleneck** identified across experiments
by replacing the vanilla softmax + epsilon-floor routing with **sparsemax**
and a **learnable per-head temperature**.  The new gating mechanism can assign
*exact zeros* to non-relevant paths, restoring sharp, head-specific selection
capability crucial for local/precision tasks (BoolQ, SQuAD, Winogrande) while
retaining the blend flexibility required by long-context tasks (LAMBADA).

Key innovations
---------------
1. **Sparsemax Gating** – encourages *sparse* path utilisation so each head can
   focus on the most relevant memory scale without mandatory probability mass on
   every path.  This directly tackles the dilution problem caused by the former
   epsilon-floor softmax.
2. **Learnable Temperature per Head** – a per-head parameter `T_h` controlling
   gate sharpness (log-parameterised for positivity).  Training can discover
   task-dependent sparsity levels; lower `T_h` → sharper (more discrete)
   selection, higher `T_h` → softer blending.
3. **Epsilon Floor Removed** – eliminates compulsory 16 % mass allocation,
   enabling *complete* suppression of non-useful paths when beneficial.
4. **Backwards Compatible API** – all public signatures remain intact.  New
   features are enabled by default yet can be toggled via **kwargs without
   touching external configs.

Computational properties and causal / O(N) guarantees of the original BMG layer
are fully preserved.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper activations
# -----------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU that is strictly positive (≈exp for x>0)."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise last dim to sum to 1 (maintains dtype/shape)."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Sparsemax implementation (Martins & Astudillo, 2016) – differentiable & O(K)
# -----------------------------------------------------------------------------

def _make_ix_like(input: torch.Tensor, dim: int) -> torch.Tensor:  # helper
    """Return 1-based indices for sorting operation along *dim*."""
    shape = [1] * input.dim()
    shape[dim] = -1
    return torch.arange(1, input.size(dim) + 1, device=input.device, dtype=input.dtype).view(shape)


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax along `dim` (returns probabilities summing to 1 with possible zeros)."""
    # 1) shift input by max for numerical stability
    input_shifted = input - input.amax(dim=dim, keepdim=True)

    # 2) sort in descending order
    zs, _ = torch.sort(input_shifted, dim=dim, descending=True)

    # 3) compute k(z)
    range_ = _make_ix_like(input_shifted, dim)
    cumsum_zs = zs.cumsum(dim)
    bound = 1 + range_ * zs
    is_gt = (bound > cumsum_zs).type(input.dtype)
    k = (is_gt * range_).amax(dim=dim, keepdim=True)

    # 4) compute tau(z)
    cumsum_zs_k = cumsum_zs.gather(dim, k.long() - 1)
    tau = (cumsum_zs_k - 1) / k

    # 5) compute output
    output = torch.clamp(input_shifted - tau, min=0.0)
    return output

# -----------------------------------------------------------------------------
# Delta-rule kernels (unchanged from BMG)
# -----------------------------------------------------------------------------

@torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):  # noqa: C901 – long but core kernel
    b, h, l, d_k = q.shape
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    padded_len = l + pad_len

    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    q, k, v, k_beta = map(lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn.to(torch.bfloat16)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    mask_exclusive = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_exclusive, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn_i @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S


@torch.compile
def ema_rule_chunkwise(
    v: torch.Tensor,  # (b h l d)
    gamma: torch.Tensor,  # (b h l)
    init_state: Optional[torch.Tensor] = None,  # (b h d)
):
    b, h, l, d = v.shape
    ema_out = torch.empty_like(v)
    state = torch.zeros((b, h, d), dtype=v.dtype, device=v.device) if init_state is None else init_state
    for t in range(l):
        g_t = gamma[:, :, t].unsqueeze(-1)
        state = g_t * state + (1.0 - g_t) * v[:, :, t]
        ema_out[:, :, t] = state
    return ema_out, state

# -----------------------------------------------------------------------------
# Multi-Scale Gate with sparsemax + learnable temperature
# -----------------------------------------------------------------------------

class MultiScaleGate(nn.Module):
    """Per-token *and* per-head gating over (1 + num_scales) paths with either softmax or sparsemax.

    Parameters
    ----------
    hidden_size: int
        Dimensionality of token representations.
    num_heads: int
        Number of attention heads.
    num_scales: int, default 3
        Number of EMA scales → total paths = 1 + num_scales (delta + EMA_k).
    gate_hid_mult: float, default 0.5
        Width multiplier for the hidden layer inside the gate MLP.
    gate_type: str, {"softmax", "sparsemax"}
        Normalisation function used to obtain the gate distribution.
    learn_temperature: bool, default True
        If *True*, a per-head temperature parameter is learned (exp(log_T_h)).
        Otherwise, temperature is fixed to 1.  Temperature multiplies logits
        *before* normalisation (lower T → sharper).
    temp_init: float, default 1.0
        Initial temperature value when learn_temperature=True.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        num_scales: int = 3,
        gate_hid_mult: float = 0.5,
        gate_type: str = "sparsemax",
        learn_temperature: bool = True,
        temp_init: float = 1.0,
    ) -> None:
        super().__init__()

        assert gate_type in {"softmax", "sparsemax"}, "gate_type must be softmax|sparsemax"
        self.gate_type = gate_type
        self.num_paths = 1 + num_scales  # delta + EMA scales
        self.num_heads = num_heads

        gate_hidden = max(8, int(hidden_size * gate_hid_mult))
        self.proj1 = nn.Linear(hidden_size, gate_hidden)
        self.act = nn.SiLU()
        self.proj2 = nn.Linear(gate_hidden, num_heads * self.num_paths)
        # Per-head, per-path bias initialised to zero
        self.bias = nn.Parameter(torch.zeros(num_heads, self.num_paths))

        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            # log-temperature so that T = exp(log_T) > 0
            init = math.log(temp_init)
            self.log_temp = nn.Parameter(torch.full((num_heads,), init))
        else:
            self.register_buffer("log_temp", torch.zeros(num_heads))

    def _apply_normalisation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply chosen normalisation (softmax / sparsemax)."""
        if self.gate_type == "softmax":
            return torch.softmax(logits, dim=-1)
        # sparsemax
        return sparsemax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, l, d)
        b, l, _ = x.shape
        raw = self.proj2(self.act(self.proj1(x)))  # (b, l, h*p)
        raw = rearrange(raw, "b l (h p) -> b l h p", h=self.num_heads, p=self.num_paths)
        raw = raw + self.bias.unsqueeze(0).unsqueeze(0)  # broadcasting over (b,l)

        # Temperature modulation (logits / T_h)
        if self.learn_temperature:
            temp = torch.exp(self.log_temp).view(1, 1, self.num_heads, 1)  # (1,1,H,1)
            raw = raw / temp

        gate = self._apply_normalisation(raw)  # (b,l,h,p) sums to 1, possibly sparse
        return gate

# -----------------------------------------------------------------------------
# DeltaNet main layer (unchanged except for gate integration params)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with **Sparsemax Multi-Scale Gated EMA Memory** (SMG)."""

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
        # ----- new gating params (enabled by default) -----
        num_scales: int = 3,
        gate_hid_mult: float = 0.5,
        gate_type: str = "sparsemax",  # "softmax" or "sparsemax"
        gate_learn_temperature: bool = True,
        gate_temp_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---------------- Parameter bookkeeping ----------------
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in {"silu", "relu", "elu", "identity"}, "Unsupported qk_activation"
        assert self.qk_norm in {"l2", "sum"}, "Unsupported qk_norm"
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
        self.layer_idx = layer_idx or 0
        self.num_scales = num_scales

        # ---------------- Dimensions ---------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "key/value dim not divisible by heads"

        # ---------------- Projections --------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- EMA decay projections ----------------
        self.dec_proj = nn.ModuleList([
            nn.Linear(hidden_size, num_heads, bias=False) for _ in range(num_scales)
        ])

        # ---------------- Gate -------------------------------
        self.ms_gate = MultiScaleGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_scales=num_scales,
            gate_hid_mult=gate_hid_mult,
            gate_type=gate_type,
            learn_temperature=gate_learn_temperature,
            temp_init=gate_temp_init,
        )

        # ---------------- Short convolution -------------------
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation="silu" if qk_activation == "silu" else None)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation="silu" if qk_activation == "silu" else None)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---------------- Output layer ------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be [batch, seq_len] padding mask"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---------------- Projections (+ optional short conv) ---------------
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # ---------------- Head split & activation ---------------------------
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

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

        # ---------------- Beta ---------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Delta path ---------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")

        recurrent_state = last_state.get("recurrent_state") if last_state else None
        o_delta, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        o_delta = rearrange(o_delta, "b h l d -> b l h d")

        # ---------------- EMA paths ----------------------------------------
        outputs_per_path = [o_delta]
        ema_states = []
        for i in range(self.num_scales):
            gamma = self.dec_proj[i](hidden_states).sigmoid()  # (b, l, h)
            gamma_d = rearrange(gamma, "b l h -> b h l")
            ema_state_prev = last_state.get(f"ema_state_{i}") if last_state is not None else None
            ema_out, ema_state = ema_rule_chunkwise(v_d, gamma_d, ema_state_prev)
            ema_out = rearrange(ema_out, "b h l d -> b l h d")
            outputs_per_path.append(ema_out)
            ema_states.append(ema_state)

        # ---------------- Gating & combination -----------------------------
        gate = self.ms_gate(hidden_states)  # (b,l,h,p)
        gate = rearrange(gate, "b l h p -> b l h p 1")  # broadcast for d
        paths = torch.stack(outputs_per_path, dim=3)  # (b,l,h,p,d)
        o = (gate * paths).sum(dim=3)  # (b,l,h,d)

        # ---------------- Cache update -------------------------------------
        if past_key_values is not None and use_cache:
            layer_state = {
                "recurrent_state": recurrent_state,
                "conv_state": (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            }
            for i, state in enumerate(ema_states):
                layer_state[f"ema_state_{i}"] = state
            layer_state["layer_idx"] = self.layer_idx
            layer_state["offset"] = q_len
            if hasattr(past_key_values, "__setitem__"):
                past_key_values[self.layer_idx] = layer_state
            else:
                past_key_values.update(layer_state)

        # ---------------- Output normalisation & projection ----------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values
