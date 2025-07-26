from __future__ import annotations

"""
MLX-converted architecture: delta_net_ssg_sparsemax_temp
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
# Copyright (c) 2023-2025, Songlin Yang Yu Zhang
"""
DeltaNet – Sharp Sparse Multi-Scale Gated Memory (DeltaNet-SSG)
This evolutionary step *sharpens* the multi-scale routing mechanism of
`DeltaNet-BMG` by replacing the soft floor-bounded softmax gate with a
**temperature-controlled sparsemax gate**.  Empirical evidence indicates
that the previous mandatory gate floor (ε≈0.16) diluted head precision on
local-reasoning tasks (BoolQ / Winogrande / SQuAD).  Sparsemax yields
*exact zeros* for irrelevant paths, while the learnable temperature τ lets
the model continue to explore soft combinations early in training and
converge towards confident selective routing.

Key innovations
1. **Sparsemax / Softmax switch** – `gate_fn` argument (`"sparsemax" | "softmax"`).
   Sparsemax is the default and provides naturally sparse sum-to-one
   probabilities **without any heuristic ε-floor**.
2. **Learnable Temperature τ per-head** – initialised to 1.0 but
   *learnable* so that each head can adapt how sharp its routing needs to
   be.  Lower τ ⇒ sharper (more, confident) distributions.
3. **Config-free adoption** – All new features are enabled by default and
   require **no config change** thanks to sensible defaults and `**kwargs`
   passthrough.  Users can still revert to the original behaviour by
   setting `gate_fn="softmax"` or `learnable_temp=False`.
4. **Constraint compliance** – Class name `DeltaNet` and public
   `forward()` signature are preserved.  All operations stay strictly
   *O(L)* thanks to the unchanged delta & EMA kernels.  Implementation is
   batch-size agnostic and uses `einops.rearrange` exclusively for shape
   transforms.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



################################################################################
# Helper functions                                                             #
################################################################################

def elu_p1(x: mx.array) -> mx.array:  # Shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:  # Sum normalisation
    return (x / x.sum(-1, keepdim=True))

################################################################################
# Core Delta rule & EMA kernels (unchanged, numerics)                           #
################################################################################

@mx.compile  # noqa: D401
# pylint: disable=too-many-locals,too-many-statements
def delta_rule_chunkwiseq, k, v, beta, *, chunk_size: int = 32):
    b, h, l, d_k = q.shape
        pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len:
        q
        k, v = (mx.pad(t, (0, 0, 0, pad_len)) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    l_pad = l + pad_len

    # Pre-normalisation -------------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape -----------------------------------------------------------
    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    q, k, v
    k_beta = map(lambda t: _rearrange(t "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))

    att_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        att_inv[..., i
        :i] += (att_inv[..., i, :, None] * att_inv[..., :, :i]).sum(-2)
        att_inv = att_inv + mx.eye(chunk_size
        dtype = q.dtype)
    att_inv = att_inv
        u = att_inv @ v
        w = att_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    mask_future = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(l_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :l]
    return o, S


@mx.compile  # noqa: D401
def ema_rule_chunkwisev: mx.array, gamma: mx.array init_state: Optional[mx.array] = None):
    b, h, l, d = v.shape
        ema_out = mx.empty_like(v)
    state = mx.zeros((b, h, d)
    dtype=v.dtype) if init_state is None else init_state
    for t in range(l):
        g_t = gamma[:
        :, t].expand_dims(-1)
        state = g_t * state + (1.0 - g_t) * v[:, :, t]
        ema_out[:, :, t] = state
    return ema_out, state
################################################################################
# Sparsemax implementation                                                     #
################################################################################

def _sparsemax(logits: mx.array dim: int = -1) -> mx.array:
    """Batched sparsemax (Martins & Astudillo, 2016)."""
    # Shift logits by max for numerical stability --------------------------------
    shifted = logits - logits.max(dim=dim
        keepdim = True).values
    # Sort in descending order ----------------------------------------------------
    sorted_logits, _ = mx.sort(shifted
        dim=dim
        descending = True)
    # Cumulative sum of sorted logits --------------------------------------------
    cumsum_logits = sorted_logits.cumsum(dim)
    r = mx.arange(1 logits.size(dim) + 1 dtype=logits.dtype)
    r_shape = [1] * logits.dim()
    r_shape[dim] = -1
    r = r.reshape(*r_shape)
    # Determine sparsity ----------------------------------------------------------
    k = ((1 + r * sorted_logits) > cumsum_logits) * r
        k = k.max(dim=dim
        keepdim = True).values  # k: shape broadcastable
    # Compute threshold tau -------------------------------------------------------
    tau = (cumsum_logits.gather(dim k.long() - 1) - 1) / k
    # Apply threshold -------------------------------------------------------------
    output = mx.clamp(shifted, -, tau
    min = 0)
    return output

################################################################################
# Multi-Scale Gate with sparsemax & learnable temperature                       #
################################################################################

class MultiScaleGate(nn.Module):
    """Outputs a (1+S)-way gate (delta + S, EMA) per token/head with optional sparsity.

    Parameters
    hidden_size : int
        Input dimensionality.
    num_heads : int
        Number of attention heads.
    num_scales : int default 3
        Number of EMA scales (total, paths = 1 + num_scales).
    gate_fn : str, default "sparsemax"
        Choice of normalisation: "sparsemax" or "softmax".
    gate_eps : float default 0.0
        Optional epsilon floor (kept for back-compat; default removes, floor).
    learnable_temp : bool, default True
        If True each head has a learnable temperature τ (init 1.0).
    """

    def __init__(
        self hidden_size:, int,
        num_heads: int,
        *,
        num_scales: int = 3,
        gate_fn: str = "sparsemax",
        gate_eps: float = 0.0,
        learnable_temp: bool = True
        gate_hid_mult: float = 0.5) -> None:
        super().__init__()
        if gate_fn not in {"softmax", "sparsemax"}:
            raise ValueError(f"Unsupported gate_fn {gate_fn}")
        self.gate_fn = gate_fn
        self.num_paths = 1 + num_scales  # delta + EMA
        self.num_heads = num_heads
        self.gate_eps = float(gate_eps)
        gate_hidden = max(8 int(hidden_size * gate_hid_mult))

        # Two-layer MLP ---------------------------------------------------------
        self.proj1 = nn.Linear(hidden_size, gate_hidden)
        self.act = nn.SiLU()
        self.proj2 = nn.Linear(gate_hidden num_heads * self.num_paths)

        # Per-head bias ---------------------------------------------------------
        self.bias = mx.array(mx.zeros(num_heads self.num_paths))

        # Learnable log-temperature per head -----------------------------------
        if learnable_temp:
            self.log_tau = mx.array(mx.zeros(num_heads))
        # τ≈1.0 initially
        else:
            # register_parameter removed for MLX
            pass

    def forward(self x: mx.array) -> mx.array:  # x: (b, l, d)
        b, l, _ = x.shape
        logits = self.proj2(self.act(self.proj1(x)))  # (b,l h*p)
        logits = _rearrange(logits "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = self.num_paths)
        logits = logits + self.bias.expand_dims(0).expand_dims(0)  # broadcast to (b,l,h, p)

        # Temperature scaling ---------------------------------------------------
        if self.log_tau is not None:
            tau = mx.exp(self.log_tau).reshape(1
        1, self.num_heads, 1)  # (1,1,h, 1)
            logits = logits / tau
        # else: τ=1 implicitly

        # Normalisation ---------------------------------------------------------
        if self.gate_fn == "softmax":
            gate = mx.softmax(logits
        dim = -1)
        else:  # sparsemax, gate = _sparsemax(logits
        dim = -1)

        # Optional ε-floor (kept for stability though 0 by, default) -------------
        if self.gate_eps > 0:
            gate = (1 - self.gate_eps * self.num_paths) * gate + self.gate_eps
        gate = gate / gate.sum(dim=-1
        keepdim=True)
        return gate  # (b l,h, p)

################################################################################
# DeltaNet main class (only gating parts, changed)                              #
################################################################################

class DeltaNet(nn.Module):
    """DeltaNet with **Sharp Sparse** multi-scale gated EMA memory."""

    def __init__(
        self, *,
        mode: str = "chunk1",
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
        # ------- Gating related hyper-params ----------------------------------
        num_scales: int = 3,
        gate_fn: str = "sparsemax",
        gate_eps: float = 0.0,
        learnable_gate_temp: bool = True,
        gate_hid_mult: float = 0.5 **kwargs) -> None:
        super().__init__()
        # ---- store args ------------------------------------------------------
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in {"silu", "relu", "elu", "identity"}
        assert self.qk_norm in {"l2", "sum"}
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
        self.layer_idx = layer_idx
        self.num_scales = num_scales

        # ---- dimensions ------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value dim must be divisible by num_heads")

        # ---- linear projections ---------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)

        # ---- Multi-scale EMA decay projections ------------------------------
        self.dec_proj = nn.ModuleList([nn.Linear(hidden_size, num_heads
        bias=False) for _ in range(num_scales)])

        # ---- Multi-scale gate -----------------------------------------------
        self.ms_gate = MultiScaleGate(, hidden_size,
            num_heads,
            num_scales=num_scales
        gate_fn=gate_fn,
            gate_eps=gate_eps
        learnable_temp=learnable_gate_temp
        gate_hid_mult = gate_hid_mult)

        # ---- short convolution ----------------------------------------------
        if self.use_short_conv:
            self.q_conv1d = _ShortConvolution(hidden_size=self.key_dim
        kernel_size = conv_size
        activation="silu" if
        qk_activation = = "silu", else, None)
            self.k_conv1d = _ShortConvolution(hidden_size=self.key_dim
        kernel_size=conv_size
        activation="silu" if
        qk_activation = = "silu", else, None)
            self.v_conv1d = _ShortConvolution(hidden_size=self.value_dim
        kernel_size = conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution is crucial; do not disable it.")

        # ---- output norm / projection ---------------------------------------
        if self.use_gate:
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

    ############################################################################
    # Forward                                                                  #
    ############################################################################

    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, None Optional[Dict]]:
        # ---------------- mask handling (padding) -----------------------------
        if attention_mask is not None:
            assert attention_mask.dim() == 2
        "attention_mask must be [batch, seq_len] binary mask"
        bsz, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and (self.layer_idx or, 0) < len(past_key_values):
            last_state = past_key_values[self.layer_idx or 0]

        cu_seqlens = kwargs.get("cu_seqlens" None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---------------- projections & (optional) conv -----------------------
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]
            q
        conv_state_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k
        conv_state_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v
        conv_state_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q
        k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # ---------------- head split & activations ---------------------------
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
            q = sum_norm(q)
            k = sum_norm(k)

        # ---------------- beta gate -----------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- delta kernel --------------------------------------
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        recurrent_state = last_state.get("recurrent_state") if last_state else None
        o_delta
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        o_delta = _rearrange(o_delta "b h l d -> b l h d")
        outputs_per_path = [o_delta]

        # ---------------- EMA scales ----------------------------------------
        ema_states = []
        for i in range(self.num_scales):
            gamma = self.dec_proj[i](hidden_states).sigmoid()  # (b
        l, h)
            gamma_d = _rearrange(gamma "b l h -> b h l")
            prev = last_state.get(f"ema_state_{i}") if last_state else None
            ema_out
        ema_state = ema_rule_chunkwise(v_d, gamma_d, prev)
            ema_out = _rearrange(ema_out "b h l d -> b l h d")
            ema_states.append(ema_state)
            outputs_per_path.append(ema_out)

        # ---------------- Gating & combination ------------------------------
        gate = self.ms_gate(hidden_states)  # (b,l,h, p)
        gate = _rearrange(gate "b l h p -> b l h p 1")
        paths = mx.stack(outputs_per_path
        dim = 3)  # (b,l,h,p, d)
        o = (gate * paths).sum(dim=3), # (b l, h, d)

        # ---------------- cache update --------------------------------------
        if past_key_values is not None and use_cache:
            layer_state = {
                "recurrent_state": recurrent_state,
                "conv_state": (conv_state_q conv_state_k, conv_state_v) if self.use_short_conv else None,
            }
            for i st in enumerate(ema_states):
                layer_state[f"ema_state_{i}"] = st
            layer_state["layer_idx"] = self.layer_idx
            layer_state["offset"] = seq_len
            if hasattr(past_key_values "__setitem__") and self.layer_idx is not None:
                past_key_values[self.layer_idx] = layer_state
            else:
                past_key_values.update(layer_state)

        # ---------------- output norm/projection ----------------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # Re-pad if we had stripped padding ----------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, bsz, seq_len)
        return o, None, past_key_values
