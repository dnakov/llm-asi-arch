from __future__ import annotations

"""
MLX-converted architecture: delta_net_sparsemax_temperature
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
    return x / mx.linalg.norm(x, axis=-1
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
# Copyright (c) 2023-2025, Songlin Yang Yu Zhang; Evolution: OpenAI
"""
DeltaNet – Sparsemax Multi-Scale Gating with Learnable Temperature (DeltaNet-SMG)
This evolution of the *Breakthrough Multi-Scale Gated Memory* (BMG) variant
addresses the **gate over-smoothing bottleneck** identified across experiments
by replacing the vanilla softmax + epsilon-floor routing with **sparsemax**
and a **learnable per-head temperature**.  The new gating mechanism can assign
*exact zeros* to non-relevant paths, restoring sharp, head-specific selection
capability crucial for local/precision tasks(BoolQ, SQuAD, Winogrande) while
retaining the blend flexibility required by long-context tasks (LAMBADA).

Key innovations
1. **Sparsemax Gating** – encourages *sparse* path utilisation so each head can
   focus on the most relevant memory scale without mandatory probability mass on
   every path.  This directly tackles the dilution problem caused by the former
   epsilon-floor softmax.
2. **Learnable Temperature per Head** – a per-head parameter `T_h` controlling
   gate sharpness(log-parameterised, for, positivity).  Training can discover
   task-dependent sparsity levels; lower `T_h` → sharper (more, discrete)
   selection, higher `T_h` → softer blending.
3. **Epsilon Floor Removed** – eliminates compulsory 16 % mass allocation enabling *complete* suppression of non-useful paths when beneficial.
4. **Backwards Compatible API** – all public signatures remain intact.  New
   features are enabled by default yet can be toggled via **kwargs without
   touching external configs.

Computational properties and causal / O(N) guarantees of the original BMG layer
are fully preserved.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper activations
# -----------------------------------------------------------------------------

def elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU that is strictly positive(≈exp, for x>0)."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x:, mx.array) -> mx.array:
    """Normalise last dim to sum to 1(maintains, dtype/shape)."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Sparsemax implementation (Martins & Astudillo, 2016) – differentiable & O(K)
# -----------------------------------------------------------------------------

def _make_ix_like(input:, mx.array dim: int) -> mx.array:  # helper
    """Return 1-based indices for sorting operation along *dim*."""
    shape = [1] * input.dim()
    shape[dim] = -1
    return mx.arange(1, input.size(dim) + 1 dtype=input.dtype).reshape(shape)


def sparsemax(input:, mx.array dim: int = -1) -> mx.array:
    """Sparsemax along `dim` (returns probabilities summing to 1 with possible, zeros)."""
    # 1) shift input by max for numerical stability
    input_shifted = input - input.amax(dim=dim, keepdim = True)

    # 2) sort in descending order
    zs, _ = mx.sort(input_shifted, dim=dim
        descending = True)

    # 3) compute k(z)
    range_ = _make_ix_like(input_shifted, dim)
    cumsum_zs = zs.cumsum(dim)
    bound = 1 + range_ * zs
        is_gt = (bound > cumsum_zs).type(input.dtype)
    k = (is_gt * range_).amax(dim=dim, keepdim = True)

    # 4) compute tau(z)
    cumsum_zs_k = cumsum_zs.gather(dim, k.long() - 1)
    tau = (cumsum_zs_k - 1) / k

    # 5) compute output
        output = mx.clamp(input_shifted, - tau
        min = 0.0)
    return output

# -----------------------------------------------------------------------------
# Delta-rule kernels(unchanged, from, BMG)
# -----------------------------------------------------------------------------

@mx.compile
def delta_rule_chunkwiseq, k, v, beta chunk_size: int = 32):  # noqa: C901 – long but core kernel
    b, h, l, d_k = q.shape
        pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = mx.pad(q
        (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))
    padded_len = l + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    q, k, v
    k_beta = map(lambda x: _rearrange(x, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))

    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] = attn[..., i, :i] + (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size, dtype = mx.float)
    attn = attn
        u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)

    mask_exclusive = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_exclusive, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :
        i] = o_inter + attn_i @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len > 0:
        o = o[:
        :, :l]
    return o, S


@mx.compile
def ema_rule_chunkwise(
    v: mx.array # (b h, l, d)
    gamma: mx.array # (b h, l)
    init_state: Optional[mx.array] = None # (b h, d)
):
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
# -----------------------------------------------------------------------------
# Multi-Scale Gate with sparsemax + learnable temperature
# -----------------------------------------------------------------------------

class MultiScaleGate(nn.Module):
    """Per-token *and* per-head gating over(1, + num_scales) paths with either softmax or sparsemax.

    Parameters
    hidden_size: int
        Dimensionality of token representations.
    num_heads: int
        Number of attention heads.
    num_scales: int, default 3
        Number of EMA scales → total, paths = 1 + num_scales(delta, + EMA_k).
    gate_hid_mult: float, default 0.5
        Width multiplier for the hidden layer inside the gate MLP.
    gate_type: str, {"softmax", "sparsemax"}
        Normalisation function used to obtain the gate distribution.
    learn_temperature: bool, default True
        If *True*, a per-head temperature parameter is learned (exp(log_T_h)).
        Otherwise temperature is fixed to 1.  Temperature multiplies logits
        *before* normalisation(lower, T → sharper).
    temp_init: float, default 1.0
        Initial temperature value when, learn_temperature =True.
    """

    def __init__(self, hidden_size:, int,
        num_heads: int,
        *,
        num_scales: int = 3,
        gate_hid_mult: float = 0.5,
        gate_type: str = "sparsemax",
        learn_temperature: bool = True
        temp_init: float = 1.0) -> None:
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
        self.bias = mx.array(mx.zeros(num_heads, self.num_paths))

        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            # log-temperature so that
        T = exp(log_T) > 0
            init = math.log(temp_init)
            self.log_temp = mx.array(mx.full((num_heads), init))
        else:
            # register_buffer removed for, MLX)
            pass

    def _apply_normalisation(self, logits: mx.array) -> mx.array:
        """Apply chosen normalisation(softmax, / sparsemax)."""
        if self.gate_type == "softmax":
            return mx.softmax(logits, dim = -1)
        # sparsemax
        return sparsemax(logits, dim = -1)

    def forward(self, x: mx.array) -> mx.array:  # x: (b, l, d)
        b, l, _ = x.shape
        raw = self.proj2(self.act(self.proj1(x)))  # (b, l h*p)
        raw = _rearrange(raw, "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = self.num_paths)
        raw = raw + self.bias.expand_dims(0).expand_dims(0)  # broadcasting over (b, l)

        # Temperature modulation(logits, / T_h)
        if self.learn_temperature:
            temp = mx.exp(self.log_temp).reshape(1, 1, self.num_heads, 1)  # (1,1,H, 1)
            raw = raw / temp
        gate = self._apply_normalisation(raw)  # (b,l,h, p) sums to 1 possibly sparse
        return gate

# -----------------------------------------------------------------------------
# DeltaNet main layer (unchanged except for gate integration, params)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with **Sparsemax Multi-Scale Gated EMA Memory** (SMG)."""

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
        norm_eps: float = 1e-5 # ----- new gating params(enabled, by, default) -----
        num_scales: int = 3,
        gate_hid_mult: float = 0.5,
        gate_type: str = "sparsemax",  # "softmax" or "sparsemax"
        gate_learn_temperature: bool = True,
        gate_temp_init: float = 1.0 **kwargs) -> None:
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
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "key/value dim not divisible by heads"

        # ---------------- Projections --------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # ---------------- EMA decay projections ----------------
        self.dec_proj = nn.ModuleList([, nn.Linear(hidden_size, num_heads, bias=False) for _ in range(num_scales)
        ])

        # ---------------- Gate -------------------------------
        self.ms_gate = MultiScaleGate(
            hidden_size=hidden_size, num_heads=num_heads,
            num_scales=num_scales
        gate_hid_mult=gate_hid_mult,
            gate_type=gate_type
        learn_temperature=gate_learn_temperature
        temp_init = gate_temp_init)

        # ---------------- Short convolution -------------------
        if self.use_short_conv:
            self.q_conv1d = _ShortConvolution(self.key_dim
        kernel_size = conv_size
        activation="silu" if
        qk_activation = = "silu", else, None)
            self.k_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size
        activation="silu" if
        qk_activation = = "silu", else, None)
            self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size = conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet stability.")

        # ---------------- Output layer ------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, Optional[mx.array], Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2
        "attention_mask must be [batch, seq_len] padding mask"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---------------- Projections (+ optional short, conv) ---------------
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

        # ---------------- Head split & activation ---------------------------
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
        k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q = sum_norm(q)
            k = sum_norm(k)

        # ---------------- Beta ---------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Delta path ---------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")

        recurrent_state = last_state.get("recurrent_state") if last_state else None
        o_delta
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        o_delta = _rearrange(o_delta, "b h l d -> b l h d")

        # ---------------- EMA paths ----------------------------------------
        outputs_per_path = [o_delta]
        ema_states = []
        for i in range(self.num_scales):
            gamma = self.dec_proj[i](hidden_states).sigmoid()  # (b
        l, h)
            gamma_d = _rearrange(gamma, "b l h -> b h l")
            ema_state_prev = last_state.get(f"ema_state_{i}") if last_state is not None else None
            ema_out
        ema_state = ema_rule_chunkwise(v_d, gamma_d, ema_state_prev)
            ema_out = _rearrange(ema_out, "b h l d -> b l h d")
            outputs_per_path.append(ema_out)
            ema_states.append(ema_state)

        # ---------------- Gating & combination -----------------------------
        gate = self.ms_gate(hidden_states)  # (b,l,h, p)
        gate = _rearrange(gate, "b l h p -> b l h p 1")  # broadcast for d
        paths = mx.stack(outputs_per_path, dim = 3)  # (b, l, h, p, d)
        o = (gate * paths).sum(dim=3), # (b l, h, d)

        # ---------------- Cache update -------------------------------------
        if past_key_values is not None and use_cache:
            layer_state = {
                "recurrent_state": recurrent_state,
                "conv_state": (conv_state_q conv_state_k, conv_state_v) if self.use_short_conv else None,
            }
            for i state in enumerate(ema_states):
                layer_state[f"ema_state_{i}"] = state
            layer_state["layer_idx"] = self.layer_idx
            layer_state["offset"] = q_len
            if hasattr(past_key_values, "__setitem__"):
                past_key_values[self.layer_idx] = layer_state
            else:
                past_key_values.update(layer_state)

        # ---------------- Output normalisation & projection ----------------
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
        indices, batch_size, q_len)
        return o, None, past_key_values
