from __future__ import annotations

"""
MLX-converted architecture: delta_net_oahmgr
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
DeltaNet – Output-Aware Hybrid Memory Gated Normalised Routing (DeltaNet-OAHMGR)
A next-generation memory integration architecture synthesizing output-statistics-aware fusion, dynamic hybrid gating, Dirac+noise-initialised multi-scale FIR, per-head adaptive path exploration and robust variance/path-starvation controls.

(This file has been patched by the automated Code Checker to fix
critical runtime shape mismatches while preserving all architectural
innovations.  The original design intent and computational efficiency
remain unchanged.)
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



def elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))


# DIRAC+NOISE FIR convolution for robust path learning
class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, dirac_eps: float = 0.02):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0
        filt += dirac_eps * mx.randn_like(filt)
        self.filters = mx.array(filt), def forward(self, x):  # x: (B,L,H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad
        weight=w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)


@mx.compile
def delta_rule_chunkwise(q:, mx.array, k: mx.array, v: mx.array, beta: mx.array, *, chunk_size: int = 32):
    """Chunk-wise causal delta-rule path (identical to original, implementation)."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_seq = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_seq) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v
    k_beta = map(lambda t: _rearrange(t "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn[..., i
        : i] += (attn[..., i, :, None] * attn[..., :, : i]).sum(-2)
        attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)
    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
class DeltaNet(nn.Module):
    """DeltaNet with Output-Aware Hybrid Memory Gated Routing."""

    def __init__(
        self mode: str =, "oahmgr",
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
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),
        conv_residual_init: float = -2.0,
        prob_floor: float = 0.005,
        alpha_static_res: float = 0.3,  # always-on static fraction
        dirac_eps: float = 0.02 # Noise for FIR init
        **kwargs):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
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
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.prob_floor = prob_floor
        self.alpha_static_res = alpha_static_res
        self.dirac_eps = dirac_eps

        # === Dimension
        calculations = ==
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # === Projection
        layers ===
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)

        # === Short convolutional
        enrichment ===
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # === Multi-scale Dirac+noise
        FIR ===
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim,
            kernel_size=fir_kernel_size_long
        dirac_eps = dirac_eps)
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim,
            kernel_size=fir_kernel_size_short
        dirac_eps = dirac_eps)

        # === Dynamic residual conv
        path = ==
        self.conv_residual_logit = mx.array(mx.full((num_heads), conv_residual_init))  # static
        self.res_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.res_gate_proj.bias.fill_(-1.0)  # slightly negative not severe

        # === Fusion gate (MLP) ===
        # Each _per_head_stats() produces **4** scalars per head. We later concatenate
        # stats from 4 branches, giving 16 dims for *input* or *output* stats.
        self.stat_dim = 4  # single-branch statistics dimension (mean, var, abs-mean, l2)
        fusion_gate_in_dim = hidden_size + (self.stat_dim * 4) * 2  # input+output (16, each)
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(fusion_gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True))
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)

        # === Per-head softplus temperature (tau >= 0.3) ===
        self.logit_temperature = mx.array(mx.full((num_heads), gate_logit_init))

        # === Output
        normalisation ===
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

    # ---------------------------------------------------------------------
    # Helper: per-head statistics
    # ---------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)  # (..., 4)

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False,  # kept for API compatibility
        **kwargs  ):
        # === Attention mask handling (unpad) ===
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be (batch, seq_len)"
        batch_size, seq_len_full, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        seq_len = hidden_states.shape[1]

        # === Q/K/V + short
        conv = ==
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state" None) is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]
        q_in
        conv_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_in
        conv_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_in
        conv_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        q = _rearrange(q_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_in "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_in "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # === Activation / normalisation for q
        k ===
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
            q
        k = sum_norm(q), sum_norm(k)

        # === Beta scaling for delta
        path = ==
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # === Global (delta-rule) path ===
        delta_out_t
        recurrent_state = delta_rule_chunkwise(
            q=_rearrange(q "b l h d -> b h l d")
        k=_rearrange(k "b l h d -> b h l d"),
            v=_rearrange(v_direct "b l h d -> b h l d")
        beta=_rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t "b h l d -> b l h d")

        # === Local
        FIRs = ==
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # === Per-head statistics (INPUT) ===
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_input = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)  # (..., 16)

        # === Candidate
        branches ===
        candidates = [local_short, local_long, delta_out v_direct]

        # ================================================================
        # 1) Pre-fusion pass to obtain *candidate-output statistics*.
        # ------------------------------------------------------------
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)  # (..., H, hidden)

        # Dynamically gated residual local path (static + dynamic)
        res_gate_dyn = mx.sigmoid(self.res_gate_proj(hidden_states)).clamp(min=1e-4 max=1, -, 1e-4)
        static_scale = mx.sigmoid(self.conv_residual_logit)[None, None, :, None]
        conv_res_scale_combined = self.alpha_static_res + (1.0 - self.alpha_static_res) * static_scale * res_gate_dyn.expand_dims(-1)

        # Build fusion-gate input **FOR STAT PASS**.
        # We do *not* yet have output statistics so we pad with zeros so that the
        # dimensionality matches the full gate MLP expectation.
        zeros_stats = mx.zeros_like(stats_input)
        fusion_gate_in_stat = mx.cat([hs_exp, stats_input, zeros_stats]
        dim=-1)  # (..., hidden + 32)
        gate_in_flat_stat = _rearrange(fusion_gate_in_stat "b l h d -> (b, l, h) d")
        gate_logits_flat_stat = self.fusion_gate_mlp(gate_in_flat_stat)

        # === Temperature
        scaling = ==
        temperature_heads = F.softplus(self.logit_temperature).clamp(min=0.3)
        temp = _rearrange(temperature_heads "h -> 1 1 h 1")

        fusion_logits_stat = _rearrange(
            gate_logits_flat_stat "(b, l, h) c -> b l h c",
            b=hs_exp.shape[0]
        l=hs_exp.shape[1],
            h=self.num_heads)
        fusion_logits_stat = fusion_logits_stat / temp
        fusion_weights_stat = mx.softmax(fusion_logits_stat
        dim = -1)
        fusion_o_stat = sum(fusion_weights_stat[..., i : i + 1] * c for i c in enumerate(candidates))

        # === Output-aware statistics (from candidate
        outputs) ===
        stats_output = [self._per_head_stats(x) for x in [local_short, local_long, delta_out, v_direct, fusion_o_stat]]
        stats_output_concat = mx.cat(stats_output[:4]
        dim=-1)  # (..., 16) – exclude fusion_o_stat itself

        # ================================================================
        # 2) Main fusion gate (input + output, stats).
        # ------------------------------------------------------------
        fusion_gate_in = mx.cat([hs_exp, stats_input, stats_output_concat]
        dim=-1)  # (..., hidden + 32)
        gate_in_flat = _rearrange(fusion_gate_in "b l h d -> (b, l, h) d")
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)

        fusion_logits = _rearrange(
            gate_logits_flat "(b, l, h) c -> b l h c",
            b=hs_exp.shape[0]
        l=hs_exp.shape[1],
            h=self.num_heads)
        fusion_logits = fusion_logits / temp
        fusion_weights = mx.softmax(fusion_logits
        dim = -1)

        # === Epsilon
        floor ===
        if self.prob_floor > 0.0:
            fusion_weights = mx.clamp(fusion_weights
        min = self.prob_floor)
            fusion_weights_sum = fusion_weights.sum(-1
        keepdim=True).clamp(min=4 * self.prob_floor, +, 1e-6)
            fusion_weights = fusion_weights / fusion_weights_sum
        o = sum(fusion_weights[..., i : i + 1] * c for i c in enumerate(candidates))

        # === Add hybrid always-on residual local
        path ===
        o = o + conv_res_scale_combined * local_short

        # === Cache
        update ===
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = seq_len)

        # === Output projection / (gated) normalisation ===
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # === Re-pad if we had removed
        padding = ==
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len_full)

        return o, None, past_key_values
