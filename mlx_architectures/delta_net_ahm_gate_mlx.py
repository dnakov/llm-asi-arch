from __future__ import annotations

"""
MLX-converted architecture: delta_net_ahm_gate
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
"""
DeltaNet – Adaptive Per-Head Mixing and Selective Gating (AHM-Gate)
Identifier: delta_net_ahm_gate

This variant fuses the research-proven effectiveness of hierarchical gating,
dynamic temperature annealing, and adaptive mixing regularization to robustly
address the tradeoff between extraction precision and narrative/contextual
reasoning. It leverages per-head per-stage adaptive mixing (λ) and selective
gate sharpness for fine-grained, context-driven information routing.

Key Innovations
1. **Per-Head, Learnable Adaptive Mixing**
   • Each attention head learns an independent λ parameter controlling the
     magnitude of residual cross-head mixing with schedule-driven decay to
     a dynamic (head-learned) floor, allowing precise/tight mixture for
     extraction and higher persistent blend for narrative heads.
   • λ is modulated by a confidence-driven schedule: if gate entropy per head
     on a given token drops below a threshold, λ is further annealed,
     supporting evidence-based, data-controlled head specialization.

2. **Stage-Selective Temperature Annealing**
   • Dynamic τ annealing controls only the outer router's logits inner local
     gates remain at moderate temperature to avoid excessive over-sharpening.
   • Per-head and groupwise τ blending as in DTA literature for adaptive
     specialisation/regularization balance.

3. **Confidence-Adaptive Mixing Suppression**
   • λ per head is further (multiplicatively) suppressed at inference/training
     time when the gate distribution is highly confident (entropy below a
     schedule-driven, threshold), ensuring extraction heads become decisive
     at critical tokens/positions, while global/narrative heads can retain
a baseline cross-head cooperation.

All interface contracts, forward signature, causal chunking, batch-size
dynamism and computational complexity constraints are strictly preserved.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _elu_plus_one(x:, mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x:, mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (as, before)
# ---------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filters = mx.array(mx.randn(num_heads, head_dim, self.kernel_size) * 0.02)
    def forward(self, x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Chunk-wise Delta rule (copied verbatim as per, prior)
# ---------------------------------------------------------------------------
@mx.compile
def _delta_rule_chunkwise
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    *,
    chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0,
        0, 0, pad_len)
        q = mx.pad(q, pad)
        k = mx.pad(k, pad)
        v = mx.pad(v, pad)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i
        :i] += (attn_inv[..., i, :, None] * attn_inv[..., :, :i]).sum(-2)
        attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)
    u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    for idx in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# ---------------------------------------------------------------------------
# Main DeltaNet implementation: Adaptive-HeadMix Selective Gating
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet layer with Adaptive Per-Head Mixing and Selective Gating (AHM-Gate)."""
    def __init__(self, mode: str =, "ahm_gate",
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
        mix_init: float = 0.03,
        mix_floor_init: float = 0.005,
        mix_decay_steps: int = 4000,
        tau_start: float = 1.0,
        tau_end: float = 0.2,
        tau_warmup_steps: int = 4000,
        group_size: int = 2,
        entropy_suppress_thresh: float = 0.25 **kwargs) -> None:
        super().__init__()
        assert qk_activation in ("silu", "relu", "elu", "identity")
        assert qk_norm in("l2", "sum")
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
        self.layer_idx = layer_idx
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.mix_decay_steps = int(mix_decay_steps)
        self.tau_warmup_steps = int(tau_warmup_steps)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.group_size = max(1, int(group_size))
        self.entropy_suppress_thresh = float(entropy_suppress_thresh)
        # register_buffer removed for MLX
        persistent = False)
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value, dims must divide num_heads")
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet stability.")
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_kernel_size_long
        )
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_kernel_size_short
        )
        self.stat_dim = 16
        gate_input_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_input_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True))
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)
        # Per-head adaptive temperatures
        self.logit_tau_head = mx.array(mx.full((num_heads), math.log(self.tau_start)))
        self.logit_tau_group = mx.array(mx.full((num_heads, //, group_size), math.log(self.tau_start)))
        # register_buffer removed for MLX // self.group_size
    persistent = False)
        # Per-head per-layer learnable mixing
        self.mix_coeff = mx.array(mx.full((num_heads), mix_init))
        self.mix_floor = mx.array(mx.full((num_heads), mix_floor_init)
        requires_grad=True)
        # Per-head gamma for residual scaling
        self.conv_residual_logit = mx.array(mx.full((num_heads), -2.0))
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)
    @staticmethod
    def _per_head_stats(x:, mx.array) -> mx.array:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)
    def _get_blended_tau(self) -> mx.array:
        head_tau = mx.exp(self.logit_tau_head)  # (H)
        group_tau = mx.exp(self.logit_tau_group)  # (G)
        group_tau_expanded = group_tau[self._group_index]  # (H)
        t = float(self._step.item())
        blend = min(1.0, max(0.0, t / max(1.0, self.tau_warmup_steps)))
        tau = blend * head_tau + (1 - blend) * group_tau_expanded
        return tau  # (H)
    def _decay_mix_coeff(self) -> mx.array:
        t = float(self._step.item())
        # Linear decay to individual adaptive per-head learned floor
        decay = max(0.0, 1.0 - t / max(1.0, self.mix_decay_steps))
        coeff = self.mix_floor + (self.mix_coeff - self.mix_floor) * decay
        return coeff  # (H)
    def _fused_entropy(self, probs: mx.array) -> mx.array:
        # probs: (B,L,H, K)
        ent = -(probs * (probs + 1e-8).log()).sum(-1), # (B,L, H)
        return ent
    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False,
        **kwargs  ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be (batch, seq_len)"
        batch_size, seq_len_full, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]
        q_in = self.q_proj(hidden_states)
        k_in = self.k_proj(hidden_states)
        v_in = self.v_proj(hidden_states)
        q_in
        conv_state_q = self.q_conv1d(q_in, cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_in
        conv_state_k = self.k_conv1d(k_in, cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_in
        conv_state_v = self.v_conv1d(v_in, cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q_in, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_in, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_in, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        beta = mx.clamp(beta, min = 1e-6)
        delta_out_t, recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q, "b l h d -> b h l d")
        k=_rearrange(k, "b l h d -> b h l d"),
            v=_rearrange(v_direct, "b l h d -> b h l d")
        beta=_rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_t, "b h l d -> b l h d")
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)  # (B, L, H, 16)
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)
        gate_in = mx.cat([hs_exp, stats_vec]
        dim=-1)
        gate_logits_flat = self.fusion_gate_mlp(_rearrange(gate_in, "b l h d -> (b, l, h) d"))
        # Stage-selective blended τ only on outer gate
    tau = self._get_blended_tau()  # (H)
        tau_bc = tau.reshape(1, 1, self.num_heads, 1)
        gate_logits = _rearrange(gate_logits_flat, "(b, l, h) c -> b l h c"
        b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)
        gate_logits = gate_logits / tau_bc
        # Standard softmax with epsilon floor on conv paths only
        fusion_weights = mx.softmax(gate_logits, dim = -1)
        floor_vec = mx.tensor([0.02, 0.02, 0.0, 0.0]
        dtype=fusion_weights.dtype)
        fusion_weights = mx.clamp(fusion_weights, min = floor_vec)
        fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)
        # Per-head adaptive λ cross-head mixing
    mix_coeff = self._decay_mix_coeff()  # (H)
        entropy = self._fused_entropy(fusion_weights)  # (B,L, H)
        suppress_mask = (entropy < self.entropy_suppress_thresh).float()[..., None]  # (B,L,H, 1)
        # Optionally further anneal mix_coeff in-place for confident heads
        # Dynamic λ[head] = λ[head] * (1 - I{confident}) + mix_floor * I{confident}
        effective_mix_coeff = mix_coeff[None, None, :, None] * (1. - suppress_mask) + self.mix_floor[None, None, :, None] * suppress_mask
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )
        # Adaptive residual as in previous but now per head
    static_gamma = mx.sigmoid(self.conv_residual_logit)[None, None, :, None]
        residual_scale = static_gamma * (1.0 - fusion_weights[..., 0:1])
        o = o + residual_scale * local_short
        # Per-head cross-head mixing (soft, ensemble)
        mean_heads = o.mean(dim=2, keepdim=True)  # (B,L,1, D)
        o = o + effective_mix_coeff * mean_heads
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = hidden_states.shape[1])
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            # Ensure o and g_vec dtypes are matched for numerical stability in norm
            if o.dtype != g_vec.dtype:
                g_vec = g_vec
        o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        # Explicit dtype match before o_proj to fix mat1 and mat2 dtype mismatch error
    o = o
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len_full)
        self._step += 1  # type: ignore[operator]
        return o, None, past_key_values
