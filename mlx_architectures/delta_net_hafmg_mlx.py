from __future__ import annotations

"""
MLX-converted architecture: delta_net_hafmg
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
DeltaNet – Hierarchical Adaptive-Floor Mixture Gating (DeltaNet-HAFMG)
Identifier: *delta_net_hafmg*

Key innovations:
1. **Hierarchical Value-vs-Context Routing:** A two-stage gate first allocates probability mass between the value/copy branch and the contextual mixture, based on hidden state and summarized content of each path. This enforces robust copy/context discrimination and prevents softmax crowding of value/confidence.

2. **Token-Adaptive Context Floor, Curriculum-Scheduled:** The context-vs-value split applies a token-adaptive minimum context floor whose schedule decays from a high initial (e.g., 0.08) to a minimal final (e.g., 0.01) over a configurable range. This ensures strong gradient signal and local routing capacity for lexical/extraction tasks, while allowing almost full copy gating later if justified.

3. **Per-Head Softplus-Constrained Context Temperature:** Probabilities among context paths (short
        long FIR Δ-rule) are computed via a per-head learnable temperature; softplus constraining prevents collapse and allows nuanced head specialization. Temperature is scheduled with optional decay and bounded below for stability.

4. **Entropy Regularization on Context Mixture:** An explicit entropy penalty targets the context submixture to guarantee sufficient guidance signal especially during ambiguous, span-level, or soft-fusion tasks.

5. **Output-Aware Summarized Gating Inputs:** Instead of concatenating all path activations, the gate MLP takes per-path statistical summaries (mean std, abs mean L2, norm), substantially reducing parameter cost and risk of overfitting while remaining output aware.

6. **Batch and Chunk Robustness:** All operations leverage einops.rearrange, preserve causal masking chunked O(N) computation and dynamic batch sizing throughout. All API and interface compatibilities are strictly preserved.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# -----------------------------------------------------------------------------
def _elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

def _per_head_stats(x: mx.array) -> mx.array:
    # x: (B,L,H, D) → (B,L,H, 4) [mean, std, abs mean, L2 norm]
    mean = x.mean(dim=-1
        keepdim=True)
    std = x.std(dim=-1
        unbiased=False
        keepdim = True)
    abs_mean = x.abs().mean(dim=-1
        keepdim=True)
    l2 = x.norm(dim=-1 keepdim=True)
    return mx.cat([mean, std, abs_mean, l2], dim=-1)

# -----------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., 0] = 1.0
            filt.add_(0.03 * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self x: mx.array) -> mx.array:  # (B,L,H, D)
        b, l, h, d = x.shape
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

@mx.compile
def _delta_rule_chunkwiseq, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
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
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size
        dtype = inv.dtype)
    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    for idx in range(L_pad // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        out = _rearrange(out "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
class DeltaNet(nn.Module):
    """DeltaNet with Hierarchical Adaptive-Floor Mixture Gating (HAFMG)."""
    def __init__(
        self, *,
        mode: str = "hafmg",
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
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        floor_start: float = 0.08,
        floor_end: float = 0.01,
        floor_decay_steps: int = 4000,
        context_temp_init: float = 0.0,
        context_temp_min: float = 0.05,
        entropy_reg_coeff: float = 0.01 **kwargs: Dict) -> None:
        super().__init__()
        # -- dimension bookkeeping
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
        self.fir_short_kernel = fir_short_kernel
        self.fir_long_kernel = fir_long_kernel
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_reg_coeff = float(entropy_reg_coeff)
        self.context_temp_min = float(context_temp_min)
        # -- core dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        # -- projections
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
        # -- short convs
        if not use_short_conv:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet variants.")
        act = "silu" if
        qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)
        # -- FIR convs
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim self.fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim self.fir_long_kernel)
        # -- two-stage gating: (statistically summarized
        inputs)
        self.gate_context_vs_value = nn.Sequential(, nn.Linear(hidden_size, +, self.num_heads * 16, hidden_size, bias = True),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_heads, bias=True))
        # value bias init: encourage copy early
        with mx.disable_grad():
            self.gate_context_vs_value[-1].bias[:] = 1.25
        # context mixture: per-head temperature
        self.context_log_tau = mx.array(mx.full((self.num_heads), context_temp_init))
        # context path mixture gate
        self.gate_context_mix = nn.Sequential(, nn.Linear(hidden_size, +, self.num_heads * 12, hidden_size, bias = True),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_heads * 3, bias=True))
        # output norm/proj
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
        # register_buffer removed for MLX
        persistent = False)

    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        r = t / max(1.0 self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start) * r

    def forward(, self,
        hidden_states: mx.array,  # (B,L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]
        q_lin
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        # -- head split/activation
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_lin "b l, (h, d) -> b l h d"
        d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)
        # -- beta
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # -- Δ-rule
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v_direct "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out_d
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d "b h l d -> b l h d")
        # -- Local FIRs
        fir_short = self.local_fir_short(v_direct)
        fir_long = self.local_fir_long(v_direct)
        # -- summarize by per-path stats (mean, std, absmean, L2): shape (B,L,H, 4)
        stats_short = _per_head_stats(fir_short)
        stats_long = _per_head_stats(fir_long)
        stats_delta = _per_head_stats(delta_out)
        stats_val = _per_head_stats(v_direct)
        gate_stat_vec = mx.cat([, stats_short, stats_long, stats_delta, stats_val
        ], dim=-1)  # (B,L,H, 16)
        flat_gate_stat = _rearrange(gate_stat_vec "b l h d -> b l, (h, d)")
        # -- 1st stage: value vs context
        gate1_in = mx.cat([hidden_states, flat_gate_stat]
        dim=-1)
        gate1_logits = self.gate_context_vs_value(gate1_in)  # (B,L, H)
        # -- context allocation (softmax not required for binary, choice):
        context_gate = mx.sigmoid(gate1_logits)  # (B,L, H), value path prob
        # -- Curriculum context floor (scheduled):
        min_context_prob = self._current_floor()
        value_prob = (1 - min_context_prob) * context_gate  # (B,L, H)
        context_prob = 1.0 - value_prob  # guaranteed >= min_context_prob
        # -- 2nd stage: context path mixture (output-aware, summarized)
        gate2_stat_vec = mx.cat([, stats_short, stats_long, stats_delta
        ], dim=-1)  # (B,L,H, 12)
        gate2_in = mx.cat([hidden_states _rearrange(gate2_stat_vec "b l h d -> b l, (h, d)")]
        dim=-1)
        mix_logits = self.gate_context_mix(gate2_in)  # (B,L H*3)
        mix_logits = _rearrange(mix_logits "b l, (h, x) -> b l h x"
        h=self.num_heads
        x = 3)
        
        context_temp = F.softplus(self.context_log_tau) + self.context_temp_min
        mix_logits = mix_logits / context_temp.reshape(1,1,self.num_heads, 1)
        context_weights = mx.softmax(mix_logits
        dim = -1)  # (B,L,H, 3)
        # -- context allocation, context_weights = context_prob.expand_dims(-1) * context_weights
        # -- output assembly
        o = (
            context_weights[..., 0:1] * fir_short
            + context_weights[..., 1:2] * fir_long
            + context_weights[..., 2:3] * delta_out
            + value_prob.expand_dims(-1) * v_direct
        )
        # -- entropy penalty (context mixture, only)
        reg_loss = None
        if self.training and self.entropy_reg_coeff > 0.0:
            context_mix_entropy = -(context_weights * (context_weights+1e-8).log()).sum(-1).mean()
        reg_loss = self.entropy_reg_coeff * context_mix_entropy
        # -- cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)
        # -- output norm/proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        self._step += 1
        return o, reg_loss, past_key_values
