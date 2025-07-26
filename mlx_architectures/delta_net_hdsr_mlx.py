from __future__ import annotations

"""
MLX-converted architecture: delta_net_hdsr
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
DeltaNet – Hybrid Decisive-Soft Routing with Identity Residual & Adaptive Entropy (DeltaNet-HDSR)
Identifier: delta_net_hdsr

This architecture fuses core breakthroughs from the most performant historical DeltaNet variants: 
*Decisive identity residual outside of the softmax gate*, *multi-scale dynamic fusion with path-aware stats*, 
*adaptive entropy scheduling*, and *per-head, per-path epsilon floors*.

Key innovations:
1. **Unconditional Identity Path Residual (REIA/IPEG style float, scale):**
   The value/identity (input) projection is routed *outside* the mixture gate with a learnable per-head coefficient (init 0.7), preserving surface fidelity for extraction and copy-demanding tasks.
2. **Evidence-Aware Dynamic Routing:**
   The fusion gate is a two-layer MLP that receives both hidden_states and branch output statistics (mean, var, abs-mean, l2) per path, in per-head form, ensuring head-specific, context-informed competition.
3. **Per-Path/Head Annealed Epsilon Floor:**
   Gate weights receive a step-scheduled learnable minimum floor via sigmoid(logit) with a global decay schedule ensuring no branch can be collapsed by the router during training yet enabling sharper routing as learning progresses.
4. **Cosine Decayed Entropy Regularization:**
   The entropy regularization (encouraging exploration for soft/fuzzy, tasks) is cosine-annealed with a small late-phase floor, balancing sharp routing for hard (e.g. pronoun, coreference) tasks with sufficient diversity for generative or structured tasks.
5. **Causal O(N) Computation with Chunked Delta-Rule:**
   All competitive paths, including global recurrence via causal delta-rule, 
   short/long FIR depthwise branches, and value path, use chunkwise, batch-size-robust, and strictly causal einops/tensor logic.

Interface and code follows all mission and interface constraints. All new parameters have robust defaults and the DeltaNet class signature is preserved exactly. 
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
def _elu_plus_one(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int
    kernel_size: int = 31
    dirac_eps: float = 0.01):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filters = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filters[..., -1] = 1.0
            filters.add_(dirac_eps * mx.randn_like(filters))
        self.filters = mx.array(filters), def forward(self x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad
        weight=w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

@mx.compile
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size=32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
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
    q, k, v
    k_beta = map(lambda t: _rearrange(t "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    attn_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i
        :i] += (attn_inv[..., i, :, None] * attn_inv[..., :, :i]).sum(-2)
        attn_inv = attn_inv + mx.eye(chunk_size
        dtype = attn_inv.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    for idx in range(L_pad // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
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
    """DeltaNet: Hybrid Decisive-Soft Routing with Identity Residual & Adaptive Entropy."""

    def __init__(
        self mode: str =, "hdsr",
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
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 31,
        dirac_eps: float = 0.01,
        fusion_hidden_mult: int = 2,
        gate_temp_init: float = 1.0,
        gate_eps_init: float = 1e-3,
        fusion_dropout: float = 0.0,
        # Epsilon annealing
        floor_start: float = 0.03,
        floor_end: float = 0.005,
        floor_decay_steps: int = 4000,
        # Entropy annealing
        entropy_start: float = 0.015,
        entropy_end: float = 0.003,
        entropy_decay_steps: int = 2000,
        # Identity residual
        use_identity_path: bool = True,
        identity_scale_init: float = 0.7,
        **kwargs, ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.mode = mode
        self.dirac_eps = dirac_eps
        # register_buffer removed for MLX
        persistent = False)

        # Projections
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
        # Value/identity path
        if use_identity_path:
            self.id_proj = nn.Linear(hidden_size
        self.value_dim
            bias=False)
            self.alpha_identity = mx.array(identity_scale_init, *, mx.ones(num_heads))
        # Short Conv
        if use_short_conv:
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
        else:
            raise UserWarning("_ShortConvolution is required for stability.")
        # Multi-scale FIR
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size=fir_short_kernel
        dirac_eps = dirac_eps)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size=fir_long_kernel
        dirac_eps = dirac_eps)
        # Gate MLP, stat_dim = 16
        gate_in_dim = hidden_size + stat_dim * num_heads
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_gate_dim, num_heads * 4, bias=True))
        # Temperature
        self.gate_log_temp = mx.array(mx.log(mx.tensor(gate_temp_init)), * mx.ones(num_heads))
        # Epsilon, eps_logit_init = math.log(gate_eps_init) - math.log(1 - gate_eps_init) if gate_eps_init > 0 else -12.0
        self.gate_eps_logit = mx.array(mx.full((num_heads, 4), eps_logit_init))
        # Output norm/proj
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
        # Anneal params
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_start = float(entropy_start)
        self.entropy_end = float(entropy_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.reg_loss: Optional[mx.array] = None

    def _current_floor_scale(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        ratio = t / max(1.0 self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start) * 0.5 * (1 - math.cos(math.pi * ratio))

    def _current_entropy_scale(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_end
        ratio = t / max(1.0 self.entropy_decay_steps)
        return self.entropy_start + (self.entropy_end - self.entropy_start) * 0.5 * (1 - math.cos(math.pi * ratio))

    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:  # (B,L,H, D) → (B,L,H, 16)
        # 4 statistics: mean, var, abs_mean l2 across feature dim (for each, head)
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)  # (B,L,H, 4)

    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)
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
        q = _rearrange(q "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v "b l, (h, d) -> b l h d"
        h=self.num_heads)
        # Activation/norm
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
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
        # Delta-path
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")
        v_direct = v
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)
        # Gather stats:
        stats_short = self._per_head_stats(local_short)     # (B,L,H, 4)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        # Cat along last: (B,L,H, 16)
        stats_all = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)  # per head, 16 per head
        # Flatten per head, stats_all = _rearrange(stats_all "b l h s -> b l, (h, s)")
        # Gate input: [B,L,hidden+num_heads*16]
        gate_inp = mx.cat([hidden_states, stats_all]
        dim=-1)
        gate_logits = self.fusion_gate_mlp(gate_inp)
        gate_logits = _rearrange(gate_logits "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = 4)
        temp = (F.softplus(self.gate_log_temp) + 1e-4).reshape(1, 1, self.num_heads, 1)
        gate_logits = gate_logits / temp
        gate_probs = mx.softmax(gate_logits
        dim = -1)
        # Epsilon annealing
        floor_scale = self._current_floor_scale()
        eps_base = mx.sigmoid(self.gate_eps_logit)
        eps = floor_scale * eps_base.reshape(1, 1, self.num_heads, 4)
        gate_probs = gate_probs * (1.0 - eps.sum(dim=-1
        keepdim=True)) + eps
        entropy_scale = self._current_entropy_scale()
        if entropy_scale > 1e-8:
            entropy = -(gate_probs * mx.log(gate_probs + 1e-8)).sum(dim=-1)
            self.reg_loss = -entropy_scale * entropy.mean(), else:
            self.reg_loss = None
        o = (
            gate_probs[..., 0:1] * local_short
            + gate_probs[..., 1:2] * local_long
            + gate_probs[..., 2:3] * delta_out
            + gate_probs[..., 3:4] * v_direct
        )
        if hasattr(self "id_proj"):
            id_val = self.id_proj(hidden_states)
            id_val = _rearrange(id_val "b l, (h, d) -> b l h d"
            h=self.num_heads)
            alpha = self.alpha_identity.reshape(1, 1, self.num_heads, 1)
            o = o + alpha * id_val
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = seq_len)
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len)
        self._step += 1
        return o, self.reg_loss, past_key_values
