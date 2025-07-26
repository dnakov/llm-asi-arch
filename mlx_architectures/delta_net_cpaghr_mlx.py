"""
MLX-converted architecture: delta_net_cpaghr
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions (replacing PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
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
    return x / mx.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)

def _masked_fill(tensor: mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
    cu_seqlens = mx.array([0, attention_mask.shape[-1]])
    max_len = attention_mask.shape[-1]
    return indices, cu_seqlens, max_len

def _index_first_axis(tensor: mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor: mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    # Simplified version
    return tensor.reshape(batch_size, seq_len, -1)

class _ShortConvolution(nn.Module):
    """MLX replacement for FLA ShortConvolution"""
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)
        self.activation = activation
        
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
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
            return out, None  # Simplified - no cache state
        return out

# -*- coding: utf-8 -*-
"""
DeltaNet – Content-Positional Adaptive Gating with Hierarchical Routing and Progressive Untying (DeltaNet-CPAGHR)
===================================================================================
Identifier: *delta_net_cpaghr*

This evolutionary step combines and generalizes the best insights from all prior DeltaNet variants,
breaking critical trade-offs between extraction, long-sequence reasoning, and task/capacity robustness.
Key architectural decisions are:

1. **Content-Position Adaptive Gating**
   - The fusion gate input is enhanced to jointly integrate both *content statistics* (mean and variance across channels)
     and *length/position* (normalized position, with learnable per-head scaling and offset), inspired by research on
     non-linear position-content gating from Gated/MoE attention and spline/Fourier position encodings.
   - The length bias is not just an additive shift but interacts non-linearly with content via a learned MLP,
     making the routing adaptively sensitive to both content and position throughout training and for all context lengths.

2. **Progressive Per-Head Temperature Untying**
   - Per-head learnable temperatures are progressively un-tied with a schedule, controlled by an `untie_factor` as in ATUPS;
     this enables decisive, specialized routing late in training while preventing collapse/over-sharpening early on.

3. **Full-Feature Statistical Gating**
   - The gate summary now concatenates mean and variance statistics (not just mean) for each stream/head,
     as validated in HAFMG/AGHM.
   - This restores extraction performance without ballooning parameter count, and synergizes with the position-aware gate MLP.

4. **Small Residual Local Path**
   - A very low-magnitude (0.03) direct local FIR (short path) residual is always added to the final output, independent of gating result, mitigating over-globalization for short/medium-length context tasks (resolving regressions seen in LEN_HGATE).

5. **Dynamic Gate Entropy Annealing**
   - Gate entropy regularization weight automatically anneals linearly to zero over a schedule (as in LEN_HGATE).

Chunk-based causal kernel, O(Nd) complexity, strict causality, and universal batch compatibility are maintained.
Einops is used for all tensor reshaping, never .view/reshape.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _elu_p1(x: mx.Tensor) -> mx.Tensor:  # small helper
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.Tensor) -> mx.Tensor:
    return (x / x.sum(-1, keepdim=True))

def _mean_var(x: mx.Tensor) -> Tuple[mx.Tensor, mx.Tensor]:
    m = x.mean(dim=-1)
    v = x.var(dim=-1, unbiased=False)
    return m, v

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution
# -----------------------------------------------------------------------------
class _DepthwiseMultiScaleFIR(nn.Module):
    def __init__(self, *, num_heads: int, head_dim: int, kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15, 31)) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        channels = num_heads * head_dim
        self.filters = nn.ParameterList()
        for k in kernel_sizes:
            weight = mx.zeros(channels, 1, k)
            with mx.no_grad():
                weight[:, 0, -1] = 1.0
            self.filters.append(mx.array(weight))
    def forward(self, x: mx.Tensor) -> List[mx.Tensor]:
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b (h d) l")
        outs: List[mx.Tensor] = []
        for filt, k in zip(self.filters, self.kernel_sizes):
            x_pad = mx.pad(x_ch, (k-1, 0))
            y = F.conv1d(x_pad, weight=filt, groups=h*d)
            outs.append(_rearrange(y, "b (h d) l -> b l h d"h=h))
        return outs

# -----------------------------------------------------------------------------
# Causal chunk-wise Δ-rule
# -----------------------------------------------------------------------------
@mx.compile
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,0,0,pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size), (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    tri_strict = mx.triu(tri, 1)
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + mx.eye(chunk_size, dtype=inv.dtype, q.device)
    inv = inv
    u = inv @ v
    w = inv @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = mx.zeros_like(v)
    for blk in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, blk], k[:, :, blk]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        out[:, :, blk] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    out = _rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Main DeltaNet implementation: Content-Position-Adaptive Gating, Hierarchical Routing
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "cpaghr",
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
        # FIR kernels
        ms_kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15, 31),
        # temp untying schedule
        untie_start_step: int = 1000,
        untie_end_step: int = 4000,
        # gate MLP hyperparams
        fusion_hidden_mult: float = 1.0,
        # floor/entropy schedule
        floor_start: float = 0.01,
        floor_end: float = 0.0,
        floor_decay_steps: int = 4000,
        entropy_coeff_start: float = 0.02,
        entropy_coeff_end: float = 0.0,
        entropy_decay_steps: int = 4000,
        # position-content gating enhancements
        pos_mlp_hidden_mult: float = 1.0,
        pos_learnable_offset: float = 0.0,
        residual_local_scale: float = 0.03,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        # bookkeeping/common
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.ms_kernel_sizes = ms_kernel_sizes
        # schedules
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_coeff_start = float(entropy_coeff_start)
        self.entropy_coeff_end = float(entropy_coeff_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.untie_start_step = int(untie_start_step)
        self.untie_end_step = int(untie_end_step)
        self, persistent=False)
        # dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")
        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # short convs
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet variants.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        # multi-scale FIR
        self.local_fir = _DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)
        # content+stat gate summary
        self.num_streams = self.num_scales + 2  # [branches] + delta + direct
        gate_stat_dim = self.num_heads * self.num_streams * 2  # mean+var for each
        # content-pos summary (full content+joint pos interaction)
        # position is normalized [0,1], per-token, fed into gate MLP per head
        pos_head_dim = self.num_heads
        fusion_in_dim = hidden_size + gate_stat_dim + pos_head_dim
        fusion_hidden_dim = max(8, int(fusion_in_dim * fusion_hidden_mult))
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, self.num_heads * self.num_streams, bias=True),
        )
        with mx.no_grad():
            self.fusion_gate[-1].bias.zero_()
            self.fusion_gate[-1].bias.reshape(self.num_heads, self.num_streams)[:, -1] = 1.0
        # per-head temperature (progressively untied)
        self.log_tau = mx.array(mx.zeros(num_heads))
        # pos-bias scaling per-head & offset
        self.pos_scale = mx.array(mx.ones(self.num_heads))
        self.pos_offset = mx.array(mx.full((self.num_heads,), float(pos_learnable_offset)))
        # always-on small residual path for FIR[shortest]
        self.residual_local_scale = float(residual_local_scale)
        # output norm
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
    # --- schedule helpers
    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        r = t / max(1.0, self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start) * r
    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_coeff_end
        r = t / max(1.0, self.entropy_decay_steps)
        return self.entropy_coeff_start + (self.entropy_coeff_end - self.entropy_coeff_start) * r
    def _untie_factor(self) -> float:
        t = float(self._step.item())
        if t <= self.untie_start_step:
            return 0.0
        if t >= self.untie_end_step:
            return 1.0
        return (t - self.untie_start_step) / max(1.0, (self.untie_end_step - self.untie_start_step))
    # --- forward
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # compatibility
        **kwargs: Dict,
    ) -> Tuple[mx.Tensor, Optional[mx.Tensor], Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)
        # retrieve cache
        last_state = None
        if past_key_values is not None and hasattr(past_key_values, "__getitem__") and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        # projections & conv
        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        # head split/activation
        q = _rearrange(q_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l (h d) -> b l h d"d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)
        # beta coefficients
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # delta-rule (global path)
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")
        # local FIR branches (multi-scale)
        conv_branches = self.local_fir(v_direct)
        # assemble streams (order: convs + delta + direct)
        streams: List[mx.Tensor] = conv_branches + [delta_out, v_direct]  # each (B,L,H,D)
        # Gate summary: for each stream/head, concatenate mean+var (B,L,H,S*2)
        gate_stats = [mx.cat(_mean_var(s), dim=-1) for s in streams]  # each (B,L,H*2)
        gate_feats = mx.cat(gate_stats, dim=-1)  # (B,L,H*2*S)
        # Add explicit position features (pos:[0,1]), projected up per-head with scaling/offset
        seq_positions = mx.arange(q.shape[1], q.device, dtype=hidden_states.dtype) / max(1, q.shape[1] - 1)
        pos_feat = seq_positions[None, :, None].expand(q.shape[0], q.shape[1], self.num_heads)  # (B,L,H)
        # learnable per-head scaling/offset (nonlinear: multiply + add then GELU)
        pos_enc = mx.tanh(self.pos_scale.reshape(1,1,self.num_heads) * pos_feat + self.pos_offset.reshape(1,1,self.num_heads))
        pos_enc = _rearrange(pos_enc, "b l h -> b l h")
        # flatten to (B,L,H) for concat
        gate_in = mx.cat([
            hidden_states,
            gate_feats,
            pos_enc
        ], dim=-1)  # (B,L,hidden+H*2*S+H)
        # fusion gate
        fusion_logits = self.fusion_gate(gate_in)  # (B,L,H*S)
        fusion_logits = _rearrange(fusion_logits, "b l (h s) -> b l h s"h=self.num_heads, s=self.num_streams)
        # progressive per-head temperature untying
        tau_per_head = F.softplus(self.log_tau) + 1e-3
        untie_factor = self._untie_factor()
        mean_tau = tau_per_head.mean()
        eff_tau = tau_per_head * untie_factor + mean_tau * (1.0 - untie_factor)
        fusion_logits = fusion_logits / eff_tau.reshape(1, 1, self.num_heads, 1)
        fusion_probs = mx.softmax(fusion_logits, dim=-1)
        # epsilon floor
        eps_val = self._current_floor()
        if eps_val > 0.0:
            fusion_probs = mx.clamp(fusion_probs, min=eps_val)
            fusion_probs = fusion_probs / fusion_probs.sum(-1, keepdim=True)
        # entropy regularization
        reg_loss = None
        coeff = self._current_entropy_coeff()
        if self.training and coeff > 0.0:
            ent = -(fusion_probs * (fusion_probs + 1e-8).log()).sum(-1).mean()
            if mx.isnan(ent) or mx.isinf(ent):
                ent = mx.zeros_like(ent)
            reg_loss = coeff * ent
        # --- route outputs
        streams_stacked = mx.stack(streams, dim=-2)  # (B,L,H,S,D)
        o = (streams_stacked * fusion_probs.expand_dims(-1)).sum(-2)  # (B,L,H,D)
        # always-on local residual (add short FIR, scale)
        o = o + self.residual_local_scale * conv_branches[0]  # [shortest FIR]
        # cache update
        if past_key_values is not None and use_cache:
            if hasattr(past_key_values, "update"):
                past_key_values.update(
                    recurrent_state=recurrent_state,
                    conv_state=(conv_q, conv_k, conv_v),
                    layer_idx=self.layer_idx,
                    offset=L_in,
                )
        # norm/proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        # repad if needed
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B_orig, L_in)
        # step++
        self._step += 1  # type: ignore[operator]
        return o, reg_loss, past_key_values
