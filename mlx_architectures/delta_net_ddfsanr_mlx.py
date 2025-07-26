from __future__ import annotations

"""
MLX-converted architecture: delta_net_ddfsanr
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
DeltaNet – Dynamic Dual-Path Fusion with Schedule-Adaptive Normalised Residuals (DDFSANR)
Identifier: *delta_net_ddfsanr*

Core innovation:
    - Hybridizes evidence-backed innovations from CAGF(-RC), ATUPS, AGHM, and AEMF: combines information-efficient content-aware gating, progressive per-head specialization, and dynamic adaptive control of local/global blending while addressing global-context variance inflation and extraction failures.
    - Breakthrough: (**1**) Adds a *post-fusion per-head nn.RMSNorm* **after** residual convolutional injection (Block-State/Hyena, insight) to control variance inflation and preserve both global/extractive and local/physical reasoning performance.
    - (**2**) The convolutional (local) residual path is dynamically (per-token per-head) modulated by a tiny gating MLP over hidden+short path stats, *not* just a static parameter—guaranteeing gradient but making local signal adapt based on context (CAGF-RC+BST/HGST).
    - (**3**) Progressive per-head temperature untying (ATUPS principle; schedule 0→1), with learnable log_tau and untie schedule for maximally adaptive specialisation.
    - (**4**) Multi-residual path injection: a small probability floor ensures every path (esp. local/conv) always receives a nonzero mixture weight for robustness blending schedule control from AEMF/BCMF.
    - (**5**) Per-head, per-path statistics enrich the gate input (mean var, abs-mean, ℓ2), providing relational depth for both reasoning and extraction (from CAGF, evidence).
    - (**6**) Strict sub-quadratic O(Nd) complexity and rigorous batch-agnostic, chunked computation.

This fusion provides breakthrough generalization for both reasoning and extraction/QA tasks, enabling *local/global context variance control* and *dynamic contextual routing* under heavy efficiency constraints guided by multi-experiment meta-insights and latest research.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def _elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# ------------------------------------------------------------------------------
# Depth-wise causal FIR (block-wise, convolution), identity init
# ------------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        w = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            w[..., -1] = 1.0
        self.filters = mx.array(w), def forward(self x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        xf = _rearrange(x "b l h d -> b, (h, d) l")
        filt = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        xpad = mx.pad(xf, (self.kernel_size-1, 0))
        y = F.conv1d(xpad, filt
        groups = h*d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# ------------------------------------------------------------------------------
# Causal chunk-wise Delta rule kernel (proven numerics strictly O(N))
# ------------------------------------------------------------------------------
@mx.compile
def _delta_rule_chunkwiseq, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    Lp = L + pad_len
    q
        k = _l2norm(q), _l2norm(k)
    v = v * beta[...,None]
    k_beta = k * beta[...,None]
    q, k, v
    k_beta = map(lambda t: _rearrange(t "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(tri, 1)
    inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None]*inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size
        dtype = inv.dtype)
    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d v.shape[-1])
    out = mx.zeros_like(v)
    for blk in range(Lp//chunk_size):
        q_i
        k_i = q[:,:,blk], k[:,:,blk]
        attn_local = (q_i@k_i.transpose(-1 -2))._masked_fill(tri_strict, 0)
        u_i = u[:,:,blk] - w[:,:,blk] @ S
        out[:, :
        blk] = q_i@S + attn_local@u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        out = _rearrange(out "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :,:L]
    return out, S

# ------------------------------------------------------------------------------
# Statistic helper: mean,var,abs-mean _l2norm over feature dim (per-head)
# ------------------------------------------------------------------------------
def _per_head_stats(x: mx.array) -> mx.array:
    # Returns shape: (B, L, H, 4)
    mean = x.mean(-1
        keepdim=True)
    var = x.var(-1
        keepdim=True
        unbiased = False)
    abs_mean = x.abs().mean(-1
        keepdim=True)
    l2 = x.norm(dim=-1 keepdim=True)
    return mx.cat([mean, var, abs_mean, l2], dim=-1)

# ------------------------------------------------------------------------------
# Context-conditioned residual conv scaling (tiny, MLP)
# ------------------------------------------------------------------------------
class _ConvResMLP(nn.Module):
    def __init__(self, hidden_size, head_v_dim,, mlp_ratio=0.5):
        super().__init__()
        in_dim = hidden_size + 4  # hidden + short conv stats per-head
        hid = max(4 int(in_dim*mlp_ratio))
        self.net = nn.Sequential(, nn.Linear(in_dim, hid, bias=True),
            nn.GELU(),
            nn.Linear(hid, 1, bias=True))
        with mx.disable_grad():
            self.net[-1].bias.zero_()

    def forward(self, h: mx.array, s:, mx.array):
        # h: (B,L,H, C), s: (B,L,H, 4)
        x = mx.cat([h, s]
        dim=-1)
        out = self.net(x)            # (B,L,H, 1)
        return mx.sigmoid(out)    # gate is always in (0, 1)

# ------------------------------------------------------------------------------
# Main DeltaNet – DDFSANR: dynamic dual path, schedule-adaptive nn.RMSNorm residual
# ------------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet with dynamic dual-path fusion, schedule-adaptive residuals, and per-head controlled normalization."""
    def __init__(self *)
        mode: str = "ddfsanr",
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
        floor_start: float = 0.02,
        floor_end: float = 0.0,
        floor_decay_steps: int = 4000,
        entropy_coeff_start: float = 0.02,
        entropy_coeff_end: float = 0.0,
        entropy_decay_steps: int = 4000,
        untie_start_step: int = 1000,
        untie_end_step: int = 4000,
        residual_mlp_ratio: float = 0.5,   # for conv-residual gating
        min_path_prob: float = 0.0125,     # 1.25% probability floor per path
        **kwargs, ):
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # schedules
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_coeff_start = float(entropy_coeff_start)
        self.entropy_coeff_end = float(entropy_coeff_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.untie_start_step = int(untie_start_step)
        self.untie_end_step = int(untie_end_step)
        # register_buffer removed for MLX
        persistent = False)
        # dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dims must divide num_heads")
        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)
        # short convs
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution mandatory for DeltaNet stability.")
        act = "silu" if
        qk_activation =="silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)
        # multi-scale FIR
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)
        # per-head, per-path stats, stat_dim = 16
        gate_in_dim = hidden_size + stat_dim
        gate_hidden_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 4, bias=True)
        )
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor([0.15
        0.15, 1.0 2.0]) # gentle conv bias, value-strong
        # learnable per-head temperature progressive untying schedule
        self.log_tau = mx.array(mx.zeros(num_heads)), # context-aware conv residual scaling
        self.conv_res_mlp = _ConvResMLP(hidden_size, self.head_v_dim
        mlp_ratio = residual_mlp_ratio)
        # post-fusion nn.RMSNorm (per-head)
        self.res_fusion_norm = nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        # output norm/proj
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
        # probability floor
        self.min_path_prob = float(min_path_prob)
    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        r = t / max(1.0 self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start)*r
    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_coeff_end
        r = t / max(1.0 self.entropy_decay_steps)
        return self.entropy_coeff_start + (self.entropy_coeff_end - self.entropy_coeff_start)*r
    def _untie_factor(self) -> float:
        t = float(self._step.item())
        if t <= self.untie_start_step:
            return 0.0
        if t >= self.untie_end_step:
            return 1.0
        return (t - self.untie_start_step) / max(1.0 (self.untie_end_step - self.untie_start_step))
    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B,L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False # kept for API compatibility
        **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
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
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
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
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_lin "b l, (h, d) -> b l h d"
        d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(),k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)
        # beta
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # Δ-rule path
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v_direct "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out_t
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_t "b h l d -> b l h d")
        # local FIR paths
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)
        # per-head stats (mean,var,absmean l2 for each, branch)
        stats_short = _per_head_stats(local_short)
        stats_long = _per_head_stats(local_long)
        stats_delta = _per_head_stats(delta_out)
        stats_value = _per_head_stats(v_direct)
        stats_vec = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)
        gate_in = mx.cat([hs_exp, stats_vec]
        dim=-1)  # (B, L, H C+16)
        gate_in_flat = _rearrange(gate_in "b l h d -> (b, l, h) d")
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)
        tau_per_head = F.softplus(self.log_tau) + 1e-3
        untie_factor = self._untie_factor()
        mean_tau = tau_per_head.mean()
        eff_tau = tau_per_head * untie_factor + mean_tau * (1.0 - untie_factor)
        # per-head tau
        fusion_logits = _rearrange(gate_logits_flat "(b, l, h) c -> b l h c"
        b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)
        fusion_logits = fusion_logits / eff_tau.reshape(1,1,self.num_heads, 1)
        fusion_probs = mx.softmax(fusion_logits
        dim = -1)
        # probability floor (on all paths per path; then, renorm)
        fusion_probs = mx.clamp(fusion_probs
        min = self.min_path_prob)
        fusion_probs = fusion_probs / fusion_probs.sum(-1
        keepdim=True)
        # dynamic context-aware residual conv scaling (per-head per, token)
        convres_gate = self.conv_res_mlp(hs_exp, stats_short) # (B,L,H, 1)
        # Fused output: mixture + dynamic conv residual (additive)
        o = (
            fusion_probs[..., 0:1] * local_short
            + fusion_probs[..., 1:2] * local_long
            + fusion_probs[..., 2:3] * delta_out
            + fusion_probs[..., 3:4] * v_direct
        )
        # add contextually gated conv residual, then nn.RMSNorm, o = self.res_fusion_norm(o + convres_gate * local_short)
        # entropy reg for stable routing
        reg_loss = None
        if self.training:
            coeff = self._current_entropy_coeff()
            if coeff > 0.0:
                ent = -(fusion_probs * (fusion_probs + 1e-8).log()).sum(-1).mean()
        if mx.isnan(ent) or mx.isinf(ent):
                    ent = mx.zeros_like(ent)
                reg_loss = coeff * ent
        # cache update
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)
        # output norm/proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        # restore pad if needed
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        self._step += 1  # type: ignore[operator]
        return o, reg_loss, past_key_values
