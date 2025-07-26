"""
MLX-converted architecture: delta_net_mafr
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
DeltaNet – Multi-Scale Adaptive Floor & Residual (delta_net_mafr)
================================================================
Identifier: *delta_net_mafr*

This evolution introduces **Multi-Scale Adaptive Floor Routing (MAFR)** that
jointly preserves local detail retrieval and global reasoning capacity while
remaining strictly O(N).

Key Innovations
---------------
1. Multi-Scale Local Memories (3× FIR)
   •  Three causal depth-wise FIR convolutions – *short*, *medium*, *long* –
      capture local patterns across 3 temporal scales (kernel sizes 3 / 15 /
      64 by default).
   •  Evidence from Hyena / RetNet shows that richer temporal spectra boosts
      both lexical extraction (very short) and phrase / paragraph coherence
      (medium).

2. Per-Head **Adaptive Probability Floors**
   •  Each head & path owns a learnable parameter `floor_logit[h,p]` that
      converts (via `sigmoid`) to a maximum floor magnitude `ε_max`.
   •  A *linear* annealing schedule drives the floor from `ε_max` →
      `ε_final` (default 0.01) over `floor_decay` steps ensuring early gradient
      flow *and* a persistent non-zero local allocation for lexical tasks.

3. Vectorised **Residual Bypass**
   •  A per-head residual weight `α[h]∈[0,1]` (sigmoid-paramised) mixes the
      *mean* of the three local FIR paths back into the fused output,
      guaranteeing irreducible local signal regardless of gate confidence.

4. Five-Path Content-Aware Gating
   •  Paths: short, medium, long, Δ-rule global, identity/value.
   •  Gating MLP ingests token embedding plus per-head statistics of each path
      (mean, var, abs-mean, L2) → logits.
   •  A single learnable temperature parameter sharpens distributions.

5. Strict O(N) Complexity & Causal Safety
   •  All ops are depth-wise 1-D convs or chunk-wise scans – no softmax
      attention.
   •  Works with arbitrary batch size; shapes always inferred at runtime via
      `einops.rearrange`.

The design directly tackles regressions observed in *dynfuse* & *parafuse*:
•  A **non-zero final ε_final** preserves SWDE / BoolQ local fidelity.
•  Additional *medium* scale plus residual bypass reinforce lexical cues.
•  Adaptive, head-specific floors prevent global over-dominance without
   hand-tuned schedules.

Interface, class name `DeltaNet`, and forward signature remain unchanged.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _elu_plus_one(x: mx.Tensor) -> mx.Tensor:  # noqa: D401
    """Shifted ELU – strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.Tensor) -> mx.Tensor:  # noqa: D401
    """L1 normalisation along the last dimension."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity + small noise init)
# -----------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR convolution for tensors shaped (B, L, H, D)."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        kernel_size: int,
        noise_std: float = 2e-3,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.no_grad():
            weight[..., -1] = 1.0  # identity (current timestep)
            if noise_std > 0:
                weight.add_(noise_std * mx.randn_like(weight))
        self.filters = mx.array(weight)  # (H, D, K)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        w = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, w, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

# -----------------------------------------------------------------------------
# Chunk-wise associative Δ-rule (@mx.compile)
# -----------------------------------------------------------------------------

@mx.compile  # noqa: D401
def _delta_rule_chunkwise(
    q: mx.Tensor,  # (B,H,L,Dk)
    k: mx.Tensor,  # (B,H,L,Dk)
    v: mx.Tensor,  # (B,H,L,Dv)
    beta: mx.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Efficient O(N) Δ-rule implementation preserving causality."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )
    tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    tri_strict = mx.triu(tri, 1)

    # Avoid mx.log2 or other log2-related ops for dynamo compatibility
    # (addressing missing OpaqueUnaryFn_log2)
    # Ensure only supported PyTorch ops are used in the dynamo-compiled region

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + mx.eye(chunk_size, dtype=inv.dtype, inv.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = mx.zeros_like(v)

    n_blocks = q.shape[2]
    for idx in range(n_blocks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = _rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Per-head stats helper
# -----------------------------------------------------------------------------
def _per_head_stats(x: mx.Tensor) -> mx.Tensor:  # (B,L,H,D) -> (B,L,H,4)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    abs_mean = x.abs().mean(dim=-1, keepdim=True)
    l2 = x.norm(dim=-1, keepdim=True)
    return mx.cat([mean, var, abs_mean, l2], dim=-1)

# -----------------------------------------------------------------------------
# Optional typing helper
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet layer with Multi-Scale Adaptive Floor & Residual
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):  # noqa: D401 – required class name
    """DeltaNet layer with *multi-scale adaptive floors and residual bypass*."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        mode: str = "mafr",
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
        # FIR kernel sizes
        fir_kernel_short: int = 3,
        fir_kernel_medium: int = 15,
        fir_kernel_long: int = 64,
        # Gating network
        gate_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float, float] = (-0.5, -0.2, -0.2, 1.0, 2.0),
        # Temperature (softplus param)
        gate_temp_init: float = 0.7,
        # Adaptive floor schedule
        floor_max: float = 0.05,
        floor_final: float = 0.01,
        floor_decay: int = 4000,
        # Residual bypass
        residual_init: float = 0.1,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        # --------------- dimension bookkeeping ------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")
        # --------------- flags & misc ---------------------------------
        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # --------------- projections ---------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # --------------- short convolutions --------------------------
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet performance.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        # --------------- multi-scale FIR memories --------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_short)
        self.fir_medium = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_medium)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_long)
        # --------------- gating network ------------------------------
        stats_dim_per_head = 4 * 5  # 5 paths × 4 statistics
        gate_in_dim = hidden_size + stats_dim_per_head  # per-head input dimension
        hidden_gate_dim = max(8, int(gate_in_dim * gate_hidden_mult // 2))
        self.gate_fc1 = nn.Linear(gate_in_dim, hidden_gate_dim, bias=True)
        self.gate_fc2 = nn.Linear(hidden_gate_dim, 5, bias=True)
        with mx.no_grad():
            self.gate_fc2.bias.zero_()
            bias_template = mx.tensor(gate_bias_init, dtype=mx.float32)
            self.gate_fc2.bias.copy_(bias_template)
        self.logit_temp = mx.array(mx.tensor([math.log(math.expm1(gate_temp_init))]))
        # --------------- adaptive floor parameters -------------------
        self.floor_max = float(floor_max)
        self.floor_final = float(floor_final)
        self.floor_decay = int(floor_decay)
        init_floor_logit = math.log(0.5)  # sigmoid ~0.5
        self.floor_param = mx.array(mx.full((num_heads, 5), init_floor_logit))
        # --------------- residual bypass -----------------------------
        self.residual_logit = mx.array(mx.full((num_heads,), math.log(residual_init / (1 - residual_init))))
        # --------------- output normalisation / proj -----------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        # step counter buffer
        self, persistent=False)

    # ------------------------------------------------------------------
    # Helper – compute current floor tensor (1,1,H,5)
    # ------------------------------------------------------------------
    def _current_floor(self) -> mx.Tensor:
        step = int(self._step.item())
        if self.floor_decay <= 0:
            factor = 0.0
        else:
            factor = max(0.0, 1.0 - step / self.floor_decay)
        eps_now = self.floor_final + (self.floor_max - self.floor_final) * factor  # scalar
        floor = mx.sigmoid(self.floor_param) * eps_now  # (H,5)
        return floor.expand_dims(0).expand_dims(0)  # (1,1,H,5)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # unused – kept for signature comp.
        **kwargs,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (B,L)"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q = _rearrange(q_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l (h d) -> b l h d"d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d, recur_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")
        local_short = self.fir_short(v_direct)
        local_medium = self.fir_medium(v_direct)
        local_long = self.fir_long(v_direct)
        stats = mx.cat([
            _per_head_stats(local_short),
            _per_head_stats(local_medium),
            _per_head_stats(local_long),
            _per_head_stats(delta_out),
            _per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H, 4*5)
        gate_token = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,D)
        gate_in = mx.cat([gate_token, stats], dim=-1)  # (B,L,H, D + 20)
        gate_in_flat = _rearrange(gate_in, "b l h d -> (b l h) d")
        x = F.gelu(self.gate_fc1(gate_in_flat))
        logits_flat = self.gate_fc2(x)  # (B*L*H,5)
        logits = _rearrange(logits_flat, "(b l h) p -> b l h p"b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads)
        temp = F.softplus(self.logit_temp) + 1e-4
        logits = logits / temp
        probs = mx.softmax(logits, dim=-1)  # (B,L,H,5)
        floor = self._current_floor()  # (1,1,H,5)
        probs = mx.clamp(probs, min=floor)
        probs = probs / probs.sum(-1, keepdim=True)
        w_short = probs[..., 0:1]
        w_medium = probs[..., 1:2]
        w_long = probs[..., 2:3]
        w_delta = probs[..., 3:4]
        w_value = probs[..., 4:5]
        fused = (
            w_short * local_short +
            w_medium * local_medium +
            w_long * local_long +
            w_delta * delta_out +
            w_value * v_direct
        )
        residual_alpha = mx.sigmoid(self.residual_logit).reshape(1, 1, self.num_heads, 1)
        local_mean = (local_short + local_medium + local_long) / 3.0
        fused = fused + residual_alpha * local_mean
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            fused = self.o_norm(fused, g_vec)
        else:
            fused = self.o_norm(fused)
        fused = _rearrange(fused, "b l h d -> b l (h d)")
        out = self.o_proj(fused)
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0), indices, B_orig, L_in)
        self._step += 1  # type: ignore[operator]
        return out, None, past_key_values
