from __future__ import annotations

"""
MLX-converted architecture: delta_net_dynfuse
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
DeltaNet – Dynamic Conv-Residual & Decaying-Floor Content-Aware Gated Fusion (delta_net_dynfuse)
This evolution of **delta_net_cagf** fixes the remaining *local–global trade-off* by
combining the strengths of Content-Aware Gated Fusion (per-head statistics Δ-rule
positive, bias) with two new mechanisms that preserve **local convolutional
capacity** *throughout* training **without** hurting global/contextual reasoning:

1. **Decaying Local-Floor Schedule**
   •   At every forward pass we enforce a *minimum share* ε(t) on both
       convolutional paths (short & long, FIR) **per-head per-token**.  ε(t)
       starts at `floor_init` (default = 0.08) to guarantee gradient flow early
       and **exponentially decays** towards `floor_final` (default = 0) with
       time-constant `floor_decay` (# steps).
   •   This protects local features early on but removes the upper-bound later,
       allowing the gate to allocate **100 %** probability to global paths when
       beneficial for tasks like coreference or completion.

2. **Learnable Conv-Residual Bypass**
   •   A tiny, *always-on* residual from the **sum of both FIR paths** is added
       to the fused output and modulated by a **single learnable scalar**
       `α ∈ [0 1]` (initialised to 0.1 in *sigmoid* space).  This prevents
       complete suppression of local information even if the gate is confident
       but keeps the residual magnitude trainable.

3. **Entropy Regularisation Hook** (optional)
   •   The layer stores an auxiliary loss `self.reg_loss` equal to
       `λ · ReLU(H_target – H_actual)` where H is the per-head gate entropy.  The
       training loop can simply add this scalar to the primary loss to maintain
       path diversity.  Defaults: λ = 0.02
       H_target = 1.0.

All other mechanics (chunk-wise Δ-rule kernel per-head statistics, causal
FIR convolutions, caching) are **unchanged** and remain *strictly O(N)*.
Interface compatibility, class name `DeltaNet`, and forward signature are
preserved – this layer is a **drop-in replacement**.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def elu_p1(x: mx.array) -> mx.array:
    """Shifted ELU – strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:
    """Normalise last dim to sum-to-one (L1)."""
    return (x / x.sum(-1, keepdim=True))


# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identical math to previous, versions)
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding (O(N) cost)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.randn(num_heads, head_dim self.kernel_size) * 0.02
        # Identity-like start for stable optimisation (weight on current, step)
        filt[..., 0] += 1.0
        self.filters = mx.array(filt), def forward(self x: mx.array) -> mx.array:  # (B, L, H, D)
        b, l, h, d = x.shape
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged still @mx.compile)
# -----------------------------------------------------------------------------

@mx.compile
def delta_rule_chunkwise(
    q: mx.array # (B H, L, D_k)
    k: mx.array,
    v: mx.array,
    beta: mx.array # (B H, L)
    *,
    chunk_size: int = 32) -> Tuple[mx.array mx.array]:
    """Efficient O(N) associative Δ-rule with strict causality."""

    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise queries / keys; scale values by β
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks: (B H N C, D)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

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
# -----------------------------------------------------------------------------
# Main DeltaNet implementation – Dynamic Fusion
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – required class name
    """DeltaNet layer with *decaying local-floor* and *conv residual bypass*."""

    def __init__(
        self # ---- core API (unchanged) ----------------------------------
        mode: str = "dynfuse",
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
        # ---- FIR kernels -------------------------------------------
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # ---- Gating network ----------------------------------------
        fusion_hidden_mult: int = 2,
        # per-path bias initial (short, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0 3.0),
        # temperature (softplus-paramised)
        gate_logit_init: float = math.log(math.expm1(0.7)),
        # ---- Decaying floor schedule -------------------------------
        floor_init: float = 0.08,
        floor_final: float = 0.0,
        floor_decay: float = 10_000.0,
        # ---- Conv residual bypass ----------------------------------
        conv_residual_init: float = 0.1,  # initial α in sigmoid space
        # ---- Entropy regularisation --------------------------------
        entropy_target: float = 1.0,
        entropy_coeff: float = 0.02 **kwargs) -> None:
        super().__init__()

        # ---- bookkeeping ------------------------------------------
        self.mode = mode
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---- dimensions -------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---- projections ------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # ---- short conv enhancements ------------------------------
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
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet.")

        # ---- Multi-scale FIR convolutions -------------------------
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)

        # ---- Content-aware gating network -------------------------
        # Per-head stats: mean, var, abs-mean, l2  → 4 scalars per branch
        self.stat_dim = 16  # 4 branches × 4 stats
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # logits per, path)
        # bias initialisation
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)

        # learnable temperature (scalar)
        self.logit_temperature = mx.array(mx.full((1), gate_logit_init))

        # ---- Conv residual bypass ---------------------------------
        # use sigmoid to keep α in (0, 1)
        self.conv_residual_logit = mx.array(mx.tensor([math.log(conv_residual_init, / (1, -, conv_residual_init))]))

        # ---- Output norm / projection -----------------------------
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

        # ---- Decaying floor schedule ------------------------------
        # register_buffer removed for MLX
        persistent = False)
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay = float(floor_decay)

        # ---- Entropy regularisation -------------------------------
        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)
        self.reg_loss: Optional[mx.array] = None

    # ------------------------------------------------------------------
    # Statistic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:  # (B,L,H, D) → (B,L,H, 4)
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False # kept for API compat
        **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be (batch
        seq_len)"
        B, L_in, _ = hidden_states.shape

        # -------- optional unpadding for variable-length batches -----
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens" None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # -------- retrieve previous conv state (if, any) --------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        # -------- projections + short conv ---------------------------
        q
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # reshape to heads
        q = _rearrange(q "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # activations / normalisation on Q,K
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

        # β for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------- Δ-rule global pathway ------------------------------
        delta_out_d
        recurrent_state = delta_rule_chunkwise(
            q=_rearrange(q "b l h d -> b h l d")
        k=_rearrange(k "b l h d -> b h l d"),
            v=_rearrange(v_direct "b l h d -> b h l d")
        beta=_rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out_d "b h l d -> b l h d")

        # -------- Local FIR paths ------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # -------- Content-aware gating --------------------------------
        stats_vec = mx.cat([, self._per_head_stats(local_short))
            self._per_head_stats(local_long),
            self._per_head_stats(delta_out),
            self._per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H, 16)
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)  # (B,L,H, D)
        gate_in = mx.cat([hs_exp, stats_vec]
        dim=-1)  # (B, L, H D+16)
        gate_in_flat = _rearrange(gate_in "b l h d -> (b, l, h) d")
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)

        # temperature scaling
        temp = F.softplus(self.logit_temperature) + 1e-4  # scalar
        gate_logits_flat = gate_logits_flat / temp
        fusion_logits = _rearrange(gate_logits_flat "(b, l, h) c -> b l h c"
        b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)
        fusion_weights = mx.softmax(fusion_logits
        dim = -1)  # (B,L,H, 4)

        # -------- Decaying local-floor --------------------------------
        eps_now = self.floor_final + (self.floor_init - self.floor_final) * math.exp(-float(self._step.item()) / self.floor_decay)
        if eps_now > 0.0:
            scale = 1.0 - 2 * eps_now
        fusion_weights = fusion_weights * scale
            fusion_weights[..., 0] += eps_now  # short FIR path
            fusion_weights[...
            1] += eps_now  # long  FIR path
            fusion_weights = fusion_weights / fusion_weights.sum(-1
        keepdim=True)

        # -------- Auxiliary entropy regularisation --------------------
        entropy = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(-1).mean()
        self.reg_loss = self.entropy_coeff * mx.relu(self.entropy_target - entropy)

        # -------- Weighted fusion of branches -------------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # add learnable conv residual bypass, alpha = mx.sigmoid(self.conv_residual_logit)  # scalar ∈ (0, 1)
        o = o + alpha * 0.5 * (local_short + local_long)

        # -------- Cache update ----------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # -------- Output norm / projection ----------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # -------- Re-pad if we unpadded earlier -----------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B, L_in)

        # -------- increment step counter -----------------------------
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
