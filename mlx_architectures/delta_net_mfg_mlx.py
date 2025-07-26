"""
MLX-converted architecture: delta_net_mfg
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
DeltaNet – Minimum-Floor Gated Multi-Scale Memory (delta_net_mfg)
================================================================
This evolution merges the most successful ingredients discovered in prior
experiments (statistics-aware 2-layer gate, per-head/per-path temperature,
identity-initialised FIR branches, entropy regularisation) **and** introduces a
*non-negotiable* **minimum probability floor** that guarantees every path keeps
receiving gradient signal throughout training.

Motivation
~~~~~~~~~~
Earlier variants suffered from *path starvation* when the routing gate became
extremely sharp – local FIR branches were sometimes reduced to ~0 weight,
causing large regressions on detail-oriented tasks (SWDE, OpenBookQA).  A fixed
minimum floor (ε) avoids this collapse without preventing decisive routing
because the softmax output is simply re-scaled so that each path obtains at
least ε probability mass.

Key Features (all enabled by default)
-------------------------------------
1. **Statistics-aware non-linear gate** (borrowed from HTNG)
   • Hidden state + 16 statistics (mean, var, abs-mean, ℓ2 of each branch).
   • 2-layer MLP with GELU; produces 4 logits per head.

2. **Per-head / per-path temperature (τ)**
   • Learnable log-temperature vector (H×4) initialised to 0 (τ≈1).
   • Allows some heads to sharpen, others to stay soft.

3. **Hard minimum floor ε=0.05** (configurable)
   • After softmax, weights are *affinely rescaled* so that
     w′ = ε + (1-4ε)·softmax(logits/τ).
   • Ensures every branch keeps ≥ε share ⇒ permanent gradient flow.

4. **Optional entropy penalty** (disabled by default)
   • Fosters diversity when enabled without relying on adaptive schedules.

5. **Strictly O(N) complexity**
   • Δ-rule global memory and depth-wise convolutions are linear.

All public APIs, shapes and computational contracts remain unchanged.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_plus_one(x: mx.Tensor) -> mx.Tensor:
    """Shifted ELU (+1) that stays strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.Tensor) -> mx.Tensor:
    """L1 normalisation so that values sum to 1 along last dim."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution with identity (δ) kernel initialisation
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR convolution for tensors of shape (B, L, H, D)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Dirac initialisation: last tap = 1, rest = 0 (+ small noise)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.no_grad():
            filt[..., -1] = 1.0
            filt.add_(0.01 * mx.randn_like(filt))
        self.filters = mx.array(filt)  # (H, D, K)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        w = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left-pad
        y = F.conv1d(x_pad, w, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

# -----------------------------------------------------------------------------
# Chunk-wise associative Δ-rule (unchanged numerics)
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: mx.Tensor,  # [B,H,L,D]
    k: mx.Tensor,  # [B,H,L,D]
    v: mx.Tensor,  # [B,H,L,Dv]
    beta: mx.Tensor,  # [B,H,L]
    *,
    chunk_size: int = 32,
):
    """Efficient O(N) Δ-rule retrieval using chunked associative processing."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalise & apply beta
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into (B,H,N,C,D)
    q, k, v, k_beta = map(lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size), (q, k, v, k_beta))
    n_blocks = q.shape[2]

    tri_full = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    tri_strict = mx.triu(tri_full, 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + mx.eye(chunk_size, dtype=inv.dtype, inv.device)
    inv = inv

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = mx.zeros_like(v)

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
# Optional typing helper
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main DeltaNet layer – Minimum-Floor Gated variant
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401  – class name must remain exactly this
    """DeltaNet with *hard minimum floor* gated multi-scale fusion."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        mode: str = "mfg",  # minimum-floor gating
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
        # ---- FIR kernels ----
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 64,
        # ---- Gate hyper-params ----
        gate_hidden_mult: int = 2,
        min_floor: float = 0.05,  # minimum probability per path (hard)
        entropy_coeff: float = 0.0,  # optional entropy regularisation
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        # dims ---------------------------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # misc ---------------------------------------------------------
        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.min_floor = float(min_floor)
        self.entropy_coeff = float(entropy_coeff)

        # projections --------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # short convs --------------------------------------------------
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet variants.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        # FIR branches -------------------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # gate MLP -----------------------------------------------------
        stat_dim = 4  # mean, var, abs-mean, l2
        gate_in_dim = hidden_size + stat_dim * 4  # hidden + 4 branches stats
        hidden_gate_dim = hidden_size * gate_hidden_mult // 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        with mx.no_grad():
            self.gate_mlp[-1].bias.zero_()
            # bias order: short, long, delta, value
            self.gate_mlp[-1].bias[2] = 0.5  # favour delta slightly
            self.gate_mlp[-1].bias[3] = 1.5  # favour identity

        # per-head/path temperature ------------------------------------
        self.log_temp = mx.array(mx.zeros(num_heads, 4))  # τ≈1 init

        # output norm / proj ------------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # aux ----------------------------------------------------------
        self, persistent=False)
        self.reg_loss: Optional[mx.Tensor] = None

    # ------------------------------------------------------------------
    # statistics helper
    # ------------------------------------------------------------------
    @staticmethod
    def _branch_stats(x: mx.Tensor) -> mx.Tensor:
        """Return concatenated stats: mean, var, abs-mean, l2 along feature dim."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,  # [B,L,D]
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape

        # optional unpadding ------------------------------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # cache --------------------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # projections + short conv ------------------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # head reshape -------------------------------------------------
        q = _rearrange(q_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l (h d) -> b l h d"d=self.head_v_dim)

        # activations --------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # beta ---------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule -------------------------------------------------------
        delta_out_d, recurrent_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"),
        )
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # FIR branches -------------------------------------------------
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # stats --------------------------------------------------------
        stats_vec = mx.cat([
            self._branch_stats(fir_short),
            self._branch_stats(fir_long),
            self._branch_stats(delta_out),
            self._branch_stats(v_direct),
        ], dim=-1)  # [B,L,H,16]

        # gate input ---------------------------------------------------
        hid_exp = hidden_states.expand_dims(2).expand(-1, -1, self.num_heads, -1)  # [B,L,H,D]
        gate_in = mx.cat([hid_exp, stats_vec], dim=-1)  # [B,L,H,D+16]

        # gate logits --------------------------------------------------
        gate_logits = self.gate_mlp(gate_in)  # [B,L,H,4]

        # temperature scaling -----------------------------------------
        temp = mx.exp(self.log_temp).clamp(0.1, 10.0)  # [H,4]
        gate_logits = gate_logits / temp.expand_dims(0).expand_dims(0)

        soft_w = mx.softmax(gate_logits, dim=-1)  # [B,L,H,4]

        # minimum floor ------------------------------------------------
        eps = self.min_floor
        # Fix: Clamp eps and the free routing mass to safeguard numerics
        max_eps = 0.24  # Four branches: ensure 1-4*eps >= 0.04 for stability
        eps = max(0.0, min(eps, max_eps))
        free_mass = 1.0 - 4 * eps
        free_mass = max(free_mass, 1e-6)  # Prevent negative/zero scale
        soft_w = eps + free_mass * soft_w  # affine rescale ensures >=eps

        # sanity (numerical) – renormalise for precision
        soft_w = soft_w / soft_w.sum(-1, keepdim=True)

        # entropy regularisation (optional) ----------------------------
        if self.entropy_coeff > 0.0 and self.training:
            ent = -(soft_w * (soft_w + 1e-8).log()).sum(-1).mean()
            self.reg_loss = -self.entropy_coeff * ent
        else:
            self.reg_loss = None

        # fuse ---------------------------------------------------------
        o = (
            soft_w[..., 0:1] * fir_short +
            soft_w[..., 1:2] * fir_long +
            soft_w[..., 2:3] * delta_out +
            soft_w[..., 3:4] * v_direct
        )

        # cache update -------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # output norm / proj ------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # re-pad -------------------------------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B_orig, L_in)

        self._step += 1  # type: ignore[operator]
        return o, None, past_key_values
