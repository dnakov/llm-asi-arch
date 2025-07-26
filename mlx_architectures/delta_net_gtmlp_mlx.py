"""
MLX-converted architecture: delta_net_gtmlp
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
DeltaNet – Group-Temperature Per-Head MLP Gating (DeltaNet-GTMLP)
=================================================================
Identifier: *delta_net_gtmlp*

This evolution merges the strongest ideas from previous DeltaNet
variants and state-of-the-art research on conditional routing :

1.  **Per-Head Two-Layer MLP Gate**
    •  Each head owns an independent two-layer GELU MLP that receives
       the token hidden state **plus rich per-branch statistics**
       (mean, power, abs-mean, L2) from all four memory pathways
       (Short-FIR, Long-FIR, Δ-rule, direct value).
    •  The head dimension is *folded into the batch* so only two small
       linear layers are required for *all* heads – parameter efficient
       yet fully decoupled.

2.  **Group-Wise Learnable Temperature**
    •  A log-temperature parameter is *shared across small groups of
       heads* (default `group_size = 2`).  This prevents over-fragmented
       specialisation while still allowing sharp routing where needed.
    •  τ is obtained with `softplus` so positivity is guaranteed and the
       gate sharpness can be learned end-to-end.

3.  **Light Probability Floor (Optional)**
    •  A tiny *global* floor ε (default `0.0`) can be enabled to ensure
       non-starvation of rarely used paths without imposing the strong
       leakage that hampered previous designs.

4.  **Richer Gate Evidence**
    •  Four statistics × four branches ⇒ 16-d evidence vector per head &
       token, giving the gate sufficient signal to discriminate between
       local, global and value memories – addressing the under-powered
       mean/std gate of BSCGF.

5.  **Everything else inherits from the proven MSDAF/HEGM core** –
   chunk-wise Δ-rule (O(N)), short convolutions, depth-wise FIR memory,
   batch-size agnosticism, strict causality and @mx.compile for the
   heavy kernel.

All new features are on **by default** and require no external config
changes.  Complexity remains linear; parameter increase is <0.3 %.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# ----------------------------------------------------------------------------
# Helper statistics ----------------------------------------------------------
# ----------------------------------------------------------------------------

def _stat_features(t: mx.Tensor) -> mx.Tensor:
    """Return 4 scalar features per token & head [μ, power, |μ|, L2]."""
    # mean over feature dim
    mean = t.mean(dim=-1, keepdim=True)
    power = (t ** 2).mean(dim=-1, keepdim=True)
    abs_mean = t.abs().mean(dim=-1, keepdim=True)
    l2 = t.norm(dim=-1, keepdim=True)
    return mx.cat([mean, power, abs_mean, l2], dim=-1)  # (B,L,H,4)


# ----------------------------------------------------------------------------
# Depth-wise causal FIR convolution -----------------------------------------
# ----------------------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D causal FIR convolution (per head/channel)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Small random filters (identity is given by value path)
        self.filters = mx.array(mx.randn(num_heads, head_dim, self.kernel_size) * 0.02)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        weight = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)


# ----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged, O(N)) ---------------------------------
# ----------------------------------------------------------------------------
@mx.compile  # noqa: D401
# pylint: disable=too-many-locals,too-many-statements

def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Associative Δ-rule with strict causal masking – O(N)."""
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
        lambda x: _rearrange(x, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_chunk = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_chunk, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)

    attn = attn + mx.eye(chunk_size, dtype=q.dtype, q.device)
    attn = attn

    u = attn @ v
    w = attn @ k_beta

    S = q.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)

    excl_mask = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(excl_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S


# ----------------------------------------------------------------------------
# Main DeltaNet layer --------------------------------------------------------
# ----------------------------------------------------------------------------
class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with Group-Temperature Per-Head MLP fusion gate (GTMLP)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        mode: str = "gtmlp",
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
        fir_kernel_short: int = 5,
        fir_kernel_long: int = 64,
        # Gate specifics
        gate_hidden_mult: float = 0.5,
        group_size: int = 2,
        min_floor: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # ---- bookkeeping ----
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
        self.group_size = max(1, int(group_size))
        self.min_floor = float(min_floor)

        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}
        assert num_heads % self.group_size == 0, "num_heads must be divisible by group_size"

        # ---- dimensions ----
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("key/value dims must divide num_heads")

        # ---- projections ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- short conv preprocessing ----
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet-GTMLP.")

        # ---- depth-wise FIR conv branches ----
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_short)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_long)

        # ---- fusion gate -------------------------------------------------
        branch_stat_dim = 4  # stats per branch per head (scalar features)
        branches = 4  # short, long, delta, value
        gate_input_dim = hidden_size + branch_stat_dim * branches
        gate_hidden_dim = max(8, int(gate_input_dim * gate_hidden_mult))

        self.gate_fc1 = nn.Linear(gate_input_dim, gate_hidden_dim, bias=True)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, 4, bias=True)  # per-head later via fold
        nn.init.zeros_(self.gate_fc1.bias)
        # Bias initialisation: favour value early, slight for delta
        with mx.no_grad():
            self.gate_fc2.bias[:] = mx.tensor([-0.5, -0.5, 0.5, 1.5])

        # group-wise temperature
        num_groups = num_heads // self.group_size
        self.log_tau = mx.array(mx.zeros(num_groups))  # τ ≈ 1.0 initially

        # optional global floor
        self)

        # ---- output norm & projection ----
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # interface placeholder
        **kwargs,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B0, L0, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L0:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # ---- projections and short conv ----
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---- head split & activation ----
        q = _rearrange(q, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k, "b l (h d) -> b l h d"d=self.head_k_dim)
        v = _rearrange(v, "b l (h d) -> b l h d"d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = (F.elu(q, 1.0, False) + 1.0), (F.elu(k, 1.0, False) + 1.0)
            # identity handled implicitly
        if self.qk_norm == "sum":
            q = q / q.sum(-1, keepdim=True)
            k = k / k.sum(-1, keepdim=True)

        v_direct = v  # value path

        # ---- beta scaling for Δ-rule ----
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule computation
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # FIR branches
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # ---- gate input features ----
        stats_short = _stat_features(fir_short)  # (B,L,H,4)
        stats_long = _stat_features(fir_long)
        stats_delta = _stat_features(delta_out)
        stats_value = _stat_features(v_direct)

        gate_in = mx.cat([
            hidden_states.expand_dims(2).expand(-1, -1, self.num_heads, -1),  # (B,L,H,D)
            stats_short,
            stats_long,
            stats_delta,
            stats_value,
        ], dim=-1)  # (B,L,H,D+16)

        B, L, H, Fin = gate_in.shape
        gate_in_flat = _rearrange(gate_in, "b l h f -> (b l h) f")

        # ---- per-head MLP (parameters shared, head folded into batch) ----
        x = F.gelu(self.gate_fc1(gate_in_flat))
        logits = self.gate_fc2(x)  # (B*L*H, 4)
        logits = _rearrange(logits, "(b l h) c -> b l h c"b=B, l=L, h=H)

        # group-wise temperature scaling
        num_groups = self.num_heads // self.group_size
        tau = F.softplus(self.log_tau) + 1e-3  # (G,)
        # map head index to group index
        head_ids = mx.arange(self.num_heads, logits.device)
        group_ids = head_ids // self.group_size  # (H,)
        tau_per_head = tau[group_ids]  # (H,)
        logits = logits / tau_per_head.reshape(1, 1, H, 1)

        probs = mx.softmax(logits, dim=-1)
        if self.min_floor > 0.0:
            probs = probs.clamp(min=self.min_floor)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # ---- fuse outputs ----
        o = (
            probs[..., 0:1] * fir_short
            + probs[..., 1:2] * fir_long
            + probs[..., 2:3] * delta_out
            + probs[..., 3:4] * v_direct
        )

        # ---- cache update ----
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L0,
            )

        # ---- output norm & projection ----
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---- re-pad if we unpadded earlier ----
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B0, L0)

        return o, None, past_key_values
