"""
MLX-converted architecture: delta_net_afef
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
DeltaNet – Adaptive Floor & Entropy Fusion (delta_net_afef)
===========================================================
Identifier: delta_net_afef

This generation focuses on solving the *late-stage over-sharpening* weakness
observed in the annealed-gate family (AEKF).  We introduce a **per-head, per-
path adaptive probability floor** that *never* fully vanishes – preserving a
small but task-critical amount of exploration signal even in the final
training phase.  The floor value follows a cosine annealing schedule from
`floor_start` → `floor_end`, where `floor_end` is strictly positive
(default = 0.01).  Each head/path additionally owns a *learnable multiplier*
(initialised so that the effective floor at *t=0* equals `floor_start`).

Key innovations (enabled by default)
-----------------------------------
1. *Adaptive non-zero floor* – prevents path starvation while still allowing
   sharp routing; the final floor magnitude is small enough (1 %) not to hurt
   precision-heavy tasks but big enough to maintain distributed reasoning.
2. *Per-head temperature* – retained from previous best variant for flexible
   sharpening.
3. *Cosine-annealed entropy regularisation* – softly keeps gate entropy above
   `entropy_target` early in training and linearly releases this pressure.

All heavy kernels (depth-wise FIR & chunked Δ-rule) remain unchanged and keep
`@mx.compile` for maximum efficiency.  The public API, constructor
arguments, and forward signature are fully preserved.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _elu_plus_one(x: mx.Tensor) -> mx.Tensor:  # noqa: D401
    """Shifted ELU so outputs are positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.Tensor) -> mx.Tensor:  # noqa: D401
    """L1 normalisation along the last dimension."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity + small noise)
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR convolution for tensors shaped (B, L, H, D)."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        kernel_size: int,
        noise_std: float = 2e-2,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.no_grad():
            weight[..., -1] = 1.0  # identity (t=0)
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
# Chunk-wise associative Δ-rule (unchanged numerics, @mx.compile)
# -----------------------------------------------------------------------------


@mx.compile  # noqa: D401 – keep optimisation
# pylint: disable=too-many-locals
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
    n_blocks = q.shape[2]

    tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    tri_strict = mx.triu(tri, 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
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
# Adaptive-floor fusion gate
# -----------------------------------------------------------------------------


class _AdaptiveFloorGate(nn.Module):
    """Fusion gate with per-head/path adaptive non-zero probability floor."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_v_dim: int,
        *,
        n_paths: int = 4,
        mlp_mult: int = 2,
        temp_init: float = 1.0,
        floor_start: float = 0.05,
        floor_end: float = 0.01,
        floor_anneal_steps: int = 2_000,
        entropy_target: float = 0.65,
        entropy_coeff: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = n_paths
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_anneal_steps = int(floor_anneal_steps)
        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)

        # step counter buffer (not a parameter) – increments each forward
        self, persistent=False)

        # learnable per-head temperature (log space)
        self.log_temp = mx.array(mx.log(mx.full((num_heads,), temp_init)))

        # learnable base logits bias (per-head, per-path)
        self.base_bias = mx.array(mx.zeros(num_heads, n_paths))
        with mx.no_grad():
            # encourage identity / value path initially (index 3)
            self.base_bias[:, 3] = 2.0

        # per-head/path raw floor parameters (sigmoid() ∈ (0,1))
        init = math.log(0.5)  # sigmoid ≈ 0.5 → initial multiplier 0.5
        self.floor_raw = mx.array(mx.full((num_heads, n_paths), init))

        # Gate MLP: inputs = hidden + flattened per-head stats (mean & var)
        stat_dim_per_path = 2  # mean & variance
        gate_in_dim = hidden_size + stat_dim_per_path * num_heads * n_paths
        hidden_dim = hidden_size * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * n_paths, bias=False),
        )

        # Exposed attributes for trainer
        self.reg_loss: Optional[mx.Tensor] = None
        self.last_entropy: Optional[float] = None

    # ----------------------------------------------
    def _cosine_anneal(self, start: float, end: float, steps: int) -> float:
        t = float(self.step.item())
        if steps <= 0 or t >= steps:
            return end
        cos_val = 0.5 * (1 + math.cos(math.pi * t / steps))
        return end + (start - end) * cos_val

    # ----------------------------------------------
    @staticmethod
    def _stats(x: mx.Tensor) -> mx.Tensor:  # (B,L,H,D) -> (B,L,H,2)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return mx.cat([mean, var], dim=-1)

    # ----------------------------------------------
    def forward(
        self,
        hidden: mx.Tensor,  # (B,L,D)
        short: mx.Tensor,  # (B,L,H,D)
        long: mx.Tensor,
        delta: mx.Tensor,
        value: mx.Tensor,
    ) -> mx.Tensor:  # returns weights (B,L,H,P)
        B, L, H, _ = short.shape
        paths = [short, long, delta, value]

        # ---------- Feature construction ----------
        stats = [self._stats(p) for p in paths]
        stats_flat = mx.cat([_rearrange(s, "b l h s -> b l (h s)") for s in stats], dim=-1)
        gate_in = mx.cat([hidden, stats_flat], dim=-1)

        logits = self.mlp(gate_in)  # (B,L,H*P)
        logits = logits + self.base_bias.reshape(1, 1, -1)
        logits = _rearrange(logits, "b l (h p) -> b l h p"h=H, p=self.n_paths)

        # temperature scaling
        temp = F.softplus(self.log_temp) + 1e-4  # (H,)
        logits = logits / temp.reshape(1, 1, H, 1)

        probs = mx.softmax(logits, dim=-1)  # (B,L,H,P)

        # ---------- adaptive floor ---------------
        floor_multiplier = mx.sigmoid(self.floor_raw)  # (H,P)
        floor_base = floor_multiplier.reshape(1, 1, H, self.n_paths)
        floor_mag = self._cosine_anneal(self.floor_start, self.floor_end, self.floor_anneal_steps)
        floor_val = floor_mag * floor_base  # (1,1,H,P)
        if floor_mag > 0:
            probs = mx.clamp(probs, min=floor_val)
            probs = probs / probs.sum(-1, keepdim=True)

        # ---------- entropy regularisation ------
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        self.last_entropy = float(entropy)
        self.reg_loss = self.entropy_coeff * mx.relu(self.entropy_target - entropy)

        # step++
        self.step += 1  # type: ignore[operator]
        return probs

# -----------------------------------------------------------------------------
# Main DeltaNet layer
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – required name
    """DeltaNet layer with Adaptive Floor & Entropy Fusion (AFEF)."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    # pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
    def __init__(
        self,
        *,
        mode: str = "afef",
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
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 63,
        # Gate hyper-params
        floor_start: float = 0.05,
        floor_end: float = 0.01,
        floor_anneal_steps: int = 2_000,
        entropy_target: float = 0.65,
        entropy_coeff: float = 0.02,
        temp_init: float = 1.0,
        fusion_mlp_mult: int = 2,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model

        # ----- basic dims -----
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # ----- flags & bookkeeping -----
        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ----- projections -----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ----- short convs -----
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution is required for DeltaNet performance.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        # ----- FIR local memories -----
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ----- Adaptive fusion gate -----
        self.fusion_gate = _AdaptiveFloorGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_v_dim=self.head_v_dim,
            temp_init=temp_init,
            floor_start=floor_start,
            floor_end=floor_end,
            floor_anneal_steps=floor_anneal_steps,
            entropy_target=entropy_target,
            entropy_coeff=entropy_coeff,
            mlp_mult=fusion_mlp_mult,
        )

        # ----- Output norm / projection -----
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

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
    ) -> Tuple[mx.Tensor, Optional[mx.Tensor], Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (B,L)"
        B_orig, L_in, _ = hidden_states.shape

        # --------------- unpadding (optional) ----------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # --------------- retrieve cache ---------------------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # --------------- projections + conv -----------------------
        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # head reshape
        q = _rearrange(q_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l (h d) -> b l h d"d=self.head_v_dim)

        # activation & norm variants
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # β factor for delta path
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------- Δ-rule global memory ---------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d, recur_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # --------------- local FIR memories ----------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # --------------- fusion gate -----------------------------
        weights = self.fusion_gate(hidden_states, local_short, local_long, delta_out, v_direct)
        mix = (
            weights[..., 0:1] * local_short
            + weights[..., 1:2] * local_long
            + weights[..., 2:3] * delta_out
            + weights[..., 3:4] * v_direct
        )
        o = mix  # residual leak removed for sharper routing

        reg_loss = self.fusion_gate.reg_loss

        # --------------- cache update ----------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # --------------- output norm / proj ----------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # --------------- re-pad if necessary ---------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B_orig, L_in)

        return o, reg_loss, past_key_values
