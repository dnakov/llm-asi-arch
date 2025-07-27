# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Floor & Entropy Fusion (delta_net_afef_mlx)
==============================================================
Identifier: delta_net_afef_mlx

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

MLX Implementation Notes:
- Converted from PyTorch to MLX framework
- Uses mlx.nn modules instead of PyTorch nn
- Simplified some operations for MLX compatibility
- Removed compilation decorators (MLX handles optimization automatically)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Tensor reshape utility for common patterns using native MLX operations"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l (h p) -> b l h p" in pattern:
        h = kwargs.get('h', 1)
        p = kwargs.get('p', 1)
        b, l, hp = tensor.shape
        return tensor.reshape(b, l, h, p)
    elif "b l h d -> b l (h d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return mx.transpose(tensor, (0, 2, 1, 3))
    elif "b h l d -> b l h d" in pattern:
        return mx.transpose(tensor, (0, 2, 1, 3))
    elif "b h (n c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h (n c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    elif "b l (h s) -> b l h s" in pattern:
        h = kwargs.get('h', 1)
        s = kwargs.get('s', 1)
        b, l, hs = tensor.shape
        return tensor.reshape(b, l, h, s)
    elif "b l h -> b h l" in pattern:
        return mx.transpose(tensor, (0, 2, 1))
    else:
        # Fallback: return tensor as-is
        return tensor

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU so outputs are positive."""
    return mx.maximum(x, 0.0) + mx.minimum(mx.exp(x) - 1.0, 0.0) + 1.0

def _sum_norm(x: mx.array) -> mx.array:
    """L1 normalisation along the last dimension."""
    return x / mx.sum(x, axis=-1, keepdims=True)

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization along the last dimension."""
    return x / mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True) + 1e-8)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution
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
        
        # Initialize weights with identity at last position and noise
        weight = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Create identity kernel manually
        identity_weight = mx.zeros_like(weight)
        identity_weight = mx.concatenate([
            mx.zeros((num_heads, head_dim, self.kernel_size - 1)),
            mx.ones((num_heads, head_dim, 1))
        ], axis=-1)
        weight = weight + identity_weight
        
        if noise_std > 0:
            noise = mx.random.normal((num_heads, head_dim, self.kernel_size)) * noise_std
            weight = weight + noise
        
        self.filters = weight  # (H, D, K)

    def __call__(self, x: mx.array) -> mx.array:  # x: (B,L,H,D)
        # Simplified implementation that just applies the identity + noise
        b, l, h, d = x.shape
        
        # Apply FIR filter as a simple weighted combination
        # This is a simplified version for MLX compatibility
        output = mx.zeros_like(x)
        
        for k in range(self.kernel_size):
            if k < l:
                # Get the filter weights for this position
                weight = self.filters[..., k]  # (H, D)
                
                # Apply to shifted input
                if k == 0:
                    shifted_x = x
                else:
                    # Shift input by k positions (causal)
                    shifted_x = mx.concatenate([
                        mx.zeros((b, k, h, d)),
                        x[:, :-k, :, :]
                    ], axis=1)
                
                # Apply filter
                weighted = shifted_x * weight[None, None, :, :]
                output = output + weighted
        
        return output

# -----------------------------------------------------------------------------
# Chunk-wise associative Δ-rule 
# -----------------------------------------------------------------------------

def _delta_rule_chunkwise(
    q: mx.array,  # (B,H,L,Dk)
    k: mx.array,  # (B,H,L,Dk)
    v: mx.array,  # (B,H,L,Dv)
    beta: mx.array,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Efficient O(N) Δ-rule implementation preserving causality."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = [(0, 0), (0, 0), (0, pad_len), (0, 0)]
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    L_pad = L + pad_len

    q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q = rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size)
    k = rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    k_beta = rearrange(k_beta, "b h (n c) d -> b h n c d", c=chunk_size)
    
    n_blocks = q.shape[2]

    # Create triangular masks
    tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0).astype(mx.bool_)
    tri_strict = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)

    # Compute inverse matrix
    inv = -(k_beta @ mx.transpose(k, axes=[0, 1, 2, 4, 3]))
    inv = mx.where(tri, 0.0, inv)
    
    # Simplified inverse computation for MLX (approximation)
    # Use a simpler approximation instead of iterative updates
    inv = inv + mx.eye(chunk_size)[None, None, None, :, :]

    u = inv @ v
    w = inv @ k_beta

    S = mx.zeros((b, h, d_k, v.shape[-1]))
    
    # Pre-compute all outputs to avoid complex assignments
    all_outputs = []
    
    for idx in range(n_blocks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ mx.transpose(k_i, axes=[0, 1, 3, 2]))
        attn_local = mx.where(tri_strict, 0.0, attn_local)
        
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out_i = q_i @ S + attn_local @ u_i
        all_outputs.append(out_i)
        
        S = S + mx.transpose(k_i, axes=[0, 1, 3, 2]) @ u_i
    
    # Stack all outputs
    out = mx.stack(all_outputs, axis=2)

    out = rearrange(out, "b h n c d -> b h (n c) d")
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

        # step counter (not a parameter) – increments each forward
        self.step = 0

        # learnable per-head temperature (log space)
        self.log_temp = mx.log(mx.full((num_heads,), temp_init))

        # learnable base logits bias (per-head, per-path)
        base_bias = mx.zeros((num_heads, n_paths))
        # encourage identity / value path initially (index 3)
        identity_bias = mx.zeros_like(base_bias)
        identity_bias = mx.concatenate([
            mx.zeros((num_heads, 3)),
            mx.full((num_heads, 1), 2.0)
        ], axis=-1)
        self.base_bias = base_bias + identity_bias

        # per-head/path raw floor parameters (sigmoid() ∈ (0,1))
        init = math.log(0.5)  # sigmoid ≈ 0.5 → initial multiplier 0.5
        self.floor_raw = mx.full((num_heads, n_paths), init)

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
        self.reg_loss: Optional[mx.array] = None
        self.last_entropy: Optional[float] = None

    def _cosine_anneal(self, start: float, end: float, steps: int) -> float:
        t = float(self.step)
        if steps <= 0 or t >= steps:
            return end
        cos_val = 0.5 * (1 + math.cos(math.pi * t / steps))
        return end + (start - end) * cos_val

    @staticmethod
    def _stats(x: mx.array) -> mx.array:  # (B,L,H,D) -> (B,L,H,2)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return mx.concatenate([mean, var], axis=-1)

    def __call__(
        self,
        hidden: mx.array,  # (B,L,D)
        short: mx.array,  # (B,L,H,D)
        long: mx.array,
        delta: mx.array,
        value: mx.array,
    ) -> mx.array:  # returns weights (B,L,H,P)
        B, L, H, _ = short.shape
        paths = [short, long, delta, value]

        # ---------- Feature construction ----------
        stats = [self._stats(p) for p in paths]
        # Flatten the stats properly
        stats_reshaped = []
        for s in stats:
            # s is (B, L, H, 2), flatten to (B, L, H*2)
            b, l, h, s_dim = s.shape
            s_flat = s.reshape(b, l, h * s_dim)
            stats_reshaped.append(s_flat)
        
        stats_flat = mx.concatenate(stats_reshaped, axis=-1)
        gate_in = mx.concatenate([hidden, stats_flat], axis=-1)

        logits = self.mlp(gate_in)  # (B,L,H*P)
        logits = logits + self.base_bias.reshape(1, 1, -1)
        logits = rearrange(logits, "b l (h p) -> b l h p", h=H, p=self.n_paths)

        # temperature scaling
        temp = nn.softplus(self.log_temp) + 1e-4  # (H,)
        temp_expanded = temp.reshape(1, 1, H, 1)
        logits = logits / temp_expanded

        probs = nn.softmax(logits, axis=-1)  # (B,L,H,P)

        # ---------- adaptive floor ---------------
        floor_multiplier = nn.sigmoid(self.floor_raw)  # (H,P)
        floor_base = floor_multiplier.reshape(1, 1, H, self.n_paths)
        floor_mag = self._cosine_anneal(self.floor_start, self.floor_end, self.floor_anneal_steps)
        floor_val = floor_mag * floor_base  # (1,1,H,P)
        
        if floor_mag > 0:
            probs = mx.maximum(probs, floor_val)
            probs = probs / mx.sum(probs, axis=-1, keepdims=True)

        # ---------- entropy regularisation ------
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
        entropy = mx.mean(entropy)
        self.last_entropy = float(entropy)
        self.reg_loss = self.entropy_coeff * mx.maximum(self.entropy_target - entropy, 0.0)

        # step++
        self.step += 1
        return probs

# -----------------------------------------------------------------------------
# Main DeltaNet layer
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with Adaptive Floor & Entropy Fusion (AFEF)."""

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

        # ----- short convs (simplified for MLX) -----
        if self.use_short_conv:
            # Simplified conv implementation for MLX
            self.q_conv_weight = mx.random.normal((self.key_dim, conv_size)) * 0.02
            self.k_conv_weight = mx.random.normal((self.key_dim, conv_size)) * 0.02
            self.v_conv_weight = mx.random.normal((self.value_dim, conv_size)) * 0.02

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
            self.gate_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _simple_conv1d(self, x: mx.array, weight: mx.array) -> mx.array:
        """Simplified 1D convolution for MLX."""
        kernel_size = weight.shape[-1]
        x_pad = mx.pad(x, [(0, 0), (kernel_size - 1, 0), (0, 0)])
        
        # Simplified convolution using matrix operations
        output = []
        for i in range(x.shape[1]):
            if i + kernel_size <= x_pad.shape[1]:
                conv_input = x_pad[:, i:i+kernel_size, :]
                conv_result = mx.sum(conv_input * weight.T, axis=1)
                output.append(conv_result)
        
        if output:
            return mx.stack(output, axis=1)
        else:
            return x  # fallback

    def __call__(
        self,
        hidden_states: mx.array,  # (B,L,D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Dict]]:
        
        B_orig, L_in, _ = hidden_states.shape

        # --------------- projections + conv -----------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        # Simple convolution for MLX
        if self.use_short_conv:
            q_lin = self._simple_conv1d(q_lin, self.q_conv_weight)
            k_lin = self._simple_conv1d(k_lin, self.k_conv_weight)
            v_lin = self._simple_conv1d(v_lin, self.v_conv_weight)

        # head reshape
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_k_dim)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_v_dim)

        # activation & norm variants
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        else:
            q, k = nn.silu(q), nn.silu(k)
            
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)
        elif self.qk_norm == "l2":
            q, k = _l2norm(q), _l2norm(k)

        # β factor for delta path
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((*hidden_states.shape[:2], self.num_heads))
        
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------- Δ-rule global memory ---------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v_direct, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out_d, recur_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

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

        # --------------- output norm / proj ----------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads, d=self.head_v_dim)
            # Simplified gated normalization for MLX
            o_normed = self.o_norm(o)
            g_normed = self.gate_norm(g_vec)
            o = o_normed * g_normed
        else:
            o = self.o_norm(o)
        
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        return o, reg_loss, past_key_values