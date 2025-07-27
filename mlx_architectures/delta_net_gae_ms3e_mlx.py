# -*- coding: utf-8 -*-
"""
DeltaNet – Head-Grouped Adaptive Multi-Statistic Gating with Explicit Entropy Regularization (delta_net_gae_ms3e)
==============================================================================================================
Breakthrough DeltaNet evolution synthesizing direct lessons from MS-DPAF, HMSMG, MSHMF, MS-GMix-RS,
magnetoresistive adaptive gating, and latest mixture/model-of-experts/GLA research. Implements these core advances:

(see original header for the detailed description of the research motivation)
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict
import mlx.core as mx
import mlx.nn as nn


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def elu_p1(x: mx.array) -> mx.array:
    """ELU(x)+1 helper used in several DeltaNet variants"""
    return mx.where(x > 0, x + 1.0, mx.exp(x))


def sum_norm(x: mx.array) -> mx.array:
    """Normalise a tensor so that the last‐dim sums to 1"""
    return x / mx.sum(x, axis=-1, keepdims=True)


def l2norm(x: mx.array) -> mx.array:
    """L2 normalize along the last dimension"""
    return x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-8)


# -----------------------------------------------------------------------------
# Depth-wise FIR block
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        kernel_size: int = 64,
        noise_std: float = 2e-2,
        alt_noise_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Identity initialisation (delta kernel)
        filters_init = mx.zeros((num_heads, head_dim, self.kernel_size))
        filters_init = filters_init.at[:, :, -1].set(1.0)
        if alt_noise_type == "orthogonal":
            # Add small signed orthogonal noise so each head starts decorrelated
            sign_flips = mx.random.randint(0, 2, filters_init.shape) * 2 - 1
            filters_init = filters_init + sign_flips * noise_std
        else:
            filters_init = filters_init + noise_std * mx.random.normal(filters_init.shape)
        self.filters = mx.array(filters_init)

    def __call__(self, x: mx.array) -> mx.array:  # (b, l, h, d)
        b, l, h, d = x.shape
        x_f = x.transpose(0, 2, 3, 1).reshape(b, h * d, l)  # (b, h*d, l)
        weight = self.filters.reshape(h * d, 1, self.kernel_size)
        
        # Causal padding
        pad_zeros = mx.zeros((b, h * d, self.kernel_size - 1))
        x_pad = mx.concatenate([pad_zeros, x_f], axis=2)
        
        # Manual grouped convolution implementation
        y = mx.zeros((b, h * d, l))
        for i in range(h * d):
            for j in range(l):
                start_idx = j
                end_idx = j + self.kernel_size
                kernel_vals = weight[i, 0, :]
                input_vals = x_pad[:, i, start_idx:end_idx]
                y = y.at[:, i, j].set(mx.sum(input_vals * kernel_vals, axis=-1))
        
        return y.reshape(b, h, d, l).transpose(0, 3, 1, 2)  # (b, l, h, d)


# -----------------------------------------------------------------------------
# Causal Chunk-wise Delta-rule core
# -----------------------------------------------------------------------------

def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Chunk-wise implementation of O(N) Delta-rule with strict causality."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_zeros_q = mx.zeros((b, h, pad_len, d_k))
        pad_zeros_beta = mx.zeros((b, h, pad_len))
        q = mx.concatenate([q, pad_zeros_q], axis=2)
        k = mx.concatenate([k, pad_zeros_q], axis=2)
        v = mx.concatenate([v, pad_zeros_q], axis=2)
        beta = mx.concatenate([beta, pad_zeros_beta], axis=2)
    L_pad = L + pad_len

    # Normalise queries / keys
    q = l2norm(q)
    k = l2norm(k)

    # Apply beta gating to values and keys
    v = v * mx.expand_dims(beta, -1)
    k_beta = k * mx.expand_dims(beta, -1)

    # Reshape into (num_chunks, chunk_size)
    num_chunks = L_pad // chunk_size
    q = q.reshape(b, h, num_chunks, chunk_size, d_k)
    k = k.reshape(b, h, num_chunks, chunk_size, d_k)
    v = v.reshape(b, h, num_chunks, chunk_size, d_k)
    k_beta = k_beta.reshape(b, h, num_chunks, chunk_size, d_k)

    # Pre-compute shared attention helper matrices (causal within chunk)
    mask_full = mx.triu(mx.ones((chunk_size, chunk_size)), k=0).astype(mx.bool_)
    attn = -(k_beta @ k.transpose(0, 1, 2, 4, 3))
    attn = mx.where(mask_full, 0, attn)
    
    for i in range(1, chunk_size):  # incremental cumulative sum trick
        prev_sum = mx.sum(attn[..., i:i+1, :] * attn[..., :, :i], axis=-2)
        attn = attn.at[..., i, :i].add(prev_sum)
    
    attn = attn + mx.eye(chunk_size)

    u = attn @ v
    w = attn @ k_beta

    # Running state S initialised to zeros
    S = mx.zeros((b, h, d_k, v.shape[-1]))
    o = mx.zeros_like(v)

    causal_mask = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)
    for idx in range(num_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = q_i @ k_i.transpose(0, 1, 3, 2)
        attn_local = mx.where(causal_mask, 0, attn_local)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o = o.at[:, :, idx].set(o_inter + attn_local @ u_i)
        S = S + k_i.transpose(0, 1, 3, 2) @ u_i

    o = o.reshape(b, h, L_pad, d_k)
    if pad_len:
        o = o[:, :, :L]
    return o, S


# -----------------------------------------------------------------------------
# Per-head Grouped Multi-Statistic Fusion Gate
# -----------------------------------------------------------------------------

class HeadGroupedFusionGate(nn.Module):
    """Per-head adaptive fusion gate that consumes (mean, rms, max) statistics."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_v_dim: int,
        fusion_hidden_mult: int = 2,
        fusion_dropout: float = 0.0,
        temp_init: float = 1.0,
        entropy_reg: float = 0.02,
        epsilon_floor_init: float = 0.01,
        eps_floor_learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_v_dim = head_v_dim
        self.entropy_reg = entropy_reg
        self.n_branches = 4
        self.stat_feat_per_branch = 3  # mean, rms, max

        gate_in_dim = (
            hidden_size  # hidden state
            + self.stat_feat_per_branch * self.head_v_dim * self.n_branches  # stats
            + self.head_v_dim * self.n_branches  # raw branch outputs
        )

        # Shared MLP that will be reused for every head (weight-tying)
        self.gate_linear1 = nn.Linear(gate_in_dim, fusion_hidden_mult * head_v_dim)
        self.gate_linear2 = nn.Linear(fusion_hidden_mult * head_v_dim, self.n_branches)
        self.fusion_dropout = fusion_dropout

        # Per-head, per-branch epsilon floor (learnable or fixed)
        if eps_floor_learnable:
            self.eps_floor = mx.array(
                mx.ones((num_heads, self.n_branches)) * epsilon_floor_init
            )
        else:
            self.eps_floor = mx.ones((num_heads, self.n_branches)) * epsilon_floor_init

        # Learnable softmax temperatures (one per head)
        self.temp = mx.array(mx.ones(num_heads) * temp_init)

        # For external logging of entropy regulariser
        self.last_entropy: Optional[mx.array] = None

    def _stat_feats(self, x: mx.array) -> mx.array:
        """Return per-feature broadcast of (mean, rms, max) statistics."""
        mean = mx.mean(x, axis=-1, keepdims=True)
        rms = mx.sqrt(mx.maximum(mx.mean(x * x, axis=-1, keepdims=True), 1e-8))
        maxv = mx.max(x, axis=-1, keepdims=True)
        # broadcast to feature dimension and concatenate => (b, l, 3*d)
        mean_expanded = mx.broadcast_to(mean, x.shape)
        rms_expanded = mx.broadcast_to(rms, x.shape)
        maxv_expanded = mx.broadcast_to(maxv, x.shape)
        return mx.concatenate([mean_expanded, rms_expanded, maxv_expanded], axis=-1)

    def __call__(self, hidden: mx.array, branches):
        b, l, h, d = branches[0].shape
        assert h == self.num_heads and d == self.head_v_dim, "Branch shape mismatch"

        fusion_weights = []
        entropy_acc: Optional[mx.array] = None

        for i in range(h):  # loop over heads to preserve numerical stability
            # Gather per-head branch outputs (b, l, d)
            pathouts = [br[:, :, i, :] for br in branches]
            # Statistics for each path (b, l, 3*d)
            stat_feats = [self._stat_feats(p) for p in pathouts]
            # Concatenate hidden state, per-branch statistics and raw outputs
            head_in = mx.concatenate([hidden, *stat_feats, *pathouts], axis=-1)  # (b, l, gate_in_dim)

            x = self.gate_linear1(head_in)
            x = nn.gelu(x)
            if self.fusion_dropout > 0.0:
                # Simple dropout approximation for MLX
                mask = mx.random.bernoulli(1 - self.fusion_dropout, x.shape)
                x = x * mask / (1 - self.fusion_dropout)
            logits = self.gate_linear2(x)

            # Temperature-scaled softmax (per-head temperature)
            t = mx.clip(self.temp[i], 0.2, 10.0)
            weights = mx.softmax(logits / t, axis=-1)

            # Apply learnable epsilon floor to keep every path alive
            floor = mx.clip(self.eps_floor[i], 1e-7, 0.1)  # (n_branches,)
            weights = mx.maximum(weights, floor)
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)

            # Entropy (for regularisation / logging)
            entropy = -mx.sum(weights * mx.log(weights + 1e-8), axis=-1)
            entropy = mx.mean(entropy)
            if entropy_acc is None:
                entropy_acc = entropy
            else:
                entropy_acc = entropy_acc + entropy

            fusion_weights.append(mx.expand_dims(weights, 2))  # (b, l, 1, n_branches)

        # Stack back to (b, l, h, n_branches)
        all_weights = mx.concatenate(fusion_weights, axis=2)
        if entropy_acc is not None:
            self.last_entropy = entropy_acc / h
        return all_weights


# -----------------------------------------------------------------------------
# DeltaNet main module
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with grouped multi-statistic adaptive fusion gating, dual FIR memory, and explicit entropy regularisation."""

    def __init__(
        self,
        mode: str = "gae_ms3e",
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
        fusion_hidden_mult: int = 2,
        fusion_dropout: float = 0.0,
        fusion_temp_init: float = 1.0,
        fusion_entropy_reg: float = 0.02,
        fusion_epsilon_floor: float = 0.01,
        fusion_eps_floor_learnable: bool = True,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        # Store config
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        # Derived dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Dimension mismatch"

        # ---------------------------------------
        # Projections
        # ---------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.use_beta = use_beta
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------------------------------
        # Short convolutions (simplified for MLX)
        # ---------------------------------------
        if use_short_conv:
            # Simplified conv implementation for MLX
            self.q_conv_weight = mx.array(mx.random.normal((self.key_dim, conv_size)) * 0.02)
            self.k_conv_weight = mx.array(mx.random.normal((self.key_dim, conv_size)) * 0.02)
            self.v_conv_weight = mx.array(mx.random.normal((self.value_dim, conv_size)) * 0.02)
            if conv_bias:
                self.q_conv_bias = mx.array(mx.zeros(self.key_dim))
                self.k_conv_bias = mx.array(mx.zeros(self.key_dim))
                self.v_conv_bias = mx.array(mx.zeros(self.value_dim))
            else:
                self.q_conv_bias = None
                self.k_conv_bias = None
                self.v_conv_bias = None

        # ---------------------------------------
        # Dual depth-wise FIR memory
        # ---------------------------------------
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_size=fir_short_kernel,
            noise_std=2e-2,
            alt_noise_type="orthogonal",
        )
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_size=fir_long_kernel,
            noise_std=2e-2,
            alt_noise_type="orthogonal",
        )

        # ---------------------------------------
        # Grouped fusion gate
        # ---------------------------------------
        self.fusion_gate = HeadGroupedFusionGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_v_dim=self.head_v_dim,
            fusion_hidden_mult=fusion_hidden_mult,
            fusion_dropout=fusion_dropout,
            temp_init=fusion_temp_init,
            entropy_reg=fusion_entropy_reg,
            epsilon_floor_init=fusion_epsilon_floor,
            eps_floor_learnable=fusion_eps_floor_learnable,
        )

        # ---------------------------------------
        # Output processing
        # ---------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _apply_conv1d(self, x: mx.array, weight: mx.array, bias: Optional[mx.array] = None) -> mx.array:
        """Simple causal 1D convolution for MLX"""
        b, l, d = x.shape
        k = weight.shape[1]
        
        # Pad for causal convolution
        pad_zeros = mx.zeros((b, k - 1, d))
        x_padded = mx.concatenate([pad_zeros, x], axis=1)
        
        # Apply convolution
        output = mx.zeros((b, l, d))
        for i in range(l):
            for j in range(k):
                if i + j < l:
                    output = output.at[:, i, :].add(x_padded[:, i + k - 1 - j, :] * weight[:, j])
        
        if bias is not None:
            output = output + bias
        
        return output

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[dict]]:
        
        batch_size, seq_len, _ = hidden_states.shape

        # --------------------------------------------------------------
        # Linear projections followed by causal short convolutions
        # --------------------------------------------------------------
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.use_short_conv:
            q = self._apply_conv1d(q, self.q_conv_weight, self.q_conv_bias)
            k = self._apply_conv1d(k, self.k_conv_weight, self.k_conv_bias)
            v = self._apply_conv1d(v, self.v_conv_weight, self.v_conv_bias)
            
            # Apply SiLU activation
            q = nn.silu(q)
            k = nn.silu(k)
            v = nn.silu(v)

        # --------------------------------------------------------------
        # Head split
        # --------------------------------------------------------------
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)

        # --------------------------------------------------------------
        # Optional activation / normalisation for q & k
        # --------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # --------------------------------------------------------------
        # Beta gating vector
        # --------------------------------------------------------------
        if self.use_beta:
            beta = mx.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((batch_size, seq_len, self.num_heads))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------------------------------------------------------
        # Delta path (O(N))
        # --------------------------------------------------------------
        q_d = q.transpose(0, 2, 1, 3)  # (b, h, l, d)
        k_d = k.transpose(0, 2, 1, 3)  # (b, h, l, d)
        v_d = v.transpose(0, 2, 1, 3)  # (b, h, l, d)
        beta_d = beta.transpose(0, 2, 1)  # (b, h, l)
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = delta_out.transpose(0, 2, 1, 3)  # (b, l, h, d)

        # --------------------------------------------------------------
        # Local memory paths: short & long FIR convolutions & direct v
        # --------------------------------------------------------------
        v_direct = v
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # --------------------------------------------------------------
        # Grouped / statistics-aware fusion gate
        # --------------------------------------------------------------
        fusion_weights = self.fusion_gate(
            hidden_states, [local_short, local_long, delta_out, v_direct]
        )  # (b, l, h, 4)

        o = (
            mx.expand_dims(fusion_weights[..., 0], -1) * local_short
            + mx.expand_dims(fusion_weights[..., 1], -1) * local_long
            + mx.expand_dims(fusion_weights[..., 2], -1) * delta_out
            + mx.expand_dims(fusion_weights[..., 3], -1) * v_direct
        )

        # --------------------------------------------------------------
        # Output projection
        # --------------------------------------------------------------
        if self.use_gate:
            g_vec = self.g_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)
            o = o * mx.sigmoid(g_vec)

        o = o.reshape(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)

        # Expose entropy for external regularisation
        self.last_fusion_entropy = self.fusion_gate.last_entropy

        return o, None, past_key_values