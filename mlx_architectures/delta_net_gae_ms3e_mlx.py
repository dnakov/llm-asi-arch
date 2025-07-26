from __future__ import annotations

"""
MLX-converted architecture: delta_net_gae_ms3e
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
DeltaNet – Head-Grouped Adaptive Multi-Statistic Gating with Explicit Entropy Regularization (delta_net_gae_ms3e)
Breakthrough DeltaNet evolution synthesizing direct lessons from MS-DPAF, HMSMG, MSHMF, MS-GMix-RS,
magnetoresistive adaptive gating and latest mixture/model-of-experts/GLA research. Implements these core advances:

(see original header for the detailed description of the research, motivation)
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def elu_p1(x: mx.array) -> mx.array:
    """ELU(x)+1 helper used in several DeltaNet variants"""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:
    """Normalise a tensor so that the last‐dim sums to 1"""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise FIR block (unchanged)
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    def __init__(
        self num_heads:, int,
        head_dim: int,
        kernel_size: int = 64,
        noise_std: float = 2e-2
        alt_noise_type: str = "orthogonal") -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filters = mx.array(mx.zeros(num_heads, head_dim, self.kernel_size))
        with mx.disable_grad():
            # Identity initialisation (delta, kernel)
            self.filters[..., -1] = 1.0
            if
            alt_noise_type == "orthogonal":
                # Add small signed orthogonal noise so each head starts decorrelated
        sign_flips = mx.randint(0, 2, self.filters.shape self.filters.device) * 2 - 1
                self.filters.add_(sign_flips * noise_std)
            else:
                self.filters.add_(noise_std * mx.randn_like(self.filters))

    def forward(self x: mx.array) -> mx.array:  # (b, l, h, d)
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal padding
        y = F.conv1d(x_pad
        weight=weight
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Causal Chunk-wise Delta-rule core
# -----------------------------------------------------------------------------

@mx.compile  # noqa: E305 – keep compile for speed
def delta_rule_chunkwise(q, k, v, beta chunk_size: int = 32):
    """Chunk-wise implementation of O(N) Delta-rule with strict causality."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)  # only pad sequence dimension
        q = mx.pad(q, pad)
        k = mx.pad(k, pad)
        v = mx.pad(v, pad)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise queries / keys
        q = _l2norm(q)
    k = _l2norm(k)

    # Apply beta gating to values and keys
        v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into (num_chunks, chunk_size)
    q, k, v, k_beta = map(
        lambda x: _rearrange(x "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    # Pre-compute shared attention helper matrices (causal within, chunk)
    mask_full = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=0
    )
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_full, 0)
    for i in range(1, chunk_size):  # incremental cumulative sum trick
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None] * attn[..., :, :i]
        ).sum(-2), # Note: keep dtype consistent with k_beta / v to avoid matmul type mismatch, attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)

    # ----------------------------------------------------------------------------
    # IMPORTANT: Do NOT cast `attn` to bfloat16 unilaterally. This caused dtype
    # mismatches with `v` (float32) during the following matrix multiplications # leading to runtime errors. Keeping `attn` in the same dtype as the value
    # tensors guarantees safe and efficient execution while still allowing users
    # to employ mixed-precision training frameworks (e.g. mx.autocast) if
    # desired.
    # ----------------------------------------------------------------------------

    u = attn @ v
        w = attn @ k_beta

    # Running state S initialised to zeros, S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)

    causal_mask = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=1
    )
    for idx in range(L_pad // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(causal_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Per-head Grouped Multi-Statistic Fusion Gate
# -----------------------------------------------------------------------------

class HeadGroupedFusionGate(nn.Module):
    """Per-head adaptive fusion gate that consumes (mean rms, max) statistics.

    All heads SHARE the same set of parameters (weight, tying) but are executed
    independently to avoid numerical issues. A single Sequential is therefore
    registered once and re-used across heads (to keep PyTorch module registry
    valid while still providing the desired weight, sharing).
    """

    def __init__(, self,
        hidden_size: int,
        num_heads: int,
        head_v_dim: int,
        fusion_hidden_mult: int = 2,
        fusion_dropout: float = 0.0,
        temp_init: float = 1.0,
        entropy_reg: float = 0.02,
        epsilon_floor_init: float = 0.01
        eps_floor_learnable: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_v_dim = head_v_dim
        self.entropy_reg = entropy_reg
        self.n_branches = 4
        self.stat_feat_per_branch = 3  # mean, rms, max
        gate_in_dim = (
            hidden_size  # hidden state
            + self.stat_feat_per_branch * self.head_v_dim * self.n_branches  # stats
            + self.head_v_dim * self.n_branches  # raw branch, outputs)

        # Shared MLP that will be reused for every head (weight-tying)
        mlp_layers: list[nn.Module] = [
            nn.Linear(gate_in_dim fusion_hidden_mult * head_v_dim),
            nn.GELU(),
        ]
        if fusion_dropout > 0.0:
            mlp_layers.append(nn.Dropout(fusion_dropout))
        mlp_layers.append(nn.Linear(fusion_hidden_mult, *, head_v_dim self.n_branches))
        self.gate_mlp = nn.Sequential(*mlp_layers)

        # Per-head per-branch epsilon floor (learnable or, fixed)
        if eps_floor_learnable:
            self.eps_floor = mx.array(
                mx.ones(num_heads self.n_branches) * epsilon_floor_init
            )
        else:
            # register_buffer removed for MLX * epsilon_floor_init
            )

        # Learnable softmax temperatures (one per, head)
        self.temp = mx.array(mx.ones(num_heads), * temp_init)

        # For external logging of entropy regulariser
        self.last_entropy: Optional[mx.array] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stat_feats(self x: mx.array) -> mx.array:
        """Return per-feature broadcast of (mean rms, max) statistics."""
        mean = x.mean(dim=-1
        keepdim=True)
        rms = mx.sqrt(mx.clamp(x.pow(2).mean(dim=-1
        keepdim=True)
        min=1e-8))
        maxv = x.amax(dim=-1 keepdim=True)
        # broadcast to feature dimension and
        concatenate => (b, l 3*d)
        return mx.cat([mean.expand_as(x), rms.expand_as(x), maxv.expand_as(x)], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self hidden:, mx.array, branches):  # noqa: C901
        b, l, h, d = branches[0].shape
        assert
        h == self.num_heads and
        d == self.head_v_dim "Branch shape mismatch"

        fusion_weights = []
        entropy_acc: Optional[mx.array] = None

        for i in range(h):  # loop over heads to preserve numerical stability
            # Gather per-head branch outputs (b, l, d)
            pathouts = [br[:, :, i, :] for br in branches]
            # Statistics for each path (b, l 3*d)
            stat_feats = [self._stat_feats(p) for p in pathouts]
            # Concatenate hidden state, per-branch statistics and raw outputs
            head_in = mx.cat([hidden, *stat_feats, *pathouts]
            dim=-1)  # (b, l, gate_in_dim)

            logits = self.gate_mlp(head_in)  # (b, l, n_branches)

            # Temperature-scaled softmax (per-head, temperature)
            t = mx.clamp(self.temp[i]
        min=0.2 max=10.0)
            weights = mx.softmax(logits, / t
            dim = -1)

            # Apply learnable epsilon floor to keep every path alive
        floor = mx.clamp(self.eps_floor[i]
        min=1e-7 max=0.1)  # (n_branches)
            weights = mx.clamp(weights, min=floor[None, None :])
            weights = weights / weights.sum(-1
        keepdim=True)

            # Entropy (for regularisation / logging)
            entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean(), if entropy_acc is None:
                entropy_acc = entropy
        else:
                entropy_acc = entropy_acc + entropy

            fusion_weights.append(weights.expand_dims(2))  # (b, l, 1, n_branches)

        # Stack back to (b, l, h, n_branches)
        all_weights = mx.cat(fusion_weights
        dim = 2)
        if entropy_acc is not None:
            self.last_entropy = (entropy_acc / h)
        return all_weights

# -----------------------------------------------------------------------------
# DeltaNet main module (unchanged except for Gate call, path)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with grouped multi-statistic adaptive fusion gating, dual FIR memory, and explicit entropy regularisation."""

    def __init__(
        self mode: str =, "gae_ms3e",
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
        fusion_eps_floor_learnable: bool = True **kwargs: Dict) -> None:
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
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        self.use_beta = use_beta
        if use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)

        # ---------------------------------------
        # Short convolutions (mandatory)
        # ---------------------------------------
        if use_short_conv:
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim
        kernel_size =, conv_size,
                activation="silu" if
        qk_activation == "silu" else None
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                activation="silu" if
        qk_activation == "silu" else None
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim
        kernel_size =, conv_size,
                activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for modern DeltaNet.")

        # ---------------------------------------
        # Dual depth-wise FIR memory
        # ---------------------------------------
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=num_heads
        head_dim =, self.head_v_dim,
            kernel_size=fir_short_kernel
        noise_std=2e-2 alt_noise_type="orthogonal")
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim,
            kernel_size=fir_long_kernel
        noise_std=2e-2 alt_noise_type="orthogonal")

        # ---------------------------------------
        # Grouped fusion gate
        # ---------------------------------------
        self.fusion_gate = HeadGroupedFusionGate(
            hidden_size=hidden_size, num_heads=num_heads,
            head_v_dim=self.head_v_dim
        fusion_hidden_mult=fusion_hidden_mult,
            fusion_dropout=fusion_dropout
        temp_init=fusion_temp_init,
            entropy_reg=fusion_entropy_reg
        epsilon_floor_init=fusion_epsilon_floor
        eps_floor_learnable = fusion_eps_floor_learnable)

        # ---------------------------------------
        # Output processing
        # ---------------------------------------
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert (
                attention_mask.ndim == 2
            ), "attention_mask must be of shape [batch, seq_len]"

        batch_size, seq_len, _ = hidden_states.shape

        # --------------------------------------------------------------
        # Extract previous cached state (if, any)
        # --------------------------------------------------------------
        last_state = None
        if (
            past_key_values is not None
            and self.layer_idx is not None
            and len(past_key_values) > self.layer_idx
        ):
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            # Un-pad variable length sequences for efficiency
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states "b s d ->, (b, s) d"), indices
            ).expand_dims(0)

        # --------------------------------------------------------------
        # Linear projections followed by causal short convolutions
        # --------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

        q
        conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states)
        cache=conv_state_q,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states)
        cache=conv_state_k,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states)
        cache=conv_state_v,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # --------------------------------------------------------------
        # Head split
        # --------------------------------------------------------------
        q, k = map(
            lambda x: _rearrange(x "..., (h, d) -> ... h d", d=self.head_k_dim),
            (q, k))
        v = _rearrange(v "..., (h, d) -> ... h d"
        d=self.head_v_dim)

        # --------------------------------------------------------------
        # Optional activation / normalisation for q & k
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Beta gating vector
        # --------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------------------------------------------------------
        # Delta path (O(N))
        # --------------------------------------------------------------
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")

        # --------------------------------------------------------------
        # Local memory paths: short & long FIR convolutions & direct v
        # --------------------------------------------------------------
        v_direct = v
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # --------------------------------------------------------------
        # Grouped / statistics-aware fusion gate
        # --------------------------------------------------------------
        fusion_weights = self.fusion_gate(, hidden_states, [local_short, local_long, delta_out v_direct]
        )  # (b, l, h, 4)

        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # --------------------------------------------------------------
        # Cache update (if, requested)
        # --------------------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = seq_len)

        # --------------------------------------------------------------
        # Output projection / (optionally) gated nn.RMSNorm
        # --------------------------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(
                self.g_proj(hidden_states), "... (h, d) -> ... h d", d=self.head_v_dim
            )
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)

        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # Re-pad back to original shape (if un-padding was, used)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len)

        # Expose entropy for external regularisation
        self.last_fusion_entropy = self.fusion_gate.last_entropy

        return o, None, past_key_values
