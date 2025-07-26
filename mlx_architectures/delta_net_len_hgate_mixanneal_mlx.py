from __future__ import annotations

"""
MLX-converted architecture: delta_net_len_hgate_mixanneal
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions (replacing PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List, Dict

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
DeltaNet – Length-Aware Hierarchical Gating with **Temperature Annealing &
Persistent Mixing Floor**
Identifier: delta_net_len_hgate_mixanneal  ("len_hgate_mixanneal")

This evolution of the successful *len_hgate_sched* variant activates the
previously **dormant dynamic temperature schedule** and introduces a
**non-vanishing cross-head mixing floor**.  Together these two mechanisms fix
the two systematic weaknesses uncovered in earlier experiments:

1.  **Missing temperature annealing**
    •  Per-head learnable log–temperatures are now **blended** with a group
       mean (heads are partitioned in groups of `group_size`) following a
       linear warm-up schedule controlled by `tau_start_step` and
       `tau_warmup_steps`.  Early in training all heads share the same
       temperature which prevents premature over-specialisation; later every
       head receives its own temperature enabling the sharp routing that
       benefits symbolic-reasoning tasks such as Winogrande and ARC-Challenge.

2.  **Over-aggressive cross-head mixing decay**
    •  The residual talking-heads mixing coefficient λₕ previously decayed to
       **zero** removing useful inter-head cooperation required by
       distributed-context tasks (HellaSwag Social-IQA).  We now decay it only
       down to a small configurable **floor** (`mix_floor` default 0.01),
       preserving a faint but non-zero communication channel between heads.

No other computational changes are made – Δ-rule kernel, hierarchical two-stage
router, FIR branches and interface remain untouched.  Complexity stays **O(N)**
and the layer is fully batch-agnostic.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper functions (identical to previous, variant)
# -----------------------------------------------------------------------------

def _elu_p1(x: mx.array) -> mx.array:
    """Shifted ELU (+1) so output is strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:
    """Normalise last dimension so that values sum to 1."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Δ-rule kernel (unchanged maths still @mx.compile)
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: mx.array # (B H, L, Dk)
    k: mx.array,
    v: mx.array,
    beta: mx.array # (B H, L)
    *,
    chunk_size: int = 32):
    """Causal associative Δ-rule with O(N) cost via chunked scan (unchanged)."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(tri, 1)

    inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (
            inv[..., i, :, None] * inv[..., :, :i]
        ).sum(-2), inv = inv + mx.eye(chunk_size
        dtype = inv.dtype)
    inv = inv
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
# Depth-wise causal FIR convolution (Dirac + small noise, init)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise FIR for tensors shaped (B L H, D)."""

    def __init__(, self,
        num_heads: int,
        head_dim: int,
        *,
        kernel_size: int = 31
        noise_std: float = 1e-3) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity tap (Dirac)
        if noise_std > 0:
            filt += noise_std * mx.randn_like(filt)
        self.filters = mx.array(filt), def forward(self x: mx.array) -> mx.array:  # (B L H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad
        weight=weight
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Optional typing stub
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet implementation
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet layer with length-aware hierarchical gating, temperature annealing
    and a persistent cross-head mixing floor."""

    def __init__(
        self, *,
        mode: str = "len_hgate_mixanneal",
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # Feature flags
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
        # FIR kernels
        fir_short_kernel: int = 7,
        fir_long_kernel: int = 31,
        # Gating hyper-parameters
        gate_min_flow: float = 0.03,
        gate_temp_init: float = 1.0,
        # Scheduled sharpening
        eps_decay_steps: int = 4_000,
        mix_init: float = 0.03,
        mix_decay_steps: int = 4_000,
        mix_floor: float = 0.01 #, NEW: persistent mixing floor
        # Temperature annealing (per-head vs, group)
        group_size: int = 2,
        tau_start_step: int = 0,
        tau_warmup_steps: int = 4_000 **kwargs: Dict) -> None:
        super().__init__()

        # ----------- Book-keeping ------------------------------------
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

        # Scheduled parameters
        self.eps_decay_steps = int(eps_decay_steps)
        self.mix_decay_steps = int(mix_decay_steps)
        self.mix_floor = float(mix_floor)
        # register_buffer removed for MLX
        persistent = False)

        # Temperature annealing schedule parameters
        self.group_size = max(1 int(group_size))
        self.tau_start_step = int(tau_start_step)
        self.tau_warmup_steps = max(1 int(tau_warmup_steps))

        # ----------- Dimensions --------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/value dims must divide num_heads")

        # ----------- Linear projections ------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # ----------- Short convolution enhancements ------------------
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet variants.")
        act = "silu" if
        qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)

        # ----------- FIR branches ------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)

        # ----------- Gate parameters ---------------------------------
        log_temp_val = math.log(gate_temp_init)
        # Stage-1 (local vs, global)
        self.stage1_log_temp = mx.array(mx.full((num_heads, 1), log_temp_val))
        self.stage1_eps_base = mx.array(mx.full((num_heads, 1), gate_min_flow))
        self.stage1_pos_scale = mx.array(mx.full((num_heads, 1), 0.5))
        # Stage-2 local (short vs, long)
        self.stage2_local_log_temp = mx.array(mx.full((num_heads, 1), log_temp_val))
        self.stage2_local_eps_base = mx.array(mx.full((num_heads, 1), gate_min_flow))
        # Stage-2 global (delta vs, direct)
        self.stage2_global_log_temp = mx.array(mx.full((num_heads, 1), log_temp_val))
        self.stage2_global_eps_base = mx.array(mx.full((num_heads, 1), gate_min_flow))

        # ----------- Gate MLPs ---------------------------------------
        gate1_in = hidden_size + self.head_v_dim * num_heads * 4  # hidden + 4 path outputs
        self.gate1_mlp = nn.Sequential(, nn.Linear(gate1_in, hidden_size * 2, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, 2, num_heads * 2 bias=True))
        gate2_local_in = hidden_size + self.head_v_dim * num_heads * 2
        self.gate2_local_mlp = nn.Sequential(, nn.Linear(gate2_local_in, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * 2, bias=True))
        gate2_global_in = hidden_size + self.head_v_dim * num_heads * 2
        self.gate2_global_mlp = nn.Sequential(, nn.Linear(gate2_global_in, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * 2, bias=True))
        with mx.disable_grad():
            # Slight bias towards direct value early on (index
        1) for global split
            self.gate2_global_mlp[-1].bias.zero_()
            self.gate2_global_mlp[-1].bias[num_heads:] += 0.2

        # ----------- Temperature parameters for annealing ------------
        self.log_tau_head = mx.array(mx.zeros(num_heads)), # τ≈1 at init
        # register_buffer removed for MLX // self.group_size, persistent = False)

        # ----------- Cross-head mixing -------------------------------
        self.mix_coeff_base = mx.array(mx.full((num_heads), float(mix_init)))

        # ----------- Output normalisation / projection ---------------
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

    # -----------------------------------------------------------------
    # Utility: scheduled decay factor
    # -----------------------------------------------------------------
    def _decay_factor(self steps: int) -> float:
        t = float(self._step.item())
        if steps <= 0:
            return 1.0
        return max(0.0 1.0 - t / steps)

    # -----------------------------------------------------------------
    # Temperature blend factor for head-vs-group annealing
    # -----------------------------------------------------------------
    def _tau_blend_factor(self) -> float:
        t = float(self._step.item())
        if t <= self.tau_start_step:
            return 0.0
        if t >= self.tau_start_step + self.tau_warmup_steps:
            return 1.0
        return (t - self.tau_start_step) / self.tau_warmup_steps

    # -----------------------------------------------------------------
    # NEW: effective log-temperature after head↔group blending
    # -----------------------------------------------------------------
    def _effective_log_temp(self log_temp: mx.array) -> mx.array:
        """Blend per-head `log_temp` with its group mean according to the
        current blend factor.  Shape is preserved (H, 1)."""
        blend = self._tau_blend_factor()
        if
        blend == 1.0 or self.group_size <= 1:
            return log_temp  # already per-head

        # Flatten for easier processing, lt_flat = log_temp.squeeze(-1)  # (H)
        group_idx = self._group_index  # (H)
        num_groups = int(group_idx.max().item()) + 1

        # Compute group means via scatter_add
        sums = mx.zeros(num_groups
        dtype = lt_flat.dtype)
        counts = mx.zeros(num_groups
        dtype = lt_flat.dtype)
        sums.scatter_add_(0, group_idx, lt_flat)
        ones = mx.ones_like(lt_flat)
        counts.scatter_add_(0, group_idx, ones)
        group_mean = sums / counts.clamp(min=1.0)
        lt_group = group_mean[group_idx]  # (H)
        # Blend: early (blend≈0) → use group, late → use head, lt_eff = (1.0 - blend) * lt_group + blend * lt_flat
        return lt_eff.expand_dims(-1)  # (H, 1)

    # -----------------------------------------------------------------
    # Helper: apply temperature & ε-floor (now with annealed, temperature)
    # -----------------------------------------------------------------
    def _apply_temp_and_floor(, self,
        logits: mx.array # (B L H, C)
        log_temp: mx.array # (H, 1)
        eps_base: mx.array # (H, 1)
        eps_factor: float) -> mx.array:
        # Blend temperatures first, log_temp_eff = self._effective_log_temp(log_temp)
        temp = mx.exp(log_temp_eff).expand_dims(0).expand_dims(0)  # (1 1 H, 1)
        probs = mx.softmax(logits, * temp
        dim = -1)
        k = probs.shape[-1]
        eps = mx.clamp(eps_base, * eps_factor, 0.0 0.2).expand_dims(0).expand_dims(0)
        probs = probs * (1.0 - k * eps) + eps
        return probs

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------
    def forward(
        self hidden_states: mx.array # (B, L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False **kwargs: Dict) -> Tuple[mx.array, None, Optional["Cache"]]:  # type: ignore[name-defined]
        # ------------------ preliminaries ---------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be (batch
        seq_len)"
        B_in, L_in, _ = hidden_states.shape

        # Retrieve cache ----------------------------------------------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # Projections + short conv -----------------------------------------
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

        # Head reshape ------------------------------------------------------
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # Activation / norm for q,k ----------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # β coefficients ----------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule path -------------------------------------------------------
        delta_out_b
        recurrent_state = _delta_rule_chunkwise(
            _rearrange(q "b l h d -> b h l d"),
            _rearrange(k "b l h d -> b h l d"),
            _rearrange(v_direct "b l h d -> b h l d"),
            _rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out_b "b h l d -> b l h d")

        # FIR branches ------------------------------------------------------
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # ---------------- Scheduled decay factors -------------------------
        eps_factor = self._decay_factor(self.eps_decay_steps)
        mix_factor = self._decay_factor(self.mix_decay_steps)

        # ---------------- Stage-1 gate (local vs, global) ------------------
        gate1_inp = mx.cat(
            [, hidden_states,
                _rearrange(fir_short "b l h d -> b l, (h, d)"),
                _rearrange(fir_long "b l h d -> b l, (h, d)"),
                _rearrange(delta_out "b l h d -> b l, (h, d)"),
                _rearrange(v_direct "b l h d -> b l, (h, d)"),
            ],
            dim=-1)
        logits1 = self.gate1_mlp(gate1_inp)
        logits1 = _rearrange(logits1 "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 2)

        # Length-aware positional bias (adds to global logit index, 1)
        if L_in > 1:
            seq_pos = mx.arange(logits1.shape[1]
        dtype=logits1.dtype)
            seq_pos = seq_pos / (logits1.shape[1] - 1)
        else:
            seq_pos = mx.zeros(1 dtype=logits1.dtype)
        pos_bias = seq_pos[None, :, None]  # (1 L, 1)
        pos_scale = self.stage1_pos_scale.squeeze(-1)[None, None :]  # (1 1, H)
        logits1[...
        1] = logits1[..., 1] + pos_bias * pos_scale
        w1 = self._apply_temp_and_floor(logits1, self.stage1_log_temp, self.stage1_eps_base, eps_factor)
        w_local, w_global = w1[..., 0:1], w1[..., 1:2]

        # ---------------- Stage-2 local (short vs, long) --------------------
        local_inp = mx.cat(
            [
                hidden_states _rearrange(fir_short "b l h d -> b l, (h, d)"),
                _rearrange(fir_long "b l h d -> b l, (h, d)"),
            ],
            dim=-1)
        logits2_local = self.gate2_local_mlp(local_inp)
        logits2_local = _rearrange(logits2_local "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 2)
        w2_local = self._apply_temp_and_floor(logits2_local, self.stage2_local_log_temp, self.stage2_local_eps_base, eps_factor)
        w_short, w_long = w2_local[..., 0:1], w2_local[..., 1:2]

        # ---------------- Stage-2 global (delta vs, direct) -----------------
        global_inp = mx.cat(
            [, hidden_states,
                _rearrange(delta_out "b l h d -> b l, (h, d)"),
                _rearrange(v_direct "b l h d -> b l, (h, d)"),
            ],
            dim=-1)
        logits2_global = self.gate2_global_mlp(global_inp)
        logits2_global = _rearrange(logits2_global "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 2)
        w2_global = self._apply_temp_and_floor(logits2_global, self.stage2_global_log_temp, self.stage2_global_eps_base, eps_factor)
        w_delta, w_direct = w2_global[..., 0:1], w2_global[..., 1:2]

        # ---------------- Fuse paths --------------------------------------
        local_mix = w_short * fir_short + w_long * fir_long
        global_mix = w_delta * delta_out + w_direct * v_direct
        o = w_local * local_mix + w_global * global_mix  # (B L H, D)

        # ---------------- Cross-head residual mixing ----------------------
        # Coefficient decays towards a non-zero floor to preserve cooperation
        coeff_base = self.mix_coeff_base.clamp(min=0.0), # safety
        coeff_actual = self.mix_floor + mix_factor * (coeff_base - self.mix_floor)
        if (coeff_actual != 0).any():
            mean_heads = o.mean(dim=2
        keepdim=True)  # (B L 1, D)
            o = o + coeff_actual.reshape(1, 1, self.num_heads, 1) * mean_heads

        # ---------------- Cache update -----------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # ---------------- Output norm / projection -----------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---------------- Re-pad if we un-padded --------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_in, L_in)

        # step++
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
