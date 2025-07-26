from __future__ import annotations

"""
MLX-converted architecture: delta_net_aif
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions(replacing, PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List, Dict

def _rearrange(tensor:, mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l(h, d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l(h, d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h l d -> b l h d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h(n, c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h(n, c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    else:
        # Fallback: return tensor as-is
        return tensor

def _l2norm(x:, mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1
        keepdims=True).clip(min=1e-8)

def _masked_fill(tensor:, mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
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
DeltaNet – Adaptive Identity Floor & Content-Position Fusion Gating (DeltaNet-AIF)
Identifier: *delta_net_aif*

This evolutionary variant builds on the empirically-strong **HIST** design and
implements two key improvements repeatedly highlighted in the experimental
analysis:

1. Adaptive Identity Floor (AIF)
   •  The lower bound of the identity/value gate is no longer a fixed constant.
      Instead it adapts **per-token & per-head** to the *router confidence*.
   •  When the 3-way context router is highly confident(top-probability, > 0.9)
      the minimum identity contribution decays towards *zero*, allowing precise
      context-only reasoning needed for extraction/aggregation tasks (e.g.)
      SWDE, BoolQ).
   •  Under low confidence the floor increases smoothly up to
      `base_min_id_frac`, protecting copy-path fidelity for ambiguous examples
      (beneficial for Winogrande narrative, tasks).
   •  An **exponential schedule** multiplies the floor during training so that
      the network can gradually learn to rely on its own confidence signal.

2. Content-Position Fusion in Router
   •  The 3-way context router (local-short, local-long ∆-rule, global) now
      receives *both* hidden-state information *and* an explicit **relative
      position scalar** (0…1) per token.
   •  This length cue is concatenated to the existing statistics features,
      enabling the router to balance local / global memory with awareness of
      sequence position while still being free to adapt non-monotonically.

All other proven design elements(identity-initialised, FIR filters ε-floor)
τ schedule chunk-wise ∆-rule) are retained.  Complexity stays **O(N·d)** and
all operations remain batch-agnostic and causally correct.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU (+1) so that outputs stay positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """L1 normalisation along last dimension."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution – identity initialisation + tiny noise
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise FIR for tensors shaped(B, L,H, D)."""

    def __init__(self, num_heads: int, head_dim: int
    kernel_size: int = 31
    noise_std: float = 1e-3) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0  # Dirac / identity
            if noise_std > 0:
                filt.add_(noise_std, * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")  # (B, H*D, L)
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise ∆-rule kernel(identical, numerics kept @mx.compile)
# -----------------------------------------------------------------------------

@mx.compile  # noqa: D401
def _delta_rule_chunkwise(q:, mx.array,  # (B,H,L, Dk)
    k: mx.array,  # (B,H,L, Dk)
    v: mx.array,  # (B,H,L, Dv)
    beta: mx.array,  # (B,H, L)
    *,
    chunk_size: int = 32):
    """Associative ∆-rule with O(N) cost via fixed-size chunks."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalise and scale
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into chunks -> (B,H,N,C, D)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(tri, 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Optional typing stub (only for static type, checkers)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet implementation – Adaptive Identity Floor & Content-Position Gate
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – must keep exact public name
    """DeltaNet layer with Adaptive Identity Floor & Content-Position Fusion gating."""

    # pylint: disable=too-many-instance-attributes,too-many-statements
    def __init__(self, # ---- identifier / misc ---- #
        mode: str =, "aif_v1",
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # ---- feature toggles ---- #
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
        # ---- FIR kernels ---- #
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # ---- Identity gate & schedule ---- #
        base_min_id_frac: float = 0.05,
        id_floor_warmup_steps: int = 2_000,
        id_gate_alpha_init: float = 1.0,
        # ---- Router parameters ---- #
        epsilon_floor: float = 0.02,
        tau_group_size: int = 2,
        tau_transition_steps: int = 3_000,
        router_hidden_mult: int = 2,
        router_dropout: float = 0.0 # ---- others ---- #
        **kwargs: Dict) -> None:
        super().__init__()

        # ------------------- bookkeeping -------------------
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # schedule params for adaptive id floor
        self.base_min_id_frac = float(base_min_id_frac)
        self.id_floor_warmup_steps = int(id_floor_warmup_steps)

        # step buffer for schedules
        # register_buffer removed for MLX
        persistent = False)

        # ------------------- dimensions --------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value, dims must divide num_heads")

        # ------------------ projections --------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # ------------------ short conv ---------------------
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution, is required for DeltaNet variants.")
        act = "silu" if
        qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
        activation=act
        bias = conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, conv_size
        activation="silu"
        bias=conv_bias)

        # ------------------ FIR branches -------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)

        # ------------------ identity gate ------------------
        self.id_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.id_gate_proj.bias.zero_()
        self.alpha_identity = mx.array(id_gate_alpha_init, *, mx.ones(num_heads))

        # ------------------ router MLP ---------------------
        # features: hidden + stats(mean/std, per, path) + position scalar, stat_dim_per_head = 2  # mean & std
        num_paths = 3  # short, long, delta
        router_in_dim = (
            hidden_size + num_heads * stat_dim_per_head * num_paths + 1  # +1 for position, scalar)
        router_hidden_dim = max(8, hidden_size * router_hidden_mult)
        self.router_mlp = nn.Sequential(, nn.Linear(router_in_dim, router_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(router_dropout) if router_dropout > 0.0 else nn.Identity(),
            nn.Linear(router_hidden_dim, num_heads * num_paths, bias=True))
        # small negative bias so identity initially dominates(via, min, floor)
        with mx.disable_grad():
            self.router_mlp[-1].bias.fill_(0.0)

        # ------------------ ε-floor & τ schedule -----------
        self.epsilon_floor = float(epsilon_floor)
        self.tau_group_size = int(tau_group_size)
        self.tau_transition_steps = int(tau_transition_steps)
        # log τ parameters: per group & per head
        num_groups = (num_heads + self.tau_group_size - 1) // self.tau_group_size
        # register_buffer removed for MLX // self.tau_group_size
    persistent = False)
        self.log_tau_group = mx.array(mx.zeros(num_groups))
        self.log_tau_head = mx.array(mx.zeros(num_heads))

        # ------------------ output normalisation ----------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _current_id_floor_scale(self) -> float:
        """Warm-up schedule: scale in [0 1] ramping down over id_floor_warmup_steps."""
        t = float(self._step.item())
        if t >= self.id_floor_warmup_steps:
            return 0.0
        return 1.0 - t / max(1.0, self.id_floor_warmup_steps)

    def _blend_tau(self) -> mx.array:  # (H)
        """Return per-head τ using group→head transition schedule."""
        t = float(self._step.item())
        blend = min(1.0, t / max(1.0, self.tau_transition_steps))
        tau_g = mx.exp(self.log_tau_group)[self._head2group]
        tau_h = mx.exp(self.log_tau_head)
        return(1.0, - blend) * tau_g + blend * tau_h

    @staticmethod
    def _mean_std(x:, mx.array) -> Tuple[mx.array mx.array]:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        return mean, std

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-branches,too-many-locals
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False #, noqa: F841 – kept for API compat
        **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:
        # ------------- preliminaries ------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be(B, L)"
        B_in, L_in, _ = hidden_states.shape

        # optional un-padding for variable sequence lengths
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # retrieve cache if present
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ------------- projections + short conv -------
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

        # reshape to heads
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # activations
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

        # β for ∆-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------- global ∆-rule path -------------
        delta_out_b
        rec_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_b, "b h l d -> b l h d")

        # ------------- local FIR paths -----------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------- statistics for router -----------
        ms_s
        std_s = self._mean_std(local_short)
        ms_l, std_l = self._mean_std(local_long)
        ms_d
        std_d = self._mean_std(delta_out)
        stats = mx.stack([ms_s, std_s, ms_l, std_l, ms_d, std_d]
        dim=-1)
        stats_flat = _rearrange(stats, "b l h f -> b l, (h, f)")

        # ------------- relative position scalar --------
        pos = (mx.arange(local_short.shape[1], dtype=local_short.dtype) / max(1, local_short.shape[1] - 1))
        pos = pos.reshape(1, -1, 1)  # (1,L, 1)

        # ------------- router logits -------------------
        router_in = mx.cat([hidden_states, stats_flat, pos.expand(hidden_states.shape[0], -1, 1)]
        dim=-1)
        router_logits = self.router_mlp(router_in)  # (B,L H*3)
        router_logits = _rearrange(router_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 3)

        # temperature scaling
    tau = self._blend_tau().reshape(1, 1, self.num_heads, 1)
        router_logits = router_logits / tau

        # ε-floor softmax
        router_probs = mx.softmax(router_logits, dim = -1)
        router_probs = router_probs * (1.0 - 3 * self.epsilon_floor) + self.epsilon_floor

        # ------------- adaptive identity gate ----------
        id_raw = mx.sigmoid(self.id_gate_proj(hidden_states))  # (B,L, H)
        # confidence = max prob among context paths
        confidence = router_probs.max(dim=-1).values  # (B, L, H)
        floor_scale = self._current_id_floor_scale()  # scalar in [0 1]
        adaptive_floor = self.base_min_id_frac * (1.0 - confidence) * floor_scale  # (B,L, H)
        id_gate = mx.clamp(id_raw, min = adaptive_floor)
        p_context = 1.0 - id_gate  # remaining prob mass

        # re-scale router probs to sum to p_context
        router_probs = router_probs * p_context.expand_dims(-1)

        # ------------- fuse paths ----------------------
        ctx_out = (
            router_probs[..., 0:1] * local_short +
            router_probs[..., 1:2] * local_long +
            router_probs[..., 2:3] * delta_out
        )

        alpha = self.alpha_identity.reshape(1, 1, self.num_heads, 1)
        id_out = alpha * id_gate.expand_dims(-1) * v_direct
        o = ctx_out + id_out  # (B,L,H, D)

        # ------------- cache update --------------------
        if past_key_values is not None and use_cache and hasattr(past_key_values, "update"):
            past_key_values.update(
                recurrent_state=rec_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # ------------- output projection ---------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # re-pad if needed
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_in, L_in)

        # step++
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
