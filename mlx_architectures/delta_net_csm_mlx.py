from __future__ import annotations

"""
MLX-converted architecture: delta_net_csm
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
    return x / mx.linalg.norm(x, axis=-1,
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
    def __init__(self, hidden_size: int,
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
DeltaNet – Hybrid Content-Sharp Multi-Scale Memory (CSM)
This **evolutionary layer** combines the most successful ideas from earlier
experiments – *content-aware gating* (CAMoE) and *temperature-sharpened routing*
(Sharp-SMG) – while remaining computationally light-weight and fully
compatible with the DeltaNet interface.

Key Innovations
1. **Per-Head Temperature-Sharpened Content Gate**
   The gate sees **token embeddings** *and* **lightweight per-path statistics**
   (mean & std per, head) of each candidate branch.  It outputs per-token per-head mixing weights and uses a **learnable per-head temperature τₕ** to
   adaptively sharpen or smooth the distribution.  This unifies the strengths
   of CAMoE (content, awareness) and SMG (adaptive, sharpness) without the
   drawbacks of hard probability floors.

2. **Dual-Statistic Path Features (mean, std)**
   For each of the four branches – short FIR, long FIR, Delta memory, identity
   – we compute the **per-head mean and standard deviation** across the channel
   dimension.  These 8 scalars capture both magnitude and variability giving
   the gate richer evidence than a single energy number while staying very
   cheap (O(N·H)).

3. **Adaptive Diversity Regulariser**
   A tiny entropy regularisation term(\alpha=0.02, by, default) discourages gate
   collapse, especially in the early phase without enforcing hard floors.  The
   coefficient decays linearly to zero over the first 30 k steps(can, be, overridden).

4. **Zero Additional Asymptotic Cost**
   All operations are linear in sequence length; the extra statistics are fast
   reductions and a small MLP.  No quadratic attention or large matrices are
   introduced.

Implementation Notes
•   Class name remains **DeltaNet**; constructor/forward signatures are kept.
•   All tensor reshaping uses `einops.rearrange` for shape safety.
•   The chunk-wise \Delta-rule kernel is copied unchanged from previous proven
    versions and kept under `@mx.compile`.
•   Short convolution preprocessing and depth-wise FIR value branches are
    preserved.
•   The new gate lives in `_ContentSharpGate` – a two-layer MLP with per-head
    temperature parameters.
•   A small helper `linear_decay()` provides the scheduling for the entropy
    regulariser.
"""


import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def elu_p1(x:, mx.array) -> mx.array:
    """ELU + 1 (always, positive)."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x:, mx.array) -> mx.array:
    """Normalise vectors so that they sum to 1 along the last dim."""
    return(x, / x.sum(dim=-1
        keepdim=True))


def linear_decay(step:, int, start: int end: int) -> float:
    """Linear decay from 1.0 at *start* to 0.0 at *end* (clamped)."""
    if step < start:
        return 1.0
    if step > end:
        return 0.0
    return 1.0 - (step - start) / float(end, - start)

# -----------------------------------------------------------------------------
# Causal chunk-wise Δ-rule(unchanged, proven, implementation)
# -----------------------------------------------------------------------------

@mx.compile  # noqa: E302
# pylint: disable=too-many-locals,too-many-statements
def delta_rule_chunkwise(
    q: mx.array # [B H L D]
    k: mx.array,  # [B H L D]
    v: mx.array,  # [B H L Dv]
    beta: mx.array,  # [B H L]
    *,
    chunk_size: int = 32):
    """Causal Δ-rule processed in fixed-size chunks – O(N) memory & compute."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_full = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_), 0
    )
    tri_strict = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_), 1
    )

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (
            inv[..., i, :, None] * inv[..., :, :i]
        ).sum(-2), inv = inv + mx.eye(chunk_size, dtype = inv.dtype)

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)

    for idx in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Depth-wise causal convolution (value, paths)
# -----------------------------------------------------------------------------

class _DepthwiseCausalConv1d(nn.Module):
    """Per-head depthwise causal 1-D convolution (kernel initialised as, Dirac)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = kernel_size
        weight = mx.zeros(num_heads, *, head_dim, 1, kernel_size)
        weight[..., -1] = 1.0  # Dirac at current token
        weight += 0.02 * mx.randn_like(weight)
        self.weight = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # [B, L, H, D]
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

# -----------------------------------------------------------------------------
# Content-sharp gate
# -----------------------------------------------------------------------------

class _ContentSharpGate(nn.Module):
    """Per-token, per-head gate with content features and learnable temperature."""

    def __init__(, self,
        hidden_size: int,
        num_heads: int,
        stat_dim: int = 8 # 4 paths × (mean+std)
        gate_hidden_mult: float = 0.5
        temp_init: float = 1.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        feat_dim = int(hidden_size, * gate_hidden_mult)
        self.in_proj = nn.Linear(hidden_size, feat_dim
        bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(feat_dim, +, stat_dim
        num_heads * 4 bias=True)

        # Per-head learnable temperature(stored, in log-space)
        init_log = math.log(math.exp(temp_init) - 1.0)
        self.log_temp = mx.array(mx.full((num_heads), init_log))

        # Bias initialisation – favour identity/value path (index, 3)
        with mx.disable_grad():
            self.out_proj.bias.zero_()
            self.out_proj.bias.reshape(num_heads, 4)[:, 3] = 2.0

    def forward(self, hid: mx.array stats: mx.array) -> mx.array:
        """Compute gate weights.

        Args:
            hid:   [B, L, D] token embeddings
            stats: [B, L, H, 8] per-head statistics
        Returns:
            weights: [B, L, H, 4] softmax weights per path
        """
        B, L, H, _ = stats.shape
        # Project hidden_state once and broadcast over heads
        h_feat = self.act(self.in_proj(hid))  # [B, L F]
        h_feat = h_feat.expand_dims(2).expand(-1, -1, H -1)  # [B, L, H, F]

        gate_inp = mx.cat([h_feat, stats]
        dim=-1)  # [B, L, H F+stat]
        gate_inp = _rearrange(gate_inp, "b l h f -> (b, l, h) f")
        logits = self.out_proj(gate_inp)  # [(B L, H), 4]
        logits = _rearrange(logits, "(b, l, h) p -> b l h p"
        b=B
        l=L
        h = H)

        temp = F.softplus(self.log_temp) + 1e-4  # ensure >0
        logits = logits * temp.reshape(1, 1, H, 1)
        weights = mx.softmax(logits, dim = -1)
        return weights

# -----------------------------------------------------------------------------
# Optional type hints for cache utilities
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                  DeltaNet
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet – Hybrid Content-Sharp Multi-Scale Memory layer."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self, *,
        mode: str = "csm",
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
        # value path kernels
        short_kernel_size: int = 3,
        long_kernel_size: int = 25,
        # gate specifics
        gate_hidden_mult: float = 0.5,
        gate_temp_init: float = 1.5,
        entropy_reg_alpha: float = 0.02,
        entropy_reg_warmup: int = 0,
        entropy_reg_decay_end: int = 30000 **kwargs: Dict) -> None:
        super().__init__()
        # ---------------- store basic params ----------------
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

        # entropy reg scheduling
        self.entropy_reg_alpha = entropy_reg_alpha
        self.entropy_reg_warmup = entropy_reg_warmup
        self.entropy_reg_decay_end = entropy_reg_decay_end

        # ---------------- derived dims ---------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---------------- projections ----------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # ---------------- short convs ----------------------
        if use_short_conv:
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
        else:
            raise UserWarning("_ShortConvolution, is mandatory for stable performance.")

        # ---------------- value path branches --------------
        self.fir_short = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, short_kernel_size)
        self.fir_long = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, long_kernel_size)

        # ---------------- content-sharp gate ---------------
        self.gate = _ContentSharpGate(
            hidden_size=hidden_size, num_heads=num_heads,
            stat_dim=8
        gate_hidden_mult=gate_hidden_mult
        temp_init = gate_temp_init)

        # ---------------- output norm / proj --------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,  # [B, L, D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        step: Optional[int] = None # current global step for scheduling
        **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        # ---------------- sanity checks --------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [B, L]"
        B0, L_in, _ = hidden_states.shape

        # ---------------- un-padding -----------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---------------- cache retrieval ------------------
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
                conv_state_q
        conv_state_k, conv_state_v = last_state.get("conv_state", (None None, None))

        # ---------------- Q/K/V projections + conv ---------
        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_raw, conv_state_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---------------- reshape to heads -----------------
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_raw, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---------------- activation / norm ---------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = sum_norm(q), sum_norm(k)

        # ---------------- β scaling ------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  # [B
        L, H]
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Delta memory --------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---------------- FIR branches --------------------
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # ---------------- stats for gate ------------------
        # mean and std across feature dim per head per token
        def _mean_std(t:, mx.array):
            mu = t.mean(dim=-1)
        std = t.std(dim=-1), return mu, std
        stats_short = mx.stack(_mean_std(fir_short)
        dim=-1)  # [B,L,H 2]
        stats_long = mx.stack(_mean_std(fir_long)
        dim=-1)
        stats_delta = mx.stack(_mean_std(delta_out)
        dim=-1)
        stats_id = mx.stack(_mean_std(v_direct)
        dim=-1)
        stats_all = mx.cat([stats_short, stats_long, stats_delta, stats_id]
        dim=-1)  # [B, L, H, 8]

        # ---------------- gate weights --------------------
        gate_w = self.gate(hidden_states, stats_all)  # [B,L,H,4]

        # ---------------- fused output -------------------
        out = (
            gate_w[..., 0:1] * fir_short
            + gate_w[..., 1:2] * fir_long
            + gate_w[..., 2:3] * delta_out
            + gate_w[..., 3:4] * v_direct
        )

        # ---------------- entropy regularisation ---------
        reg_loss = None
        if self.training and self.entropy_reg_alpha > 0.0:
            step = 0 if step is None else step
        sched = linear_decay(step, self.entropy_reg_warmup self.entropy_reg_decay_end)
            if sched > 0.0:
                ent = -(gate_w * (gate_w + 1e-8).log()).sum(-1).mean()
        reg_loss = self.entropy_reg_alpha * sched * ent

        # ---------------- cache update -------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # ---------------- output norm / proj -------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l, (h, d)")
        out = self.o_proj(out)

        # ---------------- re-pad --------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0)
        indices, B0, L_in)

        return out, reg_loss, past_key_values
