from __future__ import annotations

"""
MLX-converted architecture: delta_net_afrc
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
DeltaNet – Adaptive Floor & Rich Context-Stat Gating (delta_net_afrc)
This evolutionary variant unifies the strongest ideas from the "Dynamic
Hierarchical Gating" (DHG) and the "Context-Stat Gate" (CtxStatGate)
families **and fixes the remaining local–global trade-off** by making the
*context floor adaptive* **while enriching the router signal with higher
band-width branch statistics** and an additional *very-long* convolutional
branch.

Key Innovations (enabled by, default)
1. **Adaptive Context-Floor (ACF)** – A *learnable scalar* per head
   `logit_context_floor` initialised such that the minimum contextual mass
   equals `context_floor_init` (default *5 %*).  Because it is *learnable*
   the optimiser can freely *decrease* (or, increase) the floor when the
   network decides it no longer needs forced contextual flow removing the
   global-reasoning penalty previously caused by a *static* floor.

2. **Richer Context-Statistics (RCS)** – The fusion gate now sees *three*
   statistics (mean, RMS max-abs) from each branch instead of two.  With
   four contextual branches (short-FIR ≈3 tok long-FIR ≈31, tok)
   wide-FIR ≈64 tok Δ-memory) **plus** the identity/value branch this makes
   `5 branches × 3 stats × H` additional inputs giving the gate finer
   information to discriminate local vs. global needs without incurring
   any quadratic cost.

3. **Very-Long FIR Branch (wide-FIR)** – A new depth-wise causal FIR with, kernel = 64 tokens is introduced capturing narrative context that even
   the Δ-memory sometimes under-utilises.  The branch is initialised to an
   *identity* filter so optimisation starts from the proven baseline.

4. **Coarse-Then-Fine Routing with Temperature** – We keep the efficient
   coarse (identity vs. context) then fine (softmax over 4 contextual, branches) structure *with a learnable per-head temperature*.  This
   preserves O(N) compute, guarantees causal flow and empirically yields
   faster convergence than flat softmax.

All computations remain **O(N·d)**, strictly causal, batch-size agnostic,
`einops.rearrange` is used everywhere and the @mx.compile kernel for
chunk-wise Δ-rule is preserved.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _elu_p1(x: mx.array) -> mx.array:
    """Shifted ELU (=ELU+1) that stays strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that last dimension sums to one."""
    return (x / x.sum(-1, keepdim=True))


def _branch_stats(x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """Return (mean rms, max_abs) along the channel dimension."""
    mean = x.mean(dim=-1)
        rms = mx.sqrt(mx.clamp(x.pow(2).mean(dim=-1)
        min=1e-8))
    max_abs = x.abs().max(dim=-1).values
    return mean, rms, max_abs

# ---------------------------------------------------------------------------
# Core chunk-wise Δ-rule (identical to baseline still @mx.compile)
# ---------------------------------------------------------------------------

@mx.compile  # noqa: D401
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Associative Δ-rule retrieval processed in causal chunks (O(N))."""
    b, h, L, _ = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v
    k_beta = map(lambda t: _rearrange(t "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))

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
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        d_k = q.shape[-1]
    d_v = v.shape[-1]
    S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1 -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution with identity (Dirac) initialisation
# ---------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR convolution."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        weight[..., -1] = 1.0  # current-timestep tap (identity)
        self.filters = mx.array(weight), def forward(self x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, L, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w
        groups = h * d)
        y = _rearrange(y "b, (h, d) l -> b l h d"
        h=h)
        return y

# ---------------------------------------------------------------------------
# Optional cache typing helper
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                               DeltaNet-AFRC
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet layer with **Adaptive Floor & Rich Context-Stat Gating**."""

    def __init__(
        self, *,
        mode: str = "afrc",  # adaptive floor & rich context
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
        # FIR kernel sizes
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fir_wide_kernel: int = 64,
        # gating hyper-params
        fusion_hidden_mult: int = 2,
        context_floor_init: float = 0.05,
        value_bias_init: float = 4.0,
        gate_temp_init: float = 1.0,
        fusion_dropout: float = 0.0 **kwargs) -> None:
        super().__init__()

        # ---------------- bookkeeping & dims --------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "dims must divide num_heads"
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---------------- projections -------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # ---------------- optional short conv -----------------------
        if use_short_conv:
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
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet performance.")

        # ---------------- FIR branches ------------------------------
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)
        self.fir_wide = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_wide_kernel)

        # ---------------- fusion gate MLP ---------------------------
        # Inputs: hidden_state (D) + 5 branches * 3 stats * H = D + 15H
        gate_in_dim = hidden_size + 15 * num_heads
        gate_hidden = hidden_size * fusion_hidden_mult
        self.gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, gate_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(gate_hidden, num_heads * 5, bias=True),  # 4 contextual + 1 value, logits)
        # Warm-start bias favouring identity/value path
        with mx.disable_grad():
            self.gate_mlp[-1].bias.zero_()
            self.gate_mlp[-1].bias[4::5] = value_bias_init  # every 5th element (value, path)

        # learnable per-head value bias (added on top of MLP output for identity, path)
        self.value_bias = mx.array(mx.full((num_heads), value_bias_init))

        # learnable per-head temperature for fine gate
        self.log_temperature = mx.array(mx.full((num_heads), math.log(gate_temp_init)))

        # learnable logit for adaptive context floor (per, head)
        floor_init_logit = math.log(context_floor_init / (1.0 - context_floor_init))
        self.logit_context_floor = mx.array(mx.full((num_heads), floor_init_logit))

        # ---------------- output norm / proj ------------------------
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
    # forward
    # -----------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compatibility
        floor_schedule: Optional[float] = None,  # optional scalar ∈[0 1] to scale context floor
        **kwargs: Dict) -> Tuple[mx.array, None Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.dim() == 2 "attention_mask must be (B
        L)"
        B_orig, L_in, _ = hidden_states.shape

        # ---------------- un-padding for variable-length batches ------
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---------------- retrieve previous state ---------------------
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv and last_state.get("conv_state") is not None:
                conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

        # ---------------- projections & short conv -------------------
        q_lin, k_lin, v_lin = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q_lin, conv_state_q = self.q_conv1d(q_lin
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin, conv_state_k = self.k_conv1d(k_lin
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin, conv_state_v = self.v_conv1d(v_lin
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---------------- reshape to heads ---------------------------
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---------------- optional activation / norm ----------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---------------- beta for Δ-rule ----------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule global memory -----------------------
        delta_out
        recurrent_state = delta_rule_chunkwise(
            _rearrange(q "b l h d -> b h l d"),
            _rearrange(k "b l h d -> b h l d"),
            _rearrange(v "b l h d -> b h l d"),
            _rearrange(beta "b l h -> b h l"))
        delta_out = _rearrange(delta_out "b h l d -> b l h d")

        # ---------------- FIR branches ------------------------------
        short_out = self.fir_short(v)
        long_out = self.fir_long(v)
        wide_out = self.fir_wide(v)

        # ---------------- branch statistics -------------------------
        stats_short = _branch_stats(short_out)
        stats_long = _branch_stats(long_out)
        stats_wide = _branch_stats(wide_out)
        stats_delta = _branch_stats(delta_out)
        stats_value = _branch_stats(v)

        # concatenate stats: mean,rms max_abs -> 3*H per branch
        def _stack_stats(stats_tuple):  # (mean,rms, max) each (B,L, H)
            return mx.cat(stats_tuple, dim = -1)  # (B,L, 3H)

        stats_concat = [_stack_stats(s) for s in (stats_short, stats_long, stats_wide, stats_delta, stats_value)]
        gate_input = mx.cat([hidden_states], + stats_concat
        dim = -1)  # (B, L D + 15H)

        gate_logits = self.gate_mlp(gate_input)  # (B,L H*5)
        gate_logits = _rearrange(gate_logits "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 5)

        # ---------------- coarse gate (value vs, context) -------------
        value_logit = gate_logits[..., 4] + self.value_bias  # (B,L, H)
        context_logits = gate_logits[..., 0:4]  # (B,L,H, 4)

        # compute adaptive floor ------------------------------------
        context_floor = mx.sigmoid(self.logit_context_floor)  # (H)
        if floor_schedule is not None:
            context_floor = context_floor * max(0.0 1.0 - float(floor_schedule))
        context_floor = context_floor.reshape(1, 1 self.num_heads)  # (1,1, H)

        p_value = (1.0 - context_floor) * mx.sigmoid(value_logit)  # ensures p_value ≤ 1-floor
        others_total = 1.0 - p_value  # ≥ context_floor by construction

        # ---------------- fine gate among contextual branches --------
        temperature = mx.exp(self.log_temperature).reshape(1, 1, self.num_heads, 1)
        ctx_weights = mx.softmax(context_logits, / temperature
        dim = -1)  # (B, L, H, 4)
        ctx_weights = ctx_weights * others_total.expand_dims(-1)  # scale by available mass

        # ---------------- fuse outputs ------------------------------
        fused = (
            ctx_weights[..., 0:1] * short_out
            + ctx_weights[..., 1:2] * long_out
            + ctx_weights[..., 2:3] * wide_out
            + ctx_weights[..., 3:4] * delta_out
            + p_value.expand_dims(-1) * v
        )

        # ---------------- cache update ------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)

        # ---------------- output norm & projection ------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            fused = self.o_norm(fused, g_vec)
        else:
            fused = self.o_norm(fused)
        out = self.o_proj(_rearrange(fused "b l h d -> b l, (h, d)"))

        # ---------------- re-pad if needed ---------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0)
        indices, B_orig, L_in)

        return out, None, past_key_values
