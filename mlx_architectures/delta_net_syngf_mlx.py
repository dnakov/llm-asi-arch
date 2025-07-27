from __future__ import annotations

"""
MLX-converted architecture: delta_net_syngf
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
DeltaNet – Synergetic Local–Global Fusion (delta_net_syngf)
This evolutionary **DeltaNet** variant fuses the most successful ideas of the
AFT → CAGF → DynFuse line while explicitly fixing the residual shortcomings
identified in their evaluation:

1. Per-Head *Statistics-Aware* Gating (CAGF, strength)
   • Each head receives a 16-dim vector describing every branch
     (mean/var/abs-mean/ℓ2) enabling informed routing and avoiding premature
     path collapse.

2. *Temperature-Bound* Softmax(prevents, over-sharpening)
   • A learnable temperature τₕ is constrained to **τ ≥ 0.5** using
     `τ = 0.5 + softplus(·)`, guaranteeing minimum entropy that protected
     BoolQ & SWDE in prior studies while still allowing sharpening.

3. *Partial* Decaying Local Floor  (DynFuse, lesson)
   • Minimum probability ε(t) for the two convolutional (local) paths decays
     **exponentially** from `floor_init = 0.05` to a *non-zero* `floor_final =
     0.02`.  This preserves a thin but essential local capacity at convergence solving the late-stage lexical regression seen when ε→0.

4. *Adaptive Residual Bypass*  (new)
   • A per-head learnable residual αₕ(init, 0.1) is **scaled online** by the
     *current* lack of local allocation:

         ᾱ₍b,l h₎ = αₕ · (1 − w_local_total)

     so residual leakage is high only when the gate under-allocates local
     paths reducing output blur once the gate learns to exploit them.

5. Stronger Entropy Regulariser
   • Coefficient raised to 0.05 to further guard against path collapse during
     the long decay window.

The implementation retains strict **O(N)** complexity, uses chunk-wise Δ-rule
kernels, respect all API/signature constraints and is fully batch-agnostic via
`einops.rearrange`.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU (+1) keeping outputs positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """Normalise last dimension to L1-sum == 1."""
    return(x, / x.sum(-1
        keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity, initialisation)
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity weight on current step
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # (B,L,H, D)
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule (identical maths; kept @mx.compile for, speed)
# -----------------------------------------------------------------------------


@mx.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def _delta_rule_chunkwiseq, k, v, beta, *, chunk_size: int = 32):
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

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
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
    tri_strict = mx.triu(tri, 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        att_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + att_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Optional typing helper
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main **DeltaNet** implementation – Synergetic Fusion
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):  # noqa: D401
    """DeltaNet layer with statistics-aware gate, bounded temperature, partial
    decaying floor and adaptive residual bypass(identifier:, *syngf*)."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, # --------------------- core API ---------------------,, mode: str = "syngf",
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
        # --------------------- FIR kernels ------------------
        fir_kernel_short: int = 5,
        fir_kernel_long: int = 64,
        # --------------------- gating -----------------------
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0 3.0),
        # bounded temperature init (τ≈1)
        gate_log_temp_init: float = math.log(math.expm1(0.5)),
        # partial floor schedule
        floor_init: float = 0.05,
        floor_final: float = 0.02,
        floor_decay: float = 10_000.0,
        # residual bypass
        conv_residual_init: float = 0.1,
        # entropy reg
        entropy_target: float = 1.1,
        entropy_coeff: float = 0.05 **kwargs: Dict) -> None:  # noqa: D401
        super().__init__()

        # ----------- basic setup ----------------------------
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

        # ----------- dims -----------------------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value, dimensions must divide num_heads")

        # ----------- projections ----------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # ----------- short convs ----------------------------
        if not use_short_conv:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet.")
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

        # ----------- FIR branches ---------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ----------- gating network -------------------------
        self.stat_dim = 16  # mean/var/abs-mean/l2 × 4 branches
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # logits per, path)
        with mx.disable_grad():
            self.fusion_gate_mlp[-1].bias[:] = mx.tensor(gate_bias_init)

        # per-head log-temperature(ensure, τ>=0.5)
        self.log_temp = mx.array(mx.full((num_heads), gate_log_temp_init))

        # per-head residual bypass parameter (sigmoid)
        self.conv_residual_logit = mx.array(mx.full((num_heads), math.log(conv_residual_init, / (1 - conv_residual_init))))

        # ----------- output norm / proj ---------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

        # ----------- floor schedule & entropy ---------------
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay = float(floor_decay)
        # register_buffer removed for MLX
        persistent = False)

        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)
        self.reg_loss: Optional[mx.array] = None

    # -------------------------------------------------------
    # helpers
    # -------------------------------------------------------
    @staticmethod
    def _per_head_stats(x:, mx.array) -> mx.array:  # (B,L,H, D) -> (B,L,H, 4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    def _current_floor(self) -> float:
        t = float(self._step.item())
        return self.floor_final + (self.floor_init - self.floor_final) * math.exp(-t, / self.floor_decay)

    # -------------------------------------------------------
    # forward
    # -------------------------------------------------------
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def forward(, self,
        hidden_states: mx.array,  # (B,L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be(batch, seq_len)"
        B_orig, L_in, _ = hidden_states.shape

        # optional unpadding for variable-length sequences
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # retrieve previous conv state
    last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        # projections + short conv
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

        # head reshape
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # activations / norms on Q,K
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # β for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule path
        delta_out_d
        recur_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # local FIR paths
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # stats for gating
        stats = mx.cat([, self._per_head_stats(local_short))
            self._per_head_stats(local_long),
            self._per_head_stats(delta_out),
            self._per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H, 16)
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)
        gate_in = mx.cat([hs_exp, stats]
        dim=-1)  # (B, L, H D+16)
        gate_in_flat = _rearrange(gate_in, "b l h d -> (b, l, h) d")
        logits_flat = self.fusion_gate_mlp(gate_in_flat)
        logits = _rearrange(logits_flat, "(b, l, h) p -> b l h p"
        b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)

        # temperature scaling (τ>=0.5)
        temp = 0.5 + F.softplus(self.log_temp)  # (H)
        logits = logits / temp.reshape(1, 1, -1, 1)

        fusion_weights = mx.softmax(logits, dim = -1)  # (B,L,H, 4)

        # partial decaying local floor
    eps_now = self._current_floor()
        if eps_now > 0:
            scale = 1.0 - 2 * eps_now
        fusion_weights = fusion_weights * scale
            fusion_weights[..., 0] = fusion_weights[..., 0] + eps_now  # short
            fusion_weights[...
            1] = fusion_weights[..., 1] + eps_now  # long
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)

        # entropy regularisation
        entropy = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(-1).mean()
        self.reg_loss = self.entropy_coeff * mx.relu(self.entropy_target, - entropy)

        # weighted fusion
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # adaptive residual bypass
    alpha = mx.sigmoid(self.conv_residual_logit).reshape(1, 1, self.num_heads, 1)  # (1,1,H, 1)
        local_total = fusion_weights[..., 0:2].sum(-1, keepdim=True)  # (B, L, H, 1)
        alpha_scaled = alpha * (1.0 - local_total)
        o = o + alpha_scaled * 0.5 * (local_short + local_long)

        # cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # output norm / proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # re-pad if needed
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)

        # step++
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
