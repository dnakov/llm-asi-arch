"""
MLX-converted architecture: delta_net_afbt
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
DeltaNet – Annealed Floor & Bounded-Temperature Fusion (DeltaNet-AFBT)
=====================================================================
Identifier: delta_net_afbt

This evolutionary variant of **DeltaNet** addresses two bottlenecks discovered
in prior experiments (see *delta_net_aft* analysis):

1. **Over-Sharp / Collapsing Context Softmax**
   • Per-head temperature `τ_h` is now **lower-bounded** via a soft-plus
     transform with an additive constant `tau_min` (default **0.5**).  This
     prevents heads from collapsing to arbitrarily small temperatures that
     destroy mixture entropy and hurt span-style tasks (BoolQ, swde).

2. **Slow-Adapting Token Floor**
   • The upper bound of the token-adaptive context floor (`max_context_floor`)
     now **anneals linearly** from its initial value down to the permanent
     `min_context_floor` over `floor_decay_steps` steps (default **2 000**).
     Early in training the higher floor preserves gradient flow; as learning
     progresses the floor shrinks automatically, enabling decisive routing for
     copy-centric tasks (Winogrande, OpenBookQA) without manual scheduling.

3. **Optional Entropy Regularisation** (disabled by default)
   • An auxiliary loss `reg_loss = entropy_coeff · H(context_weights)` is stored
     as `self.reg_loss`.  Setting `entropy_coeff>0` encourages heads to keep a
     minimum amount of entropy, further mitigating premature path collapse.

All changes preserve the public API, causal O(N) complexity, chunk-wise Δ-rule,
short-convolution projections, and batch-size agnosticism.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x: mx.Tensor) -> mx.Tensor:  # shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.Tensor) -> mx.Tensor:  # sum normalisation
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise, causal FIR conv (identity initialisation – unchanged)
# -----------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 64):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # Dirac / identity kernel (causal)
        self.filters = mx.array(filt)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b (h d) l")
        weight = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))  # causal left-pad
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

# -----------------------------------------------------------------------------
# Causal chunk-wise Δ-rule  (unchanged, kept @mx.compile)
# -----------------------------------------------------------------------------
@mx.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalise & β-scale ------------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into fixed chunks ------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    att_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        att_inv[..., i, :i] += (att_inv[..., i, :, None].clone() * att_inv[..., :, :i].clone()).sum(-2)
    att_inv = att_inv + mx.eye(chunk_size, dtype=att_inv.dtype, q.device)
    att_inv = att_inv

    u = att_inv @ v
    w = att_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)
    strict_mask = mx.triu(tri_mask, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Optional static type-checking imports
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main **DeltaNet** layer – Annealed Floor & Bounded Temperature
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet with annealing context floor and lower-bounded per-head temperature."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mode: str = "afbt",  # annealed-floor bounded-temperature identifier
        d_model: int | None = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # optional components ---------------------------------------------------
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
        # FIR kernel sizes -------------------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # Fusion gate -----------------------------------------------------------
        fusion_hidden_mult: int = 2,
        fusion_include_path_outputs: bool = True,
        value_bias_init: float = 4.0,
        min_context_floor: float = 0.01,
        max_context_floor: float = 0.10,
        floor_decay_steps: int = 2000,
        # temperature bounding --------------------------------------------------
        tau_min: float = 0.5,
        # entropy regularisation -------------------------------------------------
        entropy_coeff: float = 0.0,
        fusion_dropout: float = 0.0,
        **kwargs: Dict,  # unused kwargs for compatibility
    ) -> None:
        super().__init__()

        # ---------- hyper-params ---------------------------------------------
        if d_model is not None:
            hidden_size = d_model
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

        # adaptive floor parameters
        assert 0.0 < min_context_floor < max_context_floor < 0.5, "floors must satisfy 0 < min < max < 0.5"
        self.min_context_floor = float(min_context_floor)
        self.max_context_floor = float(max_context_floor)
        self.floor_decay_steps = max(1, int(floor_decay_steps))

        # temperature parameters
        self.tau_min = float(tau_min)
        self.entropy_coeff = float(entropy_coeff)

        # ---------- dimensions ----------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")

        # ---------- projections ---------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------- short convolutions --------------------------------------
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory – do not disable.")

        # ---------- dual FIR memory branches --------------------------------
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ---------- fusion gate MLP -----------------------------------------
        fusion_in_dim = hidden_size
        self.fusion_include_path_outputs = fusion_include_path_outputs
        if fusion_include_path_outputs:
            fusion_in_dim += self.head_v_dim * self.num_heads * 3  # short + long + delta
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 4, bias=True),
        )
        # warm-start bias favouring identity path
        if self.fusion_gate_mlp[-1].bias is not None:
            with mx.no_grad():
                self.fusion_gate_mlp[-1].bias.zero_()
                self.fusion_gate_mlp[-1].bias[3::4] = value_bias_init

        # ---------- per-head log-temperature (learned) -----------------------
        self.others_log_tau = mx.array(mx.zeros(num_heads))  # log τ_h (≈0 → τ≈1)

        # ---------- output normalisation & projection -----------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ---------- step counter & reg-loss ----------------------------------
        self, persistent=False)
        self.reg_loss: Optional[mx.Tensor] = None

    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def forward(
        self,
        hidden_states: mx.Tensor,
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compatibility
        **kwargs,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape

        # ----- retrieve cached states --------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # ----- projections + short convolution ----------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ----- head split & activation ------------------------------------
        q, k = map(lambda t: _rearrange(t, "... (h d) -> ... h d"d=self.head_k_dim), (q, k))
        v = _rearrange(v, "... (h d) -> ... h d"d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        v_direct = v  # identity path

        # ----- beta coefficients -----------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ----- delta rule (global path) -----------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # ----- local FIR memories -----------------------------------------
        fir_short = self.local_fir_short(v_direct)
        fir_long = self.local_fir_long(v_direct)

        # ----- fusion gate inputs ----------------------------------------
        if self.fusion_include_path_outputs:
            gate_input = mx.cat([
                hidden_states,
                _rearrange(fir_short, "b l h d -> b l (h d)"),
                _rearrange(fir_long, "b l h d -> b l (h d)"),
                _rearrange(delta_out, "b l h d -> b l (h d)"),
            ], dim=-1)
        else:
            gate_input = hidden_states

        fusion_logits = self.fusion_gate_mlp(gate_input)  # (B,L,H*4)
        fusion_logits = _rearrange(fusion_logits, "b l (h c) -> b l h c"h=self.num_heads, c=4)

        # value/identity logit & raw probability ---------------------------
        value_logit = fusion_logits[..., 3]
        p_val_raw = mx.sigmoid(value_logit)  # (B,L,H)

        # ---- compute current max_floor (linear decay) --------------------
        step_float = float(self._step.item())
        decay_ratio = min(1.0, step_float / self.floor_decay_steps)
        current_max_floor = self.min_context_floor + (self.max_context_floor - self.min_context_floor) * (1.0 - decay_ratio)

        # ---- token-adaptive context floor --------------------------------
        floor_tok = self.min_context_floor + (current_max_floor - self.min_context_floor) * (1.0 - p_val_raw)

        # final value probability scaled so that others_total ≥ floor_tok
        p_value = (1.0 - floor_tok) * p_val_raw  # (B,L,H)
        others_total = 1.0 - p_value  # guaranteed ≥ floor_tok

        # ---- contextual softmax with bounded τ ---------------------------
        others_logits = fusion_logits[..., 0:3]  # (B,L,H,3)
        # τ_h ≥ tau_min via softplus + tau_min
        tau = F.softplus(self.others_log_tau) + self.tau_min  # (H,)
        tau = tau[None, None, :, None]  # broadcast
        others_logits_scaled = others_logits / tau
        others_weights = mx.softmax(others_logits_scaled, dim=-1)
        others_weights = others_weights * others_total.expand_dims(-1)

        # entropy reg (optional) ------------------------------------------
        if self.entropy_coeff > 0.0 and self.training:
            entropy = -(others_weights * mx.log(others_weights + 1e-8)).sum(-1).mean()
            self.reg_loss = self.entropy_coeff * entropy
        else:
            self.reg_loss = None

        # ----- final mixture ---------------------------------------------
        o = (
            others_weights[..., 0:1] * fir_short
            + others_weights[..., 1:2] * fir_long
            + others_weights[..., 2:3] * delta_out
            + p_value.expand_dims(-1) * v_direct
        )

        # ----- cache update ----------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ----- output normalisation & projection -------------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d"d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ----- restore padding if removed --------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, seq_len)

        # increment step counter -----------------------------------------
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
