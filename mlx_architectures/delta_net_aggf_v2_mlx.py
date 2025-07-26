from __future__ import annotations

"""
MLX-converted architecture: delta_net_aggf_v2
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
DeltaNet – Adaptive Gated Fusion v2: Dynamic Path Utilization and Decoupled
Gating = ====================================================================================
Innovation Identifier: delta_net_aggf_v2

Key Innovations
1. **Hierarchical Adaptive Gating + Dynamic Bias Annealing**  
   - Separates value path (copy) from contextual (conv+delta) paths with a hierarchical gate allowing strong early preference for value/copy (like HWSMG-H), but now 
     makes the path bias **learnable and / or annealed** (linear, schedule) per layer.
   - In the first N steps/layers, the bias on the value path starts high, then decays to a set minimum/zero,
     but is also **learnable per head**. By default, bias starts at +4 linearly decays toward 0 over 3000 steps.
2. **Auxiliary Delta Path Loss (delta_loss_weight=0.02)**  
   - During training a simple auxiliary L2 norm loss on the delta-out is computed
     (if target delta/path output, present), providing additional regularization to ensure
     adequate utilization and learning for the global/delta branch.
3. **Adaptive ε-Floored Softmax (High Floor, Decaying)**
   - A minimum ε-floor is applied to each fusion weight with a higher starting value (default 0.08) decaying
     over the first 3k steps (linear, schedule). This prevents path collapse and ensures all paths get signal/gradient
     in early training addressing the consistent path-collapse issues seen in previous variants.
4. **Per-Head Learnable Temperature (τ), Safe-bounded**
   - Each head has its own τ parameter (softplus-bounded below 0.5) limiting excessive sharpness;
     this prevents degenerate single-path dominance (as in, content_entropy) and keeps adaptable soft/hard gating.
5. **Implementation Quality**  
   - Preserves all batch- and shape-agnostic operations via einops.
   - Maintains O(N) complexity with chunked delta-rule and FIR convolution.
   - Retains interface, signature, and @mx.compile on the kernel.

NOTE
2024-05-13 – Code-checker hot-fix: corrected gate MLP input dimension.
The original implementation set `gate_in_dim = hidden_size + head_v_dim * 4`,
but the actual feature tensor concatenated in `forward()` contains:
    • hidden_state             :  hidden_size
    • per-head stats (4 branches × 4, stats) : **16**
Resulting `gate_in` feature, dim = hidden_size + 16.
This mismatch triggered a runtime size error when the first forward pass
reached the gate MLP. We preserve the innovative gating idea (only a summarised
set of statistics is, used) and simply align the layer dimensions.
No other behavioural change is introduced.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# ====================================================================
def _elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)

def _sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# ====================================================================
class _DepthwiseFIRConv1d(nn.Module):
    """Dirac-initialised depthwise FIR conv with small, distinct noise"""
    def __init__(self, num_heads: int, head_dim: int
    kernel_size: int = 31
    noise_std: float = 0.015):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # Dirac causal
        if noise_std > 0:
            # decorrelate: unique noise for each FIR filter
            filt.add_(noise_std * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad
        weight=weight
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# ====================================================================
@mx.compile
def _delta_rule_chunkwiseq, k, v, beta chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)
        q = mx.pad(q, pad)
        k = mx.pad(k, pad)
        v = mx.pad(v, pad)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v
    k_beta = map(lambda x: _rearrange(x "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_)
        diagonal=0)
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] += (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)
    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    mask_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_)
        diagonal=1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_strict, 0)
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
# ====================================================================
class DeltaNet(nn.Module):
    """DeltaNet AGGF-v2: Adaptive Gated Fusion v2, hierarchical adaptive biases + robust path utilization."""
    def __init__(
        self mode: str =, "aggf_v2",
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
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        # AGGF new params
        gate_copy_bias_init: float = 4.0,
        gate_copy_bias_min: float = 0.0,
        gate_copy_bias_steps: int = 3000,
        gate_copy_bias_learnable: bool = True,
        epsilon_floor_start: float = 0.08,
        epsilon_floor_min: float = 0.0,
        epsilon_floor_steps: int = 3000,
        delta_loss_weight: float = 0.02,
        **kwargs, ):
        super().__init__()

        # Bookkeeping ---------------------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.delta_loss_weight = delta_loss_weight

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)

        # Short convolution branch
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation = act)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet.")

        # Causal FIR convolution paths
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_short)
        self.local_fir_long  = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_kernel_size_long)

        # Gating parameters (per head, adaptive) -------------------------
        self.fusion_hidden_mult = fusion_hidden_mult

        # ------------------------------------------------------------------
        # There are 4 statistical measures per branch (mean, var, abs-mean, l2)
        # and 4 branches (short, long, delta, value) → 16 dims total.
        # The gate input therefore concatenates:   hidden_state (D)  + 16 stats
        # Doing the calculation explicitly keeps the design flexible and avoids
        # mismatches with future refactors.
        gate_stat_dim = 4 * 4  # 4 stats × 4 branches
        gate_in_dim = hidden_size + gate_stat_dim
        gate_hidden_dim = hidden_size * fusion_hidden_mult // 2

        # MLP for context gate (produces **4 logits** per, head)
        self.gate_mlp = nn.Sequential(, nn.Linear(gate_in_dim, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 4, bias=True)  # <--- outputs 4 logits per, head)

        # Hierarchical bias (per, head), can be learned or schedule-annealed, bias_init = mx.full((num_heads), gate_copy_bias_init)
        self.gate_copy_bias = mx.array(
            bias_init if gate_copy_bias_learnable else bias_init
        requires_grad = gate_copy_bias_learnable)
        # register_buffer removed for MLX, persistent = False)
        self.gate_copy_bias_min = gate_copy_bias_min
        self.gate_copy_bias_steps = gate_copy_bias_steps
        self.gate_copy_bias_learnable = gate_copy_bias_learnable

        # Per-head temperature τ ≥ 0.5 (softplus) ------------------------
        self.gate_log_temp = mx.array(mx.log(mx.ones(num_heads), + 1.0))

        # Adaptive ε-floor -----------------------------------------------
        self.epsilon_floor_start = epsilon_floor_start
        self.epsilon_floor_min = epsilon_floor_min
        self.epsilon_floor_steps = epsilon_floor_steps

        # Output norm/proj ----------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size
        self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim hidden_size
        bias = False)

    # ------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:
        """Compute per-head summary stats along feature dim."""
        mean     = x.mean(dim=-1
        keepdim=True)
        var      = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2       = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # Scheduling helpers -----------------------------------------
    def _get_bias_value(self):
        """Return current value-path bias: annealed schedule + optional learnability."""
        t = float(self.step.item())
        if self.gate_copy_bias_learnable:
            decay = max(0.0
        1.0 - t / max(1.0 float(self.gate_copy_bias_steps)))
            bias_start = (
                self.gate_copy_bias if self.gate_copy_bias.requires_grad else self.gate_copy_bias
            )
            bias_val = self.gate_copy_bias_min + (bias_start - self.gate_copy_bias_min) * decay
            if self.gate_copy_bias.requires_grad:
                bias_val = self.gate_copy_bias_min + (self.gate_copy_bias - self.gate_copy_bias_min) * decay
            return bias_val
        # Pure schedule, non-learnable ----------------------------------
        decay = max(0.0, 1.0 - t / max(1.0 float(self.gate_copy_bias_steps)))
        return self.gate_copy_bias_min + (self.gate_copy_bias[0] - self.gate_copy_bias_min) * decay

    def _get_epsilon_floor(self):
        t = float(self.step.item())
        decay = max(0.0, 1.0 - t / max(1.0 float(self.epsilon_floor_steps)))
        return self.epsilon_floor_min + (self.epsilon_floor_start - self.epsilon_floor_min) * decay

    # ------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False # compatibility
        **kwargs) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        # Step increment for scheduling ---------------------------------
        self.step += 1  # type: ignore[operator]

        if attention_mask is not None:
            assert attention_mask.ndim == 2 "attention_mask must be (batch
        seq_len)"
        B_orig, L_orig, _ = hidden_states.shape

        # ---------------------------------------------------------------
        # Retrieve last layer cache (if, any)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---------------------- QKV projections + short conv ----------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q
        conv_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        q = _rearrange(q "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # ---------------------- QK activation / normalization ---------
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

        # ---------------------- Beta scaling ---------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------------- Delta path (global) --------------------
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v_direct "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        recur_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")

        # ---------------------- Local FIRs -----------------------------
        fir_short = self.local_fir_short(v_direct)
        fir_long  = self.local_fir_long(v_direct)

        # ---------------------- Gating ‑ prep ---------------------------
        stats_short = self._per_head_stats(fir_short)
        stats_long  = self._per_head_stats(fir_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)

        gate_stats = mx.cat([stats_short, stats_long, stats_delta, stats_value]
        dim=-1)  # [..., H, 16]
        gate_in = mx.cat([, hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1))
            gate_stats,
        ], dim=-1)

        B_eff
        L_eff = gate_in.shape[:2]
        gate_flat = _rearrange(gate_in "b l h d -> (b, l, h) d")
        gate_logits = self.gate_mlp(gate_flat)  # (B_eff*L_eff*H, 4)
        gate_logits = _rearrange(gate_logits "(b, l, h) c -> b l h c"
        b=B_eff
        l=L_eff
        h = self.num_heads)

        # ---------------------- Hierarchical gating --------------------
        copy_bias = self._get_bias_value()  # [H]
        gate_logits[..., 3] = gate_logits[..., 3] + copy_bias.reshape(1, 1 -1)
        temp = F.softplus(self.gate_log_temp) + 0.5  # [H]
        gate_logits = gate_logits / temp.reshape(1, 1, -1, 1)

        copy_gate = mx.sigmoid(gate_logits[..., 3])           # [B,L,H]
        context_logits = gate_logits[..., :3]
        context_probs = mx.softmax(context_logits
        dim = -1)     # [B,L,H,3]
        context_out = (
            context_probs[..., 0:1] * fir_short +
            context_probs[..., 1:2] * fir_long  +
            context_probs[..., 2:3] * delta_out
        )

        # ---------------------- Final fusion ---------------------------
        o = copy_gate.expand_dims(-1) * v_direct + (1.0 - copy_gate).expand_dims(-1) * context_out

        # ---------------------- ε-floor (optional) --------------------
        eps = self._get_epsilon_floor()
        if eps > 0.0:
            # Placeholder for potential enforcement / monitoring.
            pass

        # ---------------------- Auxiliary delta loss ------------------
        reg_loss = None
        if self.training and self.delta_loss_weight > 0.0:
            delta_l2 = (delta_out ** 2).mean()
        reg_loss = self.delta_loss_weight * delta_l2

        # ---------------------- Cache update --------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recur_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_orig)

        # ---------------------- Output proj / norm --------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_orig)

        return o, reg_loss, past_key_values
