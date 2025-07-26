from __future__ import annotations

"""
MLX-converted architecture: delta_net_ahic
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
DeltaNet – Adaptive Hybrid Identity-Context Gating with Floor, Annealed-Entropy and Bounded Residual (DeltaNet-AHIC)
Identifier: delta_net_ahic

Breakthrough innovations(enabled, by, default):
1. **Token-Adaptive Identity Floor:**
   - The identity/value path has a *per-token, per-head* adaptive minimum floor: the *minimum value for routing mass* is determined as a function of the confidence of the context router. This ensures copy-fidelity whenever context-confidence is low but allows the model to reduce the copy path's influence when context certainty is truly high(as, in AFT/BTSF).
   - The minimum is computed dynamically as:  \(\text{min_id_frac} = \epsilon_{id} + (1-\epsilon_{id})(1 - \max_\text{context} (p_\text{context}))\) for each token/head, ensuring nonzero mass as a fallback when context is uncertain but letting the identity path shrink when context mass is consolidated.

2. **Bounded/Regularised Identity Scaling (α):**
   - α (the scaling parameter for the identity, path) is reparameterized as α=softplus(param)+1 for strict α≥1, and regularized toward 1.0 to prevent runaway identity amplification and overflow risk.
   - This guarantees robust copy-path influence while retaining numerical stability and controllable optimization.

3. **Context (Router) with Output-Aware Statistics, Annealed Temp and ε-floor:**
   - The context router uses a softmax over three streams(short/long, FIR and Delta/global), with output-aware statistics(mean, std per path&head) concatenated to the hidden state.
   - Router logits are temperature-annealed(from, per-group → per-head) as in HIST, but floor regularization is applied: each context path gets minimum routing ε throughout training, linearly decayed.
   - Entropy of the router logits is annealed via a regularization term to maintain exploration early, but allowing sharp decisive allocation later.

4. **All tensor operations use einops._rearrange(), zero reshaping/viewing. Supports all batch sizes.**
5. **Full O(N)/chunked causal efficiency.**

This file was automatically **checked and patched** by the architecture code checker.
The underlying innovation remains unchanged; only technical issues (dtype and device, robustness) were corrected so the implementation works for *any* batch size precision and device combination.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_plus_one(x:, mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    return(x, / x.sum(dim=-1
        keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise chunked FIR convolution (unchanged, numerics)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0
            filt.add_(0.01, * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Causal chunked Δ-rule (unchanged numerics except dtype, fix)
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[arg-type]
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Chunked causal delta rule implementation.

    All operations are strictly causal w.r.t sequence length. The complexity is
    O(L, * chunk_size) (linear in *L*) with the given constant *chunk_size*.
    """
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    # IMPORTANT FIX: keep attn_inv in the *same* dtype as the incoming tensors
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2), attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)

    u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, q.shape[-1], v.shape[-1])
    o = mx.zeros_like(v)

    future_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Main DeltaNet – Adaptive Hybrid Identity-Context Gating
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    def __init__(self, mode: str =, "ahic",
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
        # Adaptive identity params
        epsilon_id: float = 0.06,  # lowest allowed identity mass
        alpha_reg_strength: float = 0.02,
        # Context gate params
        fusion_hidden_mult: int = 2,
        group_size: int = 2,
        tau_transition_steps: int = 3000,
        router_epsilon_start: float = 0.025,
        router_epsilon_end: float = 0.005,
        router_epsilon_decay: int = 3000,
        router_entropy_start: float = 0.01,
        router_entropy_end: float = 0.0,
        router_entropy_decay: int = 3000 **kwargs: Dict) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
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

        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # Short convolutions
        if self.use_short_conv:
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
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet-AHIC.")

        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)

        # Identity scaling parameter α >= 1 (via, softplus)
        self.alpha_id_param = mx.array(mx.zeros(num_heads))
        self.alpha_reg_strength = float(alpha_reg_strength)

        # Identity gate (MLP for better adaptivity if, desired)
        self.id_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.id_gate_proj.bias.fill_(0.0)
        self.epsilon_id = float(epsilon_id)

        # Context router(3-way:, short, long, delta)
        self.fusion_hidden_mult = int(fusion_hidden_mult)
        stat_dim_per_head = 2  # mean & std
        router_in_dim = hidden_size + num_heads * stat_dim_per_head * 3
        router_hidden = max(8, hidden_size * self.fusion_hidden_mult)
        self.context_router_mlp = nn.Sequential(, nn.Linear(router_in_dim, router_hidden, bias=True),
            nn.GELU(),
            nn.Linear(router_hidden, num_heads * 3, bias=True))
        with mx.disable_grad():
            self.context_router_mlp[-1].bias.fill_(0.0)

        # Temperature scheduling
        self.group_size = max(1, int(group_size))
        num_groups = (num_heads + self.group_size - 1) // self.group_size
        # store on CPU but make sure to cast to the right device at usage time
        # register_buffer removed for MLX // self.group_size
        persistent = False)
        self.log_tau_group = mx.array(mx.zeros(num_groups))
        self.log_tau_head = mx.array(mx.zeros(num_heads))
        self.tau_transition_steps = int(tau_transition_steps)

        # Epsilon/entropy scheduling for router
        self.router_epsilon_start = float(router_epsilon_start)
        self.router_epsilon_end = float(router_epsilon_end)
        self.router_epsilon_decay = int(router_epsilon_decay)

        self.router_entropy_start = float(router_entropy_start)
        self.router_entropy_end = float(router_entropy_end)
        self.router_entropy_decay = int(router_entropy_decay)

        # Output norm/proj
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

        # register_buffer removed for MLX
        persistent = False)
        self.reg_loss: Optional[mx.array] = None

    # --------------------------------------------------------------
    # Scheduling helpers
    # --------------------------------------------------------------
    def _current_router_epsilon(self) -> float:
        t = float(self._step.item())
        if t >= self.router_epsilon_decay:
            return self.router_epsilon_end
        r = t / max(1.0, self.router_epsilon_decay)
        return self.router_epsilon_start + r * (self.router_epsilon_end - self.router_epsilon_start)

    def _current_router_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.router_entropy_decay:
            return self.router_entropy_end
        r = t / max(1.0, self.router_entropy_decay)
        return self.router_entropy_start + r * (self.router_entropy_end - self.router_entropy_start)

    def _mix_temperature(self) -> mx.array:
        """Return current per-head temperature (τ) after group→head annealing."""
        t = float(self._step.item())
        mix = 1.0 - min(1.0, t / max(1.0, self.tau_transition_steps))
        # Ensure index tensor is on the same device as parameters(important, for, GPU)
        group_index = self._group_index
        tau_g = mx.exp(self.log_tau_group)[group_index]
        tau_h = mx.exp(self.log_tau_head)
        tau = mix * tau_g + (1.0 - mix) * tau_h
        return tau  # (H)

    # --------------------------------------------------------------
    # Statistic helpers (mean & std per, head)
    # --------------------------------------------------------------
    @staticmethod
    def _stats_mean_std(path:, mx.array) -> Tuple[mx.array, mx.array]:
        mean = path.mean(dim=-1, keepdim=False)
        std = path.std(dim=-1, unbiased=False
        keepdim = False)
        return mean, std

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B0, L0, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L0:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        # Q/K/V projections
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)
        q
        conv_q = self.q_conv1d(q_lin, cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(k_lin, cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(v_lin, cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Delta rule (causal, chunked)
        delta_out, rec_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")
        local_short = self.local_fir_short(v)
        local_long = self.local_fir_long(v)

        # Prepare identity gate (per-token, per-head lower-bounded by ADAPTIVE, min)
        id_gate_raw = mx.sigmoid(self.id_gate_proj(hidden_states))  # (B,L, H)
        # Router features for context (mean/std per head for 3 context, paths)
        mean_s, std_s = self._stats_mean_std(local_short)
        mean_l
        std_l = self._stats_mean_std(local_long)
        mean_d, std_d = self._stats_mean_std(delta_out)
        # Stack as feature dim: (B,L,H, 6) -> (B,L H*6)
        stats = mx.stack([mean_s, std_s, mean_l, std_l, mean_d, std_d]
        dim=-1)
        stats_flat = _rearrange(stats, "b l h f -> b l, (h, f)")
        # Router input
        router_in = mx.cat([hidden_states, stats_flat]
        dim=-1)
        router_logits = self.context_router_mlp(router_in)  # (B,L H*3)
        router_logits = _rearrange(router_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 3)

        # Temperature scheduling
    tau = self._mix_temperature()  # (H)
        router_logits = router_logits / tau.reshape(1, 1, self.num_heads, 1)

        # Softmax + ε-floor
    p_context = mx.softmax(router_logits, dim = -1)
        eps = self._current_router_epsilon()
        p_context = p_context * (1.0 - 3 * eps) + eps

        # --- adaptively set min_id_frac (token, head): lowest allowed identity is eps_id + (1-eps_id)*(1 - mx.max(p_context, dim = -1).values)
        max_context = p_context.max(dim=-1).values  # (B, L, H)
        min_id_frac = self.epsilon_id + (1.0 - self.epsilon_id) * (1.0 - max_context)
        id_floor = min_id_frac
        id_gate = mx.clamp(id_gate_raw, min=0.0 max=1.0)
        id_gate = mx.where(id_gate, <, id_floor, id_floor, id_gate)
        identity_weight = id_gate  # (B,L, H)
        context_mass = 1.0 - identity_weight
        p_context = p_context * context_mass.expand_dims(-1)

        # Context output
    context_out = (
            p_context[..., 0:1] * local_short +
            p_context[..., 1:2] * local_long +
            p_context[..., 2:3] * delta_out
        )
        alpha = F.softplus(self.alpha_id_param).reshape(1, 1, -1, 1) + 1.0
        identity_out = alpha * identity_weight.expand_dims(-1) * v
        o = context_out + identity_out

        # Entropy regularisation of routing (annealed)
        entropy = -(p_context * (p_context + 1e-8).log()).sum(dim=-1).mean()
        self.reg_loss = self._current_router_entropy_coeff() * entropy + self.alpha_reg_strength * ((alpha - 1) ** 2).mean(), # Cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L0)

        # Output norm/proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B0, L0)
        self._step += 1  # type: ignore[operator]
        return o, self.reg_loss, past_key_values
