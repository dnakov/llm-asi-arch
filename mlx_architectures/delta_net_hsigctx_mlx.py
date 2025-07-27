# -*- coding: utf-8 -*-
"""
DeltaNet – Head-Wise Sigmoid Gating with Context Softmax (delta_net_hsigctx)
===========================================================================
This evolutionary variant unifies the strongest empirical findings from
previous DeltaNet experiments in order to *simultaneously* address the
conflicting requirements of
    • precise local reasoning & span extraction (BoolQ, PIQA, SQuAD)
    • long-range, multi-hop reasoning (ARC-Challenge, HellaSwag)
without re-introducing the path-starvation or head-collapse pathologies seen
in earlier designs.

Core innovations (all enabled **by default**)
---------------------------------------------
1. **Two-Stage, Factorised Fusion Gate – Sigmoid ⊕ Softmax**
   • Stage-A (*Sigmoid*): produces an **identity weight** `w_id ∈ (0,1)`
     for the *direct value* path **per-token & per-head**.
   • Stage-B (*Softmax*): distributes the **residual mass** `(1−w_id)`
     over the *contextual* memory paths **(short-FIR, long-FIR, Δ-rule)**
     via a temperature-controlled softmax.
   • This removes the *zero-sum* trade-off between identity and contextual
     paths that limited both global reasoning (need large w_id) and local
     detail (need FIR / Δ).  Identity can dominate when required, yet the
     contextual trio still receives unconstrained probability mass.

2. **Head-Wise, Output-Aware Gate Parameters**
   • Each attention head owns *independent* (tiny) parameter matrices,
     enabling specialisation while avoiding destructive cross-head
     interference identified in global-MLP gates.
   • Gate inputs combine the token's hidden embedding with the *actual
     branch outputs* of that head, giving the controller direct feedback
     about path saliency.

3. **Strong Warm-Start Bias for Identity Path (+4)**
   • Initial identity-path bias is set to `+4.0`, yielding `w_id ≈ 0.982`
     at step-0 – empirically proven to preserve optimization stability on
     deep-reasoning tasks and prevent early gradient starvation of the
     recurrent Δ-rule.

4. **Dual Depth-Wise FIR Local Paths (Dirac + noise)**
   • Short (k=3) and Long (k=31) depth-wise FIR convolutions are
     initialised to a causal identity filter plus small Gaussian noise,
     guaranteeing information preservation at initialization whilst
     providing minimal diversity for the gate to exploit.

5. **Strict O(N) Complexity & Batch-Agnostic Implementation**
   • All heavy computations (Δ-rule kernel, FIR convolutions) operate in
     causal, chunk-wise linear time; gating adds only **O(1)** per token.
   • `einops.rearrange()` is used universally; no shape assumptions are
     hard-coded – the layer works with *any* batch size / sequence length.

The public class name (`DeltaNet`) and its constructor / `forward` signature
remain **unchanged**, ensuring full drop-in compatibility with existing
pipelines and checkpoints.

MLX Conversion Notes:
- Converted from PyTorch to MLX framework
- Replaced torch.nn with mlx.nn modules
- Converted tensor operations to MLX array operations
- Maintained all architectural innovations and complexity
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

# -----------------------------------------------------------------------------
# External helper modules (imported from project) – we keep the same contracts
# -----------------------------------------------------------------------------
# Note: These imports would need to be adapted for MLX versions
# For now, we'll implement minimal versions or skip complex dependencies
try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
    from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
    from fla.modules.l2norm import l2norm
except ImportError:
    # Fallback implementations for MLX
    def get_unpad_data(attention_mask):
        # Simplified implementation
        return None, None, None
    
    def index_first_axis(x, indices):
        return x
    
    def pad_input(x, indices, batch_size, seq_len):
        return x
    
    def l2norm(x):
        return x / mx.linalg.norm(x, axis=-1, keepdims=True)
    
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = mx.ones((hidden_size,))
        
        def __call__(self, x):
            variance = mx.mean(x * x, axis=-1, keepdims=True)
            x = x / mx.sqrt(variance + self.eps)
            return self.weight * x
    
    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.norm = RMSNorm(hidden_size, eps)
        
        def __call__(self, x, gate=None):
            x = self.norm(x)
            if gate is not None:
                x = x * gate
            return x
    
    class ShortConvolution(nn.Module):
        def __init__(self, hidden_size, kernel_size=4, activation=None, bias=False):
            super().__init__()
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)
            self.activation = activation
        
        def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
            # x shape: [B, L, D]
            x_t = mx.transpose(x, (0, 2, 1))  # [B, D, L]
            out = self.conv(x_t)
            out = mx.transpose(out, (0, 2, 1))  # [B, L, D]
            out = out[:, :x.shape[1]]  # Remove padding
            
            if self.activation == "silu":
                out = nn.silu(out)
            elif self.activation == "relu":
                out = nn.relu(out)
            
            final_state = None
            if output_final_state:
                final_state = out[:, -1:]  # Simple state approximation
            
            return out, final_state

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU (=ELU+1) that stays strictly positive."""
    return nn.elu(x, alpha=1.0) + 1.0


def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that the last dimension sums to one."""
    return x / mx.sum(x, axis=-1, keepdims=True)

# -----------------------------------------------------------------------------
# Causal, chunk-wise Δ-rule kernel (identical numerics to proven baseline)
# -----------------------------------------------------------------------------

def _delta_rule_chunkwise(
    q: mx.array,  # [B,H,L,Dk]
    k: mx.array,  # [B,H,L,Dk]
    v: mx.array,  # [B,H,L,Dv]
    beta: mx.array,  # [B,H,L]
    *,
    chunk_size: int = 32,
):
    """Associative retrieval using the Delta rule in causal chunks."""
    b, h, L, d_k = q.shape
    # Optional padding to multiple of *chunk_size*
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, pad_len)
        q = mx.pad(q, [(0, 0), (0, 0), pad_cfg, (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), pad_cfg, (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), pad_cfg, (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), pad_cfg])
    L_pad = L + pad_len

    # Normalise keys / queries and apply β scaling to values & keys
    q = l2norm(q)
    k = l2norm(k)
    v = v * mx.expand_dims(beta, -1)
    k_beta = k * mx.expand_dims(beta, -1)

    # Reshape into chunks
    q = rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size)
    k = rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    k_beta = rearrange(k_beta, "b h (n c) d -> b h n c d", c=chunk_size)

    tri_mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=0)
    inv = -(k_beta @ mx.transpose(k, [0, 1, 2, 4, 3]))
    inv = mx.where(tri_mask, 0, inv)
    
    for i in range(1, chunk_size):
        inv_i = mx.expand_dims(inv[..., i, :], -1) * inv[..., :, :i]
        inv = inv.at[..., i, :i].add(mx.sum(inv_i, axis=-2))
    
    eye = mx.eye(chunk_size)
    inv = inv + eye

    u = inv @ v
    w = inv @ k_beta

    S = mx.zeros((b, h, d_k, v.shape[-1]))
    out = mx.zeros_like(v)
    future_mask = mx.triu(tri_mask, k=1)
    
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = q_i @ mx.transpose(k_i, [0, 1, 3, 2])
        attn_local = mx.where(future_mask, 0, attn_local)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out = out.at[:, :, idx].set(q_i @ S + attn_local @ u_i)
        S = S + mx.transpose(k_i, [0, 1, 3, 2]) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S  # (B,H,L,Dv), recurrent state

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac initialisation)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR with Dirac-delta initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, init_std: float = 0.02):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Parameter shape: (H, D, K)
        weight = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Causal identity (Dirac) at the last position
        weight = weight.at[..., -1].set(1.0)
        # Add small Gaussian noise
        noise = mx.random.normal((num_heads, head_dim, self.kernel_size)) * init_std
        weight = weight + noise
        self.filters = weight

    def __call__(self, x: mx.array) -> mx.array:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")  # [B, H*D, L]
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = mx.pad(x_f, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        
        # Manual grouped convolution for depth-wise operation
        y_list = []
        for i in range(h * d):
            y_i = mx.conv1d(x_pad[:, i:i+1], w[i:i+1], padding=0)
            y_list.append(y_i)
        y = mx.concatenate(y_list, axis=1)
        
        y = rearrange(y, "b (h d) l -> b l h d", h=h)
        return y

# -----------------------------------------------------------------------------
# Optional typing helpers
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401 – runtime import optional

# -----------------------------------------------------------------------------
# Main DeltaNet layer – Head-Wise Sigmoid + Context Softmax gating
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – keep public name
    """DeltaNet with two-stage head-wise fusion gate (Sigmoid ⊕ Softmax)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        mode: str = "hsigctx",
        d_model: int | None = None,
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
        layer_idx: int | None = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # FIR kernel sizes
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # Gate hyper-params
        warm_start_bias: float = 4.0,
        gate_temp_init: float = 1.0,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        # ----- dimensional bookkeeping --------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value dims must divide num_heads")

        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ----- linear projections ------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ----- mandatory ShortConvolution enhancement -----------------------
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")

        # ----- local FIR paths ---------------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ----- two-stage head-wise gate parameters --------------------------
        gate_in_dim_per_head = hidden_size + 3 * self.head_v_dim  # hidden + (short,long,delta)
        # Stage-A (sigmoid) identity logit parameters
        self.id_weight = mx.zeros((num_heads, gate_in_dim_per_head))
        self.id_bias = mx.full((num_heads,), warm_start_bias)

        # Stage-B (softmax) context logits parameters (3 context paths)
        self.ctx_weight = mx.zeros((num_heads, gate_in_dim_per_head, 3))
        self.ctx_bias = mx.zeros((num_heads, 3))

        # per-head temperature (positive scalar)
        self.tau_log = mx.full((num_heads,), math.log(gate_temp_init))

        # ----- output normalisation & projection ---------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(
        self,
        hidden_states: mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore  # noqa: F821
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # retained for API compatibility
        **kwargs: Dict,
    ) -> Tuple[mx.array, None, Optional["Cache"]]:  # noqa: F821
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B0, L_in, _ = hidden_states.shape

        # ----- optional unpadding ------------------------------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices)
            hidden_states = mx.expand_dims(hidden_states, 0)

        # ----- load past conv state ----------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = last_state.get("conv_state", (None, None, None))

        # ----- Q/K/V projections + short conv ------------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin, conv_state_q = self.q_conv1d(q_lin, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_state_k = self.k_conv1d(k_lin, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_state_v = self.v_conv1d(v_lin, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # head reshape -------------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # activations / norms -----------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # beta coefficients --------------------------------------------------
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global memory ----------------------------------------------
        delta_out_d, recurrent_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
            chunk_size=32,
        )
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # Local FIR paths -----------------------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------------------------------------------------------------
        # Two-stage head-wise fusion gate
        # ------------------------------------------------------------------
        # Gate input per head: hidden + branch outputs (short,long,delta)
        h_exp = mx.expand_dims(hidden_states, 2)
        h_exp = mx.broadcast_to(h_exp, (h_exp.shape[0], h_exp.shape[1], self.num_heads, h_exp.shape[3]))
        gate_in = mx.concatenate([h_exp, local_short, local_long, delta_out], axis=-1)  # [B,L,H,F]

        # Stage A – identity sigmoid weight
        id_logits = mx.einsum("blhf,hf->blh", gate_in, self.id_weight) + self.id_bias  # [B,L,H]
        w_id = nn.sigmoid(id_logits)  # (0,1)

        # Stage B – context softmax over (short,long,delta)
        ctx_logits = mx.einsum("blhf,hfc->blhc", gate_in, self.ctx_weight) + self.ctx_bias  # [B,L,H,3]
        tau = mx.exp(self.tau_log).reshape(1, 1, self.num_heads, 1)
        ctx_weights = nn.softmax(ctx_logits / tau, axis=-1)  # [B,L,H,3]
        w_short, w_long, w_delta = mx.split(ctx_weights, 3, axis=-1)
        w_short, w_long, w_delta = mx.squeeze(w_short, -1), mx.squeeze(w_long, -1), mx.squeeze(w_delta, -1)

        # Combine outputs -----------------------------------------------------
        context_combined = (
            mx.expand_dims(w_short, -1) * local_short
            + mx.expand_dims(w_long, -1) * local_long
            + mx.expand_dims(w_delta, -1) * delta_out
        )
        o = mx.expand_dims(w_id, -1) * v_direct + mx.expand_dims(1.0 - w_id, -1) * context_combined

        # ------------------------------------------------------------------
        # Cache update (if requested)
        # ------------------------------------------------------------------
        if use_cache and past_key_values is not None and hasattr(past_key_values, "update"):
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ------------------------------------------------------------------
        # Output norm / projection
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # re-pad if we removed padding earlier
        if attention_mask is not None:
            o = pad_input(mx.squeeze(o, 0), indices, B0, L_in)

        return o, None, past_key_values