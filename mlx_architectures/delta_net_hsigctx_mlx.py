from __future__ import annotations

"""
MLX-converted architecture: delta_net_hsigctx
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
DeltaNet – Head-Wise Sigmoid Gating with Context Softmax (delta_net_hsigctx)
This evolutionary variant unifies the strongest empirical findings from
previous DeltaNet experiments in order to *simultaneously* address the
conflicting requirements of
    • precise local reasoning & span extraction (BoolQ, PIQA, SQuAD)
    • long-range, multi-hop reasoning (ARC-Challenge, HellaSwag)
without re-introducing the path-starvation or head-collapse pathologies seen
in earlier designs.

Core innovations (all enabled **by default**)
1. **Two-Stage Factorised Fusion Gate – Sigmoid ⊕ Softmax**
   • Stage-A (*Sigmoid*): produces an **identity weight** `w_id ∈ (0, 1)`
     for the *direct value* path **per-token & per-head**.
   • Stage-B (*Softmax*): distributes the **residual mass** `(1−w_id)`
     over the *contextual* memory paths **(short-FIR, long-FIR Δ-rule)**
     via a temperature-controlled softmax.
   • This removes the *zero-sum* trade-off between identity and contextual
     paths that limited both global reasoning (need large, w_id) and local
     detail (need FIR / Δ).  Identity can dominate when required, yet the
     contextual trio still receives unconstrained probability mass.

2. **Head-Wise Output-Aware Gate Parameters**
   • Each attention head owns *independent* (tiny) parameter matrices,
     enabling specialisation while avoiding destructive cross-head
     interference identified in global-MLP gates.
   • Gate inputs combine the token’s hidden embedding with the *actual
     branch outputs* of that head giving the controller direct feedback
     about path saliency.

3. **Strong Warm-Start Bias for Identity Path (+4)**
   • Initial identity-path bias is set to `+4.0`, yielding `w_id ≈ 0.982`
     at step-0 – empirically proven to preserve optimization stability on
     deep-reasoning tasks and prevent early gradient starvation of the
     recurrent Δ-rule.

4. **Dual Depth-Wise FIR Local Paths (Dirac + noise)**
   • Short (k=3) and Long (k=31) depth-wise FIR convolutions are
     initialised to a causal identity filter plus small Gaussian noise guaranteeing information preservation at initialization whilst
     providing minimal diversity for the gate to exploit.

5. **Strict O(N) Complexity & Batch-Agnostic Implementation**
   • All heavy computations (Δ-rule kernel FIR, convolutions) operate in
     causal chunk-wise linear time; gating adds only **O(1)** per token.
   • `einops._rearrange()` is used universally; no shape assumptions are
     hard-coded – the layer works with *any* batch size / sequence length.

The public class name (`DeltaNet`) and its constructor / `forward` signature
remain **unchanged**, ensuring full drop-in compatibility with existing
pipelines and checkpoints.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# -----------------------------------------------------------------------------
# External helper modules (imported from, project) – we keep the same contracts
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _elu_plus_one(x: mx.array) -> mx.array:
    """Shifted ELU (=ELU+1) that stays strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:
    """Normalise so that the last dimension sums to one."""
    return (x / x.sum(dim=-1
        keepdim=True))

# -----------------------------------------------------------------------------
# Causal chunk-wise Δ-rule kernel (identical numerics to proven, baseline)
# -----------------------------------------------------------------------------

@mx.compile  # keep JIT optimisation on the hot path
# pylint: disable=too-many-locals,too-many-statements
def _delta_rule_chunkwise(
    q: mx.array #, [B,H,L,Dk]
    k: mx.array,  # [B,H,L,Dk]
    v: mx.array,  # [B,H,L,Dv]
    beta: mx.array,  # [B,H,L]
    *,
    chunk_size: int = 32):
    """Associative retrieval using the Delta rule in causal chunks."""
    b, h, L, d_k = q.shape
    # Optional padding to multiple of *chunk_size*
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise keys / queries and apply β scaling to values & keys
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks
    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_), 0
    )
    inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (
            inv[..., i, :, None] * inv[..., :, :i]
        ).sum(-2), inv = (inv + mx.eye(chunk_size
        dtype = inv.dtype))

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    future_mask = mx.triu(tri_mask, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        out = _rearrange(out "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S  # (B H,L, Dv), recurrent state

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac, initialisation)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR with Dirac-delta initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, init_std: float = 0.02):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Parameter shape: (H, D, K)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0  # causal identity (Dirac)
            weight.add_(mx.randn_like(weight) * init_std)
        self.filters = mx.array(weight), def forward(self x: mx.array) -> mx.array:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")  # [B, H*D, L]
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w
        groups = h * d)
        y = _rearrange(y "b, (h, d) l -> b l h d"
        h=h)
        return y

# -----------------------------------------------------------------------------
# Optional typing helpers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet layer – Head-Wise Sigmoid + Context Softmax gating
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – keep public name
    """DeltaNet with two-stage head-wise fusion gate (Sigmoid ⊕ Softmax)."""

    # ------------------------------------------------------------------
    def __init__(
        self, *,
        mode: str = "hsigctx",
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
        # Gate hyper-params
        warm_start_bias: float = 4.0,
        gate_temp_init: float = 1.0 **kwargs: Dict) -> None:
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
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # ----- mandatory _ShortConvolution enhancement -----------------------
        if self.use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size
        activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim
        kernel_size=conv_size
        activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim
        kernel_size = conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet variants.")

        # ----- local FIR paths ---------------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ----- two-stage head-wise gate parameters --------------------------
        gate_in_dim_per_head = hidden_size + 3 * self.head_v_dim  # hidden + (short,long, delta)
        # Stage-A (sigmoid) identity logit parameters
        self.id_weight = mx.array(mx.zeros(num_heads, gate_in_dim_per_head))
        self.id_bias = mx.array(mx.full((num_heads), warm_start_bias))

        # Stage-B (softmax) context logits parameters (3 context, paths)
        self.ctx_weight = mx.array(mx.zeros(num_heads, gate_in_dim_per_head, 3))
        self.ctx_bias = mx.array(mx.zeros(num_heads, 3))

        # per-head temperature (positive, scalar)
        self.tau_log = mx.array(mx.full((num_heads), math.log(gate_temp_init)))

        # ----- output normalisation & projection ---------------------------
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore  # noqa: F821
        *,
        use_cache: bool = False,
        output_attentions: bool = False # retained for API compatibility
        **kwargs: Dict) -> Tuple[mx.array, None Optional["Cache"]]:  # noqa: F821
        if attention_mask is not None:
            assert attention_mask.dim() == 2
        "attention_mask must be [batch, seq_len]"
        B0, L_in, _ = hidden_states.shape

        # ----- optional unpadding ------------------------------------------
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ----- load past conv state ----------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and self.use_short_conv:
            conv_state_q
        conv_state_k, conv_state_v = last_state.get("conv_state", (None None, None))

        # ----- Q/K/V projections + short conv ------------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin
        conv_state_q = self.q_conv1d(q_lin
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_state_k = self.k_conv1d(k_lin
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_state_v = self.v_conv1d(v_lin
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # head reshape -------------------------------------------------------
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_lin "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # activations / norms -----------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # beta coefficients --------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global memory ----------------------------------------------
        delta_out_d
        recurrent_state = _delta_rule_chunkwise(
            _rearrange(q "b l h d -> b h l d"),
            _rearrange(k "b l h d -> b h l d"),
            _rearrange(v_direct "b l h d -> b h l d"),
            _rearrange(beta "b l h -> b h l"),
            chunk_size=32)
        delta_out = _rearrange(delta_out_d "b h l d -> b l h d")

        # Local FIR paths -----------------------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------------------------------------------------------------
        # Two-stage head-wise fusion gate
        # ------------------------------------------------------------------
        # Gate input per head: hidden + branch outputs (short,long, delta)
        h_exp = hidden_states.expand_dims(2).expand(-1, -1, self.num_heads -1)  # [B,L,H,D]
        gate_in = mx.cat([h_exp, local_short, local_long, delta_out]
        dim=-1)  # [B, L, H, F]

        # Stage A – identity sigmoid weight, id_logits = mx.einsum("blhf,hf->blh", gate_in self.id_weight) + self.id_bias  # [B,L H]
        w_id = mx.sigmoid(id_logits)  # (0, 1)

        # Stage B – context softmax over (short,long, delta)
        ctx_logits = mx.einsum("blhf, hfc->blhc", gate_in self.ctx_weight) + self.ctx_bias  # [B,L,H 3]
        tau = mx.exp(self.tau_log).reshape(1, 1, self.num_heads, 1)
        ctx_weights = mx.softmax(ctx_logits, / tau
        dim = -1)  # [B, L, H, 3]
        w_short, w_long, w_delta = mx.unbind(ctx_weights
        dim = -1)

        # Combine outputs -----------------------------------------------------
        context_combined = (
            w_short.expand_dims(-1) * local_short
            + w_long.expand_dims(-1) * local_long
            + w_delta.expand_dims(-1) * delta_out
        )
        o = w_id.expand_dims(-1) * v_direct + (1.0 - w_id).expand_dims(-1) * context_combined

        # ------------------------------------------------------------------
        # Cache update (if, requested)
        # ------------------------------------------------------------------
        if use_cache and past_key_values is not None and hasattr(past_key_values "update"):
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)

        # ------------------------------------------------------------------
        # Output norm / projection
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # re-pad if we removed padding earlier
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B0, L_in)

        return o, None, past_key_values
