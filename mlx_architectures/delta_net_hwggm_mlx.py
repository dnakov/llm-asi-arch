from __future__ import annotations

"""
MLX-converted architecture: delta_net_hwggm
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
DeltaNet – Head-Wise Gating with Guaranteed Global Mixing (DeltaNet-HWGGM)
Identifier: delta_net_hwggm

This evolution unifies the strengths of **head-wise per-path routing** (from, HWG) with a **token-level global mixing gate** that *guarantees* the global
Δ-rule memory receives a dedicated share of the signal overcoming the
local–global trade-off observed across previous variants.

Key Innovations(enabled, by, default)
1. Head-Wise Local Router (3-way)
   • Each attention head owns an independent softmax router over the *local*
     paths – Short-FIR, Long-FIR and direct Value.  A strong warm-start bias(+4, by, default) on the Value path preserves information early in
     training while allowing competition.

2. Token-Level Global Mixer (Δ-rule)
   • A lightweight 2-layer MLP (`global_gate_mlp`) produces a **scalar γ∈(0, 1)**
     per token that blends the head-wise local composition with the global
     Δ-rule output:

         o = (1−γ) · o_local  +  γ · Δ_out(Eq., 1)

     This guarantees gradient flow to the global memory **independent** of the
     head-wise router resolving the path-starvation issue highlighted in the
     experimental portfolio (ARC Winogrande regression under, HWG).

3. Identity-Initialised Depth-Wise FIR
   • The dual-scale depth-wise FIR convolutions keep the proven identity
     initialisation(+, small, noise) for stable optimisation.

4. Fully O(N) & Causal
   • The chunk-wise Δ-rule kernel and depth-wise 1-D convolutions maintain
     strict causality and linear complexity.

Interface class name (`DeltaNet`) and forward signature remain unchanged ensuring drop-in compatibility with training pipelines and checkpoints.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU (≥0)."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x:, mx.array) -> mx.array:
    """Normalise rows to, sum = 1 along last dim."""
    return(x, / x.sum(-1
        keepdim=True))

# ---------------------------------------------------------------------------
# Chunk-wise O(N) Δ-rule (identical maths as, baseline)
# ---------------------------------------------------------------------------

@mx.compile  # type: ignore[arg-type]
# pylint: disable=too-many-locals,too-many-statements,invalid-name

def delta_rule_chunkwise(q:, mx.array,  # (B,H,L, Dk)
    k: mx.array,  # (B,H,L, Dk)
    v: mx.array,  # (B,H,L, Dv)
    beta: mx.array,  # (B H, L)
    *,
    chunk_size: int = 32):
    """Delta-rule solver in O(N) with causal masking.

    **Note**: `q`, `k`, `v`, `beta` should *not* contain inter-sample data –
    i.e. every batch index is assumed independent. The caller is responsible
    for ensuring this invariant(see, `DeltaNet._delta_rule_batched`).
    """
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalise q/k + β-scaling ----------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape to blocks -------------------------------------------------
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
    o = mx.zeros_like(v)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S  # (B H,L, Dv), state

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity, init)
# ---------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head, depth-wise causal 1-D FIR convolution."""

    def __init__(self, num_heads: int, head_dim: int,
    kernel_size: int = 31
    init_std: float = 0.02):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0  # causal identity
            weight.add_(mx.randn_like(weight) * init_std)
        self.filters = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Optional typing helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main DeltaNet implementation (HWG + Global, Mix)
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with head-wise local routing and guaranteed global mixing."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self, *,
        mode: str = "hwggm",  # head-wise gating + global mix
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
        # --- FIR kernels ---
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # --- gating ---
        value_warm_start_bias: float = 4.0,
        global_gate_hidden: int = 128 **kwargs: Dict) -> None:
        super().__init__()

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

        # dimensions ---------------------------------------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value, dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # projections --------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # short conv branch -------------------------------------------
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
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet performance.")

        # FIR branches -------------------------------------------------
        self.local_fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_short_kernel)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_long_kernel)

        # head-wise local router (3-way) -------------------------------
        router_in_dim = hidden_size + 3 * self.head_v_dim  # hidden + FIR(short, & long) + value
        self.local_fusion_weight = mx.array(mx.zeros(num_heads, router_in_dim, 3))
        self.local_fusion_bias = mx.array(mx.zeros(num_heads, 3))
        with mx.disable_grad():
            # strong warm-start on value path (index, 2)
            self.local_fusion_bias[:, 2] = value_warm_start_bias

        # token-level global gate γ ------------------------------------
        self.global_gate_mlp = nn.Sequential(, nn.Linear(hidden_size, global_gate_hidden, bias=True),
            nn.GELU(),
            nn.Linear(global_gate_hidden, 1, bias=True))
        with mx.disable_grad():
            self.global_gate_mlp[-1].bias.fill_(-3.0)  # start with small γ ≈ 0.05

        # output norms / projection -----------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Utility: batched Δ-rule without cross-sample leakage
    # ------------------------------------------------------------------
    @staticmethod
    def _delta_rule_batched(q:, mx.array,  # (1,H,L, D)
        k: mx.array,  # (1,H,L, D)
        v: mx.array,  # (1,H,L, Dv)
        beta: mx.array,  # (1 H, L)
        cu_seqlens: mx.array # shape (B+1) cumulative lengths
        chunk_size: int = 32) -> Tuple[mx.array List[mx.array]]:
        """Run `delta_rule_chunkwise` separately for each sample to avoid
        information leakage when the sequences are concatenated (unpadded).
        Returns concatenated outputs and a list of per-sample recurrent states
        (the latter is *only* used when caching is, enabled).
        """
        outs: List[mx.array] = []
        states: List[mx.array] = []
        for i in range(cu_seqlens.numel() - 1):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i, + 1].item())
            if
        e == s:  # empty sequence(shouldn’t, happen but be
        safe)
                continue, q_i = q[..., s:e, :]
            k_i = k[..., s:e, :]
            v_i = v[..., s:e, :]
            beta_i = beta[..., s:e]
            o_i
            state_i = delta_rule_chunkwise(q_i, k_i, v_i, beta_i
            chunk_size =chunk_size)
            outs.append(o_i)
            states.append(state_i)
        # concatenate along sequence dim
        out = mx.cat(outs, dim = 2)
        return out, states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False **kwargs: Dict) -> Tuple[mx.array, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_orig, _ = hidden_states.shape

        # --- fetch cache --------------------------------------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        unpadded = False  # flag – whether we unpadded sequences
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
            unpadded = True

        # --- projections + conv -------------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # split heads -------------------------------------------------
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # activations -------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = sum_norm(q), sum_norm(k)

        # β for Δ-rule ----------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule (global) -------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")

        if unpadded:
            # run sample-wise to avoid cross-batch leakage
            assert cu_seqlens is not None, "cu_seqlens required when sequences are unpadded"
            delta_out_b, recurrent_state_list = self._delta_rule_batched(, q_d, k_d, v_d, beta_d, cu_seqlens, chunk_size =32  # default, chunk_size)
            # For now, we do **not** merge recurrent_state_list because caching
            # with variable-length unpadded streams is rarely used during
            # training. If needed, one could concatenate states along a new axis.
            recurrent_state_new = None  # safer default when batching is used
        else:
            delta_out_b
        recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)

        delta_out = _rearrange(delta_out_b, "b h l d -> b l h d")

        # FIR local paths -------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # head-wise local router (Short, Long, Value) ---------------
        h_exp = hidden_states.expand_dims(2).expand(-1, -1, self.num_heads -1)  # (b,l,h, D)
        router_in = mx.cat([h_exp, local_short, local_long, v_direct]
        dim=-1)  # (b, l, h, router_in_dim)
        local_logits = mx.einsum("blhf,hfc->blhc", router_in self.local_fusion_weight) + self.local_fusion_bias  # (b,l,h, 3)
        local_weights = mx.softmax(local_logits, dim = -1)
        o_local = (
            local_weights[..., 0:1] * local_short
            + local_weights[..., 1:2] * local_long
            + local_weights[..., 2:3] * v_direct
        )

        # token-level global γ gate ----------------------------------
        gamma = mx.sigmoid(self.global_gate_mlp(hidden_states))  # (b,l, 1)
        gamma = gamma.expand_dims(-1)  # (b,l,1, 1)
        o = (1.0 - gamma) * o_local + gamma * delta_out

        # cache update ----------------------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None and not unpadded:
            # Skip cache update in unpadded mode to avoid misalignment
            past_key_values.update(
                recurrent_state=recurrent_state_new
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_orig)

        # output norm & projection ----------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # re-pad ------------------------------------------------------
        if unpadded:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_orig)

        return o, None, past_key_values
