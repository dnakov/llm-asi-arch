"""
MLX-converted architecture: delta_net_mor
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
DeltaNet – Multi-Scale Output-Aware Routing (MOR)
================================================
This evolution integrates the strengths of prior *dual-scale* convolutional
branches while fixing the router myopia that previously starved the long-range
**delta** memory pathway.  The router now conditions its decision **both** on
input token representation **and** lightweight *statistics* of candidate path
outputs (local, mid, delta, identity).  These output-aware logits enable the
network to dynamically balance locality and globality per token & head.

Key Innovations
---------------
1. **Tri-Path Value Space** –  *Local* (k=7) and *Mid* (k=31) depth-wise causal
   convolutions complement the associative **delta** memory and the *identity*
   (direct value) path.  This preserves proven local precision while retaining
   robust long-range reasoning.
2. **Output-Aware Softmax Router** –  A two-layer MLP on the input embedding
   produces preliminary logits which are *modulated* by per-path statistics
   (mean absolute activation) drawn from the candidate outputs themselves.
   This cheap but expressive feedback loop prevents systematic under-selection
   of any branch (especially the delta path) and has theoretical grounding in
   recent MoE/Router and SSM literature.
3. **Identity-Favoured Yet Flexible Bias** –  The router bias initialisation
   still favours the identity path for early stability, but the statistics
   modulation term learns quickly (init=0) allowing the model to re-allocate
   probability mass as each branch matures.
4. **Strict Causality & O(N)** –  All added ops are depth-wise 1-D convolutions
   or per-token projections; computational complexity remains linear in
   sequence length and fully batch-agnostic.

Interface, class name (`DeltaNet`), forward signature and parameter schema are
unchanged, satisfying drop-in compatibility requirements.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def elu_p1(x: mx.Tensor) -> mx.Tensor:  # ELU+1 keeps positive domain
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.Tensor) -> mx.Tensor:  # L1 normalisation along last dim
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Core chunk-wise delta rule (identical to baseline – O(N))
# -----------------------------------------------------------------------------

@mx.compile
def delta_rule_chunkwise(
    q: mx.Tensor,  # [B H L Dk]
    k: mx.Tensor,  # [B H L Dk]
    v: mx.Tensor,  # [B H L Dv]
    beta: mx.Tensor,  # [B H L]
    *,
    chunk_size: int = 32,
):
    """Associative delta memory evaluated in causal chunks (O(N))."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)  # pad sequence dimension
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # feature normalisation ----------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into chunks [B H N C D]
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    mask_future = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + mx.eye(chunk_size, dtype=inv.dtype, q.device)
    inv = inv

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Depth-wise causal 1-D convolution (per-head) – O(N·k)
# -----------------------------------------------------------------------------

class _DepthwiseCausalConv1d(nn.Module):
    """Per-head depth-wise causal convolution used for local / mid branches."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.randn(num_heads * head_dim, 1, self.kernel_size) / math.sqrt(self.kernel_size)
        self.weight = mx.array(weight)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # x: [B L H D]
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b (h d) l")
        padding = (self.kernel_size - 1, 0)  # left pad for causality
        x_pad = mx.pad(x_ch, padding)
        y = F.conv1d(x_pad, self.weight, groups=h * d)
        y = _rearrange(y, "b (h d) l -> b l h d"h=h)
        return y

# -----------------------------------------------------------------------------
# Optional cache type hints
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                DeltaNet – MOR
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with *Multi-Scale Output-Aware Routing* (MOR)."""

    def __init__(
        self,
        mode: str = "mora",  # mode name for debugging
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
        # ---- new MOR params --------------------------------------------
        local_kernel_size: int = 7,
        mid_kernel_size: int = 31,
        router_hidden_mult: int = 2,
        router_identity_bias: float = 1.5,  # favours identity path at init (~70%)
        stats_weight_init: float = 0.0,
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        # ---------------- basic setup ----------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value dims must be divisible by num_heads")

        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---------------- projections ----------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # optional short convs in q/k/v space ---------------------------
        if use_short_conv:
            activation = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=activation, bias=conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=activation, bias=conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # depth-wise conv branches --------------------------------------
        self.local_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, kernel_size=local_kernel_size)
        self.mid_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, kernel_size=mid_kernel_size)

        # ---------------- output-aware router --------------------------
        # order of paths: local, mid, delta, identity
        router_out_dim = num_heads * 4
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * router_hidden_mult, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * router_hidden_mult, router_out_dim, bias=True),
        )
        # init bias so identity starts dominant
        with mx.no_grad():
            self.router_mlp[-1].bias.zero_()
            bias_view = self.router_mlp[-1].bias.reshape(num_heads, 4)
            bias_view[:, 3] = router_identity_bias  # identity path bias

        # learnable weights for statistics modulation (per head, per path)
        self.stats_weight = mx.array(mx.full((num_heads, 4), stats_weight_init))

        # ---------------- output norm / projection ---------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: mx.Tensor,  # [B L D]
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[mx.Tensor, None, Optional["Cache"]]:
        # ---------------- sanity & unpad ------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, L_in, _ = hidden_states.shape

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # ---------------- fetch cache ---------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
                conv_state_q, conv_state_k, conv_state_v = last_state.get("conv_state", (None, None, None))

        # ---------------- projections & short conv --------------------
        q_lin, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k_lin, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v_lin, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # head reshape --------------------------------------------------
        q = _rearrange(q_lin, "b l (h d) -> b l h d"h=self.num_heads)
        k = _rearrange(k_lin, "b l (h d) -> b l h d"h=self.num_heads)
        v = _rearrange(v_lin, "b l (h d) -> b l h d"h=self.num_heads)  # direct value path

        # activations ---------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # beta gate -----------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # delta rule ----------------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d, chunk_size=32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # convolutional branches ---------------------------------------
        local_out = self.local_conv(v)
        mid_out = self.mid_conv(v)
        identity_out = v

        # ---------------- path statistics (mean absolute) -------------
        def _stat(x: mx.Tensor) -> mx.Tensor:  # [B L H D] -> [B L H]
            return x.abs().mean(dim=-1)

        stat_local = _stat(local_out)
        stat_mid = _stat(mid_out)
        stat_delta = _stat(delta_out)
        stat_identity = _stat(identity_out)
        stats_stack = mx.stack([stat_local, stat_mid, stat_delta, stat_identity], dim=-1)  # [B L H 4]
        stats_term = stats_stack * _rearrange(self.stats_weight, "h p -> 1 1 h p")  # broadcast

        # ---------------- router logits & weights ----------------------
        router_logits = self.router_mlp(hidden_states)  # [B L H*4]
        router_logits = _rearrange(router_logits, "b l (h p) -> b l h p"h=self.num_heads, p=4)
        router_logits = router_logits + stats_term  # output-aware modulation
        router_weights = F.softmax(router_logits, dim=-1)  # [B L H 4]

        # ---------------- fuse outputs --------------------------------
        fused = (
            router_weights[..., 0:1] * local_out
            + router_weights[..., 1:2] * mid_out
            + router_weights[..., 2:3] * delta_out
            + router_weights[..., 3:4] * identity_out
        )  # [B L H D]

        # cache update --------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # output norm / proj -------------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"h=self.num_heads)
            fused = self.o_norm(fused, g_vec)
        else:
            fused = self.o_norm(fused)
        out = self.o_proj(_rearrange(fused, "b l h d -> b l (h d)"))

        # re-pad --------------------------------------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0), indices, B_orig, L_in)
        return out, None, past_key_values
