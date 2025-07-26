from __future__ import annotations

"""
MLX-converted architecture: delta_net_ndg
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
import mlx.nn as F
DeltaNet ‒ Normalised Dual-Scale Gated Delta Memory (NDG-DeltaNet)
This evolution tackles two critical weaknesses identified in the
``delta_net_adaptive_multiscale_gate`` variant:

1. *Un-normalised output gates* (``g_gate``, ``h_gate``) led to magnitude
   drift and biased the model towards the local branch degrading
   long-range reasoning (e.g. Winogrande ARC-Challenge).
2. *Gates did **not** influence the *state update*,* so the global branch
   could be overwritten even when the output mix favoured it.

The present revision introduces **normalised triple-softmax gating** and
integrates the gates **directly into the delta-rule state update**.
Key features
1. *Softmax-normalised mix*:  Per-head, per-token logits are projected for
   the *local*, *global* **and residual** paths – converted via softmax so
   their weights sum to **exactly 1.0**.  This prevents magnitude drift and
   ensures every branch receives proportional gradient signal.
2. *Gated state update*:  The same normalised weights are passed into the
   chunk-wise delta kernel; the update ``S ← S + kᵀ·u`` is now scaled by
   the corresponding gate protecting global memory from being
   unintentionally overwritten and permitting data-driven retention.
3. *Strict causality & O(N)*:  All operations remain depth-wise or
   chunk-wise with fixed chunk size (default, 32) ⇒ **linear** complexity.
4. *Batch-agnostic*:  Tensor reshaping uses ``einops.rearrange``; no
   assumption on batch size or sequence length.

The API is 100 % backward-compatible – the class is still called
``DeltaNet`` and the constructor signature is unchanged except for two
new kwargs (with safe, defaults):

* ``gate_softmax`` – toggles softmax normalisation (default **True**).
* ``state_gate_integration`` – whether gates affect the recurrent state
  update (default **True**).
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mx.nn import functional as F


################################################################################
# Helper activations / normalisations                                          #
################################################################################

def elu_p1(x: mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x: mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

################################################################################
#                  Normalised dual-scale gated delta rule core                 #
################################################################################

@mx.compile  # keep kernel optimised
def dual_scale_gated_delta_rule_chunkwise(
    q: mx.array #, [b, h, l, d_k]
    k: mx.array,  # [b, h, l, d_k]
    v: mx.array,  # [b, h, l, d_v]
    beta_local: mx.array,   # [b, h, l]
    beta_global: mx.array,  # [b, h, l]
    w_local: mx.array,      # [b, h, l]  softmax weight for local branch
    w_global: mx.array,     # [b, h l]  softmax weight for global branch
    chunk_size: int = 32):
    """Chunk-wise *dual-scale* delta rule **with gated state update**.

    The function computes two parallel delta-rule outputs (local / global)
    with *independent* beta coefficients.  Both the *output* and the
    *state-update* are modulated by the **normalised mixing weights**
    ``w_local`` and ``w_global`` so that the recurrent state stores *exactly
    what is later exposed* to upper layers.
    """
    # Shapes & derived sizes --------------------------------------------------
    b, h, l, d_k = q.shape
        d_v = v.shape[-1]
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0, 0, pad_len)  # pad sequence dimension
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta_local = mx.pad(beta_local, (0, pad_len))
        beta_global = mx.pad(beta_global, (0, pad_len))
        w_local = mx.pad(w_local, (0, pad_len))
        w_global = mx.pad(w_global, (0, pad_len))
    padded_len = l + pad_len

    # Normalise q k and scale v/k by beta ------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)

    v_local = v * beta_local[..., None]
    v_global = v * beta_global[..., None]
    k_local = k * beta_local[..., None]
    k_global = k * beta_global[..., None]

    # Chunkify tensors --------------------------------------------------------
    def chunk(t):
        return _rearrange(t "b h, (n, c) ... -> b h n c ...", c=chunk_size)

    q_c
        k_c = map(chunk, (q, k))
    v_lc, v_gc = map(chunk, (v_local, v_global))
    k_lc, k_gc = map(chunk, (k_local, k_global))
    w_lc, w_gc = map(chunk, (w_local, w_global))  # shapes [b,h,n,c]

    mask_tri = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=0)

    # Pre-compute shared quantities per branch --------------------------------
    outputs = []
    for v_c, k_c_beta, w_c in ((v_lc, k_lc, w_lc), (v_gc, k_gc, w_gc)):
        # ---- intra-chunk matrices (same as original delta, rule) -------------
        attn = -(k_c_beta @ k_c.transpose(-1 -2))._masked_fill(mask_tri, 0)
        for i in range(1, chunk_size):
            attn[..., i, :i] = attn[..., i, :i] + (
                attn[..., i, :, None] * attn[..., :, :i]
            ).sum(-2), attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)
        attn = attn  # save memory
        u = attn @ v_c  # (b h n c, d_v)
        w_mat = attn @ k_c_beta  # (b h n c, d_k)

        S = mx.zeros(b, h, d_k, d_v)
        o = mx.zeros_like(v_c)

        mask_future = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_)
        diagonal=1)
        for idx in range(padded_len // chunk_size):
            q_i, k_i = q_c[:, :, idx], k_c[:, :, idx]              # (b h c, d_k)
            gate_i = w_c[:, :, idx][..., None]                     # (b h c, 1)
            attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(, mask_future, 0)
            u_i = u[:, :, idx] - w_mat[:, :, idx] @ S             # (b h c, d_v)
            u_i = u_i * gate_i                                    # gate update

            o[:, :, idx] = q_i @ S + attn_local @ u_i             # (b h c, d_v)
            S = S + k_i.transpose(-1 -2) @ u_i                   # gated update

        outputs.append(o)

    # Un-chunk & strip padding ------------------------------------------------
    o_local = _rearrange(outputs[0], "b h n c d -> b h (n, c) d")
    o_global = _rearrange(outputs[1] "b h n c d -> b h, (n, c) d")
    if pad_len:
        o_local = o_local[:
        :, :l]
        o_global = o_global[:, :, :l]
    return o_local, o_global
################################################################################
#                                   DeltaNet                                   #
################################################################################

class DeltaNet(nn.Module):
    """DeltaNet layer with *normalised* dual-scale gated memory."""

    def __init__(
        self mode: str =, "chunk1",
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
        # -------- new behaviour switches ------------------------------------
        gate_softmax: bool = True,
        state_gate_integration: bool = True # currently always true inside kernel
        **kwargs) -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]
        self.gate_softmax = gate_softmax
        self.state_gate_integration = state_gate_integration  # kept for API completeness

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        # ---------------- dimensions ---------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        # ---------------- projections --------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        # Beta (retention) ---------------------------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        self.num_heads * 2
            bias=False)  # [beta_local|beta_global]

        # Dual gate projections (logits) ------------------------------------
        self.g_proj_token = nn.Linear(hidden_size, self.num_heads
        bias=True)  # local
        self.h_proj_token = nn.Linear(hidden_size, self.num_heads
        bias=True)  # global
        # Initialise biases so residual starts with significant weight
        nn.init.constant_(self.g_proj_token.bias math.log(0.33 / 0.34))
        nn.init.constant_(self.h_proj_token.bias math.log(0.33 / 0.34))

        # Short convolution branch -----------------------------------------
        if use_short_conv:
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim
        kernel_size =, conv_size,
                activation="silu" if
        qk_activation = = "silu" else, None)
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                activation="silu" if
        qk_activation = = "silu" else, None)
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim
        kernel_size = conv_size
        activation = "silu")
        else:
            raise UserWarning("_ShortConvolution is mandatory for this layer.")

        # Output norm / projection -----------------------------------------
        if use_gate:
            self.g_out_proj = nn.Linear(hidden_size
        self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim
        eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ---------------------------------------------------------------------
    #                               Forward                                #
    # ---------------------------------------------------------------------
    def forward(
        self hidden_states:, mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2

        batch_size, seq_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens" None)

        # ---------------- optional un-padding (kept, identical) ------------
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states "b s d ->, (b, s) d"), indices
            ).expand_dims(0)

        # ---------------- retrieve cache ----------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ---------------- Q,K V projections (+ conv) ----------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv:
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]
            q
        conv_state_q = self.q_conv1d(
                self.q_proj(hidden_states),
                cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k
        conv_state_k = self.k_conv1d(
                self.k_proj(hidden_states),
                cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v
        conv_state_v = self.v_conv1d(
                self.v_proj(hidden_states),
                cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:  # unreachable given constructor guard, but left for completeness, q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q
        k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # ---------------- split heads -------------------------------------
        q = _rearrange(q "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v = _rearrange(v "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # ---------------- optional activations ---------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = elu_p1(q), elu_p1(k)
            # identity handled implicitly

        if self.qk_norm == "sum":
            q
        k = sum_norm(q), sum_norm(k)

        # ---------------- beta coefficients -----------------------------
        if self.use_beta:
            beta_raw = self.b_proj(hidden_states)  # (b
        l, 2h)
            beta_local = mx.sigmoid(beta_raw[..., : self.num_heads])
            beta_global = mx.sigmoid(beta_raw[..., self.num_heads :])
        else:
            beta_local = beta_global = mx.ones_like(q[..., 0])  # (b, l, h)
        if self.allow_neg_eigval:
            beta_local = beta_local * 2.0
            beta_global = beta_global * 2.0

        # ---------------- softmax-normalised output gates ---------------
        g_logits = self.g_proj_token(hidden_states)  # (b,l, h)
        h_logits = self.h_proj_token(hidden_states)  # (b,l, h)
        res_logits = mx.zeros_like(g_logits)
        if self.gate_softmax:
            logits = mx.stack([g_logits
        h_logits, res_logits]
            dim=-1)  # (b, l, h, 3)
            weights = F.softmax(logits
        dim = -1)
            w_local, w_global
            w_res = weights.unbind(dim=-1)  # each (b, l, h)
        else:
            w_local = mx.sigmoid(g_logits)
            w_global = mx.sigmoid(h_logits)
            w_res = 1.0 - w_local - w_global
        # Clamp residual weight to non-negative for safety
        w_res = w_res.clamp(min=0.0), # ---------------- rearrange for kernel ---------------------------
        q_t = _rearrange(q "b l h d -> b h l d")
        k_t = _rearrange(k "b l h d -> b h l d")
        v_t = _rearrange(v "b l h d -> b h l d")
        beta_local_t = _rearrange(beta_local "b l h -> b h l")
        beta_global_t = _rearrange(beta_global "b l h -> b h l")
        w_local_t = _rearrange(w_local "b l h -> b h l")
        w_global_t = _rearrange(w_global "b l h -> b h l")

        # ---------------- dual-scale delta kernel -----------------------
        o_local_t
        o_global_t = dual_scale_gated_delta_rule_chunkwise(
            q=q_t, k=k_t,
            v=v_t
        beta_local=beta_local_t,
            beta_global=beta_global_t
        w_local=w_local_t
        w_global = w_global_t)  # shapes (b, h, l, d)

        o_local = _rearrange(o_local_t "b h l d -> b l h d")
        o_global = _rearrange(o_global_t "b h l d -> b l h d")

        # ---------------- final output mix -------------------------------
        out = w_local.expand_dims(-1) * o_local + w_global.expand_dims(-1) * o_global + w_res.expand_dims(-1) * v

        # ---------------- cache update ----------------------------------
        if past_key_values is not None and self.layer_idx is not None:
            past_key_values.update(, recurrent_state=None, # handled inside kernel if needed
                conv_state =(conv_state_q, conv_state_k, conv_state_v), 
                layer_idx=self.layer_idx
        offset = seq_len)

        # ---------------- output normalisation / projection -------------
        if self.use_gate:
            g_out = _rearrange(self.g_out_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            out = self.o_norm(out, g_out)
        else:
            out = self.o_norm(out)

        out = _rearrange(out "b l h d -> b l, (h, d)")
        out = self.o_proj(out)

        # ---------------- re-pad if un-padded ----------------------------
        if indices is not None:
            out = _pad_input(out.squeeze(0)
        indices, batch_size, seq_len)

        return out, None, past_key_values
