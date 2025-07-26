"""
MLX-converted architecture: delta_net_selm
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
DeltaNet – Selective Multi-Scale Hybrid Memory (DeltaNet-SELM)
=============================================================
This evolution integrates research-driven advances to balance global context, multi-scale local precision, and dynamic selectivity, directly targeting the core HMGM, DCM, and HSM bottlenecks identified in experimental evidence and research.

Major Innovations
-----------------
1. **True Multi-Scale Convolutional Memory (Dynamic Local Branches)**
   - Adds both a large FIR (long-range) and a small FIR (high-resolution, e.g. kernel=3) path to the value branch.
   - Both are strictly causal, depthwise, and are batch/shape-agnostic.
   - Employs a per-branch, per-head, per-token fusion gate, enabling token-wise selection among local detail, mid/global context, and bypass.

2. **Input & Output-Conditioned Dynamic Gating**
   - Projection for fusion gating now receives not only the token input but also summary statistics of each branch output (mean, std, or L2-norm per head/branch), as inspired by selective SSMs (Mamba, Hyena) and TransNormerLLM.
   - Gate MLP concatenates input embedding and branch summaries for each token.
   - This allows the model to dynamically correct for over/under-smoothing and competitive multi-scale fusion.

3. **Convex Fusion with Gate Temperature**
   - Adds a per-layer, learnable gate temperature to control gate sharpness, initialized such that the identity (direct v) path is favored early.
   - This ensures that at the start of training, the model cannot over-smooth via FIR or otherwise dominate with non-bypass paths, directly addressing observed instability for local tasks.
   - Temperature is applied to fusion logits before softmax.

4. **Chunked Causal Recurrence**
   - Core chunkwise delta-rule path is preserved (unchanged, efficient, O(N)).

5. **Batch & Sequence Agnostic**
   - einops.rearrange used everywhere for robust shape handling, no batch/sequence assumptions.

6. **Full Evidence-Driven & Research-Aligned Implementation**
   - Directly resolves: over-smoothing/blur from fixed-kernel, underselectivity from input-only gating, loss of QA/local/structured task recall.
   - Draws architectural and mathematical framework from Mamba (input+state selective fusion), Hyena (MS gating), Gated Attention (ICLR’24), and TransNormerLLM (temperature/init strategies).

Interface compatibility, all batch/shape safety, and chunkwise O(N) processing are strictly preserved.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
from mx.nn import functional as F



# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------

def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True))


def branch_l2(x):
    # x: [b, l, h, d] -> [b, l, h, 1] (token, head-wise L2 norm)
    return x.norm(dim=-1, keepdim=True)


def branch_mean(x):
    # Mean pooling over hidden_dim per token/head
    return x.mean(dim=-1, keepdim=True)


def branch_std(x):
    return x.std(dim=-1, keepdim=True)

# --------------------------------------------------------------------------
# Depthwise Causal FIR Convolution Layer (generalized for variable kernel)
# --------------------------------------------------------------------------


class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads, head_dim, kernel_size=64):
        super().__init__()
        self.kernel_size = kernel_size
        # Parameter shape: (groups, in_channel_per_group, kernel_size)
        self.filters = mx.array(mx.randn(num_heads, head_dim, kernel_size) * 0.02)

    def forward(self, x):  # [b, l, h, d]
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        weight = _rearrange(self.filters, "h d k -> (h d) 1 k")
        # Causal padding – pad only on the left (past) side
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        y = _rearrange(y, "b (h d) l -> b l h d"h=h)
        return y

# --------------------------------------------------------------------------
# Core chunkwise delta rule (O(N), baseline)
# --------------------------------------------------------------------------


@mx.compile
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Chunkwise (block) implementation of the delta-rule.
    Complexity is O(N * chunk_size^2) which is linear w.r.t sequence length for a fixed chunk_size.
    """

    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

    # ------------------------------------------------------------------
    # Padding so that sequence length % chunk_size == 0
    # ------------------------------------------------------------------
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q = mx.pad(q, pad)
        k = mx.pad(k, pad)
        v = mx.pad(v, pad)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # ------------------------------------------------------------------
    # Normalisation & re-shaping
    # ------------------------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Shape: (b, h, n_chunks, chunk_size, d)
    q, k, v, k_beta = map(
        lambda x: _rearrange(x, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    # ------------------------------------------------------------------
    # Pre-compute block-level attention terms (strictly causal within block)
    # ------------------------------------------------------------------
    mask_full = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), diagonal=0
    )
    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_full, 0)

    # Cumulative summation (delta rule mechanics)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)

    attn = attn + mx.eye(chunk_size, dtype=mx.float, q.device)
    attn = attn

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)

    mask_strict = mx.triu(
        mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), diagonal=1
    )

    # ------------------------------------------------------------------
    # Main recurrence – iterate over blocks in sequence order
    # ------------------------------------------------------------------
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# --------------------------------------------------------------------------
# Main DeltaNet: Selective Multi-Scale Hybrid Memory
# --------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with Selective Multi-Scale Hybrid Memory (SELM).

    Innovations:
    • Small & large FIR convolutional value branches
    • Input + branch-statistic driven gating with learnable temperature
    • Chunkwise delta-rule global memory
    """

    def __init__(
        self,
        mode: str = "selm",
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
        fir_large_kernel: int = 64,
        fir_small_kernel: int = 3,
        fusion_hidden_mult: int = 2,
        gate_init_temp: float = 0.33,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

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

        # ------------------------------------------------------------------
        # Dimension bookkeeping
        # ------------------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ------------------------------------------------------------------
        # Linear projections
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Beta predictor for delta rule weighting
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------------------------------------------------------------
        # Short convolutional enhancer (mandatory)
        # ------------------------------------------------------------------
        if use_short_conv:
            self.q_conv1d = _ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.k_conv1d = _ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.v_conv1d = _ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation="silu",
            )
        else:
            raise UserWarning("_ShortConvolution is mandatory.")

        # ------------------------------------------------------------------
        # Multi-scale FIR convolutions (value pathway)
        # ------------------------------------------------------------------
        self.fir_large = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_large_kernel
        )
        self.fir_small = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_small_kernel
        )

        # ------------------------------------------------------------------
        # Fusion gate – input + branch statistics
        #   Stats per branch  : 3 (mean, std, l2)
        #   Branches considered: 4 (fir_small, fir_large, delta_out, direct)
        #   Total statistic dim: 3 * 4 * num_heads
        # ------------------------------------------------------------------
        branch_stats_per_head = 3  # mean / std / l2
        num_branches_for_stats = 4  # small FIR, large FIR, delta, direct
        stats_dim = branch_stats_per_head * num_branches_for_stats * self.num_heads
        gate_input_dim = hidden_size + stats_dim

        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 3, bias=True),
        )

        # Learnable softmax temperature (>0)
        self.gate_log_temp = mx.array(mx.log(mx.tensor([gate_init_temp])))  # Make 1D tensor, not scalar

        # ------------------------------------------------------------------
        # Output normalisation / projection
        # ------------------------------------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: mx.Tensor,
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.Tensor, Optional[mx.Tensor], Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"

        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # --------------------------------------------------------------
        # Padding-aware un-padding (Flash-like contractors)
        # --------------------------------------------------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).expand_dims(0)

        # --------------------------------------------------------------
        # Projections + short convolutional enhancement
        # --------------------------------------------------------------
        conv_state_q, conv_state_k, conv_state_v = (None, None, None)
        if last_state is not None and last_state.get("conv_state", None) is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        q, k = map(lambda x: _rearrange(x, "... (h d) -> ... h d"d=self.head_k_dim), (q, k))
        v = _rearrange(v, "... (h d) -> ... h d"d=self.head_v_dim)

        # --------------------------------------------------------------
        # Activation / normalisation configs for q,k
        # --------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        v_direct = v  # [b, l, h, d]

        # --------------------------------------------------------------
        # Beta for delta rule (sigmoid-restricted if allow_neg_eigval False)
        # --------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------------------------------------------------------
        # Delta-rule global memory path
        # --------------------------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=q_d, k=k_d, v=v_d, beta=beta_d, chunk_size=32
        )
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # --------------------------------------------------------------
        # FIR paths (multi-scale local memory)
        # --------------------------------------------------------------
        fir_small = self.fir_small(v_direct)
        fir_large = self.fir_large(v_direct)

        # --------------------------------------------------------------
        # Branch statistics for dynamic gating
        # --------------------------------------------------------------
        summaries = []
        for branch in [fir_small, fir_large, delta_out, v_direct]:
            summaries.append(branch_mean(branch))
            summaries.append(branch_std(branch))
            summaries.append(branch_l2(branch))
        summary_cat = mx.cat(summaries, dim=-1)  # [b, l, h, num_stats]
        summary_cat_flat = _rearrange(summary_cat, "b l h c -> b l (h c)")

        # --------------------------------------------------------------
        # Gating – input embedding + branch summaries
        # --------------------------------------------------------------
        fusion_gate_inp = mx.cat([hidden_states, summary_cat_flat], dim=-1)
        fusion_logits = self.fusion_gate_mlp(fusion_gate_inp)  # [b, l, (h*3)]
        fusion_logits = rearrange(
            fusion_logits, "b l (h c) -> b l h c", h=self.num_heads, c=3
        )

        gate_temp = mx.exp(self.gate_log_temp)[0].clamp(min=1e-4)  # Now 1D tensor, get scalar with [0]
        fusion_logits = fusion_logits / gate_temp
        fusion_weights = mx.softmax(fusion_logits, dim=-1)  # [b, l, h, 3]

        # --------------------------------------------------------------
        # Compose outputs
        #   Gate order: [0] local (small+large), [1] delta, [2] direct
        # --------------------------------------------------------------
        fir_local = fir_small + fir_large
        outputs = [fir_local, delta_out, v_direct]
        o = (
            fusion_weights[..., 0:1] * outputs[0]
            + fusion_weights[..., 1:2] * outputs[1]
            + fusion_weights[..., 2:3] * outputs[2]
        )

        # --------------------------------------------------------------
        # Cache update (if requested)
        # --------------------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v)
                if self.use_short_conv
                else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # --------------------------------------------------------------
        # Output normalisation & projection
        # --------------------------------------------------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d"d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # --------------------------------------------------------------
        # Re-pad back to original shape (if un-padded earlier)
        # --------------------------------------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
