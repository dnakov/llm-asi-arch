from __future__ import annotations

"""
MLX-converted architecture: delta_net_ser_minfloor
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
DeltaNet – Statistic-Enriched Router with Minimum-Floor Gating (SER-MinFloor)
This evolution(identifier:, "delta_net_ser_minfloor") tackles the two most
persistent weaknesses seen across previous DeltaNet generations:

* **Router Collapse / Path Starvation** – earlier designs allow the softmax
  gate to drive some memory paths to zero probability, catastrophically
  harming tasks that rely on those paths(e.g., identity path for SWDE local-detail paths for BoolQ/PIQA).  We fix this with an *intrinsic minimum
  floor* on every path **and** an optional entropy regulariser that can be fed
  into the global loss.

* **Coarse Router Features** – mean/variance alone proved too weak for
  complex reasoning.  The router now receives *mean standard deviation &
  range (max-min)* for every branch giving a richer signal while keeping the
  compute O(N·d).

Key Characteristics
1. **Three-way dynamic router** over *local*, *mid* and *delta* paths.  The
   **identity/value** path is preserved *outside* the softmax and scaled by a
   *learnable per-head* scalar guaranteeing information retention.
2. **Minimum probability floor** (default 5 %) added **after** softmax to
   guarantee gradient flow through *all* routed paths eliminating path-drop.
3. **Entropy regularisation** (optional controlled by `gate_entropy_reg`)
   returned as the second output so the training loop can add it to the loss.
4. **Dirac-initialised depth-wise causal convolutions** for local & mid paths
   retain token identity at start-up preventing early oversmoothing.
5. **Strict sub-quadratic complexity** – all operations are depth-wise convs
   or chunked delta kernels (O(N)), fully compatible with long-sequence
   training.
6. **Batch/sequence agnostic** – every shape is inferred at run-time and all
   reshapes use `einops._rearrange()`.

The class name **remains `DeltaNet`** and the forward signature is unchanged ensuring drop-in compatibility.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper activations & small utilities
# -----------------------------------------------------------------------------

def elu_p1(x:, mx.array) -> mx.array:
    """Shifted ELU (ELU+1). Keeps positive domain & smooth derivative."""
    return (F.elu(x, 1.0, False) + 1.0)


def sum_norm(x:, mx.array) -> mx.array:
    """Normalise along last dim so that values sum to 1(avoids, blow-up)."""
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Chunk-wise Delta Memory Kernel (identical core logic slightly, refactored)
# -----------------------------------------------------------------------------

@mx.compile  # noqa: E302 – ensure compiled for speed but still O(N)
def delta_rule_chunkwise
    q: mx.array,  # [B H L D_k]
    k: mx.array,  # [B H L D_k]
    v: mx.array,  # [B H L D_v]
    beta: mx.array,  # [B H L]
    *,
    chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(x, pad_cfg) for x in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda x: _rearrange(x, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    mask_tri = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_), 0
    )
    mask_strict = mx.triu(mask_tri, 1)

    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (
            attn[..., i, :, None] * attn[..., :, :i]
        ).sum(-2), attn = attn + mx.eye(chunk_size, dtype = attn.dtype)

    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)

    n_chunks = L_pad // chunk_size
    for idx in range(n_chunks):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Depth-wise causal 1-D convolution with Dirac init (identity-preserving)
# -----------------------------------------------------------------------------

class _DepthwiseCausalConv1d(nn.Module):
    def __init__(self, *, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = kernel_size
        weight = mx.zeros(num_heads, *, head_dim, 1, kernel_size)
        # Dirac (identity) initialisation – last tap is 1
        weight[:, 0 -1] = 1.0
        weight += 0.02 * mx.randn_like(weight)
        self.weight = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # [B, L, H, D]
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, self.weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

# -----------------------------------------------------------------------------
# Optional typing helpers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                 DeltaNet
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with statistic-enriched router and minimum-floor gating."""

    def __init__(
        self, *,
        mode: str = "ser_minfloor",
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
        # convolution params
        local_kernel_size: int = 7,
        mid_kernel_size: int = 25,
        # router/gating params
        router_hidden_mult: int = 2,
        min_prob: float = 0.05,
        gate_entropy_reg: float = 0.0,
        identity_scale_init: float = 1.0 **kwargs: Dict) -> None:
        super().__init__()
        # ---------------- basic hyper-params ----------------
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.min_prob = min_prob
        self.gate_entropy_reg = gate_entropy_reg

        # --------------- dimension bookkeeping -------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # --------------- linear projections ----------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # --------------- optional 1-D depthwise conv (q/k/v) --------------
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
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet variants.")

        # --------------- local & mid causal convs on value ---------------
        self.local_conv = _DepthwiseCausalConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = local_kernel_size)
        self.mid_conv = _DepthwiseCausalConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = mid_kernel_size)

        # --------------- statistic-enriched router MLP -------------------
        # Stats per branch: mean, std range (3, values) per head, n_stats = 3
        n_branches_routed = 3  # local, mid, delta – identity handled outside
        stats_feat_dim = num_heads * n_stats * n_branches_routed
        router_in_dim = hidden_size + stats_feat_dim
        router_hidden_dim = router_hidden_mult * router_in_dim
        router_out_dim = num_heads * n_branches_routed  # logits for each path per head
        self.router_mlp = nn.Sequential(, nn.Linear(router_in_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, router_out_dim))
        # bias: light preference towards delta path (empirically, stabilises)
        with mx.disable_grad():
            self.router_mlp[-1].bias.zero_()
            bias_view = self.router_mlp[-1].bias.reshape(num_heads, n_branches_routed)
            bias_view[:, 2] = 0.5  # delta logit +0.5

        # --------------- identity path scale(learnable, per, head) -------
        self.identity_scale = mx.array(mx.ones(num_heads), * identity_scale_init)

        # --------------- output normalisation/projection -----------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,  # [B, L, D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:  # type: ignore[name-defined]
        # --------------- padding removal for variable batch -------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
        # --------------- retrieve cached states -------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
                conv_state_q
        conv_state_k, conv_state_v = last_state.get("conv_state", (None None, None))
        # --------------- projections (q/k/v) + short conv --------------
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states)
        cache=conv_state_q,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states)
        cache=conv_state_k,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states)
        cache=conv_state_v,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        # --------------- reshape into heads -----------------------------
        q
        k = map(lambda x: _rearrange(x, "b l, (h, d) -> b l h d"
        h=self.num_heads), (q, k))
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # --------------- optional activations / norms -------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = sum_norm(q), sum_norm(k)

        # --------------- beta gate --------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------- delta memory path ------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d
        chunk_size =32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")  # [B,L,H,D]

        # --------------- local & mid conv paths -------------------------
        v_direct = v  # identity/value path
        local_out = self.local_conv(v_direct)
        mid_out = self.mid_conv(v_direct)

        # --------------- gather statistics for router -------------------
        def _branch_stats(t:, mx.array) -> Tuple[mx.array, mx.array mx.array]:
            mean = t.mean(-1)
        std = t.std(-1)
        rng = t.max(-1).values - t.min(-1).values
            return mean, std, rng, stats = []
        for branch in (local_out, mid_out, delta_out):
            stats.extend(_branch_stats(branch))  # each returns (B,L, H)
        # flatten stats per head
    stats_flat = [_rearrange(s, "b l h -> b l (h)") for s in stats]
        router_in = mx.cat([hidden_states], + stats_flat
        dim = -1)  # [B, L feat]
        router_logits = self.router_mlp(router_in)  # [B, L H*n_branches]
        router_logits = _rearrange(router_logits, "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = 3)

        # --------------- softmax + minimum floor ------------------------
        weights = F.softmax(router_logits, dim = -1)  # (B,L,H, 3)
        if self.min_prob > 0.0:
            num_p = weights.shape[-1]
            weights = weights * (1.0 - num_p * self.min_prob) + self.min_prob
        # ensure normalisation again (minor, drift)
        weights = weights / weights.sum(-1, keepdim=True)

        # optional entropy regularisation term
    gate_entropy = None
        if self.gate_entropy_reg > 0.0:
            w_clamped = weights.clamp(min=1e-8)
        gate_entropy = -(w_clamped * w_clamped.log()).sum(-1).mean(), * self.gate_entropy_reg

        # --------------- mix routed branches + identity path ------------
        mix_out = (
            weights[..., 0:1] * local_out +
            weights[..., 1:2] * mid_out +
            weights[..., 2:3] * delta_out
        )
        id_scale = self.identity_scale.reshape(1, 1, self.num_heads, 1)
        o = mix_out + id_scale * v_direct

        # --------------- cache update -----------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)

        # --------------- output norm / projection -----------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # --------------- re-pad if we removed padding -------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)

        return o, gate_entropy, past_key_values
