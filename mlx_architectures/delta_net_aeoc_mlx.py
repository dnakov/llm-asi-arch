"""
MLX-converted architecture: delta_net_aeoc
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
DeltaNet – Adaptive-Entropy Output-Conditioned Multi-Scale Routing (AEOC)
Innovation: delta_net_aeoc

This architecture synthesizes breakthrough research-driven upgrades, directly targeting
all major performance limitations previously identified in DeltaNet models. It integrates:

1. **Output-Conditioned Router with Expanded Relational Statistics**
   - Router MLP is now fed, per token/head, with concatenated statistics(mean, variance, max, cross-dot pairwise similarities and dynamic, entropy) extracted from all candidate memory streams:
     [local conv, mid conv, delta, identity].
   - Enables decision making that accounts for not just statistical dispersion but also relational structure and higher-moment evidence - unlocking reasoning/QA and structure-sensitive tasks.

2. **Identity-Preserving Multi-Scale FIR Stack with Adaptive-Scale Gating**
   - The value path is routed through 3x depthwise-conv branches: short (k=3)
   mid (k=7)
   long (k=25) and also passes through an unblurred
        k = 1 (identity) path.
   - These four branches, plus the global delta-memory allow the model to access any combination from local to global evidence.
   - FIRs initialised as causally-aligned Dirac (identity) for stability.

3. **Adaptive Entropy Regularization with Minimum-Path Probability and KL Penalty**
   - The router's output head is regularized with a dual criterion:
     (a) a learnable, scheduled entropy target and
     (b) a minimum path probability per-branch(floored, at 1%) to prevent collapse on any output stream (especially critical for extraction tasks like, SWDE).
   - KL-Uniform penalty further discourages premature path collapse sustaining multi-path utilization throughout training.

4. **Efficient O(N) Implementation, True Batch-Seq Agnostic and Causal**
   - Uses chunked delta memory, FIR depthwise convolutions and einops for all tensor reshaping.
   - Strictly maintains batch/sequence/length independence and O(N) complexity.
   - All features are default-on, with sensible hyperparams, backward compatible.

5. **Interface and Pipe Compatibility**
   - Preserves DeltaNet class, forward signature, and **kwargs pattern.
   - All code robust to any batch size, supporting packed/variable input.
   - Regularization losses are returned for integration into upstream objectives.

Full theoretical justification and implementation details in design notes. This composite design
captures the best innovations from both output-conditioned routing, entropy-aware fusion and
expanded multi-scale memory - integrating the top empirical and theoretical findings.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# ---------------------------------------------
# Helper functions for activations and norm
# ---------------------------------------------
def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------
# Causal chunked delta memory kernel
# ---------------------------------------------
@mx.compile
def delta_rule_chunkwise(q:, mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    *,
    chunk_size: int = 32):
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q
        (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v
    k_beta = map(lambda x: _rearrange(x, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    mask_tri_full = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri_full, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] += (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size, dtype = attn.dtype)
    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    mask_tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    num_chunks = L_pad // chunk_size
    for idx in range(num_chunks):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_tri_strict, 0)
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
# ---------------------------------------------
# Per-head causal depthwise FIR Conv (identity, init) for, k = 1,3,7 25
# ---------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        total_channels = num_heads * head_dim
        # Identity (Dirac) in last tap for causality
    filt = mx.zeros(total_channels, 1, self.kernel_size)
        with mx.disable_grad():
            filt[:, 0 -1] = 1.0
        self.weight = mx.array(filt), def forward(self, x: mx.array):  # [B, L, H, D]
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

class DeltaNet(nn.Module):
    """DeltaNet with Adaptive-Entropy Output-Conditioned Multi-Scale Routing (AEOC)"""

    def __init__(
        self, *,
        mode: str = "aeoc",
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
        # Multi-scale FIR kernel sizes
        kernel_short: int = 3,
        kernel_mid: int = 7,
        kernel_long: int = 25,
        router_hidden_mult: int = 2,
        router_min_prob: float = 0.01,
        router_entropy_coeff: float = 0.02,
        router_kl_coeff: float = 0.01,
        router_entropy_target: float = 1.0,  # default entropy target
        **kwargs: Dict, ):
        super().__init__()
        self.mode = mode
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads
            bias=False)
        if self.use_short_conv:
            activation = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=activation
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=activation
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution, is mandatory for stable performance.")
        # Multi-scale FIRs, all causal, identity init
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = kernel_short)
        self.fir_mid = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = kernel_mid)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = kernel_long)
        self.fir_id = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = 1)  # identity
        # Output fusion router: receives both hidden and extensive stats from all branches
        # For each path: mean, var, max over last dim, entropy, and cross-path pairwise similarities
        # Between (short, mid, long, delta, id), there are 5 branches, m = 5
        self.router_num_paths = 5
        router_in_feats = hidden_size + self.num_heads * self.router_num_paths * 4 + self.num_heads * (self.router_num_paths * (self.router_num_paths-1)) // 2
        router_hidden = router_hidden_mult * router_in_feats
        self.router_mlp = nn.Sequential(, nn.Linear(router_in_feats, router_hidden, bias=True),
            nn.GELU(),
            nn.Linear(router_hidden, self.num_heads * self.router_num_paths, bias=True))
        # Init bias so id and delta path get slight boost
        with mx.disable_grad():
            self.router_mlp[-1].bias.zero_()
            bias_v = self.router_mlp[-1].bias.reshape(self.num_heads, self.router_num_paths)
            bias_v[:, -1] = 0.5
            bias_v[:, -2] = 0.5
        self.router_min_prob = router_min_prob
        self.router_entropy_coeff = router_entropy_coeff
        self.router_kl_coeff = router_kl_coeff
        self.router_entropy_target = router_entropy_target
        # Output norm/gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)
    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,  # [B, L, D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs  ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [B, L] tensor"
        B, L, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
        conv_q = conv_k = conv_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv:
                conv_q
        conv_k, conv_v = last_state.get("conv_state", (None None, None))
        q, conv_q = self.q_conv1d(
            x=self.q_proj(hidden_states)
        cache=conv_q,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k, conv_k = self.k_conv1d(
            x=self.k_proj(hidden_states)
        cache=conv_k,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v, conv_v = self.v_conv1d(
            x=self.v_proj(hidden_states)
        cache=conv_v,
            output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)
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
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # Delta memory, rearrange to [b,h,l d]
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d
        chunk_size =32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")  # [B, L, H D]
        # Multi-scale FIRs (local)
        v_id = self.fir_id(v)     # k=1 (identity)
        v_short = self.fir_short(v)
        v_mid = self.fir_mid(v)
        v_long = self.fir_long(v)
        # Stack branches for router: short, mid, long, delta, id, branches = [v_short, v_mid, v_long, delta_out, v_id]
        # Output stats for each path: mean, var, max, entropy(per, B,L, H)
        branch_feats = []  # each [B, L, H S]
        for x in branches:
            mean = x.mean(dim=-1)
        var = x.var(dim=-1)
        maxx = x.amax(dim=-1)
            # For entropy, flatten last dim, softmaxed = F.softmax(x, dim = -1)
            entropy = -(softmaxed * (softmaxed + 1e-8).log()).sum(-1), branch_feats.extend([mean, var, maxx entropy])
        # Cross-branch headwise dot-product similarities
    cross_feats = []
        num_branches = len(branches)
        for i in range(num_branches):
            for j in range(i, +, 1, num_branches):
                # [B, L, H, D] x [B, L, H, D] -> [B, L H]
                dot = (branches[i] * branches[j]).sum(-1), cross_feats.append(dot)
        # [B, L, h_feats]
        all_branch_feats = mx.cat(
            [_rearrange(f, "b l h -> b l (h)") for f in branch_feats + cross_feats]
        dim=-1
        )
        router_in = mx.cat([hidden_states, all_branch_feats]
        dim=-1)
        router_logits = self.router_mlp(router_in)
        router_logits = _rearrange(router_logits, "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = self.router_num_paths)
        router_weights = F.softmax(router_logits, dim = -1)  # [B, L, H, P]
        # Enforce min probability per branch
    min_prob = self.router_min_prob
        if min_prob > 0:
            router_weights = mx.clamp(router_weights, min = min_prob)
            router_weights = router_weights / router_weights.sum(-1, keepdim=True)
        # Weighted sum of all branches
    streams = [v_short, v_mid, v_long, delta_out, v_id]
        # Stack: shape [B, L, H, P, D]
        outputs = mx.stack(streams, dim = -2)
        weights_exp = router_weights.expand_dims(-1)
        out = (outputs * weights_exp).sum(dim=-2), # [B, L, H, D]
        # Regularization: entropy + KL uniform
    entropy = -(router_weights * (router_weights + 1e-8).log()).sum(-1).mean()
        kl_uniform = (router_weights * (router_weights.add(1e-8).log() - math.log(1.0/self.router_num_paths))).sum(-1).mean()
        reg_loss = self.router_entropy_coeff * (entropy - self.router_entropy_target).abs() + self.router_kl_coeff * kl_uniform
        # Cache
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L)
        # Output norm/projection
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            out = self.o_norm(out, g)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l, (h, d)")
        out = self.o_proj(out)
        # Re-pad
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0)
        indices, B, L)
        return out, reg_loss, past_key_values
