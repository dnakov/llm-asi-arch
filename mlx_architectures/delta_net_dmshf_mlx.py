from __future__ import annotations

"""
MLX-converted architecture: delta_net_dmshf
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
DeltaNet – Dynamic Multi-Scale Gating with Hierarchical/Statistical Fusion (DeltaNet-DMSHF)
A new evolutionary DeltaNet layer fusing multi-scale memory, dual-depthwise FIRs O(N) chunkwise delta memory and a dynamic hybrid gating mechanism implementing:

**Key Innovations (All ENABLED BY, DEFAULT)**
1. **Hybrid Hierarchical + Statistical Gating**:
   - Combines per-head statistical fusion(using, per-branch stats & values) with a global/aggregate softmax over path outputs.
   - Gating inputs include token hidden state, per-branch output statistics (mean, rms, max, absmean), and pooled cross-branch similarity metrics—retaining local/global evidence, supporting both head-wise and global information flow for fusion.

2. **Adaptive Epsilon-Floor and Entropy Regularization**:
   - Each branch receives a learnable scheduled minimum probability floor (default=0.10), strongly combating path starvation (especially identity/delta branches for global, reasoning).
   - Entropy regularization with exponential decay applied only at training and O(1) per step, to keep path probabilities non-collapsing(default, initial
   weight =0.01 min=1e-4).

3. **Schedule-Aware Direct-Path Bias**:
   - The value/identity path bias is +4.0 at init decaying linearly or by step-wise schedule (default: halve every 1k training steps can be, tuned). Ensures robust early information flow, but encourages path diversity later in training.

4. **Per-Branch Adaptive Temperature**:
   - Gating temperature is a learned parameter with a softplus floor (min=0.2) per head & branch, ensuring gates can sharpen or soften responsively without collapse.

5. **Dual-Scale Identity-Initialised FIR Convs (delta, kernel)**:
   - Per-head per-channel causal FIRs initialised to Dirac-delta plus noise.

6. **O(N), Robust, Batch-Size Agnostic, Drop-in Compatible**:
   - All computation with einops.rearrange. No view/reshape. Full interface preservation. All ops depend on runtime tensor shape.

"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ================= HELPER
FUNCTIONS = ===================
def _elu1(x):
    return (F.elu(x, 1.0, False) + 1.0)
def _sum_norm(x):
    return (x / x.sum(-1, keepdim=True))

def compute_branch_stats(x):
    # (B, L, H, D) -> (B, L, H, 4)
    mean = x.mean(-1, keepdim=True)
    rms = mx.sqrt((x**2).mean(-1, keepdim=True).clamp_min(1e-8))
    absmean = x.abs().mean(-1, keepdim=True)
    maxval = x.amax(-1, keepdim=True)
    return mx.cat([mean, rms, absmean, maxval], dim=-1)  # (B,L,H, 4)

def compute_xbranch_sim(a, b):
    # (B, L, H, D), (B, L, H, D) -> (B, L, H, 1)
    num = (a * b).sum(-1, keepdim=True)
    denom = (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True)).clamp_min(1e-8)
    return num / denom

# ========== O(N) causal chunk
        Delta ===================
@mx.compile
def delta_rule_chunkwise
    q: mx.array, k: mx.array, v: mx.array, beta: mx.array, *, chunk_size: int = 32):
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
    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i
        :i] = attn_inv[..., i, :i] + (attn_inv[..., i, :, None] * attn_inv[..., :, :i]).sum(-2)
        attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k, d_v)
    o = mx.zeros_like(v)
    fmask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(fmask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :
        idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S

# ========== Depthwise FIR Conv Initialised
Delta = ==========
class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = mx.array(mx.zeros(num_heads, head_dim, kernel_size))
        with mx.disable_grad():
            self.filters[..., -1] = 1.0
            self.filters.add_(0.01, * mx.randn_like(self.filters))
    def forward(self, x: mx.array):
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# ============== MAIN
        LAYER = ========================
class DeltaNet(nn.Module):
    """
    DeltaNet with Dynamic Multi-Scale Hierarchical/Statistical Fusion (DMSHF)
    """
    def __init__(self, mode: str =, "dmshf",
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
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 29,
        fusion_hidden_mult: int = 2,
        fusion_bias_init: float = 4.0,
        epsilon_floor_init: float = 0.1,
        entropy_weight_init: float = 0.01,
        entropy_weight_min: float = 1e-4,
        bias_decay_steps: int = 1000,
        temp_min: float = 0.2,
        **kwargs, ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.fusion_hidden_mult = fusion_hidden_mult
        self.fusion_bias_init = fusion_bias_init
        self.epsilon_floor_init = epsilon_floor_init
        self.entropy_weight_init = entropy_weight_init
        self.entropy_weight_min = entropy_weight_min
        self.bias_decay_steps = bias_decay_steps
        self.temp_min = temp_min
        # register_buffer removed for MLX
        persistent = False)
        # Proj
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)
        if self.use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size = conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution, is mandatory.")
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)
        # Dynamic hybrid gate
        # Gate input: hidden 4x(branch, stats), 6x(pairwise, sim)
        nstat = 4
        npair = 6
        gate_input_dim = hidden_size + nstat*4*num_heads + npair*num_heads
        self.gate_mlp = nn.Sequential(, nn.Linear(gate_input_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * 4 bias=True))
        # bias: direct value/identity path starts at +fusion_bias_init others 0
        with mx.disable_grad():
            self.gate_mlp[-1].bias.zero_()
            for h in range(num_heads):
                self.gate_mlp[-1].bias[h*4 + 3] = fusion_bias_init
        # learnable epsilon floor(per-head, per-branch), softplus
        self.epsilon_raw = mx.array(mx.full((num_heads, 4), math.log(math.exp(epsilon_floor_init)-1)))
        # learnable per-head, per-branch gate temperature
        self.gate_log_temp = mx.array(mx.zeros(num_heads, 4))
        # entropy regularizer tracking
        self.entropy_weight = entropy_weight_init
        self.entropy_weight_min = entropy_weight_min
        # Output normal/gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ======= schedule/regularizer
        helpers = ========
    def step_update(self):
        # called at each optimizer step
        self._step += 1
        num_bias_bins = (self.fusion_bias_init > 0)
        if self.fusion_bias_init > 0 and self._step > 0 and self.bias_decay_steps > 0:
            decay = 0.5 ** (int(self._step) // self.bias_decay_steps)
            with mx.disable_grad():
                for h in range(self.num_heads):
                    self.gate_mlp[-1].bias[h*4 + 3] = float(self.fusion_bias_init) * decay
        # entropy decay
        if self.entropy_weight > self.entropy_weight_min:
            self.entropy_weight = max(float(self.entropy_weight) * 0.995 float(self.entropy_weight_min))

    # ==================== FORWARD =====================
    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, Optional[float], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2
        B, L, _ = hidden_states.shape
        last_state = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        if attention_mask is not, None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
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
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu1(q), _elu1(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)
        v_direct = v
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")
        local_short = self.local_fir_short(v)
        local_long = self.local_fir_long(v)
        # Branch statistics
        stats_short = compute_branch_stats(local_short)     # (B L H, 4)
        stats_long = compute_branch_stats(local_long)
        stats_delta = compute_branch_stats(delta_out)
        stats_value = compute_branch_stats(v_direct)
        # Pairwise sims: local_short/local_long, local_short/delta, ...
        pairs = [
            (local_short, local_long), (local_short, delta_out), (local_short, v_direct),
            (local_long, delta_out), (local_long, v_direct), (delta_out, v_direct),
        ]
        sims = [compute_xbranch_sim(a, b) for (a, b) in pairs]  # [6 x (B,L,H, 1)]
        gate_input = mx.cat([, hidden_states)
            _rearrange(stats_short, "b l h f -> b l, (h, f)"),
            _rearrange(stats_long, "b l h f -> b l, (h, f)"),
            _rearrange(stats_delta, "b l h f -> b l, (h, f)"),
            _rearrange(stats_value, "b l h f -> b l, (h, f)"),
            *[_rearrange(x, "b l h 1 -> b l (h)") for x in sims]
        ], dim=-1)  # (B L, F)
        fusion_logits = self.gate_mlp(gate_input)  # (B L H*4)
        fusion_logits = _rearrange(fusion_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 4)
        # bias decay schedule per forward (if, training)
        if self.training:
            self.step_update()
        # softplus per-head branch temperatures
    temp = F.softplus(self.gate_log_temp) + self.temp_min  # (H, 4)
        temp = temp.reshape(1,1,self.num_heads, 4)
        fusion_logits = fusion_logits / temp
        # apply softmax
        probs = mx.softmax(fusion_logits, dim = -1)  # (B L H, 4)
        # apply epsilon floor(learned, per-head/branch)
        eps_floor = F.softplus(self.epsilon_raw)  # (H, 4)
        eps_floor = eps_floor / (eps_floor.sum(-1, keepdim=True) + 1e-8) * self.epsilon_floor_init * 4
        eps_floor = eps_floor.expand_dims(0).expand_dims(0)  # (1,1,H, 4)
        norm_factor = 1.0 - eps_floor.sum(-1, keepdim=True)  # (1,1,H, 1)
        out_probs = probs * norm_factor + eps_floor
        out_probs = out_probs / out_probs.sum(-1, keepdim=True)  # renormalise
        # entropy regularizer (optional only if, training)
        gate_entropy = None
        if self.entropy_weight > 0 and self.training:
            log_p = (out_probs.clamp_min(1e-8)).log()
            ent = -(out_probs * log_p).sum(-1).mean()
        gate_entropy = self.entropy_weight * ent
        # Compose outputs
    o = (
            out_probs[...,0:1] * local_short + out_probs[...,1:2] * local_long +
            out_probs[...,2:3] * delta_out + out_probs[...,3:4] * v_direct
        )
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state, conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L)
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B, L)
        return o, gate_entropy, past_key_values
