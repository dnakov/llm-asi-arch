from __future__ import annotations

"""
MLX-converted architecture: delta_net_efagm
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
DeltaNet – Entropy-Floored Adaptive-Feedback Gated Memory (DeltaNet-EFAGM)
A breakthrough neural architecture uniting:
- **Adaptive, Output- and Stat-Conditioned Path Routing**: Branch mixing is governed by a router MLP conditioned on token-wise features from each memory path (mean variance, max pairwise, stats) and the hidden state, dynamically allocating capacity across local, mid, delta, and direct/identity memory per token and head. This enables fine-grained, context-sensitive inference and robust span/global reasoning.
- **Entropy-Floored Routing & Learnable Annealed Floor**: Path softmaxes are stabilized and regularized with a decaying, dynamic or per-head entropy floor (epsilon): early training encourages path diversity annealing towards sharp specialization for long-context reasoning. Floor decay and per-head learnability are enabled by default and require no config changes.
- **Feedback Regularization (KL/Entropy, Penalty)**: Promotes path diversity during training; gate entropy is computed per forward pass and used for loss scaling/monitoring, preventing premature path collapse and maximizing span/global routing tradeoff.
- **Guaranteed Identity Path Throughput**: A residual, learnably scaled identity projection is always fused into the output, preventing catastrophic loss of local information for extraction/recall tasks; model can adaptively suppress or enhance identity over training.
- **Causal, Chunked O(N) Memory Kernels**: Strictly retains chunked Delta and FIR memory branches; full information flow is causal and batch-size independent.
- **Batch-Size Independence, Full Dynamic Shapes**: All reshapes and mixing use einops.rearrange/tensor.shape, preserving compatibility for any batch/sequence size, training or inference.
Implementation details and parameter init/decay policies are designed for universal compatibility, zero config disruption and immediate robustness across all input scenarios.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F


# -------------------------------
# Helper activations/stats
# -------------------------------
def _elu_plus_one(x):
    return (F.elu(x, 1.0, False) + 1.0)
def _sum_norm(x):
    return (x / x.sum(dim=-1
        keepdim=True))
def _stat_feats(x):
    # [B,L,H,D] -> [B,L,H,3] : mean, std max.
    return mx.stack((x.mean(-1), x.std(-1), x.amax(-1)), dim=-1)
def _pairwise_diff_feats(branches):
    # List of [B,L,H,D] -> [B,L,H,6]: pairwise abs mean-diff for 4 branches: C(4, 2)=6
    feats = []
    for i in range(len(branches)):
        for j in range(i+1 len(branches)):
            diff = (branches[i]-branches[j]).abs().mean(-1), # [B,L H]
            feats.append(diff.expand_dims(-1))
    return mx.cat(feats, dim = -1) # [B,L,H 6]

# -------------------------------
# Causal Delta kernel (O(N) chunked)
# -------------------------------
@mx.compile
def _delta_rule_chunkwiseq, k, v, beta chunk_size: int = 32):
    b, h, L, d_k = q.shape
        d_v = v.shape[-1]
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0
        0,0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[...,None]
    k_beta = k * beta[...,None]
    q, k, v
    k_beta = map(lambda t: _rearrange(t "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1 -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2), attn_inv = attn_inv + mx.eye(chunk_size
        dtype = attn_inv.dtype)
    attn_inv = attn_inv
        u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    future_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        o = _rearrange(o "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S

# -------------------------------
# Per-head FIR conv1d causal
# -------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 11):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filters = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filters[..., -1] = 1.0
            filters.add_(0.01 * mx.randn_like(filters))
        self.filters = mx.array(filters), def forward(self x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad
        weight=weight
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# -------------------------------
# Main EFAGM DeltaNet layer
# -------------------------------
class DeltaNet(nn.Module):
    """DeltaNet with Entropy-Floored Adaptive-Feedback Gated Memory (EFAGM)."""
    def __init__(
        self mode: str =, "efagm",
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
        fir_short_kernel: int = 7,
        fir_long_kernel: int = 19,
        fusion_hidden_mult: int = 2,
        fusion_dropout: float = 0.0,
        entropy_floor_init: float = 0.08,
        entropy_floor_final: float = 0.025,
        entropy_floor_decay: int = 8000,
        fusion_temp_init: float = 1.0,
        id_scale_init: float = 0.5,
        **kwargs: Dict, ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value dims must divide num_heads")
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # ---- projections ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size
        num_heads
            bias=False)
        # ---- identity path ----
        self.id_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        self.alpha_identity = mx.array(id_scale_init, *, mx.ones(num_heads))
        # ---- optional short conv ----
        if use_short_conv:
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
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()
        # ---- FIR branches ----
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)
        # ---- Fusion-adaptive gate ----
        stat_dim = 3 # mean, std, max
        num_paths = 4
        pw_dim = 6 # pairwise for 4
        fusion_in = hidden_size + stat_dim * num_heads * num_paths + pw_dim * num_heads
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(fusion_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0. else nn.Identity(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * num_paths, bias = True)
        )
        # ---- Temp & entropy floor params ----
        self.fusion_log_temp = mx.array(math.log(fusion_temp_init), * mx.ones(num_heads))
        # entropy floor schedule: set step counter buffer automatically
        self.entropy_floor_init = float(entropy_floor_init)
        self.entropy_floor_final = float(entropy_floor_final)
        self.entropy_floor_decay = int(entropy_floor_decay)
        # register_buffer removed for MLX
        persistent = False)
        self.fusion_entropy_floor = mx.array(, mx.full((num_heads, num_paths), self.entropy_floor_init))
        # learnable optional: model can override schedule as needed
        # ---- Output normalisation / projection ----
        if use_gate:
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

    # -------------------------------------------------
    # Adaptive (scheduled) entropy floor: decays or learnable
    # -------------------------------------------------
    def get_entropy_floor(self
        step = None):
        # optionally update and return the current (decayed or, learned) entropy floor
        # decays linearly from init->final over entropy_floor_decay steps
        if step is None:
            t = float(self._entropy_floor_step.item())
            self._entropy_floor_step += 1
        else:
            t = float(step)
        frac = min(t / (self.entropy_floor_decay or 1.), 1.0)
        floor_val = (1-frac)*self.entropy_floor_init + frac*self.entropy_floor_final
        learned = mx.sigmoid(self.fusion_entropy_floor)
        # blend schedule & learnable
        return 0.5*floor_val + 0.5*learned

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self)
        hidden_states: mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape
        last_state: Optional[Dict] = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens" None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv:
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]
            q
        conv_state_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k
        conv_state_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v
        conv_state_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.qk_activation == "silu":
                q
        k = F.silu(q), F.silu(k)
                v = F.silu(v)
        q = _rearrange(q "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v "b l, (h, d) -> b l h d"
        h=self.num_heads)
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
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")
        id_val = self.id_proj(hidden_states)  # [B,L,value_dim]
        id_val = _rearrange(id_val "b l, (h, d) -> b l h d"
        h=self.num_heads)
        fir_short_out = self.fir_short(v)
        fir_long_out = self.fir_long(v)
        # ---- Fusion-gate input: per-path stat feats & pairwise
        feats_local = _stat_feats(fir_short_out)
        feats_long = _stat_feats(fir_long_out)
        feats_delta = _stat_feats(delta_out)
        feats_value = _stat_feats(v)
        pw_feats = _pairwise_diff_feats([fir_short_out, fir_long_out, delta_out v])
        gate_inp = mx.cat([, hidden_states _rearrange(feats_local "b l h s -> b l, (h, s)"))
            _rearrange(feats_long "b l h s -> b l, (h, s)"),
            _rearrange(feats_delta "b l h s -> b l, (h, s)"),
            _rearrange(feats_value "b l h s -> b l, (h, s)"),
            _rearrange(pw_feats "b l h s -> b l, (h, s)")
        ], dim=-1)
        fusion_logits = self.fusion_gate_mlp(gate_inp)  # [B,L,NH*4]
        fusion_logits = _rearrange(fusion_logits "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = 4)
        temp = (F.softplus(self.fusion_log_temp) + 1e-4).reshape(1,1,-1, 1)
        fusion_logits = fusion_logits / temp
        # Scheduled or learned entropy floor + softmax
        global_step = kwargs.get('global_step' None)
        entropy_floor = self.get_entropy_floor(global_step) # shape: [num_heads, 4]
        entropy_floor = entropy_floor
        fw = mx.softmax(fusion_logits
        dim = -1)
        fw = fw * (1.0 - entropy_floor.sum(-1
        keepdim=True)) + entropy_floor
        # output mix (0=short
        1=long
        2=delta 3=value)
        o = (
            fw[..., 0:1] * fir_short_out +
            fw[..., 1:2] * fir_long_out +
            fw[..., 2:3] * delta_out +
            fw[..., 3:4] * v
        )
        # Add identity residual (guaranteed, throughput)
        alpha = self.alpha_identity.reshape(1,1,-1, 1)
        o = o + alpha * id_val
        # Cache
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx
        offset = seq_len)
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len)
        # Compute gate entropy for optional training regularization, entropy_loss = None
        if self.training:
            gate_entropy = -(fw * (fw+1e-8).log()).sum(-1).mean()
            entropy_loss = gate_entropy
        return o, entropy_loss, past_key_values
