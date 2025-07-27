from __future__ import annotations

"""
MLX-converted architecture: delta_net_htfr
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
DeltaNet – Hierarchical Temperature-and-Floor Regularised Gating (DeltaNet-HTFR)
This evolution unifies the strongest empirical findings from previous DeltaNet
variants into a *single* architecture that simultaneously:

1.  Maintains *dual-scale* causal FIR convolutions for rich local context
    modelling (short + long kernels **identity-initialised** with small, noise).
2.  Integrates a *global* recurrent **Δ-rule** path for unlimited context
    propagation while preserving **O(N)** complexity via chunkwise scan.
3.  Employs a **three-way hierarchical fusion gate** with *learnable per-head
    temperature* **and** a small **ε-floor** at **all stages** to prevent early
    collapse and gradient starvation.
4.  Adds an always-on **entropy regularisation loss** that discourages overly
    sharp gating distributions and promotes balanced path utilisation.

The class name and public interface remain **DeltaNet**; all changes are
internal and enabled by default ensuring seamless drop-in compatibility.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



################################################################################
# Helper utilities                                                             #
################################################################################

def _elu_plus_one(x:, mx.array) -> mx.array:  # shifted ELU keeps >0
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:  # row-sum normalisation
    return (x / x.sum(-1, keepdim=True))

################################################################################
# Core chunk-wise Δ-rule implementation(unchanged, O(N·d))                    #
################################################################################

@mx.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def delta_rule_chunkwise(q:, mx.array,  # (B,H,L, D_k)
    k: mx.array,  # (B,H,L, D_k)
    v: mx.array,  # (B,H,L, D_v)
    beta: mx.array,  # (B,H, L)
    *,
    chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisations ----------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks ----------------------------------------------
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)
    inv = inv
        u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)
    excl_mask = mx.triu(tri_mask, 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(excl_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
################################################################################
# Depth-wise causal FIR convolution                                            #
################################################################################

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal 1-D FIR convolution with identity init."""

    def __init__(, self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_size: int, noise_std: float = 1e-2) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # causal identity (current, timestep)
        if noise_std > 0:
            filt.add_(mx.randn_like(filt) * noise_std)
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, L, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

################################################################################
# Optional typing imports -----------------------------------------------------#
################################################################################
################################################################################
# Main DeltaNet class                                                          #
################################################################################

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with hierarchical temperature- & ε-floor regularised gating."""

    def __init__(self, # ===== baseline
        args =====
        mode: str =, "htfr",  # hierarchical temperature-floor regularised
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
        # ===== new hyper-parameters =====
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 64,
        fusion_hidden_mult: int = 2,
        gate_epsilon: float = 0.05,  # ε-floor for *all* gates
        gate_temp_init: float = 1.0,  # initial temperature(per-head, log-space, param)
        entropy_reg_weight: float = 0.01 **kwargs: Dict) -> None:
        super().__init__()
        # ---------------- basic bookkeeping ----------------
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.gate_eps = gate_epsilon
        self.entropy_reg_weight = entropy_reg_weight

        # ---------------- dimensions ---------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---------------- projections --------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # ---------------- short conv --------------
        if use_short_conv:
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
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet-HTFR.")

        # ---------------- FIR branches -------------
        self.local_fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_short_kernel)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim
        kernel_size = fir_long_kernel)

        # ---------------- hierarchical gates -------
        fused_in_dim = hidden_size + self.head_v_dim * num_heads * 4  # hidden + all path outputs
        self.stage1_mlp = nn.Sequential(, nn.Linear(fused_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * 2 bias=True))
        local_in_dim = hidden_size + self.head_v_dim * num_heads * 2
        self.stage2_local_mlp = nn.Sequential(, nn.Linear(local_in_dim, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * 2, bias=True))
        global_in_dim = hidden_size + self.head_v_dim * num_heads * 2
        self.stage2_global_mlp = nn.Sequential(, nn.Linear(global_in_dim, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * 2, bias=True))

        # Warm-start bias favouring *direct value* path (index 1 of global, gate)
        with mx.disable_grad():
            if self.stage2_global_mlp[-1].bias is not None:
                self.stage2_global_mlp[-1].bias.zero_()
                self.stage2_global_mlp[-1].bias[num_heads:] = 4.0  # direct value branch bias

        # Per-head temperatures (log-param) – shared across all gates
    log_temp = math.log(gate_temp_init)
        self.log_temp_stage1 = mx.array(mx.full((num_heads, 1), log_temp))
        self.log_temp_stage2_local = mx.array(mx.full((num_heads, 1), log_temp))
        self.log_temp_stage2_global = mx.array(mx.full((num_heads, 1), log_temp))

        # ---------------- output norm/proj ----------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    ############################################################################
    # forward                                                                  #
    ############################################################################

    # pylint: disable=too-many-statements,too-many-branches,too-many-locals
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False # kept for API compat
        **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.dim() == 2 "attention_mask must be(batch, seq_len)"

        B_orig, L_orig, _ = hidden_states.shape

        # ------------- unpadding for variable length -------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ------------- linear projections + short conv -----------
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

        # ------------- head split & activations -------------------
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

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

        v_direct = v  # identity/value path

        # ------------- β for Δ-rule -------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------- Δ-rule path --------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d
        chunk_size =32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # ------------- FIR branches -------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ------------- Hierarchical gating ------------------------
        # Stage-1: local (short+long) vs global (delta+direct)
        stage1_in = mx.cat([, hidden_states)
            _rearrange(local_short, "b l h d -> b l, (h, d)"),
            _rearrange(local_long, "b l h d -> b l, (h, d)"),
            _rearrange(delta_out, "b l h d -> b l, (h, d)"),
            _rearrange(v_direct, "b l h d -> b l, (h, d)"),
        ], dim=-1)
        logits1 = self.stage1_mlp(stage1_in)  # (B,L H*2)
        logits1 = _rearrange(logits1, "b l, (h, s) -> b l h s"
        h=self.num_heads
        s = 2)
        temp1 = mx.exp(self.log_temp_stage1).expand_dims(0).expand_dims(0)  # (1,1,H, 1)
        w1 = mx.softmax(logits1, * temp1
        dim = -1)
        w1 = w1 * (1.0 - 2 * self.gate_eps) + self.gate_eps  # ε-floor

        # Stage-2 local: short vs long
        stage2_local_in = mx.cat([, hidden_states _rearrange(local_short, "b l h d -> b l, (h, d)"))
            _rearrange(local_long, "b l h d -> b l, (h, d)"),
        ], dim=-1)
        logits2l = self.stage2_local_mlp(stage2_local_in)
        logits2l = _rearrange(logits2l, "b l, (h, s) -> b l h s"
        h=self.num_heads
        s = 2)
        temp2l = mx.exp(self.log_temp_stage2_local).expand_dims(0).expand_dims(0)
        w2l = mx.softmax(logits2l, * temp2l
        dim = -1)
        w2l = w2l * (1.0 - 2 * self.gate_eps) + self.gate_eps

        # Stage-2 global: delta vs direct
        stage2_global_in = mx.cat([, hidden_states)
            _rearrange(delta_out, "b l h d -> b l, (h, d)"),
            _rearrange(v_direct, "b l h d -> b l, (h, d)"),
        ], dim=-1)
        logits2g = self.stage2_global_mlp(stage2_global_in)
        logits2g = _rearrange(logits2g, "b l, (h, s) -> b l h s"
        h=self.num_heads
        s = 2)
        temp2g = mx.exp(self.log_temp_stage2_global).expand_dims(0).expand_dims(0)
        w2g = mx.softmax(logits2g, * temp2g
        dim = -1)
        w2g = w2g * (1.0 - 2 * self.gate_eps) + self.gate_eps

        # Compose outputs --------------------------------------------------
        local_comb = w2l[..., 0:1] * local_short + w2l[..., 1:2] * local_long
        global_comb = w2g[..., 0:1] * delta_out + w2g[..., 1:2] * v_direct
        out = w1[..., 0:1] * local_comb + w1[..., 1:2] * global_comb

        # ------------- cache update --------------------------------------
        if use_cache and past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state_new
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_orig)

        # ------------- output norm & projection ---------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = _rearrange(out, "b l h d -> b l, (h, d)")
        out = self.o_proj(out)

        # ------------- repad if necessary --------------------------------
        if attention_mask is not None:
            out = _pad_input(out.squeeze(0)
        indices, B_orig, L_orig)

        # ------------- entropy regularisation ----------------------------
        # Compute average negative entropy across all gates
        probs = mx.cat([w1.flatten(-2), w2l.flatten(-2), w2g.flatten(-2)]
        dim=-1)  # (..., H*2*3)
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        reg_loss = self.entropy_reg_weight * (-entropy)  # maximise
        entropy => minimise negative

        return out, reg_loss, past_key_values
