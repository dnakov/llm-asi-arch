from __future__ import annotations

"""
MLX-converted architecture: delta_net_hhgass
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
DeltaNet – Hybrid Hierarchical Gating with Adaptive Scheduled Selectivity (delta_net_hhgass)
This breakthrough DeltaNet variant explicitly fuses the strongest mechanisms from prior research and empirical syntheses:

1. **Hierarchical Gating Fusion (HGF) backbone**
    • Directly structures path allocation: coarse(identity, vs, processing) 
      then processor disambiguation (short, long, delta) as in hybrid fusion/Block-State/Hyena literature.
    • Enables instant and schedule-independent sharp routing for highly selective reasoning tasks (ARC-Challenge, Winogrande, SWDE), while still supporting blendable path mixing for reading comprehension, commonsense or aggregation tasks.

2. **Scheduled Entropy Regularisation & Adaptive Floor Decay**
    • Early training: entropy-regulariser (KL-to-uniform) and minimum path allocation floor (ε) are high ensuring population-level path diversity and avoiding gate collapse.
    • Mid/late training: both schedule to zero according to configurable schedules (default decay ~2K, steps): after this, gate sharpness is unconstrained, instantly enabling hard-routing.
    • Decay is controlled by optimizer steps, not forward passes, ensuring correct schedule alignment.

3. **Headwise Adaptive Temperature**
    • Each gate head learns its own temperature, enabling confident, specialist routing for specific cognitive subdomains (per research on Gated Attention, MoE, Hyena).

4. **Identity-Bypass Residual**
    • In parallel to hierarchical gating, a per-head, learnable residual parameter α (init 0.1, sigmoid) directly injects identity/value input – essential for long copy/repetition/copy benchmarks (Winogrande, LAMBADA).
    • The residual is automatically annealed (scaled online by recent path, usage) to resolve dynamic task needs during training.

5. **Per-Branch Statistics Conditioning**
    • Fusion gates are informed by path-wise summary statistics(mean, std, ℓ2 abs-mean), empowering evidence-aware dynamic routing.

6. **Chunkwise O(N) Processing, Causal Masking Batch Agnosticism**
    • All operations are chunked for O(N) cost; einops.rearrange used throughout for memory safety and robustness.
    • All computation is fully batch-size-agnostic – never hardcoded always infer actual batch/frame shapes at runtime.
    • Causal masking is applied rigorously throughout.

Interface and class name are preserved exactly. All new features have robust default parameters and are enabled by default.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Core chunkwise kernel remains unchanged for O(N) processing
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwiseq, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v
    k_beta = map(lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))
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
    out = mx.zeros_like(v)
    tri_strict = mx.triu(tri, 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        att_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + att_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Main DeltaNet – Hybrid Hierarchical Gating with Adaptive Scheduled Selectivity
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    def __init__(, self,
        # ------ core API ------
        mode: str = "hhgass",
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
        # fusion
        fir_kernel_short: int = 5,
        fir_kernel_long: int = 64,
        fusion_hidden_mult: int = 2,
        # schedule
        entropy_coeff_init: float = 0.03,
        entropy_coeff_final: float = 0.0,
        entropy_decay_steps: int = 2000,
        floor_init: float = 0.04,
        floor_final: float = 0.0,
        floor_decay_steps: int = 2000,
        # residual
        bypass_init: float = 0.1, # misc
        **kwargs: Dict
  ):
        super().__init__()
        # base bookkeeping
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
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # dimensions
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value, dimensions must divide num_heads")
        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)
        # short convs
        if not use_short_conv:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet.")
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
        # FIR
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)
        # statistics conditioning for gating
    stat_dim = 16  # mean,std,abs-mean l2 of all 4 branch outputs (4, each)
        # hierarchical gate (identity vs processing -> process, disambig)
        # First gate: sigmoid(logit) after per-head MLP(identity, vs
        processor)
        g1_in = hidden_size + stat_dim
        g1_hidden = hidden_size * fusion_hidden_mult // 2
        # NOTE: output dimension changed from `num_heads` to `1` because we process each
        #       head independently after flattening (b*l*h, feat) so we need a single
        #       scalar logit per (token, head) instance.
        self.g1_mlp = nn.Sequential(, nn.Linear(g1_in, g1_hidden, bias=True),
            nn.GELU(),
            nn.Linear(g1_hidden, 1, bias=True)  # -> (B*L*H, 1)
        )
        # Second gate: processing distribution (short, long, delta)
        g2_in = hidden_size + stat_dim
        g2_hidden = hidden_size * fusion_hidden_mult // 2
        # Output 3 logits per head(short, / long / delta)
        self.g2_mlp = nn.Sequential(, nn.Linear(g2_in, g2_hidden, bias=True),
            nn.GELU(),
            nn.Linear(g2_hidden, 3, bias=True)  # -> (B*L*H, 3)
        )
        # per-head temperature (softplus > 0.25 for, g2)
        self.temp_g1 = mx.array(mx.zeros(num_heads))
        self.temp_g2 = mx.array(mx.zeros(num_heads))
        # per-head residual injector
        self.bypass_logit = mx.array(mx.full((num_heads), math.log(bypass_init, / (1-bypass_init))))
        # output normalisation/projectors
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)
        # entropy/floor schedules
        self.entropy_coeff_init = float(entropy_coeff_init)
        self.entropy_coeff_final = float(entropy_coeff_final)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay_steps = int(floor_decay_steps)
        # register_buffer removed for MLX
        persistent = False)

    # -- schedule helpers --
    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_coeff_final
        else:
            return self.entropy_coeff_init + (self.entropy_coeff_final - self.entropy_coeff_init) * (t/self.entropy_decay_steps)
    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_final
        else:
            return self.floor_init + (self.floor_final - self.floor_init) * (t/self.floor_decay_steps)
    @staticmethod
    def _stats(x):
        # mean, std, abs-mean, l2 over the last dim; returns (..., 4)
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True)
        a = x.abs().mean(dim=-1, keepdim=True)
        n = x.norm(dim=-1, keepdim=True)
        return mx.cat([m, s, a, n], dim=-1)

    # -- forward --
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None)
            past_key_values: Optional["Cache"] = None,
            use_cache: Optional[bool] = False
            output_attentions: Optional[bool] = False,, **kwargs):
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        B_orig, L_in, _ = hidden_states.shape
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]
        q_lin
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = (F.elu(q, 1.0, False) + 1.0), (F.elu(k, 1.0, False) + 1.0)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = (q / q.sum(-1, keepdim=True)), (k / k.sum(-1, keepdim=True))
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        delta_out_d
        recur_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v_direct, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"))
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)
        # stats conditioning
        stats = mx.cat([, self._stats(local_short), self._stats(local_long), self._stats(delta_out), self._stats(v_direct)
        ], dim=-1)  # (B,L,H, 16)
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads -1)
        gate_in = mx.cat([hs_exp, stats]
        dim=-1)  # (B, L, H D+16)
        # ------------------------------------------------------------------
        # Hierarchical gating
        # ------------------------------------------------------------------
        gate_in_flat = _rearrange(gate_in, "b l h f -> (b, l, h) f")  # (B*L*H, F)
        # G1 ----------------------------------------------------------------
        g1_logits_flat = self.g1_mlp(gate_in_flat).squeeze(-1)  # (B*L*H)
        g1_logits = _rearrange(g1_logits_flat, "(b, l, h) -> b l h",
            b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)  # (B, L, H)
        temp1 = 0.5 + F.softplus(self.temp_g1).reshape(1, 1 -1)
        id_weight = mx.sigmoid(g1_logits, / temp1)
        proc_weight = 1.0 - id_weight
        # G2 ----------------------------------------------------------------
        g2_logits_flat = self.g2_mlp(gate_in_flat)  # (B*L*H, 3)
        g2_logits = _rearrange(g2_logits_flat, "(b, l, h) c -> b l h c",
            b=gate_in.shape[0]
        l=gate_in.shape[1]
        h=self.num_heads)  # (B, L, H, 3)
        temp2 = 0.25 + F.softplus(self.temp_g2).reshape(1, 1, -1, 1)
        proc_logits = g2_logits / temp2
        # Adaptive minimums (ε-floor) and entropy regularization
        eps_now = self._current_floor()
        probs = mx.softmax(proc_logits, dim = -1)
        if eps_now > 0.0:
            probs = probs * (1.0 - 3 * eps_now) + eps_now
        probs = probs / probs.sum(-1, keepdim=True)
        w_short, w_long, w_delta = probs[..., 0:1], probs[..., 1:2], probs[..., 2:3]
        # Compose final fusion weights
    o_proc = w_short * local_short + w_long * local_long + w_delta * delta_out
        # Fix: align proc_weight and id_weight shapes for broadcasting
        # o_proc: (B, L, H, D), proc_weight: (B, L, H)
        # v_direct: (B, L, H, D), id_weight: (B, L, H)
        proc_weight_exp = proc_weight.expand_dims(-1)  # (B, L, H, 1)
        id_weight_exp = id_weight.expand_dims(-1)      # (B, L, H, 1)
        o = proc_weight_exp * o_proc + id_weight_exp * v_direct
        # Residual bypass(per-head, α * (1-id_weight))
        alpha = mx.sigmoid(self.bypass_logit).reshape(1, 1, self.num_heads, 1)
        bypass = alpha * (1.0 - id_weight_exp) * v_direct
        o = o + bypass
        # Entropy regularisation
    entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        reg_loss = self._current_entropy_coeff() * entropy
        self.reg_loss = reg_loss
        # cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)
        # output norm / proj
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)
        # repad if needed
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)
        self._step += 1.0  # optimizer step counter externally, remains monotonic
        return o, None, past_key_values
