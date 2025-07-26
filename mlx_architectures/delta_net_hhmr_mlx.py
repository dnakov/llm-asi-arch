"""
MLX-converted architecture: delta_net_hhmr
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
DeltaNet – Hierarchical Hybrid Multi-Scale Routing (DeltaNet-HHMR)
==================================================================
Identifier: *delta_net_hhmr*

This evolved architecture targets the dual bottleneck revealed by empirical
analysis: (1) over-compressed gating destroys extraction and comprehension,
while (2) lacking adaptive/decoupled local-global routing starves global
reasoning (coreference, ARC-Challenge). Integrating state-of-the-art research
and concrete ablations, this model introduces:

Key Innovations (Enabled by Default)
------------------------------------
1. **Hierarchical Hybrid Gating (H²-Gate)**
   • Decouples local-vs-global routing into a two-stage, *hierarchical* gate:
     - Stage 1: Head- and token-specific MLP determines the local vs global 
       pathway (scalar gate per (B,L,H)) using context-adaptive features.
     - Stage 2: On the "local" path (where local routing is dominant), a *rich-stats* 
       gate (MLP over both mean and variance of each local branch, per head) selects 
       among local FIR scales. On the "global" path, queries select between Delta-rule 
       and direct value via a high-resolution output-aware gate (MLP on mean/var/stdev).
   • This allows ultra-local, factual content to use high-fidelity gates and 
     challenging long-span tasks to benefit from full context/decisive global selection.

2. **Richer Stream Statistics for Gating**
   • Gating MLP inputs for all choices now concatenate *mean and variance* 
     per head and stream, not just mean. This restores fine-grained, entity-level 
     awareness for extraction without reverting to (prohibitively expensive) 
     full-feature flattening.

3. **Progressive Temperature Untying (Preserved)**
   • Retain proven per-head, scheduled τ untying: early, mean-τ for stable learning; 
     late, per-head τ allowing sharp specialisation for ARCs, Winogrande.

4. **Chunked/Batch-Agnostic, Causal Processing**
   • All paths implemented with chunked, strictly causal patterns and einops 
     handling for universal batch/seq compatibility.

5. **Adaptive Schedule Alignment**
   • All schedule lengths reduced to 2k steps by default, ensuring τ untying and 
     gating specialisation matches observed training durations.

All O(N·d) complexity and strict batch/sequence agnosticism maintained.
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F


# --- Helper functions ---
def _elu_p1(x: mx.Tensor) -> mx.Tensor:
    return (F.elu(x, 1.0, False) + 1.0)
def _sum_norm(x: mx.Tensor) -> mx.Tensor:
    return (x / x.sum(-1, keepdim=True))
def _mean_var(x: mx.Tensor) -> Tuple[mx.Tensor, mx.Tensor]:
    mu = x.mean(-1)
    var = x.var(-1, unbiased=False)
    return mu, var

# --- Depth-wise multi-scale causal FIR, as before ---
class _DepthwiseMultiScaleFIR(nn.Module):
    def __init__(self, *, num_heads: int, head_dim: int, kernel_sizes: Tuple[int, ...]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        channels = num_heads * head_dim
        self.filters = nn.ParameterList()
        for k in kernel_sizes:
            weight = mx.zeros(channels, 1, k)
            with mx.no_grad():
                weight[:, 0, -1] = 1.0
            self.filters.append(mx.array(weight))
    def forward(self, x: mx.Tensor) -> List[mx.Tensor]:
        b, L, h, d = x.shape
        x_ch = _rearrange(x, "b l h d -> b (h d) l")
        return [_rearrange(F.conv1d(mx.pad(x_ch, (k - 1, 0)), filt, groups=h*d), "b (h d) l -> b l h d"h=h)
                for filt, k in zip(self.filters, self.kernel_sizes)]

# --- Chunkwise Delta-Rule ---
@mx.compile
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size), (q, k, v, k_beta))
    tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    tri_strict = mx.triu(tri, 1)
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + mx.eye(chunk_size, dtype=inv.dtype, q.device)
    inv = inv
    u = inv @ v
    w = inv @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = mx.zeros_like(v)
    for blk in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, blk], k[:, :, blk]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        out[:, :, blk] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    out = _rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

class DeltaNet(nn.Module):
    """DeltaNet with Hierarchical Hybrid Multi-Scale Routing (HHMR)"""
    def __init__(
        self,
        *,
        mode: str = 'hhmr',
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
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        ms_kernel_sizes: Tuple[int, ...] = (1, 7, 15, 31),
        untie_start_step: int = 0,
        untie_end_step: int = 2000,
        fusion_hidden_mult: float = 1.0,
        floor_start: float = 0.02,
        floor_end: float = 0.0,
        floor_decay_steps: int = 2000,
        entropy_coeff_start: float = 0.02,
        entropy_coeff_end: float = 0.0,
        entropy_decay_steps: int = 2000,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
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
        self.ms_kernel_sizes = ms_kernel_sizes

        # Schedules
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_coeff_start = float(entropy_coeff_start)
        self.entropy_coeff_end = float(entropy_coeff_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.untie_start_step = int(untie_start_step)
        self.untie_end_step = int(untie_end_step)
        self, persistent=False)

        # Dimensions
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError('Key/Value dimensions must divide num_heads.')

        # Projections & convs
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        if not self.use_short_conv:
            raise UserWarning('_ShortConvolution is mandatory for DeltaNet variants.')
        act = 'silu' if qk_activation == 'silu' else None
        self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size=conv_size, activation='silu', bias=conv_bias)

        # Multi-scale FIR
        self.local_fir = _DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)

        # Hierarchical gates ---
        gate1_in_dim = hidden_size + self.num_heads * 2  # means and variances (local/global summary)
        gate1_hidden = max(8, int(gate1_in_dim * fusion_hidden_mult))
        self.gate1 = nn.Sequential(
            nn.Linear(gate1_in_dim, gate1_hidden, bias=True), nn.GELU(),
            nn.Linear(gate1_hidden, self.num_heads, bias=True),  # scalar gate per head (pre-sigmoid)
        )
        # Local branch gate: decide between local FIR scales
        gate_local_in_dim = hidden_size + 2 * self.num_heads * self.num_scales  # mean+var per scale
        gate_local_hidden = max(8, int(gate_local_in_dim * fusion_hidden_mult))
        self.gate_local = nn.Sequential(
            nn.Linear(gate_local_in_dim, gate_local_hidden, bias=True), nn.GELU(),
            nn.Linear(gate_local_hidden, self.num_heads * self.num_scales, bias=True),
        )
        # Global branch gate: decide between delta and direct value (mean+var each)
        gate_global_in_dim = hidden_size + 4 * self.num_heads  # mean/var delta, mean/var direct value
        gate_global_hidden = max(8, int(gate_global_in_dim * fusion_hidden_mult))
        self.gate_global = nn.Sequential(
            nn.Linear(gate_global_in_dim, gate_global_hidden, bias=True), nn.GELU(),
            nn.Linear(gate_global_hidden, self.num_heads * 2, bias=True),
        )

        # Temperature params (per-head, untied schedule)
        self.log_tau = mx.array(mx.zeros(num_heads))

        # Output norm/proj
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # --- schedule helpers ---
    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        r = t / max(1.0, self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start) * r
    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_coeff_end
        r = t / max(1.0, self.entropy_decay_steps)
        return self.entropy_coeff_start + (self.entropy_coeff_end - self.entropy_coeff_start) * r
    def _untie_factor(self) -> float:
        t = float(self._step.item())
        if t <= self.untie_start_step:
            return 0.0
        if t >= self.untie_end_step:
            return 1.0
        return (t - self.untie_start_step) / max(1.0, (self.untie_end_step - self.untie_start_step))

    # --- Forward ---
    def forward(
        self,
        hidden_states: mx.Tensor,  # (B,L,D)
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Dict,
    ) -> Tuple[mx.Tensor, Optional[mx.Tensor], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q = _rearrange(q_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k_lin, "b l (h d) -> b l h d"d=self.head_k_dim)
        v = _rearrange(v_lin, "b l (h d) -> b l h d"d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)
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
        delta_out_d, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")
        v_direct = v
        local_branches = self.local_fir(v)
        # --- Hierarchical Gating ---
        # Prepare local/global summary stats (mean, var over D per head/stream)
        sum_stats_local = []
        for x in local_branches:
            mu, var = _mean_var(x)
            sum_stats_local.append(mu)
            sum_stats_local.append(var)
        local_stats = mx.cat(sum_stats_local, dim=-1)  # (B,L,H*S*2)
        mu_delta, var_delta = _mean_var(delta_out)
        mu_direct, var_direct = _mean_var(v_direct)
        global_stats = mx.cat([mu_delta, var_delta, mu_direct, var_direct], dim=-1)  # (B,L,H*4)
        # Hierarchical gate, stage 1: local vs global decision (per-head token)
        gate1_feats = mx.cat([
            hidden_states,
            # mean + var across all local and global pathways (just means for efficiency)
            mx.cat([mu_direct, mu_delta], dim=-1),  # global means
        ], dim=-1)  # (B,L, D + H*2)
        gate1_logits = self.gate1(gate1_feats)   # (B,L,H)
        gate1_s = mx.sigmoid(gate1_logits)    # (B,L,H): 0=global, 1=local
        # Stage 2: local path (choose among local scales)
        local_feats = mx.cat([
            hidden_states,
            local_stats,
        ], dim=-1)  # (B,L, D + H*S*2)
        gate_local_logits = self.gate_local(local_feats)  # (B,L,H*S)
        gate_local_logits = _rearrange(gate_local_logits, "b l (h s) -> b l h s"h=self.num_heads, s=self.num_scales)
        # Stage 2: global path (choose delta vs direct value)
        global_feats = mx.cat([
            hidden_states,
            mu_delta, var_delta, mu_direct, var_direct,
        ], dim=-1)  # (B,L, D + H*4)
        gate_global_logits = self.gate_global(global_feats)
        gate_global_logits = _rearrange(gate_global_logits, "b l (h k) -> b l h k"h=self.num_heads, k=2)
        # Progressive τ untying for all gating stages
        tau_per_head = F.softplus(self.log_tau) + 1e-3  # (H,)
        untie_factor = self._untie_factor()
        mean_tau = tau_per_head.mean()
        eff_tau = tau_per_head * untie_factor + mean_tau * (1.0 - untie_factor)
        gate_local_logits = gate_local_logits / eff_tau.reshape(1,1,self.num_heads,1)
        gate_global_logits = gate_global_logits / eff_tau.reshape(1,1,self.num_heads,1)
        gate1_s = gate1_s  # logistic, no temperature needed
        gate_local_probs = mx.softmax(gate_local_logits, dim=-1)
        gate_global_probs = mx.softmax(gate_global_logits, dim=-1)
        # --- Gate floors & entropy regularisation ---
        eps_val = self._current_floor()
        if eps_val > 0.0:
            gate_local_probs = mx.clamp(gate_local_probs, min=eps_val)
            gate_local_probs = gate_local_probs / gate_local_probs.sum(-1, keepdim=True)
            gate_global_probs = mx.clamp(gate_global_probs, min=eps_val)
            gate_global_probs = gate_global_probs / gate_global_probs.sum(-1, keepdim=True)
        reg_loss = None
        if self.training:
            coeff = self._current_entropy_coeff()
            if coeff > 0.0:
                ent_local = -(gate_local_probs * (gate_local_probs+1e-8).log()).sum(-1).mean()
                ent_global = -(gate_global_probs * (gate_global_probs+1e-8).log()).sum(-1).mean()
                reg_loss = coeff * (ent_local + ent_global) / 2
        # --- Final fusion
        # Local: weighted sum of local FIRs
        local_stack = mx.stack(local_branches, dim=-2)  # (B,L,H,S,D)
        local_out = (local_stack * gate_local_probs.expand_dims(-1)).sum(-2) #(B,L,H,D)
        # Global: weighted sum of delta and direct
        global_stack = mx.stack([delta_out, v_direct], dim=-2)  # (B,L,H,2,D)
        global_out = (global_stack * gate_global_probs.expand_dims(-1)).sum(-2) # (B,L,H,D)
        # Blend local/global per (B,L,H) gate
        o = gate1_s.expand_dims(-1) * local_out + (1.0 - gate1_s).expand_dims(-1) * global_out
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B_orig, L_in)
        self._step += 1  # type: ignore[operator]
        return o, reg_loss, past_key_values
