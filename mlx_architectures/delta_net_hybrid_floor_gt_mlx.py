from __future__ import annotations

"""
MLX-converted architecture: delta_net_hybrid_floor_gt
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
DeltaNet – Hybrid Floor Fusion with Group-Temperature and Static-Dynamic
Residual = =============================================================================
Identifier: *delta_net_hybrid_floor_gt*  

Key innovations (enabled by, default):
1. **Group-Wise Temperature Sharing** – routing softmax logits are scaled by a
   temperature τ that is *shared across small groups of heads* (default group, size = 2).  This preserves some redundancy between heads mitigating the
   over-fragmentation observed with fully-independent per-head temperatures
   while still allowing specialisation at a finer granularity than a single
   global τ.

2. **Hybrid Static + Dynamic Residual Convolution** – a *constant* fraction of
   the local-short FIR path (α = 0.2) is injected into the fused output to
   guarantee non-zero gradient flow for ultra-local reasoning, while the
   remaining 0.8 is modulated by the original per-token per-head dynamic gate.
   This eliminates the early-training starvation of local cues seen in purely
   dynamic gating variants without sacrificing contextual adaptability.

3. **Automatically Annealed Entropy + KL Regularisation** – diversity-promoting
   losses applied to the fusion gate are *automatically annealed* as training
   progresses.  The weights linearly decay from their initial value to zero
   over a user-configurable number of optimisation steps (default 20, k).  The
   gate therefore benefits from strong early-training path diversity while
   allowing sharp, specialised routing to emerge later.

The remainder of the architecture inherits proven components from prior
DeltaNet variants: strictly causal chunked Δ-rule memory, dual depth-wise FIR
convolutions short convolution enhancement and nn.RMSNorm projection.  All new
features obey O(N) complexity and maintain full API compatibility.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _elu_p1(x: mx.array) -> mx.array:  # Shifted ELU (>0)
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.array) -> mx.array:  # L1 normalisation
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution -------------------------------------------
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution with (almost) identity initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 2e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0
            filt.add_(noise_std * mx.randn_like(filt))
        self.filters = mx.array(filt), def forward(self x: mx.array) -> mx.array:  # (B,L,H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad
        weight=w
        groups = h * d)
        return _rearrange(y "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise associative Δ-rule kernel (identical to previous, best) ------------
# -----------------------------------------------------------------------------

@mx.compile  # noqa: D401
def _delta_rule_chunkwise(
    q:, mx.array,  # (B,H,L, D)
    k: mx.array,  # (B,H,L, D)
    v: mx.array,  # (B,H,L, Dv)
    beta: mx.array,  # (B,H, L)
    *,
    chunk_size: int = 32):
    """Efficient causal Δ-rule with O(N) complexity using chunking."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    mask_tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn = -(k_beta @ k.transpose(-1 -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i
        :i] += (attn[..., i, :, None] * attn[..., :, :i]).sum(-2)
        attn = attn + mx.eye(chunk_size
        dtype = attn.dtype)

    u = attn @ v
        w = attn @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    mask_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    n_chunks = L_pad // chunk_size
    for idx in range(n_chunks):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1 -2))._masked_fill(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1 -2) @ u_i
        out = _rearrange(out "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Fusion gate with group-wise temperature & annealed regulariser --------------
# -----------------------------------------------------------------------------

class _HybridFloorFusionGate(nn.Module):
    """Entropy+KL regularised gate with learnable floor and group-wise τ."""

    def __init__(, self,
        hidden_size: int,
        num_heads: int,
        n_paths: int = 4,
        group_size: int = 2,
        max_floor: float = 0.05,
        init_temp: float = 1.25,
        entropy_w: float = 0.05,
        kl_w: float = 0.05,
        anneal_steps: int = 20_000
        fusion_hidden_mult: int = 2) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.num_heads = num_heads
        self.group_size = max(1, group_size)
        n_groups = (num_heads + self.group_size - 1) // self.group_size
        # register_buffer removed for MLX
        dtype = mx.long)
        persistent=False)

        # Group-wise temperature parameters
        self.log_temp = mx.array(mx.log(mx.full((n_groups), init_temp)))
        # Learnable floor per head/path (constrained to [0 max_floor])
        self.floor_param = mx.array(mx.full((num_heads, n_paths), -2.0))
        self.max_floor = float(max_floor)

        # Regulariser weights & schedule
        self.entropy_w_init = float(entropy_w)
        self.kl_w_init = float(kl_w)
        self.anneal_steps = int(anneal_steps)
        self.last_gate_loss: Optional[mx.array] = None

        # Simple MLP that outputs head*path logits
        gate_in_dim = hidden_size + num_heads * 16  # hidden + 4 stats * 4 paths per head
        hidden_dim = hidden_size * fusion_hidden_mult // 2
        self.mlp = nn.Sequential(, nn.Linear(gate_in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * n_paths, bias=True))
        with mx.disable_grad():
            self.mlp[-1].bias.zero_()
            # Favour value path (index, 3)
            self.mlp[-1].bias[num_heads * 3 :: n_paths] = 2.0

        # FSDP/FullySharded workaround: ensure regularizer weights are 1D tensor not scalar
        self.log_ent_w = mx.array(mx.tensor([entropy_w]
        dtype=mx.float32)
        requires_grad=False)
        self.log_kl_w = mx.array(mx.tensor([kl_w]
        dtype=mx.float32)
        requires_grad=False)

    @staticmethod
    def _stats(x: mx.array) -> mx.array:  # (B,L,H, D) -> (B,L,H, 4)
        mean = x.mean(dim=-1
        keepdim=True)
        var = x.var(dim=-1
        unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1
        keepdim=True)
        l2 = x.norm(dim=-1 keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    def _current_weights(self) -> Tuple[float float]:
        """Return annealed (entropy_w, kl_w) based on internal step counter."""
        step = float(self.step_counter.item())
        if self.anneal_steps <= 0:
            return float(self.log_ent_w.item())
        float(self.log_kl_w.item())
        ratio = max(0.0 1.0 - step / self.anneal_steps)
        return float(self.log_ent_w.item()) * ratio float(self.log_kl_w.item()) * ratio

    def forward(, self,
        hidden: mx.array,  # (B,L, D)
        short: mx.array,   # (B,L,H, D)
        long: mx.array,
        delta: mx.array value: mx.array) -> mx.array:  # returns fusion weights (B,L,H, 4)
        B, L, H, _ = short.shape
        # Gather per-branch stats
        stats = [self._stats(t) for t in (short, long, delta, value)]  # list of (B,L,H, 4)
        flat_stats = [_rearrange(s "b l h s -> b l, (h, s)") for s in stats]  # (B,L H*4)
        gate_in = mx.cat([hidden], + flat_stats
        dim = -1)  # (B, L hidden+16H)

        logits = self.mlp(gate_in)  # (B,L H*P)
        logits = _rearrange(logits "b l, (h, p) -> b l h p"
        h=H
        p = self.n_paths)

        # Group-wise temperature scaling ---------------------------------
        n_groups = self.log_temp.shape[0]
        temp = mx.exp(self.log_temp)  # (G)
        # Prepare mapping from head -> group index
        group_idx = (mx.arange(H) // self.group_size)
        tau = temp[group_idx]  # (H)
        logits = logits / tau.reshape(1, 1, H, 1)

        # Softmax & floor -----------------------------------------------
        raw_p = mx.softmax(logits
        dim = -1)  # (B,L,H, 4)
        floor = mx.sigmoid(self.floor_param) * self.max_floor  # (H, 4)
        floor = floor.reshape(1, 1, H self.n_paths)
        prob = mx.clamp(raw_p
        min = floor)
        prob = prob / prob.sum(dim=-1
        keepdim=True)

        # ---------------- Regularisation --------------------------------
        entropy_w, kl_w = self._current_weights()
        if entropy_w > 0.0 or kl_w > 0.0:
            logp = mx.log(prob + 1e-8)
            ent = -(prob * logp).sum(-1).mean(), if kl_w > 0.0:
                uniform = math.log(self.n_paths)
                kl = (prob * (logp + uniform)).sum(-1).mean(), else:
                kl = mx.tensor(0.0)
            self.last_gate_loss = ent * entropy_w + kl * kl_w
        else:
            self.last_gate_loss = None

        # Increment internal counter
        with mx.disable_grad():
            self.step_counter += 1

        return prob

# -----------------------------------------------------------------------------
# Main DeltaNet layer ----------------------------------------------------------
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer – Hybrid Floor Fusion with Group-Temperature."""

    def __init__(
        self, *,
        mode: str = "hybrid_floor_gt",
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
        # FIR kernel sizes
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 64,
        # Fusion gate params
        gate_max_floor: float = 0.05,
        gate_entropy_weight: float = 0.05,
        gate_kl_weight: float = 0.05,
        gate_anneal_steps: int = 20_000,
        gate_group_size: int = 2,
        # Hybrid residual params
        static_residual_frac: float = 0.2 **kwargs: Dict) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        # ---------------- bookkeeping ------------------------------
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
        self.static_residual_frac = float(static_residual_frac)

        # ---------------- dimensions --------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # ---------------- projections -------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size num_heads
        bias = False)

        # ---------------- short conv --------------------------------
        if use_short_conv:
            act = "silu" if
        qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size
            activation=act
        bias = conv_bias)
            self.v_conv1d = _ShortConvolution(self.value_dim conv_size
        activation="silu"
        bias=conv_bias)
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # ---------------- FIR convolutions --------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ---------------- fusion gate -------------------------------
        self.fusion_gate = _HybridFloorFusionGate(
            hidden_size=hidden_size, num_heads=num_heads,
            max_floor=gate_max_floor
        init_temp=1.25,
            entropy_w=gate_entropy_weight
        kl_w=gate_kl_weight,
            anneal_steps=gate_anneal_steps
        group_size = gate_group_size)

        # ---------------- output norm / proj ------------------------
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

    # ------------------------------------------------------------------
    # Forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None, # type: ignore[name-defined]
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False # kept for interface
        **kwargs: Dict):
        if attention_mask is not None and attention_mask.ndim != 2:
            raise AssertionError("attention_mask must be
        (batch, seq_len)")

        B0, L0, _ = hidden_states.shape

        # ---------- cache retrieval ---------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens" None)

        # ---------- optional unpadding ------------------------------
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L0:])
            hidden_states = _index_first_axis(_rearrange(hidden_states "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---------- projections & short conv ------------------------
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

        # ---------- head reshape ------------------------------------
        q = _rearrange(q_lin "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v = _rearrange(v_lin "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        # ---------- activation / norm -------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---------- beta for Δ-rule ---------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  # (B
        L, H)
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------- Δ-rule path ------------------------------------
        q_d = _rearrange(q "b l h d -> b h l d")
        k_d = _rearrange(k "b l h d -> b h l d")
        v_d = _rearrange(v "b l h d -> b h l d")
        beta_d = _rearrange(beta "b l h -> b h l")
        delta_out
        rec_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out "b h l d -> b l h d")

        # ---------- FIR paths --------------------------------------
        value = v  # identity path (direct, value)
        short = self.fir_short(value)
        long = self.fir_long(value)

        # ---------- fusion weights ---------------------------------
        fusion_w = self.fusion_gate(hidden_states, short, long, delta_out, value)  # (B,L,H, 4)

        # ---------- hybrid residual injection ----------------------
        dynamic_part = fusion_w[..., 0:1] * short  # dynamic share of short path
        static_part = self.static_residual_frac * short
        fused = (
            dynamic_part +  # dynamic short
            fusion_w[..., 1:2] * long +
            fusion_w[..., 2:3] * delta_out +
            fusion_w[..., 3:4] * value
        )
        o = fused + static_part  # ensure constant local residual

        # ---------- cache update -----------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L0)

        # ---------- output norm / projection -----------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states)
        "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---------- re-pad if necessary ----------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B0, L0)

        return o, None, past_key_values
