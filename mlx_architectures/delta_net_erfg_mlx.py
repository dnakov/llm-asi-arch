from __future__ import annotations

"""
MLX-converted architecture: delta_net_erfg
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
DeltaNet – Entropy-Regularised Floor-Gated Multi-Scale Memory (ERFG)
Identifier: delta_net_erfg

This evolution unifies the strongest empirical elements of prior DeltaNet
variants while *directly fixing* the two key residual bottlenecks that were
identified across experiments:

1. **Early Path-Collapse caused by un-regularised gating**
   •  A new *Entropy-Regularised Fusion Gate* (ERFG) applies an explicit
      entropy + KL penalty to the per-token / per-head routing probabilities.
      The penalty is returned as the `reg_loss` from `forward()` so the
      training loop can incorporate it seamlessly.
   •  A learnable probability floor(as, in `adaptive_floor_gate`) remains
      but is now *trainable* through a bounded parameter – the entropy term
      prevents the floor from decaying to zero and collapsing unused paths.

2. **Premature Memory Truncation via unconstrained λ (forget, gate)**
   •  The per-head forget parameter λ is now *scheduled* by a simple
      monotonic function that starts at 1 (no, forgetting) and only decays
      toward the learnable target value after `warmup_steps` (default = 30, k)
      – eliminating early long-context degradation while retaining
      adaptability later in training.  The schedule is implemented on-the-fly
      inside `forward()` using the `step` kwarg (optionally supplied by the
      training, loop).

All other strengths – dual FIR branches, chunked Δ-rule kernel, adaptive
probability floor per-head temperature – are preserved.  Complexity remains
O(N) and the public interface is unchanged.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
# Helper activations & norms
# ---------------------------------------------------------------------------

def _elu_plus_one(x:, mx.array) -> mx.array:
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------------------------------------
# Depth-wise FIR conv (Dirac + orthogonal, noise)
# ---------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 5e-3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = mx.zeros(num_heads, head_dim, self.kernel_size)
        weight[..., -1] = 1.0  # identity (Dirac) at latest time-step
        if noise_std > 0:
            noise = mx.randn_like(weight) * noise_std
            # make noise orthogonal to identity direction for stability
        proj = (noise * weight).sum(-1, keepdim=True)
            noise = noise - proj * weight
        weight = weight + noise
        self.filters = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # x: [B, L, H, D]
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel with optional forgetting
# ---------------------------------------------------------------------------

@mx.compile  # retain high-performance compilation
# pylint: disable=too-many-locals, too-many-statements

def _delta_rule_chunkwise
    q: mx.array,  # [B H L Dk]
    k: mx.array,  # [B H L Dk]
    v: mx.array,  # [B H L Dv]
    beta: mx.array,  # [B H L]
    forget: Optional[mx.array] = None,  # [B H]
    *,
    chunk_size: int = 32):
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # chunk reshape --------------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    mask_tri = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_), 0
    )
    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (
            inv[..., i, :, None] * inv[..., :, :i]
        ).sum(-2), inv = inv + mx.eye(chunk_size, dtype = inv.dtype)

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)
    mask_future = mx.triu(, mx.ones(chunk_size, chunk_size, dtype=mx.bool_), 1
    )

    lam = None
    if forget is not None:
        lam = forget[...
        None None]  # [B H 1 1]

    n_chunks = q.shape[2]
    for idx in range(n_chunks):
        q_i
        k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        if lam is None:
            S = S + k_i.transpose(-1, -2) @ u_i
        else:
            S = S * lam + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# ---------------------------------------------------------------------------
# Entropy-Regularised Fusion Gate
# ---------------------------------------------------------------------------

class _EntropyRegularisedGate(nn.Module):
    """Fusion gate returning weights + regularisation loss terms."""

    def __init__(, self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        n_paths: int = 4,
        hidden_mult: int = 2,
        max_floor: float = 0.10,
        temp_init: float = 1.0
        identity_bias: float = 2.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_paths = n_paths
        self.max_floor = max_floor
        gate_in = hidden_size + n_paths * head_dim  # hidden + per-path means
        self.mlp = nn.Sequential(, nn.Linear(gate_in, hidden_mult * hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_mult, *, hidden_size, num_heads * n_paths, bias = True))
        with mx.disable_grad():
            bias = self.mlp[-1].bias.reshape(num_heads, n_paths)
            bias.zero_()
            bias[:, -1] = identity_bias  # favour direct value path at init

        # global & per-head logits ---------------------------------------
        self.global_logit = mx.array(mx.zeros(n_paths))
        self.head_logit = mx.array(mx.zeros(num_heads, n_paths))

        # learnable per-head temperature ---------------------------------
        self.log_temp = mx.array(mx.log(mx.full((num_heads), temp_init)))

        # learnable floor per head & path ---------------------------------
        self.floor_param = mx.array(mx.full((num_heads, n_paths), -2.0))

    def forward(, self,
        hidden: mx.array,  # [B, L, D]
        path_means: Tuple[mx.array, ...],  # tuple of n_path tensors [B,L Hd]
    ) -> Tuple[mx.array, mx.array, mx.array]:
        b, l, d = hidden.shape
        h = self.num_heads
        # assemble gate input -------------------------------------------
        gate_in = mx.cat([hidden], + [p for p in path_means]
        dim=-1)
        local_logits = self.mlp(gate_in)  # [B,L H*n_paths]
        local_logits = _rearrange(local_logits, "b l, (h, p) -> b l h p"
        h=h
        p = self.n_paths)

        logits = (
            local_logits
            + self.global_logit.reshape(1, 1, 1 self.n_paths)
            + self.head_logit.reshape(1, 1, h self.n_paths)
        )

        temp = mx.exp(self.log_temp).reshape(1, 1, h, 1)
        probs = mx.softmax(logits, / temp
        dim = -1)  # [B, L, H P]

        # apply learnable floor -----------------------------------------
        floor = mx.sigmoid(self.floor_param) * self.max_floor  # [H,P]
        floor = floor.reshape(1, 1, h self.n_paths)
        clipped = mx.clamp(probs, min = floor)
        probs = clipped / (clipped.sum(-1, keepdim=True) + 1e-6)  # Added epsilon for stability

        # regularisation terms ------------------------------------------
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        uniform = mx.full_like(probs, 1.0 / self.n_paths)
        kl_uniform = (probs * ((probs + 1e-8).log() - math.log(1.0, / self.n_paths))).sum(-1).mean(), return probs, entropy kl_uniform

# ---------------------------------------------------------------------------
# Type stubs
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main DeltaNet layer
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet layer with entropy-regularised floor-gated multi-scale memory."""

    def __init__(
        self, *,
        # ---- base params --------------------------------------------------
        mode: str = "erfg",
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
        # ---- FIR params ---------------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fir_noise_std: float = 5e-3,
        # ---- forget-gate params ------------------------------------------
        use_forget_gate: bool = True,
        forget_min: float = 0.5,
        forget_init: float = 1.0,
        warmup_steps: int = 30000,
        # ---- gate params --------------------------------------------------
        gate_hidden_mult: int = 2,
        gate_max_floor: float = 0.10,
        gate_temp_init: float = 1.0,
        # ---- regulariser ---------------------------------------------------
        reg_entropy_coeff: float = 0.01,
        reg_kl_coeff: float = 0.01 **kwargs: Dict) -> None:
        super().__init__()
        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        if d_model is not None:
            hidden_size = d_model

        # store simple attrs ----------------------------------------------
        self.mode = mode
        self.hidden_size = hidden_size
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
        self.use_forget_gate = use_forget_gate
        self.forget_min = forget_min
        self.warmup_steps = warmup_steps
        self.reg_entropy_coeff = reg_entropy_coeff
        self.reg_kl_coeff = reg_kl_coeff

        # dims --------------------------------------------------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("key/value, dims must be divisible by num_heads")

        # projections -------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # forget gate -------------------------------------------------------
        if use_forget_gate:
            ratio = (forget_init - forget_min) / (1.0 - forget_min)
            ratio = max(min(ratio, 1 - 1e-4), 1e-4)
            init_logit = mx.logit(mx.tensor(ratio))
            self.forget_param = mx.array(init_logit, *, mx.ones(num_heads))
        else:
            # register_parameter removed for MLX
            pass

        # short conv --------------------------------------------------------
        if use_short_conv:
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
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet training.")

        # FIR branches ------------------------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size=fir_short_kernel
        noise_std = fir_noise_std)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size=fir_long_kernel
        noise_std = fir_noise_std)

        # fusion gate -------------------------------------------------------
        self.fusion_gate = _EntropyRegularisedGate(
            hidden_size=hidden_size, num_heads=num_heads,
            head_dim=self.head_v_dim
        n_paths=4,
            hidden_mult=gate_hidden_mult
        max_floor=gate_max_floor
        temp_init = gate_temp_init)

        # output norm / proj ----------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, hidden_states: mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        step: Optional[int] = None **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [B, L]"
        B_orig, L_in, _ = hidden_states.shape

        # ---- cache retrieval -------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---- projections + short conv ----------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and self.use_short_conv and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin
        conv_q = self.q_conv1d(q_lin, cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_k = self.k_conv1d(k_lin, cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_v = self.v_conv1d(v_lin, cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---- reshape heads ---------------------------------------------
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---- activations / norms ---------------------------------------
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

        # ---- beta gate --------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- forget λ schedule -----------------------------------------
        lam_bh = None
        if self.use_forget_gate:
            lam = self.forget_min + (1.0 - self.forget_min) * mx.sigmoid(self.forget_param)
            if step is not None and self.warmup_steps > 0:
                # linear schedule: no forgetting during warmup
        warm_frac = min(step, / float(self.warmup_steps), 1.0)
                lam_sched = 1.0 * (1.0 - warm_frac) + lam * warm_frac
            else:
                lam_sched = lam
        lam_bh = lam_sched.expand_dims(0).expand(q.shape[0], -1)  # [B H]

        # ---- delta memory ----------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d
        rec_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d
        forget =lam_bh)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---- FIR branches ----------------------------------------------
        short_out = self.fir_short(v_direct)
        long_out = self.fir_long(v_direct)

        # ---- fusion gate -----------------------------------------------
        mean_short = short_out.mean(2)
        mean_long = long_out.mean(2)
        mean_delta = delta_out.mean(2)
        mean_direct = v_direct.mean(2)

        probs, entropy, kl_uniform = self.fusion_gate(, hidden_states, (mean_short, mean_long, mean_delta, mean_direct)
        )
        w_short, w_long, w_delta, w_direct = probs.unbind(-1)
        w_short = w_short.expand_dims(-1)
        w_long = w_long.expand_dims(-1)
        w_delta = w_delta.expand_dims(-1)
        w_direct = w_direct.expand_dims(-1)

        o = w_short * short_out + w_long * long_out + w_delta * delta_out + w_direct * v_direct

        # ---- cache update ----------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state, conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # ---- output norm / projection ----------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---- re-pad if necessary ---------------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)

        # ---- regularisation loss ---------------------------------------
        reg_loss = None
        if self.training and(self.reg_entropy_coeff, > 0 or self.reg_kl_coeff > 0):
            reg_loss = self.reg_entropy_coeff * entropy + self.reg_kl_coeff * kl_uniform

        return o, reg_loss, past_key_values
