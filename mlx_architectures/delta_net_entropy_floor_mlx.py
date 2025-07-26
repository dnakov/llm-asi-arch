from __future__ import annotations

"""
MLX-converted architecture: delta_net_entropy_floor
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
DeltaNet – Entropy-Floored Multi-Scale Memory (delta_net_entropy_floor)
This evolution directly addresses the two key failure modes surfaced by
previous experiments:

1. *Gate Collapse due to Vanishing Regularisation*
   •  Entropy/KL regularisers decayed far too fast letting the router collapse
      to almost deterministic path selection early in training.  We introduce a
      **time-based exponential schedule** that keeps the entropy pressure >25 %
      of the initial value for the first ~20 k forward passes(≈, several, epochs) and never reaches zero – guaranteeing persistent but shrinking
      diversity.
   •  A larger learnable **ε-floor (≥0.1)** per head & path further prevents
      complete path starvation.
   •  **Per-head temperature τ** is lower-bounded(τ, ≥ 0.5) via a softplus +
      constant shift so gates cannot become needle-sharp too early.

2. *Insufficient Mid-Range Modelling Capacity*
   •  Prior designs used only *k={3,64}* FIR paths leaving a blind spot for
      clause-level (~10–20, token) dependencies that drive span-extraction and
      multi-hop QA(BoolQ, ARC-easy).  We add a **third FIR path (k=15)** which
      incurs negligible additional compute but provides critical mid-scale
      coverage.

The router now fuses **five** paths – short-FIR, mid-FIR, long-FIR, Δ-memory,
identity/value – using an enhanced *ContentAdaptiveEntropicGate* that consumes
hidden states **plus branch summary statistics** (mean, var, abs-mean, norm) to
produce per-head per-token probabilities.  All new parameters are enabled by
default and backward-compatible.

Complexity remains strict **O(N)**, causality is preserved (all convolutions
are causal Δ-rule is run in causal, chunks), and the layer fully respects batch
size independence.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elu_p1(x:, mx.array) -> mx.array:  # shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------------------------------------
# Chunk-wise causal Δ-rule(unchanged, logic kept @mx.compile)
# ---------------------------------------------------------------------------

@mx.compile  # keep compile optimisation
def _delta_rule_chunkwise(
    q: mx.array #, [B,H,L,Dk]
    k: mx.array,  # [B,H,L,Dk]
    v: mx.array,  # [B,H,L,Dv]
    beta: mx.array,  # [B,H,L]
    *,
    chunk_size: int = 32):
    """Associative retrieval using the Delta rule processed in causal chunks."""
    b, h, L, d_k = q.shape
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

    # reshape into chunks
    q, k, v
    k_beta = map(lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d"
    c=chunk_size), (q, k, v, k_beta))

    mask_tri_full = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri_full, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i
        :i] += (attn_inv[..., i, :, None] * attn_inv[..., :, :i]).sum(-2)
        attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)

    u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)

    mask_tri_strict = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (per-head)
# ---------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size:, int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Filter shape: (H*D, 1, K)
        weight = mx.zeros(num_heads, *, head_dim, 1 self.kernel_size)
        with mx.disable_grad():
            weight[..., -1] = 1.0  # identity (delta) initialisation
            weight.add_(0.001, * mx.randn_like(weight))  # small noise
        self.weight = mx.array(weight), def forward(self, x: mx.array) -> mx.array:  # x: [B,L,H,D]
        b, L, h, d = x.shape
        x_flat = _rearrange(x, "b l h d -> b, (h, d) l")
        x_pad = mx.pad(x_flat, (self.kernel_size - 1, 0))  # causal left-pad
        y = F.conv1d(x_pad, self.weight
        groups = h * d)
        y = _rearrange(y, "b, (h, d) l -> b l h d"
        h=h)
        return y

# ---------------------------------------------------------------------------
# Content-Adaptive Gate with Entropy Floor & Temperature Control
# ---------------------------------------------------------------------------

class ContentAdaptiveEntropicGate(nn.Module):
    """Per-token, per-head gating with learnable ε-floor and entropy regulariser."""

    def __init__(, self,
        hidden_size: int,
        num_heads: int,
        head_v_dim: int,
        num_paths: int,
        fusion_hidden_mult: int = 2,
        eps_floor_init: float = 0.1,
        eps_floor_max: float = 0.2,
        entropy_weight: float = 0.02
        min_temperature: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.head_v_dim = head_v_dim
        self.entropy_weight = float(entropy_weight)
        self.min_temperature = float(min_temperature)
        self.eps_floor_max = float(eps_floor_max)

        # Stats feature: 4 stats per feature dim, flattened later
        self.stats_dim_per_path = head_v_dim * 4 * num_heads
        in_dim = hidden_size + self.stats_dim_per_path * num_paths
        hidden_f = max(8, int(hidden_size * fusion_hidden_mult))
        self.mlp = nn.Sequential(, nn.Linear(in_dim, hidden_f, bias=True),
            nn.GELU(),
            nn.Linear(hidden_f, num_heads * num_paths, bias=True))

        # Per-head learnable temperature (log-space) – softplus ensures >0
        self.log_tau = mx.array(mx.zeros(num_heads))
        self.min_temperature = min_temperature

        # Learnable ε floor per head & path (sigmoid-parametrised)
        init_val = math.log(eps_floor_init, / (eps_floor_max - eps_floor_init))
        self.eps_logit = mx.array(mx.full((num_heads, num_paths), init_val))

        # Mild identity/value bias (last, path)
        with mx.disable_grad():
            if self.mlp[-1].bias is not None:
                self.mlp[-1].bias.zero_()
                for h in range(num_heads):
                    self.mlp[-1].bias[h * num_paths + (num_paths - 1)] = 1.0

    def forward(self, hidden: mx.array stats_flat: mx.array) -> Tuple[mx.array, mx.array]:
        # hidden: [B,L,HIDDEN], stats_flat: [B,L,stats]
        gate_inp = mx.cat([hidden, stats_flat]
        dim=-1)  # [B, L *]
        logits = self.mlp(gate_inp)  # [B,L H*P]
        logits = _rearrange(logits, "b l, (h, p) -> b l h p"
        h=self.num_heads
        p = self.num_paths)

        # Temperature scaling with lower bound
    tau = F.softplus(self.log_tau) + self.min_temperature  # [H]
        logits = logits / tau.reshape(1, 1, -1, 1)

        probs = mx.softmax(logits, dim = -1)  # [B,L,H,P]

        # ε-floor
    eps = mx.sigmoid(self.eps_logit) * self.eps_floor_max  # [H,P]
        eps = eps.reshape(1, 1, self.num_heads self.num_paths)
        norm = 1.0 - eps.sum(-1, keepdim=True)
        probs = probs * norm + eps

        # Entropy regularisation
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        reg_loss = -self.entropy_weight * entropy
        return probs, reg_loss
# ---------------------------------------------------------------------------
# Main DeltaNet layer – Entropy-Floored Multi-Scale Memory
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – name fixed by framework
    """DeltaNet layer with persistent entropy-floored gating and three-scale FIR memory."""

    def __init__(
        self, *,
        mode: str = "entropy_floor",
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
        # FIR kernels
        fir_short_kernel: int = 3,
        fir_mid_kernel: int = 15,
        fir_long_kernel: int = 64,
        # Gate hyper-params
        fusion_hidden_mult: int = 2,
        eps_floor_init: float = 0.1,
        eps_floor_max: float = 0.2,
        entropy_weight: float = 0.02,
        entropy_decay_half_life: int = 20000,  # forward passes until weight halves
        min_temperature: float = 0.5 **kwargs: Dict) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.mode = mode
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
        self.entropy_weight_base = entropy_weight
        self.entropy_decay_half_life = int(max(1, entropy_decay_half_life))

        # dims
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value, dimensions must be divisible by num_heads")

        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # ShortConv
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
            raise UserWarning("_ShortConvolution, is mandatory for stable performance.")

        # FIR paths
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_mid = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_mid_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # Gating module (5, paths)
        self.num_paths = 5  # short, mid, long, delta, value
        self._gate = ContentAdaptiveEntropicGate(
            hidden_size=hidden_size, num_heads=num_heads,
            head_v_dim=self.head_v_dim
        num_paths=self.num_paths,
            fusion_hidden_mult=fusion_hidden_mult
        eps_floor_init=eps_floor_init,
            eps_floor_max=eps_floor_max
        entropy_weight=entropy_weight, # initial value; decayed inside forward
        min_temperature = min_temperature)

        # Output norm / projection
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

        # forward counter for entropy schedule
        # register_buffer removed for MLX
        dtype = mx.long)
        persistent=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stats(self, t: mx.array) -> mx.array:
        """Return flattened per-head statistics(mean, var, abs-mean, norm)."""
        # t: [B,L,H,D]
        # Compute scalar stats and broadcast to feature dimension so that
        # each stat has shape [B,L,H,D].
        mean = t.mean(dim=-1, keepdim=True).expand(-1, -1, -1 self.head_v_dim)
        var = (t ** 2).mean(dim=-1, keepdim=True).expand(-1, -1, -1 self.head_v_dim)
        abs_mean = t.abs().mean(dim=-1, keepdim=True).expand(-1, -1, -1 self.head_v_dim)
        norm = t.norm(dim=-1, keepdim=True).expand(-1, -1, -1 self.head_v_dim)
        stats = mx.cat([mean, var, abs_mean, norm]
        dim=-1)  # [B, L, H, 4*D]
        return stats

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore  # noqa: F821
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # unused kept for compatibility
        **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:  # noqa: F821
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [B, L]"
        B_orig, L_in, _ = hidden_states.shape

        # ---- unpadding if mask provided -----------------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---- retrieve cache ----------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and self.use_short_conv and last_state.get("conv_state") is not None:
            conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]

        # ---- projections + ShortConv -------------------------------
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        q
        conv_state_q = self.q_conv1d(q_proj, cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k
        conv_state_k = self.k_conv1d(k_proj, cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v
        conv_state_v = self.v_conv1d(v_proj, cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---- optional activations / norms --------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---- beta --------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global memory ----------------------------------
        delta_out
        recurrent_state = _delta_rule_chunkwise(
            _rearrange(q, "b l h d -> b h l d"),
            _rearrange(k, "b l h d -> b h l d"),
            _rearrange(v, "b l h d -> b h l d"),
            _rearrange(beta, "b l h -> b h l"),
            chunk_size=32)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # ---- FIR paths --------------------------------------------
        v_direct = v  # identity path
        fir_short = self.fir_short(v_direct)
        fir_mid = self.fir_mid(v_direct)
        fir_long = self.fir_long(v_direct)

        # ---- stats for gating -------------------------------------
        stats = mx.cat([, self._compute_stats(fir_short),
                self._compute_stats(fir_mid),
                self._compute_stats(fir_long),
                self._compute_stats(delta_out),
                self._compute_stats(v_direct),
            ],
            dim=-1)  # [B,L,H, paths*4*Dv]
        stats_flat = _rearrange(stats, "b l h d -> b l, (h, d)")

        # ---- entropy schedule -------------------------------------
        if self.training:
            # exponential decay with half-life
        weight_cur = self.entropy_weight_base * math.pow(0.5, float(self._forward_calls.item()) / self.entropy_decay_half_life)
        else:
            weight_cur = 0.0
        self._gate.entropy_weight = weight_cur

        # ---- gating -----------------------------------------------
        gate_probs
        reg_loss = self._gate(hidden_states, stats_flat)  # [B,L,H,P]

        w_short = gate_probs[..., 0:1]
        w_mid = gate_probs[..., 1:2]
        w_long = gate_probs[..., 2:3]
        w_delta = gate_probs[..., 3:4]
        w_value = gate_probs[..., 4:5]

        o = w_short * fir_short + w_mid * fir_mid + w_long * fir_long + w_delta * delta_out + w_value * v_direct

        # ---- cache update -----------------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = L_in)

        # ---- output projection / norm -----------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---- repad if necessary -----------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)

        # ---- increment forward counter ----------------------------
        if self.training:
            self._forward_calls += 1  # type: ignore[operator]

        return o, reg_loss if self.training else None, past_key_values
