"""
MLX-converted architecture: delta_net_aegf_br
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
DeltaNet – Annealed Entropic Gated Fusion with Balanced Residual Injection (AEGF-BR)
===================================================================================
Identifier: delta_net_aegf_br

Motivation (brief):
-------------------
This evolution merges the strengths of *CAGF-BR* (stable residual variance
handling) with the superior gating strategy of *AEKF* (annealed entropy / KL
regularisation, decaying probability floor and per-head temperature).  The new
fusion gate maintains early training exploration – guaranteeing gradient flow
through ALL memory paths – while still allowing late-stage specialisation that
benefits global reasoning tasks.  At the same time, the proven **Balanced
Residual Conv Injection** is preserved to stabilise variance without harming
local detail.

Key features enabled **by default**
----------------------------------
1. Annealed Entropy-KL gate regularisation with decaying ε-floor.
2. Per-head learnable temperature controlling gate sharpness.
3. Balanced residual injection tied to the suppression of the short-conv path.
4. Strict O(N) complexity, causal chunking, batch-size agnostic operations.

All public interfaces, forward-signature and configurability remain unchanged –
this class is a drop-in replacement for previous `DeltaNet` layers.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _elu_plus_one(x: mx.Tensor) -> mx.Tensor:
    """Shifted ELU so output is strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.Tensor) -> mx.Tensor:
    """Normalise last dim so values sum to one."""
    return (x / x.sum(-1, keepdim=True))

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution – unchanged
# ---------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # (H, D, K)
        self.filters = mx.array(mx.randn(num_heads, head_dim, self.kernel_size) * 0.02)

    def forward(self, x: mx.Tensor) -> mx.Tensor:  # (B, L, H, D)
        b, l, h, d = x.shape
        w = _rearrange(self.filters, "h d k -> (h d) 1 k")  # (H*D,1,K)
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left padding
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

# ---------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged)
# ---------------------------------------------------------------------------
@mx.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: mx.Tensor,
    k: mx.Tensor,
    v: mx.Tensor,
    beta: mx.Tensor,
    *,
    chunk_size: int = 32,
):
    """Efficient chunk-wise associative Δ-rule with O(N) cost."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)  # pad length dimension
        q = mx.pad(q, pad)
        k = mx.pad(k, pad)
        v = mx.pad(v, pad)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into (chunks, chunk_size)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    tri_strict = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1)

    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()).sum(-2)
    attn_inv = attn_inv + mx.eye(chunk_size, dtype=attn_inv.dtype, q.device)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = mx.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = _rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# ---------------------------------------------------------------------------
# Annealed fusion gate implementation
# ---------------------------------------------------------------------------
class _AnnealedFusionGate(nn.Module):
    """Content-aware fusion gate with annealed entropy/KL regularisation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        stat_dim: int,
        n_paths: int = 4,
        fusion_hidden_mult: int = 2,
        # Annealing / regularisation ---------------------------------
        floor_start: float = 0.05,
        floor_end: float = 0.005,
        entropy_weight: float = 0.02,
        kl_weight: float = 0.02,
        anneal_steps: int = 10_000,
        # Bias & temperature inits -----------------------------------
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        temp_init: float = 0.7,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = n_paths
        self.stat_dim = stat_dim
        self.hidden_size = hidden_size
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.entropy_weight = float(entropy_weight)
        self.kl_weight = float(kl_weight)
        self.anneal_steps = int(anneal_steps)

        # Per-head temperature (softplus-param)
        self.log_temp = mx.array(mx.full((num_heads,), math.log(math.expm1(temp_init))))

        # Base bias per head / path – helps steer early routing
        self.base_bias = mx.array(mx.tensor(gate_bias_init).repeat(num_heads, 1))  # (H, P)

        # MLP ----------------------------------------------------------------
        gate_in_dim = hidden_size + stat_dim  # per-head dimensions are handled later
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, n_paths, bias=True),
        )

        # Step buffer --------------------------------------------------------
        self, persistent=False)

        # Exposed losses for trainer ----------------------------------------
        self.last_gate_loss: Optional[mx.Tensor] = None

    # ------------------------------------------------------------------
    def _current_alpha(self) -> float:
        """Linear annealing factor α ∈ [1, 0]."""
        step = float(self._step.item())
        if step >= self.anneal_steps:
            return 0.0
        return 1.0 - step / self.anneal_steps

    # ------------------------------------------------------------------
    def forward(self, hidden_exp: mx.Tensor, stats: mx.Tensor) -> mx.Tensor:
        """Compute fusion weights.

        Args:
            hidden_exp: (B, L, H, D) – hidden states broadcasted per head.
            stats:      (B, L, H, stat_dim)
        Returns:
            fusion_weights: (B, L, H, n_paths)
        """
        B, L, H, D = hidden_exp.shape  # type: ignore[unpacking]
        # Prepare input ----------------------------------------------------
        gate_in = mx.cat([hidden_exp, stats], dim=-1)  # (B,L,H,D+stat_dim)
        gate_in_flat = _rearrange(gate_in, "b l h d -> (b l h) d")
        logits_flat = self.mlp(gate_in_flat)  # (B*L*H, P)
        logits = _rearrange(logits_flat, "(b l h) p -> b l h p"b=B, l=L, h=H)
        logits = logits + self.base_bias.reshape(1, 1, H, self.n_paths)

        # Temperature scaling --------------------------------------------
        temp = F.softplus(self.log_temp) + 1e-4  # (H,)
        logits = logits / temp.reshape(1, 1, H, 1)

        # Softmax ---------------------------------------------------------
        p = mx.softmax(logits, dim=-1)

        # ε-floor with linear decay --------------------------------------
        alpha = self._current_alpha()
        eps = self.floor_end + alpha * (self.floor_start - self.floor_end)
        if eps > 0.0:
            floor_vec = mx.tensor([eps, eps, 0.0, 0.0], dtype=p.dtype, p.device)
            p = mx.clamp(p, min=floor_vec)
            p = p / p.sum(dim=-1, keepdim=True)

        # Regularisation losses -----------------------------------------
        if self.entropy_weight > 0.0 or self.kl_weight > 0.0:
            entropy = -(p * (p + 1e-8).log()).sum(-1).mean()
            uniform = 1.0 / self.n_paths
            kl = (p * ((p + 1e-8).log() - math.log(uniform))).sum(-1).mean()
            ent_w = self.entropy_weight * alpha
            kl_w = self.kl_weight * alpha
            self.last_gate_loss = ent_w * entropy + kl_w * kl
        else:
            self.last_gate_loss = None

        # Step ++ ---------------------------------------------------------
        self._step += 1  # type: ignore[assignment]
        return p

# ---------------------------------------------------------------------------
# Typing helper – only for mypy / static checkers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main DeltaNet implementation
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet layer with **Annealed Entropic Gated Fusion & Balanced Residual** (AEGF-BR)."""

    def __init__(
        self,
        # ---- Legacy / common kwargs -----------------------------------
        mode: str = "aegf_br",
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
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # ---- FIR kernel sizes ----------------------------------------
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        # ---- Gate hyper-params --------------------------------------
        fusion_floor_start: float = 0.05,
        fusion_floor_end: float = 0.005,
        fusion_entropy_weight: float = 0.02,
        fusion_kl_weight: float = 0.02,
        anneal_steps: int = 10_000,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        temp_init: float = 0.7,
        # ---- Residual scaling ---------------------------------------
        conv_residual_init: float = -2.0,  # logit ⇒ σ ≈ 0.12
        **kwargs,
    ) -> None:
        super().__init__()

        assert qk_activation in ("silu", "relu", "elu", "identity")
        assert qk_norm in ("l2", "sum")

        # Book-keeping ----------------------------------------------------
        if d_model is not None:
            hidden_size = d_model  # alias
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # Dimensions ------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # Linear projections ---------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Beta projection -------------------------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Optional short conv enhancements -------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = _ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = _ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("_ShortConvolution is mandatory for DeltaNet stability.")

        # FIR convolutions -------------------------------------------------
        self.local_fir_long = _DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_long
        )
        self.local_fir_short = _DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_short
        )

        # Gating network (annealed entropy / KL) --------------------------
        self.stat_dim = 16  # 4 paths × 4 stats each
        self.fusion_gate = _AnnealedFusionGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            stat_dim=self.stat_dim,
            fusion_hidden_mult=fusion_hidden_mult,
            floor_start=fusion_floor_start,
            floor_end=fusion_floor_end,
            entropy_weight=fusion_entropy_weight,
            kl_weight=fusion_kl_weight,
            anneal_steps=anneal_steps,
            gate_bias_init=gate_bias_init,
            temp_init=temp_init,
        )

        # Residual conv scaling γ_h (per head) ----------------------------
        self.conv_residual_logit = mx.array(mx.full((num_heads,), conv_residual_init))

        # Output RMSNorm / projection ------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Exposed gate loss ----------------------------------------------
        self.last_gate_loss: Optional[mx.Tensor] = None

    # ------------------------------------------------------------------
    # Per-head statistics helper
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: mx.Tensor) -> mx.Tensor:  # (B,L,H,D) → (B,L,H,4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: mx.Tensor,
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compat
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"

        batch_size, seq_len_full, _ = hidden_states.shape

        # Retrieve cache --------------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # Optional unpadding ---------------------------------------------
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).expand_dims(0)

        # Q/K/V projections + optional short conv ------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q_in = self.q_proj(hidden_states)
        k_in = self.k_proj(hidden_states)
        v_in = self.v_proj(hidden_states)

        q_in, conv_state_q = self.q_conv1d(q_in, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_in, conv_state_k = self.k_conv1d(k_in, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_in, conv_state_v = self.v_conv1d(v_in, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # Head reshape ----------------------------------------------------
        q = _rearrange(q_in, "b l (h d) -> b l h d"d=self.head_k_dim)
        k = _rearrange(k_in, "b l (h d) -> b l h d"d=self.head_k_dim)
        v_direct = _rearrange(v_in, "b l (h d) -> b l h d"d=self.head_v_dim)

        # Activation / normalisation on Q/K ------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # Beta for Δ-rule -------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        beta = mx.clamp(beta, min=1e-6)

        # Global Δ-rule pathway ------------------------------------------
        delta_out_t, recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q, "b l h d -> b h l d"),
            k=_rearrange(k, "b l h d -> b h l d"),
            v=_rearrange(v_direct, "b l h d -> b h l d"),
            beta=_rearrange(beta, "b l h -> b h l"),
        )
        delta_out = _rearrange(delta_out_t, "b h l d -> b l h d")

        # Local FIR paths --------------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # Build gating input ---------------------------------------------
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = mx.cat([stats_short, stats_long, stats_delta, stats_value], dim=-1)  # (B,L,H,16)

        # Hidden expanded per head ---------------------------------------
        hs_exp = hidden_states.expand_dims(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,D)

        # Fusion weights via annealed gate -------------------------------
        fusion_weights = self.fusion_gate(hs_exp, stats_vec)
        self.last_gate_loss = self.fusion_gate.last_gate_loss

        # Weighted fusion -------------------------------------------------
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # Balanced residual conv injection ------------------------------
        static_gamma = mx.sigmoid(self.conv_residual_logit)  # (H,)
        static_gamma = static_gamma[None, None, :, None]  # (1,1,H,1)
        residual_scale = static_gamma * (1.0 - fusion_weights[..., 0:1])  # (B,L,H,1)
        o = o + residual_scale * local_short

        # Cache update ---------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
            )

        # Output norm / projection --------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d"d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # Re-pad if we previously un-padded -----------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, seq_len_full)

        return o, None, past_key_values
