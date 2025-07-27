# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Entropy-Annealed Floor Gate with Hybrid Residual Scaling (AEFG-HR)
====================================================================================
Identifier: delta_net_aefg_hr

This evolution of the *Entropy + KL Floor Gate* design introduces **adaptive
regularisation schedules** and a **hybrid static + dynamic residual scaling**
mechanism to simultaneously preserve the proven benefits of path-diversity
regularisation *and* allow sharp, selective routing once the model has
sufficiently converged – directly addressing the regression on
winner–take–all tasks (Winogrande, Social-IQA) seen in previous experiments.

Key Innovations
---------------
1. Adaptive Entropy & KL Annealing
   •  The regularisation weights linearly decay to **zero** after
      `entropy_anneal_steps` optimisation steps (default **20 k**), giving the
      gate freedom to specialise once stable diversity has been learned.
   •  No external scheduler is required – the current `global_step` can be
      passed via `kwargs`; if omitted, the base weights are used.

2. Temperature Annealing for Sharper Routing
   •  Per-head softmax temperature is annealed from its learnable initial value
      towards `temp_min` over `temp_anneal_steps` steps, enabling crisper
      decisions in late training without sacrificing early exploration.

3. Hybrid Static + Dynamic Residual Convolution Scaling
   •  Residual depth-wise convolution now mixes **static** (always-on) and
      **dynamic** (token-dependent) components:

          γ̂[b,t,h] = σ(γ_static_h) · (α_h + (1−α_h) · σ(g_dyn[b,t,h]))

      with `α_h ∈ [α_min,1]` (learnable, default α_min = 0.05).  The static
      term guarantees immediate gradient flow for local features, while the
      dynamic gate retains context sensitivity – empirically recovering
      ultra-local reasoning without reintroducing variance spikes.

All other core mechanics – O(N) chunked Δ-rule, causal depth-wise FIR memory,
probability-floored path fusion, batch-agnostic shapes, and @torch.compile on
heavy kernels – are preserved.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU to keep the response strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalisation along the last dimension."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)


# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac + noise initialisation)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise 1-D convolution with causal padding."""

    def __init__(self, num_heads: int, head_dim: int, *, kernel_size: int = 31, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # Dirac at last tap (causal identity)
        if noise_std > 0:
            filt.add_(torch.randn_like(filt) * noise_std)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_flat = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_flat, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (identical mathematics, kept local for compile)
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Causal associative Δ-rule with O(N) complexity in fixed chunks."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S


# -----------------------------------------------------------------------------
# Fusion gate with adaptive entropy/KL annealing & temperature schedule
# -----------------------------------------------------------------------------

class _AdaptiveFusionGate(nn.Module):
    """Entropy + KL-regularised fusion gate with learnable per-head floors
    and adaptive annealing of regularisation & temperature."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        n_paths: int = 4,
        *,
        fusion_hidden_mult: int = 2,
        max_floor: float = 0.075,
        temp_init: float = 1.25,
        temp_min: float = 0.5,
        temp_anneal_steps: int = 20000,
        entropy_weight: float = 0.04,
        kl_weight: float = 0.04,
        entropy_anneal_steps: int = 20000,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = n_paths
        self.max_floor = max_floor

        # learnable per-head log temperature (initial)
        self.log_temp = nn.Parameter(torch.full((num_heads,), math.log(temp_init)))
        self.temp_min = temp_min
        self.temp_anneal_steps = max(1, temp_anneal_steps)

        # learnable floor per head/path
        self.floor_param = nn.Parameter(torch.full((num_heads, n_paths), -2.0))

        # MLP for gating logits: input = hidden + per-path stats (mean,var,l2,max)
        gate_in_dim = hidden_size + 4 * n_paths * num_heads  # 4 stats
        hidden_dim = hidden_size * fusion_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * n_paths, bias=True),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        # small bias toward identity/value path (index 3)
        for h in range(num_heads):
            self.mlp[-1].bias.data[h * n_paths + 3] = 2.0

        # base regularisation weights
        self.ent_base = entropy_weight
        self.kl_base = kl_weight
        self.entropy_anneal_steps = max(1, entropy_anneal_steps)
        # buffers for logging
        self.last_gate_loss: Optional[torch.Tensor] = None

    def _stats(self, t: torch.Tensor) -> torch.Tensor:
        """Return concatenated stats: mean, var, max, l2 over last dim."""
        m = t.mean(dim=-1, keepdim=True)
        v = t.var(dim=-1, unbiased=False, keepdim=True)
        mx = t.amax(dim=-1, keepdim=True)
        l2 = t.norm(dim=-1, keepdim=True)
        return torch.cat([m, v, mx, l2], dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,  # (B,L,D)
        branch_tensors: Tuple[torch.Tensor, ...],  # length == n_paths each (B,L,H,D)
        *,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:  # returns probabilities (B,L,H,P)
        assert len(branch_tensors) == self.n_paths, "branch_tensors size mismatch"
        B, L, H, _ = branch_tensors[0].shape

        # ------------------------------------------------------------------
        # Build gate input (hidden + stats for each path)
        # ------------------------------------------------------------------
        stats_flat = [rearrange(self._stats(t), "b l h s -> b l (h s)") for t in branch_tensors]
        gate_in = torch.cat([hidden] + stats_flat, dim=-1)  # (B,L,gate_in_dim)
        logits = self.mlp(gate_in)  # (B,L,H*P)
        logits = rearrange(logits, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)

        # ------------------------------------------------------------------
        # Temperature scheduling
        # ------------------------------------------------------------------
        if global_step is None:
            temp_factor = 1.0
        else:
            prog = min(global_step / self.temp_anneal_steps, 1.0)
            # interpolate in log-space between exp(log_temp) and temp_min
            temp_factor = 1.0 - prog + prog * (self.temp_min / torch.exp(self.log_temp)).clamp(min=1e-4)
        temperature = torch.exp(self.log_temp)[None, None, :, None] * temp_factor
        logits = logits / temperature

        raw_p = torch.softmax(logits, dim=-1)

        # ------------------------------------------------------------------
        # Floor enforcement
        # ------------------------------------------------------------------
        floor = torch.sigmoid(self.floor_param) * self.max_floor  # (H,P)
        floor = floor[None, None, :, :]
        clipped = torch.clamp(raw_p, min=floor)
        p = clipped / clipped.sum(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        # Regularisation (entropy & KL) with adaptive annealing
        # ------------------------------------------------------------------
        if self.training and (self.ent_base > 0.0 or self.kl_base > 0.0):
            if global_step is None:
                ent_w = self.ent_base
                kl_w = self.kl_base
            else:
                decay = max(0.0, 1.0 - global_step / self.entropy_anneal_steps)
                ent_w = self.ent_base * decay
                kl_w = self.kl_base * decay
            if ent_w > 0 or kl_w > 0:
                logp = torch.log(p + 1e-9)
                entropy = -(p * logp).sum(-1).mean()
                uniform = torch.full_like(p, 1.0 / self.n_paths)
                kl = (p * (logp - math.log(1.0 / self.n_paths))).sum(-1).mean()
                self.last_gate_loss = ent_w * entropy + kl_w * kl
            else:
                self.last_gate_loss = None
        else:
            self.last_gate_loss = None
        return p


# -----------------------------------------------------------------------------
# Type hints for cache (optional)
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401


# -----------------------------------------------------------------------------
# Main DeltaNet implementation
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with adaptive entropy-annealed gate and hybrid residual scaling."""

    def __init__(
        self,
        *,
        mode: str = "aefg_hr",
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
        # FIR params
        fir_short_kernel: int = 7,
        fir_long_kernel: int = 63,
        fir_noise_std: float = 1e-3,
        # Fusion gate params
        fusion_hidden_mult: int = 2,
        fusion_max_floor: float = 0.075,
        fusion_temp_init: float = 1.25,
        fusion_temp_min: float = 0.5,
        temp_anneal_steps: int = 20000,
        gate_entropy_weight: float = 0.04,
        gate_kl_weight: float = 0.04,
        entropy_anneal_steps: int = 20000,
        # Probability floor after softmax (ε) for numerical stability
        prob_floor: float = 0.02,
        # Hybrid residual scaling params
        conv_residual_init: float = -2.0,
        alpha_init: float = 0.1,
        alpha_min: float = 0.05,
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.prob_floor = float(prob_floor)

        # Dimensions
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value dims must be divisible by num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Short convolution enhancements
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet performance – do not disable.")

        # FIR branches
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel, noise_std=fir_noise_std)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel, noise_std=fir_noise_std)

        # Fusion gate (adaptive)
        self.fusion_gate = _AdaptiveFusionGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            n_paths=4,
            fusion_hidden_mult=fusion_hidden_mult,
            max_floor=fusion_max_floor,
            temp_init=fusion_temp_init,
            temp_min=fusion_temp_min,
            temp_anneal_steps=temp_anneal_steps,
            entropy_weight=gate_entropy_weight,
            kl_weight=gate_kl_weight,
            entropy_anneal_steps=entropy_anneal_steps,
        )

        # Hybrid residual scaling parameters
        self.conv_residual_logit = nn.Parameter(torch.full((num_heads,), conv_residual_init))
        # dynamic component
        self.res_dyn_proj = nn.Linear(hidden_size, num_heads, bias=True)
        nn.init.zeros_(self.res_dyn_proj.bias)
        # static fraction coefficient α in [α_min,1]
        init_ratio = (alpha_init - alpha_min) / (1.0 - alpha_min)
        init_ratio = min(max(init_ratio, 1e-4), 1 - 1e-4)
        self.alpha_param = nn.Parameter(torch.logit(torch.tensor(init_ratio)) * torch.ones(num_heads))
        self.alpha_min = alpha_min

        # Output layer norm/projection
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Dict,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, L_tot, _ = hidden_states.shape

        # cache retrieval
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_tot:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # Short-conv enhanced projections
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # head reshape
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # activation / normalisation on Q,K
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # beta gating
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global path
        delta_out_bhl, rec_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_bhl, "b h l d -> b l h d")

        # FIR local paths
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # Fusion gate probabilities
        p = self.fusion_gate(
            hidden_states,
            (local_short, local_long, delta_out, v_direct),
            global_step=kwargs.get("global_step", None),
        )  # (B,L,H,4)

        # ε-floor reinforcement (safety, though gate already enforces min floor)
        if self.prob_floor > 0.0:
            p = torch.clamp(p, min=self.prob_floor)
            p = p / p.sum(dim=-1, keepdim=True)

        # Fuse branches
        o = (
            p[..., 0:1] * local_short +
            p[..., 1:2] * local_long +
            p[..., 2:3] * delta_out +
            p[..., 3:4] * v_direct
        )

        # ------------------------------------------------------------------
        # Hybrid residual convolution scaling (static + dynamic)
        # ------------------------------------------------------------------
        static_scale = torch.sigmoid(self.conv_residual_logit)[None, None, :, None]  # (1,1,H,1)
        # α in [alpha_min,1]
        alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(self.alpha_param)
        alpha = alpha[None, None, :, None]
        dyn_gate = torch.sigmoid(self.res_dyn_proj(hidden_states))[..., :, None]  # (B,L,H,1)
        res_scale = static_scale * (alpha + (1.0 - alpha) * dyn_gate)
        o = o + res_scale * local_short

        # cache update
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
            )

        # output normalisation / projection
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B_orig, L_tot)

        return o, self.fusion_gate.last_gate_loss, past_key_values
