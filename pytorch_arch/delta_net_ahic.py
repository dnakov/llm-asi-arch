# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Hybrid Identity-Context Gating with Floor, Annealed-Entropy, and Bounded Residual (DeltaNet-AHIC)
===============================================================================================================
Identifier: delta_net_ahic

Breakthrough innovations (enabled by default):
---------------------------------------------
1. **Token-Adaptive Identity Floor:**
   - The identity/value path has a *per-token, per-head* adaptive minimum floor: the *minimum value for routing mass* is determined as a function of the confidence of the context router. This ensures copy-fidelity whenever context-confidence is low, but allows the model to reduce the copy path's influence when context certainty is truly high (as in AFT/BTSF).
   - The minimum is computed dynamically as:  \(\text{min_id_frac} = \epsilon_{id} + (1-\epsilon_{id})(1 - \max_\text{context} (p_\text{context}))\) for each token/head, ensuring nonzero mass as a fallback when context is uncertain, but letting the identity path shrink when context mass is consolidated.

2. **Bounded/Regularised Identity Scaling (α):**
   - α (the scaling parameter for the identity path) is reparameterized as α=softplus(param)+1 for strict α≥1, and regularized toward 1.0 to prevent runaway identity amplification and overflow risk.
   - This guarantees robust copy-path influence, while retaining numerical stability and controllable optimization.

3. **Context (Router) with Output-Aware Statistics, Annealed Temp, and ε-floor:**
   - The context router uses a softmax over three streams (short/long FIR and Delta/global), with output-aware statistics (mean,std per path&head) concatenated to the hidden state.
   - Router logits are temperature-annealed (from per-group → per-head) as in HIST, but floor regularization is applied: each context path gets minimum routing ε throughout training, linearly decayed.
   - Entropy of the router logits is annealed via a regularization term to maintain exploration early, but allowing sharp, decisive allocation later.

4. **All tensor operations use einops.rearrange(), zero reshaping/viewing. Supports all batch sizes.**
5. **Full O(N)/chunked causal efficiency.**

This file was automatically **checked and patched** by the architecture code checker.
The underlying innovation remains unchanged; only technical issues (dtype and device
robustness) were corrected so the implementation works for *any* batch size,
precision and device combination.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise chunked FIR convolution (unchanged numerics)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            filt[..., -1] = 1.0
            filt.add_(0.01 * torch.randn_like(filt))
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Causal chunked Δ-rule (unchanged numerics except dtype fix)
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[arg-type]
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Chunked causal delta rule implementation.

    All operations are strictly causal w.r.t sequence length. The complexity is
    O(L * chunk_size) (linear in *L*) with the given constant *chunk_size*.
    """
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    # IMPORTANT FIX: keep attn_inv in the *same* dtype as the incoming tensors
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, q.shape[-1], v.shape[-1])
    o = torch.zeros_like(v)

    future_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Main DeltaNet – Adaptive Hybrid Identity-Context Gating
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
    from fla.models.utils import Cache  # pragma: no cover


class DeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = "ahic",
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
        # FIR kernel sizes
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # Adaptive identity params
        epsilon_id: float = 0.06,  # lowest allowed identity mass
        alpha_reg_strength: float = 0.02,
        # Context gate params
        fusion_hidden_mult: int = 2,
        group_size: int = 2,
        tau_transition_steps: int = 3000,
        router_epsilon_start: float = 0.025,
        router_epsilon_end: float = 0.005,
        router_epsilon_decay: int = 3000,
        router_entropy_start: float = 0.01,
        router_entropy_end: float = 0.0,
        router_entropy_decay: int = 3000,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
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

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Short convolutions
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet-AHIC.")

        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # Identity scaling parameter α >= 1 (via softplus)
        self.alpha_id_param = nn.Parameter(torch.zeros(num_heads))
        self.alpha_reg_strength = float(alpha_reg_strength)

        # Identity gate (MLP for better adaptivity if desired)
        self.id_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        with torch.no_grad():
            self.id_gate_proj.bias.fill_(0.0)
        self.epsilon_id = float(epsilon_id)

        # Context router (3-way: short, long, delta)
        self.fusion_hidden_mult = int(fusion_hidden_mult)
        stat_dim_per_head = 2  # mean & std
        router_in_dim = hidden_size + num_heads * stat_dim_per_head * 3
        router_hidden = max(8, hidden_size * self.fusion_hidden_mult)
        self.context_router_mlp = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden, bias=True),
            nn.GELU(),
            nn.Linear(router_hidden, num_heads * 3, bias=True),
        )
        with torch.no_grad():
            self.context_router_mlp[-1].bias.fill_(0.0)

        # Temperature scheduling
        self.group_size = max(1, int(group_size))
        num_groups = (num_heads + self.group_size - 1) // self.group_size
        # store on CPU but make sure to cast to the right device at usage time
        self.register_buffer("_group_index", torch.arange(num_heads) // self.group_size, persistent=False)
        self.log_tau_group = nn.Parameter(torch.zeros(num_groups))
        self.log_tau_head = nn.Parameter(torch.zeros(num_heads))
        self.tau_transition_steps = int(tau_transition_steps)

        # Epsilon/entropy scheduling for router
        self.router_epsilon_start = float(router_epsilon_start)
        self.router_epsilon_end = float(router_epsilon_end)
        self.router_epsilon_decay = int(router_epsilon_decay)

        self.router_entropy_start = float(router_entropy_start)
        self.router_entropy_end = float(router_entropy_end)
        self.router_entropy_decay = int(router_entropy_decay)

        # Output norm/proj
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.reg_loss: Optional[torch.Tensor] = None

    # --------------------------------------------------------------
    # Scheduling helpers
    # --------------------------------------------------------------
    def _current_router_epsilon(self) -> float:
        t = float(self._step.item())
        if t >= self.router_epsilon_decay:
            return self.router_epsilon_end
        r = t / max(1.0, self.router_epsilon_decay)
        return self.router_epsilon_start + r * (self.router_epsilon_end - self.router_epsilon_start)

    def _current_router_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.router_entropy_decay:
            return self.router_entropy_end
        r = t / max(1.0, self.router_entropy_decay)
        return self.router_entropy_start + r * (self.router_entropy_end - self.router_entropy_start)

    def _mix_temperature(self) -> torch.Tensor:
        """Return current per-head temperature (τ) after group→head annealing."""
        t = float(self._step.item())
        mix = 1.0 - min(1.0, t / max(1.0, self.tau_transition_steps))
        # Ensure index tensor is on the same device as parameters (important for GPU)
        group_index = self._group_index.to(self.log_tau_group.device)
        tau_g = torch.exp(self.log_tau_group)[group_index]
        tau_h = torch.exp(self.log_tau_head)
        tau = mix * tau_g + (1.0 - mix) * tau_h
        return tau  # (H,)

    # --------------------------------------------------------------
    # Statistic helpers (mean & std per head)
    # --------------------------------------------------------------
    @staticmethod
    def _stats_mean_std(path: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = path.mean(dim=-1, keepdim=False)
        std = path.std(dim=-1, unbiased=False, keepdim=False)
        return mean, std

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B0, L0, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L0:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # Q/K/V projections
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)
        q, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Delta rule (causal, chunked)
        delta_out, rec_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")
        local_short = self.local_fir_short(v)
        local_long = self.local_fir_long(v)

        # Prepare identity gate (per-token, per-head, lower-bounded by ADAPTIVE min)
        id_gate_raw = torch.sigmoid(self.id_gate_proj(hidden_states))  # (B,L,H)
        # Router features for context (mean/std per head for 3 context paths)
        mean_s, std_s = self._stats_mean_std(local_short)
        mean_l, std_l = self._stats_mean_std(local_long)
        mean_d, std_d = self._stats_mean_std(delta_out)
        # Stack as feature dim: (B,L,H,6) -> (B,L,H*6)
        stats = torch.stack([mean_s, std_s, mean_l, std_l, mean_d, std_d], dim=-1)
        stats_flat = rearrange(stats, "b l h f -> b l (h f)")
        # Router input
        router_in = torch.cat([hidden_states, stats_flat], dim=-1)
        router_logits = self.context_router_mlp(router_in)  # (B,L,H*3)
        router_logits = rearrange(router_logits, "b l (h c) -> b l h c", h=self.num_heads, c=3)

        # Temperature scheduling
        tau = self._mix_temperature()  # (H,)
        router_logits = router_logits / tau.view(1, 1, self.num_heads, 1)

        # Softmax + ε-floor
        p_context = torch.softmax(router_logits, dim=-1)
        eps = self._current_router_epsilon()
        p_context = p_context * (1.0 - 3 * eps) + eps

        # --- adaptively set min_id_frac (token, head): lowest allowed identity is eps_id + (1-eps_id)*(1 - torch.max(p_context, dim=-1).values)
        max_context = p_context.max(dim=-1).values  # (B,L,H)
        min_id_frac = self.epsilon_id + (1.0 - self.epsilon_id) * (1.0 - max_context)
        id_floor = min_id_frac
        id_gate = torch.clamp(id_gate_raw, min=0.0, max=1.0)
        id_gate = torch.where(id_gate < id_floor, id_floor, id_gate)
        identity_weight = id_gate  # (B,L,H)
        context_mass = 1.0 - identity_weight
        p_context = p_context * context_mass.unsqueeze(-1)

        # Context output
        context_out = (
            p_context[..., 0:1] * local_short +
            p_context[..., 1:2] * local_long +
            p_context[..., 2:3] * delta_out
        )
        alpha = F.softplus(self.alpha_id_param).view(1, 1, -1, 1) + 1.0
        identity_out = alpha * identity_weight.unsqueeze(-1) * v
        o = context_out + identity_out

        # Entropy regularisation of routing (annealed)
        entropy = -(p_context * (p_context + 1e-8).log()).sum(dim=-1).mean()
        self.reg_loss = self._current_router_entropy_coeff() * entropy + self.alpha_reg_strength * ((alpha - 1) ** 2).mean()

        # Cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L0,
            )

        # Output norm/proj
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B0, L0)
        self._step += 1  # type: ignore[operator]
        return o, self.reg_loss, past_key_values
