# -*- coding: utf-8 -*-
"""
DeltaNet – Unified Dynamic Memory with Output-Aware Annealed Gating (DeltaNet-UDMAG)
==================================================================================
Identifier: delta_net_udmag

This evolution unifies the strongest mechanisms verified across the prior research
line while directly fixing the bottlenecks surfaced in the experimental portfolio:

1.  Per-head, per-token **dynamic decay (γ)** for the Δ-rule global memory – proven to
    boost selective passage retention and narrative tasks.
2.  **Output-aware fusion gate** that conditions routing on hidden-state features *and*
    real path statistics (mean/std).  This addresses the abstract relational
    reasoning deficit (SocialIQA / Winogrande) by giving the router visibility into
    its own outputs, as validated in the OAGATE studies.
3.  **Annealed ε-floor** – guarantees gradient flow early while permitting sharp,
    decisive routing later, eliminating the over-soft gating weakness identified in
    HMDG-v4 and APEX.
4.  **Correctly-signed entropy regulariser** – encourages path diversity during the
    initial phase, fixing the sign bug that harmed HellaSwag/SWDE in APEX.
5.  Always-on **identity residual path** outside any softmax competition, ensuring
    verbatim copying capacity for extraction tasks (Winogrande/SQuAD) without
    starving other branches – a lesson from AHAG.
6.  Strict **O(N)** complexity via chunk-wise processing, complete batch-size
    agnosticism, and universal use of einops.rearrange for shape manipulations.

All new features come with sensible defaults *enabled by default* and do **not**
modify the external interface.  The class name remains `DeltaNet`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict

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
    """Shifted ELU (+1) keeps activations strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so that elements along the last dimension sum to one."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule with per-token, per-head dynamic decay (γ)
# -----------------------------------------------------------------------------


@torch.compile  # type: ignore[misc]
def _delta_rule_chunkwise_gamma(
    q: torch.Tensor,  # [B, H, L, Dk]
    k: torch.Tensor,  # [B, H, L, Dk]
    v: torch.Tensor,  # [B, H, L, Dv]
    beta: torch.Tensor,  # [B, H, L]
    gamma: torch.Tensor,  # [B, H, L]  (forget factor 0–1)
    *,
    chunk_size: int = 32,
):
    """Associative retrieval with dynamic decay processed in causal chunks (O(N))."""
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
        gamma = F.pad(gamma, (0, pad_len))
    L_pad = L + pad_len

    # Unit-length normalisation on q/k for numerical stability
    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks -> (B H N C D)
    q, k, v, k_beta, gamma = map(
        lambda t: rearrange(t, "b h (n c) ... -> b h n c ...", c=chunk_size),
        (q, k, v, k_beta, gamma),
    )
    n_chunks = q.shape[2]

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones_like(tri), 1)

    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    # Woodbury identity recursion inside the chunk
    for i in range(1, chunk_size):
        # Following line was causing inplace modification error in autograd
        # attn_inv[..., i, :i] += (
        #     attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        # ).sum(-2)
        # Replace inplace addition with out-of-place operation to avoid interfering with autograd
        attn_inv_slice = (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
        attn_inv = attn_inv.clone()  # ensure previous refs are not reused
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + attn_inv_slice
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)

    u = attn_inv @ v  # (B,H,N,C,Dv)
    w = attn_inv @ k_beta  # (B,H,N,C,Dk)

    S = q.new_zeros(b, h, d_k, d_v)
    out = torch.zeros_like(v)

    for idx in range(n_chunks):
        q_i = q[:, :, idx]  # (B,H,C,Dk)
        k_i = k[:, :, idx]
        gamma_i = gamma[:, :, idx]  # (B,H,C)

        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S  # (B,H,C,Dv)
        out[:, :, idx] = q_i @ S + attn_local @ u_i

        # Aggregate new state with decay – average γ across chunk for efficiency
        gamma_factor = gamma_i.mean(-1).unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
        S = S * gamma_factor + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac+noise init)
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR with identity initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31, init_eps: float = 0.02):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # causal identity
            weight.add_(init_eps * torch.randn_like(weight))
        self.filters = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Optional typing helper for cache
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore[name-defined]

# -----------------------------------------------------------------------------
# Main DeltaNet implementation
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):
    """DeltaNet layer with Unified Dynamic Memory & Output-Aware Annealed Gating (UDMAG)."""

    def __init__(
        self,
        *,
        mode: str = "udmag",
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
        fir_long_kernel: int = 31,
        # Dynamic decay initial bias (sigmoid(bias) ≈ retain_rate)
        gamma_bias_init: float = 1.2,  # sigmoid(1.2)≈0.77 retain
        # Fusion gate hyper-params
        fusion_hidden_mult: int = 2,
        gate_temp_init: float = 1.0,
        # Annealed epsilon floor
        epsilon_start: float = 0.02,
        epsilon_end: float = 0.002,
        epsilon_decay_steps: int = 4000,
        # Entropy regularisation weight schedule
        entropy_start: float = 0.02,
        entropy_end: float = 0.0,
        entropy_decay_steps: int = 3000,
        # Identity residual scale
        identity_scale_init: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__()

        # --- bookkeeping ----------------------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key / Value dims must divide num_heads")

        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # --- projections ----------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Identity residual projection (outside any gate)
        self.id_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.alpha_identity = nn.Parameter(identity_scale_init * torch.ones(num_heads))

        # --- optional short conv -------------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()

        # --- FIR convolutions ----------------------------------------------
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # --- dynamic gamma projection --------------------------------------
        self.gamma_proj = nn.Linear(hidden_size, num_heads, bias=True)
        nn.init.constant_(self.gamma_proj.bias, gamma_bias_init)

        # --- fusion gate MLP -----------------------------------------------
        # stats: mean+std for 4 paths (short,long,delta,value) → 8 * H dims
        stats_dim = 8 * num_heads
        fusion_in = hidden_size + stats_dim
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 3, bias=True),  # 3 context paths
        )
        # Temp per head
        self.gate_log_temp = nn.Parameter(torch.log(torch.tensor(gate_temp_init)) * torch.ones(num_heads))

        # --- epsilon annealing buffer --------------------------------------
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # --- entropy annealing ---------------------------------------------
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.entropy_decay_steps = entropy_decay_steps

        # step counter
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

        # --- output norm / projection --------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # helper schedules
    # ---------------------------------------------------------------------
    def _current_epsilon(self) -> float:
        t = float(self._step.item())
        if t >= self.epsilon_decay_steps:
            return self.epsilon_end
        ratio = t / max(1.0, self.epsilon_decay_steps)
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    def _current_entropy_scale(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_end
        ratio = t / max(1.0, self.entropy_decay_steps)
        return self.entropy_start + ratio * (self.entropy_end - self.entropy_start)

    # ---------------------------------------------------------------------
    # Utility – compute mean & std across D for each head
    # ---------------------------------------------------------------------
    @staticmethod
    def _mean_std(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # x: [B,L,H,D]
        mean = x.mean(dim=-1, keepdim=False)
        std = x.std(dim=-1, unbiased=False, keepdim=False)
        return mean, std

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compat
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"

        batch_size, seq_len_full, _ = hidden_states.shape

        # --- cache retrieval ----------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # --- optional unpadding -------------------------------------------
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # --- projections + optional short conv ----------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv and last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q_proj_out = self.q_proj(hidden_states)
        k_proj_out = self.k_proj(hidden_states)
        v_proj_out = self.v_proj(hidden_states)

        q_in, conv_state_q = self.q_conv1d(q_proj_out, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_in, conv_state_k = self.k_conv1d(k_proj_out, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_in, conv_state_v = self.v_conv1d(v_proj_out, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # --- reshape to heads --------------------------------------------
        q = rearrange(q_in, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_in, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = rearrange(v_in, "b l (h d) -> b l h d", h=self.num_heads)

        # --- activation / norm on Q,K ------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # --- beta for Δ-rule ---------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --- gamma decay --------------------------------------------------
        gamma = torch.sigmoid(self.gamma_proj(hidden_states))  # [B,L,H]

        # --- Δ-rule global path -------------------------------------------
        delta_out_t, recurrent_state = _delta_rule_chunkwise_gamma(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
            gamma=rearrange(gamma, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_t, "b h l d -> b l h d")

        # --- local FIR paths ---------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # --- compute output-aware statistics -----------------------------
        mean_s, std_s = self._mean_std(local_short)
        mean_l, std_l = self._mean_std(local_long)
        mean_d, std_d = self._mean_std(delta_out)
        mean_v, std_v = self._mean_std(v_direct)
        stats_concat = torch.cat(
            [mean_s, std_s, mean_l, std_l, mean_d, std_d, mean_v, std_v], dim=-1
        )  # [B,L,8H]

        # merge heads into feature dim
        stats_feat = rearrange(stats_concat, "b l h8 -> b l (h8)")

        # --- fusion gate --------------------------------------------------
        gate_inp = torch.cat([hidden_states, stats_feat], dim=-1)  # [B,L,hs+stats]
        gate_logits_flat = self.fusion_gate_mlp(gate_inp)  # [B,L,H*3]
        gate_logits = rearrange(gate_logits_flat, "b l (h p) -> b l h p", h=self.num_heads, p=3)

        # temperature scaling per head
        temp = (F.softplus(self.gate_log_temp) + 1e-4).view(1, 1, -1, 1)
        gate_logits = gate_logits / temp

        gate_soft = torch.softmax(gate_logits, dim=-1)  # [B,L,H,3]

        # epsilon floor
        eps = self._current_epsilon()
        gate_soft = gate_soft * (1.0 - 3 * eps) + eps

        # optional entropy regularisation
        reg_loss = None
        entropy_scale = self._current_entropy_scale()
        if self.training and entropy_scale > 0.0:
            entropy = -(gate_soft * torch.log(gate_soft + 1e-8)).sum(dim=-1).mean()
            # maximise entropy ⇒ add *negative* coefficient to loss later; here we return positive value
            reg_loss = -entropy * entropy_scale

        # --- weighted fusion of context paths ----------------------------
        o_context = (
            gate_soft[..., 0:1] * local_short + gate_soft[..., 1:2] * local_long + gate_soft[..., 2:3] * delta_out
        )

        # --- identity residual ------------------------------------------
        id_val = self.id_proj(hidden_states)
        id_val = rearrange(id_val, "b l (h d) -> b l h d", h=self.num_heads)
        alpha = self.alpha_identity.view(1, 1, -1, 1)

        o = o_context + alpha * id_val

        # --- cache update ------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            if hasattr(past_key_values, "update"):
                past_key_values.update(
                    recurrent_state=recurrent_state,
                    conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                    layer_idx=self.layer_idx,
                    offset=hidden_states.shape[1],
                )

        # --- output norm / projection -----------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # --- re-pad if we un-padded -------------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len_full)

        # step++
        self._step += 1  # type: ignore[operator]

        return o, reg_loss, past_key_values
