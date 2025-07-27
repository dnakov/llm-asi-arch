# -*- coding: utf-8 -*-
"""
DeltaNet – Head-Adaptive Floor & Sparsity (HAFS)
==============================================
Identifier: delta_net_hafs

This evolution unifies the strongest ideas across previous DeltaNet variants
(IPEG, HTNG, Adaptive-ε, DynFuse) while eliminating their respective
weaknesses:

1. Head-Adaptive, Annealed Floor  
   •   Each *head / path* owns a learnable **floor parameter** (sigmoid in
       [0,1]).  A global exponential **annealing schedule** multiplies this
       floor, starting at ``floor_init`` (default 5 %) and converging to
       ``floor_final`` (default 1 %).  This guarantees **persistent local
       capacity** (unlike HTNG) while still enabling near-exclusive routing
       (unlike IPEG’s fixed ε-floor).

2. Per-Head Temperature (τ)  
   •   Retains the proven benefits of *per-head/*per-path temperature – sharp
       when useful, smooth otherwise – without extra runtime cost.

3. Identity & Conv Residual Safeguards  
   •   A minimal, learnable **identity residual** (copies token surface form)
       and a **conv residual** (averaged FIR outputs) ensure gradient flow and
       local information retention even if the softmax gate collapses.

4. Expressive Gate with Branch Statistics  
   •   A two-layer GELU MLP receives the hidden state plus 16 statistics
       (mean, var, abs-mean, ℓ2) aggregated from the four branches, producing
       per-head logits.

The layer remains **strictly O(N)** thanks to chunk-wise Δ-rule retrieval and
FIR convolutions.  All tensor reshaping uses **einops.rearrange** for dynamic,
batch-agnostic shapes.  Class name and forward signature are unchanged – this
is a drop-in replacement.
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
# Helper utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU so output stays strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1-normalise along the last dimension."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule (identical math, still @torch.compile) ----------------------
# -----------------------------------------------------------------------------


@torch.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: torch.Tensor,  # [B,H,L,Dk]
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,  # [B,H,L]
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal associative retrieval with O(N) complexity via chunking."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    # L2 normalise Q/K, scale V with β
    q, k = l2norm(q), l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # reshape into chunks: [B,H,N,C,D]
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=inv.device)
    u = inv @ v
    w = inv @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    n_chunks = q.shape[2]
    for idx in range(n_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution -------------------------------------------
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    """Per-head, per-channel causal FIR with identity (Dirac) initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31, noise_std: float = 1e-2) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            filt[..., -1] = 1.0  # identity tap
            if noise_std > 0:
                filt.add_(noise_std * torch.randn_like(filt))
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Typing helper ---------------------------------------------------------------
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # pylint: disable=ungrouped-imports

# -----------------------------------------------------------------------------
# Main DeltaNet – HAFS variant -------------------------------------------------
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):  # noqa: D401 – class name required by framework
    """DeltaNet layer with *Head-Adaptive Floor & Sparsity* (HAFS)."""

    def __init__(
        self,
        # ---- generic args --------------------------------------------------
        mode: str = "hafs",
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
        # ---- FIR kernels ---------------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 63,
        # ---- gating hyper-params ------------------------------------------
        gate_hidden_mult: int = 2,
        floor_init: float = 0.05,
        floor_final: float = 0.01,
        floor_decay: float = 8_000.0,
        # temperatures
        temp_init: float = 1.0,
        # ---- residual safeguards -----------------------------------------
        use_identity_path: bool = True,
        identity_scale_init: float = 0.5,
        conv_residual_init: float = 0.05,
        # ---- entropy regularisation --------------------------------------
        entropy_target: float = 1.0,
        entropy_coeff: float = 0.02,
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        # ---------------- bookkeeping -------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        # dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must be divisible by num_heads")

        # ---------------- linear projections ------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # identity path projection
        if use_identity_path:
            self.id_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.alpha_identity = nn.Parameter(identity_scale_init * torch.ones(num_heads))
        else:
            self.register_parameter("id_proj", None)
            self.register_parameter("alpha_identity", None)

        # ---------------- optional short conv -----------------------------
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            # retain compatibility (Identity layer)
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()

        # ---------------- local FIR convolutions --------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ---------------- gate MLP ---------------------------------------
        stats_dim = 4  # mean, var, abs-mean, l2-norm
        gate_in_dim = hidden_size + stats_dim * 4  # hidden + 16 stats
        hidden_gate_dim = hidden_size * gate_hidden_mult // 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # per-head later via broadcast
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.zero_()
            # bias order: short, long, delta, value – favour value a bit
            self.gate_mlp[-1].bias[3] = 1.5

        # per-head / path temperature & floor
        self.log_temp = nn.Parameter(torch.log(torch.full((num_heads, 4), temp_init)))
        self.floor_param = nn.Parameter(torch.zeros(num_heads, 4))  # sigmoid → (0,1)
        # annealing schedule
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay = float(floor_decay)

        # conv residual bypass (per-head scalar)
        init_logit = math.log(conv_residual_init / (1.0 - conv_residual_init))
        self.conv_residual_logit = nn.Parameter(init_logit * torch.ones(num_heads))

        # ---------------- output normalisation / projection --------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ---------------- entropy regularisation -------------------------
        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)
        self.reg_loss: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # statistic helper (mean, var, abs-mean, l2)
    # ------------------------------------------------------------------
    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D) → (B,L,H,4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None and attention_mask.ndim != 2:
            raise AssertionError("attention_mask must be [batch, seq_len]")
        B0, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)

        # ----- unpad variable-length batches ----------------------------
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ----- retrieve cached conv state ------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # ----- projections + optional conv -----------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)
        if self.use_short_conv:
            q_lin, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k_lin, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v_lin, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ----- head reshape -------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ----- activation / normalisation ----------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # β for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Clamp beta for numerical safety
        beta = beta.clamp(min=1e-4, max=1.0)

        # ----- Δ-rule global memory -----------------------------------
        delta_out_d, recur_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # ----- local FIR paths ---------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ----- gate computation --------------------------------------
        stats_vec = torch.cat([
            self._stats(local_short),
            self._stats(local_long),
            self._stats(delta_out),
            self._stats(v_direct),
        ], dim=-1)  # [B,L,H,16]
        hid_exp = hidden_states.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        gate_in = torch.cat([hid_exp, stats_vec], dim=-1)  # [B,L,H,D+16]
        gate_logits = self.gate_mlp(gate_in)  # [B,L,H,4] – shared MLP per head

        # temperature scaling
        temp = torch.exp(self.log_temp).clamp(0.05, 10.0)  # [H,4]
        gate_logits = gate_logits / temp.unsqueeze(0).unsqueeze(0)
        soft_w = torch.softmax(gate_logits, dim=-1)  # [B,L,H,4]

        # head-adaptive floor (annealed)
        floor_sched = self.floor_final + (self.floor_init - self.floor_final) * math.exp(-float(self._step.item()) / self.floor_decay)
        floor = floor_sched * torch.sigmoid(self.floor_param)  # [H,4]
        floor = floor.unsqueeze(0).unsqueeze(0)
        residual_mass = 1.0 - floor.sum(-1, keepdim=True)
        # Clamp residual_mass for numerical safety (avoid negative or 0 after floor sum)
        residual_mass = torch.clamp(residual_mass, min=1e-3)  # WAS 1e-6, raised for more stability
        # Also clamp floor for numerical stability (avoid 0/negative values)
        floor = torch.clamp(floor, min=1e-5, max=1.0)
        weights = floor + residual_mass * soft_w  # [B,L,H,4]
        # Clamp weights for safety to avoid nan in log, normalize for sum-to-one
        weights = torch.clamp(weights, min=1e-5, max=1.0)
        weights = weights / weights.sum(-1, keepdim=True)

        # entropy regularisation -------------------------------------
        entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean()
        self.reg_loss = self.entropy_coeff * torch.relu(self.entropy_target - entropy)

        # ----- weighted fusion --------------------------------------
        out = (
            weights[..., 0:1] * local_short
            + weights[..., 1:2] * local_long
            + weights[..., 2:3] * delta_out
            + weights[..., 3:4] * v_direct
        )

        # conv residual bypass
        alpha = torch.sigmoid(self.conv_residual_logit).view(1, 1, -1, 1)
        out = out + 0.5 * alpha * (local_short + local_long)

        # identity additive path
        if self.id_proj is not None:
            id_val = self.id_proj(hidden_states)
            id_val = rearrange(id_val, "b l (h d) -> b l h d", d=self.head_v_dim)
            out = out + self.alpha_identity.view(1, 1, -1, 1) * id_val

        # ----- cache update -----------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ----- output norm & projection -----------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # ----- re-pad if needed -------------------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, B0, L_in)

        # increment step counter
        self._step += 1  # type: ignore[operator]

        return out, None, past_key_values
