# -*- coding: utf-8 -*-
"""
DeltaNet – Tri-Scale FIR Memory with Per-Head Residual & Persistent Local Floor (delta_net_triscale)
===============================================================================================
This evolution introduces **mid-range convolutional memory** to close the gap
between short-range (≤5 tokens) and long-range (≥64 tokens) dependencies that
previous variants struggled with (see BoolQ regressions).  Concretely we add a
*mid* depth-wise FIR branch (default **kernel_size_mid = 15**) and extend the
content-aware fusion gate from 4 → 5 paths.

Key innovations (enabled by default)
-----------------------------------
1. Tri-scale *causal* FIR memory  –  short / **mid** / long kernels per head.
2. Persistent non-zero local floor ε(t) applied to **all three** FIR paths.
3. Per-head learnable residual bypass mixing **all three** FIR outputs.
4. Content-aware 5-way softmax gate with temperature and entropy regulariser.

All additions keep *O(N·d)* complexity, are batch-agnostic, and preserve the
public interface (class name `DeltaNet`, identical `forward` signature).
The implementation copies heavily from `delta_net_dynfloor_reshead` while
extending it to a 5-path setting.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

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

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU – strictly positive output."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise last dim to sum-to-one (L1)."""
    return (x / x.sum(-1, keepdim=True)).to(x)


# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (dirac initialised)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding (sub-linear memory)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads, head_dim, self.kernel_size)
        # Dirac (identity) – last tap = 1  +  small noise for symmetry break
        weight[..., -1] = 1.0
        weight.add_(0.01 * torch.randn_like(weight))
        self.filters = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, H, D)
        b, l, h, d = x.shape
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged numerics, still compiled)
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
def _delta_rule_chunkwise(
    q: torch.Tensor,  # (B H L Dk)
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,  # (B H L)
    *,
    chunk_size: int = 32,
):
    """Efficient O(N) associative Δ-rule with strict causality."""

    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Unit-norm projections and gated values
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
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=inv.device)
    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
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
# Main DeltaNet – tri-scale FIR + 5-way gated fusion
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore


class DeltaNet(nn.Module):  # noqa: D401 – required class name
    """DeltaNet with tri-scale FIR memory, persistent local floor, and per-head residual."""

    def __init__(
        self,
        # Core API -----------------------------------------------------------
        mode: str = "triscale",
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
        # FIR kernels -------------------------------------------------------
        fir_kernel_size_short: int = 5,
        fir_kernel_size_mid: int = 15,
        fir_kernel_size_long: int = 64,
        # Gating network ----------------------------------------------------
        fusion_hidden_mult: int = 2,
        # per-path bias initial (short, mid, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float, float] = (-0.5, -0.5, -0.5, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),
        # Local floor schedule ---------------------------------------------
        floor_init: float = 0.08,
        floor_final: float = 0.02,
        floor_decay: float = 10_000.0,
        # Per-head residual bypass -----------------------------------------
        conv_residual_init: float = 0.1,
        # Entropy regularisation -------------------------------------------
        entropy_target: float = 1.0,
        entropy_coeff: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---------------- bookkeeping -------------------------------------
        self.mode = mode
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

        # ---------------- dimensions --------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value dims must divide num_heads")

        # ---------------- projections -------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv enhancements -------------------------
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        # ---------------- FIR memories ------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)
        self.fir_mid = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_mid)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)

        # ---------------- Content-aware gating ----------------------------
        # per-head statistics (mean, var, abs-mean, l2) × 5 paths = 20 dims per head
        self.stat_dim = 20
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 5, bias=True),  # logits per path (5)
        )
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias[:] = torch.tensor(gate_bias_init)

        # learnable temperature (scalar)
        self.logit_temperature = nn.Parameter(torch.full((1,), gate_logit_init))

        # ---------------- Per-head residual bypass ------------------------
        init_logit = math.log(conv_residual_init / (1 - conv_residual_init))
        self.conv_residual_logit = nn.Parameter(torch.full((num_heads,), init_logit))

        # ---------------- Output norm / projection ------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ---------------- Floor schedule ----------------------------------
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay = float(floor_decay)

        # ---------------- Entropy regularisation -------------------------
        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)
        self.reg_loss: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Statistic helper
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D) → (B,L,H,4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B, L_in, _ = hidden_states.shape

        # -------- optional unpadding for variable-length batches ---------
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # -------- retrieve previous conv state (if any) ------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # -------- projections + short conv -------------------------------
        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # reshape to heads
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # Ensure input and projection dtypes match
        if v_direct.dtype != self.fir_short.filters.dtype:
            v_direct = v_direct.to(self.fir_short.filters.dtype)
        if q.dtype != self.fir_short.filters.dtype:
            q = q.to(self.fir_short.filters.dtype)
        if k.dtype != self.fir_short.filters.dtype:
            k = k.to(self.fir_short.filters.dtype)

        # activations / normalisations on Q,K
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # β for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------- Δ-rule global pathway ----------------------------------
        delta_out_d, recurrent_state = _delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # -------- FIR paths ----------------------------------------------
        local_short = self.fir_short(v_direct)
        local_mid = self.fir_mid(v_direct)
        local_long = self.fir_long(v_direct)

        # -------- Content-aware gating -----------------------------------
        stats_vec = torch.cat([
            self._per_head_stats(local_short),
            self._per_head_stats(local_mid),
            self._per_head_stats(local_long),
            self._per_head_stats(delta_out),
            self._per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H,20)
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,D)
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)  # (B,L,H,D+20)
        gate_logits = self.fusion_gate_mlp(rearrange(gate_in, "b l h d -> (b l h) d"))

        temp = F.softplus(self.logit_temperature) + 1e-4
        gate_logits = gate_logits / temp
        fusion_logits = rearrange(gate_logits, "(b l h) c -> b l h c", b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads)
        fusion_weights = torch.softmax(fusion_logits, dim=-1)  # (B,L,H,5)

        # -------- Persistent local floor ----------------------------------
        eps_now = self.floor_final + (self.floor_init - self.floor_final) * math.exp(-float(self._step.item()) / self.floor_decay)
        if eps_now > 0.0:
            scale = 1.0 - 3 * eps_now  # three FIR paths share floor mass
            fusion_weights = fusion_weights * scale
            fusion_weights[..., 0] += eps_now  # short
            fusion_weights[..., 1] += eps_now  # mid
            fusion_weights[..., 2] += eps_now  # long
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)

        # -------- Entropy regularisation ----------------------------------
        entropy = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(-1).mean()
        self.reg_loss = self.entropy_coeff * torch.relu(self.entropy_target - entropy)

        # -------- Weighted fusion of branches -----------------------------
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_mid +
            fusion_weights[..., 2:3] * local_long +
            fusion_weights[..., 3:4] * delta_out +
            fusion_weights[..., 4:5] * v_direct
        )

        # -------- Per-head residual bypass --------------------------------
        alpha = torch.sigmoid(self.conv_residual_logit).view(1, 1, -1, 1)  # (1,1,H,1)
        o = o + alpha * (local_short + local_mid + local_long) / 3.0

        # -------- Cache update --------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # -------- Output norm / projection --------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = o.to(self.o_proj.weight.dtype)  # Ensure dtype compatibility for o_proj
        o = self.o_proj(o)

        # -------- Re-pad if unpadded earlier ------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B, L_in)

        # -------- step counter -------------------------------------------
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
