# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Floor & Per-Head Linear Gate (AFP)
====================================================
This evolution combines the strongest ideas from previous variants while
explicitly fixing the two residual bottlenecks that repeatedly limited
performance:

1. *Rigid / global gate parameters* – previous single-MLP gates were shared
   across all heads which constrained specialisation.  Here we introduce **a
   true per-head linear gate** (implemented as an efficient batched einsum)
   giving each head its own set of weights *and* bias.

2. *Fixed minimum floor* – non-zero floor was helpful for stability but hurt
   tasks that demand pure single-path routing.  We replace it with a **learnable
   adaptive floor**: each head-path pair has its own parameter that is mapped
   through a `sigmoid` into `[0, base_floor]` so it *starts* with a gentle
   minimum but the network can learn to reduce it to ~0 when beneficial.

Extra features
--------------
* **Per-path temperature** – a learnable, path-specific temperature controls
  sharpness of the softmax, enabling automatic entropy tuning during
  training.
* **Improved initial bias** – biases are initialised to favour the identity
  path (+1.5) and delta path (+0.5) while slightly discouraging the two FIR
  branches (-0.5).  This provides the proven warm-start without starving local
  branches.
* **All computations remain O(N)** – we reuse the proven chunk-wise Δ-rule and
  depth-wise FIR convolutions.
* **einops everywhere** – every reshape / transpose is performed with
  `einops.rearrange` for dynamic shape safety.

The public interface (`DeltaNet` signature, **kwargs, batch agnosticism)
remains unchanged and *all features are on by default* so no external config
changes are necessary.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper activations
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise FIR convolution (identical to earlier proven implementation)
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Causal depth-wise FIR convolution with per-head filters."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = nn.Parameter(
            torch.randn(num_heads, head_dim, kernel_size) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, H, D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left-pad
        out = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(out, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule (unchanged, battle-tested)
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
def delta_rule_chunkwise(
    q: torch.Tensor,  # [B, H, L, Dk]
    k: torch.Tensor,  # [B, H, L, Dk]
    v: torch.Tensor,  # [B, H, L, Dv]
    beta: torch.Tensor,  # [B, H, L]
    *,
    chunk_size: int = 32,
):
    """Efficient, strictly causal Δ-rule implementation (O(N))."""
    b, h, L, d_k = q.shape

    # Pad so length is a multiple of chunk_size
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # L2-norm normalisation (stable cosine sim)
    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into (chunk) blocks of size C
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] = inv[..., i, :i] + (
            inv[..., i, :, None].clone() * inv[..., :, :i].clone()
        ).sum(-2)

    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)
    inv = inv.to(torch.bfloat16)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    mask_future = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Main DeltaNet with Adaptive Floor & Per-Head Gate
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover – for type tooling only
    from fla.models.utils import Cache


class DeltaNet(nn.Module):
    """DeltaNet layer with multi-scale memory and adaptive per-head gating."""

    # pylint: disable=too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        *,
        mode: str = "afp",  # Adaptive Floor & Per-head gate
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
        # --- Multi-scale FIR params -------------------------------------
        fir_kernel_size_short: int = 5,
        fir_kernel_size_long: int = 64,
        # --- Gate params -------------------------------------------------
        base_floor: float = 0.05,  # maximum floor value (learnable param in [0, base_floor])
        warm_start_bias_value: float = 1.5,
        warm_start_bias_delta: float = 0.5,
        gate_hidden_mult: int = 2,
        **kwargs,
    ):  # noqa: D401
        super().__init__()

        # -------- Basic attribute bookkeeping ---------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key / value dimensions must be divisible by num_heads")

        # Save flags
        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.base_floor = base_floor

        # -------- Projection layers ------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # -------- Optional Short Convolutions --------------------------
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
                bias=conv_bias,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
                bias=conv_bias,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation="silu",
                bias=conv_bias,
            )
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")

        # -------- Multi-scale FIR convolutions --------------------------
        self.fir_short = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_short
        )
        self.fir_long = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_long
        )

        # -------- Statistic helper (returns list of 4 tensors) ----------
        def _stat_f(t: torch.Tensor):  # type: ignore[override]
            m1 = t.mean(dim=-2, keepdim=True).expand_as(t)
            m2 = (t ** 2).mean(dim=-2, keepdim=True).expand_as(t)
            m3 = t.abs().mean(dim=-2, keepdim=True).expand_as(t)
            m4 = t.norm(dim=-1, keepdim=True).expand_as(t)
            return [m1, m2, m3, m4]

        self.stat_f = _stat_f  # type: ignore[assignment]

        # -------- Per-head linear gate ----------------------------------
        branch_stat_dim = self.head_v_dim * 4  # 4 stats per branch
        total_stats_dim = branch_stat_dim * 3  # we feed stats of 3 branches (short,long,delta)
        fusion_in_dim = hidden_size + total_stats_dim  # per head concat

        # Weight: [H, F_in, 4] ; Bias: [H, 4]
        self.gate_weight = nn.Parameter(torch.empty(num_heads, fusion_in_dim, 4))
        self.gate_bias = nn.Parameter(torch.zeros(num_heads, 4))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        # Warm-start bias initialisation
        with torch.no_grad():
            self.gate_bias[:, 0] = -0.5  # short FIR
            self.gate_bias[:, 1] = -0.5  # long  FIR
            self.gate_bias[:, 2] = warm_start_bias_delta  # delta
            self.gate_bias[:, 3] = warm_start_bias_value  # value / identity

        # Per-path temperature  (log-temp so positivity is guaranteed)
        self.log_temp = nn.Parameter(torch.zeros(4))  # init temp = 1.0 for all paths

        # Adaptive floor parameter per head-path (initial 0, sigmoid→0.5)
        # floor = base_floor * sigmoid(param)  ∈ (0, base_floor)
        self.floor_param = nn.Parameter(torch.zeros(num_heads, 4))

        # -------- Output normalisation & projection ---------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ----------------------------------------------------------------- #
    # Forward pass                                                     #
    # ----------------------------------------------------------------- #

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # not used, kept for API parity
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:
        # ------------ Optional unpadding for variable-length batches -------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).unsqueeze(0)

        # ------------ Linear projections + optional short conv -------------------
        conv_q = conv_k = conv_v = None
        if self.use_short_conv and last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # Apply projection then short conv which already includes silu for v path
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        if self.use_short_conv:
            q, conv_q = self.q_conv1d(q_proj, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_k = self.k_conv1d(k_proj, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_v = self.v_conv1d(v_proj, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:  # pragma: no cover – shouldn't happen
            q, k = q_proj, k_proj
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(v_proj)

        # ------------ Head split & activations -----------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            # identity handled implicitly
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        v_direct = v  # identity / value path

        # ------------ Beta scaling for Δ-rule -------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global memory -----------------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # Local FIR memory paths ---------------------------------------------------
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # ------------ Feature statistics per head ---------------------------------
        stats_short = torch.cat(self.stat_f(fir_short), dim=-1)
        stats_long = torch.cat(self.stat_f(fir_long), dim=-1)
        stats_delta = torch.cat(self.stat_f(delta_out), dim=-1)

        # Build gate input  [B, L, H, fusion_in_dim]
        hidden_exp = hidden_states.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [B,L,H,D]
        fusion_in = torch.cat([
            hidden_exp,
            stats_short,
            stats_long,
            stats_delta,
        ], dim=-1)

        # ------------ Per-head linear gate ----------------------------------------
        gate_logits = einsum(
            fusion_in,
            self.gate_weight,  # type: ignore[arg-type]
            "b l h f, h f c -> b l h c",
        ) + self.gate_bias  # [B, L, H, 4]

        temp = torch.exp(self.log_temp).clamp(min=0.1, max=10.0)  # [4]
        gate_logits = gate_logits / temp  # broadcast over last dim

        soft = torch.softmax(gate_logits, dim=-1)  # [B,L,H,4]

        # Adaptive floor per head-path
        floor = self.base_floor * torch.sigmoid(self.floor_param)  # [H,4] in (0, base_floor)
        floor = floor[None, None, :, :]  # broadcast to [B,L,H,4]

        residual = 1.0 - floor.sum(-1, keepdim=True)
        gate_weights = floor + residual * soft  # convex combination with adaptive floor

        # Save for potential external regularisation
        self.last_fusion_weights = gate_weights  # [B,L,H,4]

        # ------------ Fuse branches ---------------------------------------------
        o = (
            gate_weights[..., 0:1] * fir_short
            + gate_weights[..., 1:2] * fir_long
            + gate_weights[..., 2:3] * delta_out
            + gate_weights[..., 3:4] * v_direct
        )

        # ------------ Cache management ------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ------------ Output norm and projection ---------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)

        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ------------ Re-pad if unpadded earlier ---------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
