# -*- coding: utf-8 -*-
"""
DeltaNet – Content-Aware Gated Fusion (CAGF)
===========================================
This evolution upgrades the original *Multi-Scale Dynamic Adaptive Fusion*
variant by introducing **content-aware, per-head gating** that directly
leverages path statistics, correcting the two major weaknesses identified in
prior experiments:

1.  *Head-flattened statistics* prevented head specialisation.  We now compute
    **per-head statistics** (mean, variance, abs-mean, ℓ2-norm) for every
    memory path so the gating network can route information on a head-by-head
    basis.
2.  *Uniform bias initialisation* diluted the long-range Δ-rule path.  We fix
    this with path-specific bias (+3 for direct value, +1 for Δ-rule, –1 for
    convolutional paths) **and** a learnable temperature that sharpens the
    softmax over training.

All changes preserve the original API, maintain **O(N)** complexity, stay
strictly causal and remain fully batch-agnostic.  The layer is drop-in
compatible with previous DeltaNet variants.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ================================================================
# Utility helpers
# ================================================================

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU so output is strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise last dim to sum-to-one."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# ================================================================
# Depth-wise causal FIR convolution (unchanged from MSDAF)
# ================================================================
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left-padding."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = nn.Parameter(
            torch.randn(num_heads, head_dim, kernel_size) * 0.02
        )  # (H, D, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        y = rearrange(y, "b (h d) l -> b l h d", h=h)
        return y

# ================================================================
# Core chunk-wise Δ-rule kernel (identical to previous versions)
# ================================================================
@torch.compile
def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficient chunk-wise associative Δ-rule with O(N) cost."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation and scaling ------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks (B H N C D)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    # In-chunk inverse (I − tril(K β Kᵀ))⁻¹ -----------------------
    mask_tri = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, : i] += (
            attn[..., i, :, None].clone() * attn[..., :, : i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# ================================================================
# Main DeltaNet with Content-Aware Gated Fusion
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

class DeltaNet(nn.Module):
    """DeltaNet layer with **Content-Aware, Per-Head Gated Fusion**."""

    def __init__(
        self,
        mode: str = "cagf",
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
        # ─── Multi-scale FIR kernel sizes ────────────────────────────
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        # Path-specific initial biases: (short, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 3.0),
        # Temperature initial (softplus-paramised → τ≈0.7)
        gate_logit_init: float = math.log(math.expm1(0.7)),
        **kwargs,
    ) -> None:
        super().__init__()

        # ---- Book-keeping & basic dims ----------------------------------
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ("silu", "relu", "elu", "identity")
        assert self.qk_norm in ("l2", "sum")

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

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        # --- FIX: assert statement must be on a single line ---------
        assert (
            self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        ), "Key/Value dims must divide num_heads"

        # ---- Linear projections --------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Beta projection for Δ-rule -----------------------------------
        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- Mandatory short convolutions ----------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- Multi-scale local FIR convolutions ----------------------
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_long
        )
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=self.num_heads, head_dim=self.head_v_dim, kernel_size=fir_kernel_size_short
        )

        # ---- Content-aware gating network ----------------------------
        # Per-head stats: 4 metrics per branch, 4 branches = 16 scalars
        self.stat_dim = 16
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2  # moderate size
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # produces logits for 4 paths
        )
        # Path-specific bias initialisation
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias[:] = torch.tensor(gate_bias_init)

        # Learnable temperature (softplus parameterisation to ensure >0)
        self.logit_temperature = nn.Parameter(torch.full((1,), gate_logit_init))

        # ---- Output normalisation / projection ----------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------
    # Statistic helpers (per-head)
    # ---------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:
        """Return per-token, per-head statistics vector of length 4.

        Stats: mean, variance, mean(|x|), ℓ2-norm over feature dim.
        Output shape: [B, L, H, 4]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compat
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"

        batch_size, seq_len, _ = hidden_states.shape

        # Retrieve cache -------------------------------------------------
        last_state = None
        if (
            past_key_values is not None
            and self.layer_idx is not None
            and len(past_key_values) > self.layer_idx
        ):
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # Optional unpadding for efficiency ----------------------------
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).unsqueeze(0)

        # --------------------------------------------------------------
        # Q/K/V projections + short conv enhancements
        # --------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state", None) is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # Head reshape --------------------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # Activation on Q/K --------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # Beta for Δ-rule ----------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global pathway ----------------------------------------
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # Local FIR paths ---------------------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # --------------------------------------------------------------
        # Content-aware gating (per-head)
        # --------------------------------------------------------------
        # Per-head stats (B, L, H, 4)
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = torch.cat(
            [stats_short, stats_long, stats_delta, stats_value], dim=-1
        )  # (B, L, H, 16)

        # Build gating input -----------------------------------------
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,C)
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)  # (B,L,H,C+16)
        gate_in_flat = rearrange(gate_in, "b l h d -> (b l h) d")
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)
        # Temperature scaling
        temperature = F.softplus(self.logit_temperature) + 1e-4
        gate_logits_flat = gate_logits_flat / temperature
        fusion_logits = rearrange(
            gate_logits_flat,
            "(b l h) c -> b l h c",
            b=gate_in.shape[0],
            l=gate_in.shape[1],
            h=self.num_heads,
        )
        fusion_weights = torch.softmax(fusion_logits, dim=-1)

        # --------------------------------------------------------------
        # Weighted fusion of memory paths
        # --------------------------------------------------------------
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # --------------------------------------------------------------
        # Cache update -----------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # Normalisation / projection ---------------------------------
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # Re-pad sequence if we unpadded earlier ----------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
