# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Content & Memory Gating (ACMG)
=================================================
This evolutionary variant combines the strongest ideas from prior experiments
(BCMF, HWSMG-Hier, HMCF) while *resolving* their residual trade-offs through a
**dynamic, confidence-conditioned minimum-leak mechanism** and *output-aware*
softmax gating.

Key Innovations – all enabled by default
---------------------------------------
1. Output-Aware Gating
   •  The fusion gate conditions on **both** the incoming hidden state *and* a
      per-path *summary* (mean across heads) of each candidate branch output
      (local-short, local-long, Δ-memory).  Experiments show this additional
      information enables sharper, context-sensitive routing without blowing up
      parameter count.

2. Learnable Temperature
   •  A single positive scalar τ (initialised ≈0.7) modulates gate sharpness.
      The model learns whether to mix softly or route hard, layer-wise.

3. Confidence-Conditioned Minimum-Leak (Adaptive Floor)
   •  Previous *static* minimum-leak (BCMF) guaranteed 5 % flow through each
      convolutional path, rescuing local reasoning *but* capping global routing.
      We generalise this idea:  the minimum floor is **proportional to the
      gate’s own confidence in the identity path** – i.e.

          floor = κ · w_value        with κ = min_local_weight_base (0.05)

      •  When the value/identity path dominates (   w_value → 1.0  ) the floor
         equals κ, protecting local branches from starvation.
      •  When the gate already allocates little mass to the value path
         (   w_value → 0.0  ) the floor vanishes, lifting the earlier upper-
         bound on contextual routing.  Thus we retain local robustness during
         the crucial early-training phase *without* sacrificing mature
         long-range capacity.

4. Gentle Bias Initialisation
   •  Branch-specific biases (short, long, Δ, value) = (-0.2, ‑0.2, +1.0, +3.0)
     – proven in BCMF to keep optimisation stable while avoiding early
       conv-path suppression.

5. Identity FIR Initialisation
   •  All depth-wise causal FIR filters start as exact δ-kernels (identity)
     – preserves information at step 0, accelerates convergence.

Complexity, causal masking, and interface are *unchanged*: the design remains
O(N) and a drop-in replacement for any earlier DeltaNet layer.
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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:  # shifted ELU(+1)
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalisation along the last dimension."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity init)
# ---------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head, per-channel causal FIR convolution with **identity** init."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Parameter shape: (H, D, K)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # δ-kernel for causality (tap at current time-step)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")  # depth-wise groups
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # left pad for causality
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (identical core, kept @torch.compile)
# ---------------------------------------------------------------------------

@torch.compile  # noqa: D401 – keep high-perf compilation
# pylint: disable=too-many-locals,too-many-statements

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B, H, L, D_k)
    k: torch.Tensor,  # (B, H, L, D_k)
    v: torch.Tensor,  # (B, H, L, D_v)
    beta: torch.Tensor,  # (B, H, L)
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficient **O(N)** associative Δ-rule with strict causality."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise q/k and apply beta scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape: (B H N C D)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )
    mask_tri = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    eye = torch.eye(chunk_size, dtype=attn_inv.dtype, device=attn_inv.device)
    attn_inv = attn_inv + eye

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    mask_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S.detach()

# ---------------------------------------------------------------------------
# Typing helper (for static checkers only)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

# ---------------------------------------------------------------------------
# Main DeltaNet layer – ACMG variant
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with **Adaptive Content & Memory Gating** (ACMG)."""

    def __init__(
        self,
        # ---------- base args ---------- #
        mode: str = "acmg",
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
        # ---------- branch params ---------- #
        fir_kernel_short: int = 3,
        fir_kernel_long: int = 31,
        # ---------- gating params ---------- #
        fusion_hidden_mult: int = 2,
        gate_dropout: float = 0.1,
        min_local_weight_base: float = 0.05,  # κ in description
        # bias order: short, long, delta, value
        gate_bias_init: Tuple[float, float, float, float] = (-0.2, -0.2, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),  # τ≈0.7 via softplus−1
        **kwargs,
    ) -> None:
        super().__init__()

        # ---------------- bookkeeping ---------------- #
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.allow_neg_eigval = allow_neg_eigval
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx or 0
        self.min_local_weight_base = min_local_weight_base
        self.gate_dropout = gate_dropout

        # ---------------- dimensions ----------------- #
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---------------- projections ---------------- #
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short convs ----------------- #
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---------------- local FIR convs ------------- #
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---------------- gating network -------------- #
        gate_in_dim = hidden_size + 3 * self.head_v_dim  # hidden + mean of 3 branch outputs
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # logits for 4 paths
        )
        with torch.no_grad():
            self.fusion_gate[-1].bias[:] = torch.tensor(gate_bias_init)

        # dropout on gate logits
        self.gate_dropout_layer = nn.Dropout(p=gate_dropout)
        # learnable temperature τ  (via softplus for positivity)
        self.logit_temperature = nn.Parameter(torch.full((1,), gate_logit_init))

        # ---------------- output normalisation -------- #
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compat
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        # -------------- mask / padding handling ------------------- #
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B, L_in, _ = hidden_states.shape

        # fetch previous layer state if any
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # -------------- Q K V projections (+ conv) ---------------- #
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # activation & optional normalisation on q/k
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ---------------- beta for delta -------------------------- #
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule path ----------------------------- #
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---------------- local FIR paths ------------------------- #
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---------------- gating --------------------------------- #
        # Build gate input (hidden + per-path means)
        gate_inp = torch.cat(
            (
                hidden_states,
                rearrange(local_short.mean(dim=2), "b l d -> b l d"),
                rearrange(local_long.mean(dim=2), "b l d -> b l d"),
                rearrange(delta_out.mean(dim=2), "b l d -> b l d"),
            ),
            dim=-1,
        )
        gate_logits = self.fusion_gate(gate_inp)  # (B, L, 4)

        # dropout on logits during training
        if self.training and self.gate_dropout > 0.0:
            gate_logits = self.gate_dropout_layer(gate_logits)

        # Temperature scaling
        temperature = F.softplus(self.logit_temperature) + 1e-4
        gate_logits = gate_logits / temperature
        gate_logits = rearrange(gate_logits, "b l c -> b l 1 c").expand(-1, -1, self.num_heads, -1)  # (B,L,H,4)

        fusion_weights = torch.softmax(gate_logits, dim=-1)  # (B,L,H,4)

        # ---------- adaptive minimum-leak local floor ------------- #
        if self.min_local_weight_base > 0.0:
            value_w = fusion_weights[..., 3:4]  # (B,L,H,1)
            floor = self.min_local_weight_base * value_w  # proportional to confidence
            # Add floor to conv paths, re-normalise
            fusion_weights = fusion_weights + torch.zeros_like(fusion_weights)  # clone for safety
            fusion_weights[..., 0:1] = fusion_weights[..., 0:1] + floor
            fusion_weights[..., 1:2] = fusion_weights[..., 1:2] + floor
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)

        # ---------------- fuse outputs ---------------------------- #
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # ---------------- cache update --------------------------- #
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
            )

        # ---------------- output normalisation ------------------- #
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # re-pad if we had removed padding tokens
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B, L_in)

        return o, None, past_key_values
