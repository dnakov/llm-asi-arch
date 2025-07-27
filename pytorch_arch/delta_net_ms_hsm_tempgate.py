# -*- coding: utf-8 -*-
"""
DeltaNet – Multi-Scale Convolution + Hierarchical Memory with Temperature-Controlled Gating
==========================================================================================
Identifier: delta_net_ms_hsm_tempgate

This evolution unifies the strongest ideas discovered so far:

1.   **Multi-Scale Local Convolution (MS-Conv)** – retains the efficient
     depth-wise causal convolutions (kernel sizes default [3,7,15]) and **adds
     a lightweight point-wise (1×1) channel-mix projection** so information can
     flow *across* channels, fixing the inter-channel isolation weakness that
     hurt global reasoning in previous purely depth-wise designs.

2.   **Global Associative Memory (Delta Rule)** – keeps the proven
     chunk-wise Δ-rule path for precise, order-sensitive long-range reasoning.

3.   **Hierarchical Segment Memory (HSM)** – provides inexpensive pooled
     context at exponentially increasing scales.  A *content-aware* softmax
     over scales lets every token choose its preferred receptive field.

4.   **Temperature–Controlled Per-Head Gating** – a *single* softmax gates the
     three branches *per token & per head*, but **each head owns a
     learnable temperature and bias**.  This allows some heads to make
     extremely sharp (near-hard) selections (good for Winogrande-style local
     precision) while others keep soft blends for broad discourse (needed for
     BoolQ / HellaSwag).  Temperatures are enforced positive via *softplus*.

All operations remain strictly causal and sub-quadratic (O(N log N) from the
HSM pooling, O(N) elsewhere).  Interfaces, class name, and forward signature
are fully preserved.  Every tensor manipulation is batch-agnostic and uses
`einops.rearrange` for safety.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

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

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (+1) used as positive kernel feature map."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so that values along the last dim sum to 1."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Delta Rule (unchanged numerics)
# -----------------------------------------------------------------------------

@torch.compile
def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Original DeltaNet associative scan kernel (linear time, causal)."""
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

    # reshape into chunks ------------------------------------------------------
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)
    attn_inv = attn_inv.to(torch.bfloat16)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S.detach()

# -----------------------------------------------------------------------------
# Hierarchical Segment Memory (HSM) utilities
# -----------------------------------------------------------------------------

@torch.compile
def _hierarchical_context(
    v: torch.Tensor,          # (B,H,L,Dv)
    gates: torch.Tensor,      # (B,H,L,S)
    scales: List[int],
) -> torch.Tensor:           # (B,H,L,Dv)
    """Multi-scale causal average pooling with content gates."""
    b, h, L, d = v.shape
    out = torch.zeros_like(v)
    v_flat = rearrange(v, "b h l d -> (b h) d l")  # for conv pooling

    for idx, win in enumerate(scales):
        if win == 1:
            pooled = v_flat  # identity – preserves exact local details
        else:
            pad = win - 1
            pooled = F.avg_pool1d(F.pad(v_flat, (pad, 0)), kernel_size=win, stride=1, padding=0)
        pooled = rearrange(pooled, "(b h) d l -> b h l d", b=b, h=h)
        gate = gates[..., idx].unsqueeze(-1)  # (B,H,L,1)
        out = out + pooled * gate
    return out


def _get_scales(max_len: int, max_scales: int) -> List[int]:
    """Return powers-of-two scales up to *max_len* (always includes 1)."""
    scales: List[int] = [1]
    w = 2
    while len(scales) < max_scales and w <= max_len:
        scales.append(w)
        w <<= 1
    return scales

# -----------------------------------------------------------------------------
# Multi-Scale Depth-wise Convolution with Point-wise Mixing
# -----------------------------------------------------------------------------

class MultiScaleDepthwiseConv1d(nn.Module):
    """Depth-wise causal convolutions at multiple kernel sizes + channel mix."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        kernel_sizes: List[int] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_heads * head_dim,
                out_channels=num_heads * head_dim,
                kernel_size=k,
                groups=num_heads * head_dim,
                bias=False,
            )
            for k in self.kernel_sizes
        ])
        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.02)

        # Point-wise mixing (1×1) across channels to restore feature coupling
        self.channel_mix = nn.Linear(head_dim * len(self.kernel_sizes), head_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_flat = rearrange(x, "b l h d -> b (h d) l")
        outs = []
        for k_size, conv in zip(self.kernel_sizes, self.convs):
            pad = k_size - 1
            out = conv(F.pad(x_flat, (pad, 0)))  # causal pad left
            outs.append(out)
        y = torch.cat(outs, dim=1)  # (B, H*D*lenK, L)
        y = rearrange(
            y,
            "b (h d_mult) l -> b l h (d_mult)",
            h=h,
            d_mult=d * len(self.kernel_sizes),
        )
        y = self.channel_mix(y)  # (B,L,H,D)
        return y

# -----------------------------------------------------------------------------
# Main DeltaNet class
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache
    from transformers.processing_utils import Unpack


class DeltaNet(nn.Module):
    """DeltaNet layer with Multi-Scale Conv, HSM and Temperature-Gated Fusion."""

    def __init__(
        self,
        *,
        mode: str = "ms_hsm_tempgate",
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
        # --- new hyper-parameters -----------------------------------
        ms_kernel_sizes: Tuple[int, ...] | List[int] = (3, 7, 15),
        hsm_max_scales: int = 6,
        gate_hidden_mult: int = 2,
        # -------------------------------------------------------------
        **kwargs: "Unpack[Dict]",
    ) -> None:
        super().__init__()
        # store basic flags
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ("silu", "relu", "elu", "identity")
        assert self.qk_norm in ("l2", "sum")

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
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
        self.hsm_max_scales = hsm_max_scales

        # dimensions ---------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value dim must be divisible by num_heads")

        # linear projections ------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # beta ---------------------------------------------------------
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # short conv ---------------------------------------------------
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
            raise UserWarning("ShortConvolution is mandatory for DeltaNet performance.")

        # multi-scale conv --------------------------------------------
        self.local_conv = MultiScaleDepthwiseConv1d(
            hidden_size=self.value_dim,
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_sizes=ms_kernel_sizes,
        )

        # content-aware HSM scale gate (token, head)
        self.hsm_scale_gate = nn.Linear(self.head_k_dim, hsm_max_scales, bias=False)

        # gating MLP (token-wise) producing per-head 3-path logits ------
        gate_in_dim = hidden_size + 3 * num_heads  # hidden + branch norms
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_size * gate_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * gate_hidden_mult, num_heads * 3, bias=True),
        )

        # per-head temperature (>0 via softplus) and bias per branch ----
        self.gate_log_temp = nn.Parameter(torch.zeros(num_heads))          # temp log
        self.gate_bias = nn.Parameter(torch.zeros(num_heads, 3))           # bias for each branch

        # output normalisation / projection ----------------------------
        if self.use_gate:
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
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # unused but kept for API
        **kwargs: "Unpack[Dict]",
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        # ---------------- input validation ---------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---------------- projections + short conv -------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens
        )

        # ---------------- split heads --------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ---------------- activation / normalisation ----------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ---------------- beta gate ----------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- delta path ---------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")

        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---------------- local conv path ----------------------------
        local_out = self.local_conv(v)  # (B,L,H,D)

        # ---------------- HSM path -----------------------------------
        scales = _get_scales(seq_len, self.hsm_max_scales)
        hsm_gate_logits = self.hsm_scale_gate(q)  # (B,L,H,S)
        hsm_gate_logits = hsm_gate_logits[..., : len(scales)]
        hsm_gates = F.softmax(rearrange(hsm_gate_logits, "b l h s -> b h l s"), dim=-1)
        hsm_out = _hierarchical_context(v_d, hsm_gates, scales)  # (B,H,L,D)
        hsm_out = rearrange(hsm_out, "b h l d -> b l h d")

        # ---------------- compute branch norms -----------------------
        def _norm(x: torch.Tensor) -> torch.Tensor:
            return x.abs().mean(dim=-1)  # (B,L,H)

        feat = torch.cat(
            [
                hidden_states,
                rearrange(_norm(local_out), "b l h -> b l (h)"),
                rearrange(_norm(delta_out), "b l h -> b l (h)"),
                rearrange(_norm(hsm_out), "b l h -> b l (h)"),
            ],
            dim=-1,
        )

        gate_logits = self.fusion_gate_mlp(feat)  # (B,L,H*3)
        gate_logits = rearrange(gate_logits, "b l (h c) -> b l h c", h=self.num_heads, c=3)

        # apply per-head temperature and bias -------------------------
        temp = F.softplus(self.gate_log_temp) + 1e-3  # ensure >0
        gate_logits = gate_logits * temp.view(1, 1, self.num_heads, 1) + self.gate_bias.view(1, 1, self.num_heads, 3)

        gate_weights = F.softmax(gate_logits, dim=-1)  # (B,L,H,3)

        # ---------------- fuse ---------------------------------------
        out = (
            gate_weights[..., 0:1] * local_out +
            gate_weights[..., 1:2] * delta_out +
            gate_weights[..., 2:3] * hsm_out
        )

        # ---------------- cache update -------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ---------------- output norm & proj -------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)

        out = rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # ---------------- re-pad if needed ---------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, batch_size, seq_len)

        return out, None, past_key_values
