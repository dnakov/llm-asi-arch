# -*- coding: utf-8 -*-
"""
DeltaNet – Multi-Scale Depthwise Convolution **with Cross-Channel Pointwise Mixing**
==============================================================================
This evolutionary step addresses the main weakness found in the previous
`delta_net_mshiergate` variant: **lack of cross-feature / cross-head mixing**
after the depth-wise multi-scale convolution branch.  Depth-wise kernels pick
up local patterns but, being *channel–wise*, cannot combine information across
features or heads, which is essential for higher-order reasoning benchmarks
(HellaSwag, Winogrande, etc.).

Key innovations
---------------
1. **Point-wise (1×1) Channel Mixer**
   After the multi-scale depth-wise convolutions we add a *single* linear layer
   that operates on the *(head×channel)* dimension (`hidden_size`) for every
   token independently.  This is equivalent to a point-wise `Conv1d` with
   kernel-size 1 and *no* groups, and therefore mixes **both** channels *and*
   heads at negligible cost (O(L·D²) where D≈1K).

2. **Lean Kernel Set**
   Practical experiments showed diminishing returns beyond 3 scales.  We now
   use a compact kernel list `[3, 15, 31]` by default, keeping the receptive
   field diversity while reducing parameter footprint and memory.

3. **Gentler Gating Bias**
   The strong identity-path bias (+2.0) previously delayed specialisation.
   It is relaxed to `+1.0`, empirically allowing faster utilisation of the new
   local / delta features without destabilising early training.

All computational constraints remain unchanged:
• **O(N·K)** complexity (depth-wise) + **O(N·D²)** for the 1×1 mix (token-wise).
• Strict causality via left-padding.
• Full batch/sequence agnosticism using `einops.rearrange`.
• Interface, class name, and forward signature stay **identical** so the layer
  can be dropped into existing checkpoints.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict

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

def elu_p1(x: torch.Tensor) -> torch.Tensor:  # shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0).to(x)

def sum_norm(x: torch.Tensor) -> torch.Tensor:  # sum-normalisation
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Core chunk-wise delta-rule kernel (unchanged numerics)
# -----------------------------------------------------------------------------

@torch.compile
def delta_rule_chunkwise(
    q: torch.Tensor,  # (B H L Dk)
    k: torch.Tensor,  # (B H L Dk)
    v: torch.Tensor,  # (B H L Dv)
    beta: torch.Tensor,  # (B H L)
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal associative scan used by DeltaNet – identical to the previous impl."""
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

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

    # reshape to chunks ------------------------------------------------------
    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask_inc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask_inc, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    attn = attn.to(torch.bfloat16)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    tri_mask_exc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_mask_exc, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Multi-scale depth-wise causal conv **with channel mixer**
# -----------------------------------------------------------------------------

class MultiScaleDepthwiseConv1d(nn.Module):
    """Depth-wise causal conv at multiple scales + point-wise (1×1) channel mix."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: List[int] = (3, 15, 31),
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        hidden_per_head = head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        in_channels = num_heads * head_dim  # depth-wise → groups = in_channels

        # depth-wise convs ----------------------------------------------------
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=k,
                    groups=in_channels,
                    bias=False,
                )
                for k in self.kernel_sizes
            ]
        )
        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.02)

        # per-head projection to original dim (mix kernels) -------------------
        self.kernel_mix = nn.Linear(len(self.kernel_sizes) * head_dim, head_dim, bias=False)

        # cross-head/channel mixer (1×1) --------------------------------------
        total_hidden = num_heads * head_dim  # full hidden size of value stream
        self.channel_mixer = nn.Linear(total_hidden, total_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B L H D)
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")  # to (B C L) for conv

        outs: List[torch.Tensor] = []
        for k_size, conv in zip(self.kernel_sizes, self.convs):
            pad = (k_size - 1)
            x_pad = F.pad(x_f, (pad, 0))  # causal left-pad
            outs.append(conv(x_pad))  # (B C L)
        y = torch.cat(outs, dim=1)  # (B C*|K| L)

        # back to (B L H (|K|*D)), careful with einops!
        y = rearrange(
            y,
            "b (h kd) l -> b l h kd",
            h=self.num_heads,
            kd=len(self.kernel_sizes) * d,
        )

        # mix kernels inside each head --------------------------------------
        y = self.kernel_mix(y)  # (B L H D)

        # cross-channel/head 1×1 mixing ------------------------------------
        y_flat = rearrange(y, "b l h d -> b l (h d)")  # (B L H*D)
        y_mixed = self.channel_mixer(y_flat)
        y = rearrange(y_mixed, "b l (h d) -> b l h d", h=h)
        return y

# -----------------------------------------------------------------------------
# DeltaNet main layer (with updated local branch & gentler gate bias)
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # forward-compat placeholder

class DeltaNet(nn.Module):
    """DeltaNet layer – Multi-Scale Depth-wise Conv + Channel Mixer + Soft Gating."""

    def __init__(
        self,
        *,
        mode: str = "mscmix",  # identifier
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
        # --- new local branch settings --------------------------------------
        ms_kernel_sizes: Tuple[int, int, int] = (3, 15, 31),
        fusion_hidden_mult: int = 2,
        gate_bias_init: float = 1.0,  # softer identity bias
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        # ---------------- dimensions ---------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0, "key_dim must divide num_heads"
        assert self.value_dim % num_heads == 0, "value_dim must divide num_heads"

        # ---------------- projection layers --------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv branch --------------------------------
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory – do not disable.")

        # ---------------- local multi-scale branch --------------------------
        self.local_conv = MultiScaleDepthwiseConv1d(
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_sizes=list(ms_kernel_sizes),
        )

        # ---------------- output-aware fusion gate --------------------------
        gate_in_dim = hidden_size + 3 * num_heads  # token features + per-branch norms
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 3, bias=True),
        )
        # initialise bias so identity/value path is preferred early on -------
        nn.init.constant_(self.fusion_gate_mlp[-1].bias[num_heads * 2 :], gate_bias_init)

        # ---------------- output norm / projection --------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward pass (interface unchanged)
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B L D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"

        batch_size, seq_len, _ = hidden_states.shape

        # --------------------------------------------------------------
        # Retrieve layer-specific cache entry (if any)
        # --------------------------------------------------------------
        last_state = None
        if (
            past_key_values is not None
            and self.layer_idx is not None
            and len(past_key_values) > self.layer_idx
        ):
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).unsqueeze(0)

        # --------------------------------------------------------------
        # Short conv projections
        # --------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
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

        # head split ---------------------------------------------------
        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim),
            (q, k),
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # activation ---------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        v_direct = v  # keep for fusion

        # beta ---------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # delta rule (global) -----------------------------------------
        q_g = rearrange(q, "b l h d -> b h l d")
        k_g = rearrange(k, "b l h d -> b h l d")
        v_g = rearrange(v, "b l h d -> b h l d")
        beta_g = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(q_g, k_g, v_g, beta_g)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # local multi-scale branch ------------------------------------
        ms_out = self.local_conv(v_direct)

        # branch norms (per-token, per-head) ---------------------------
        def branch_norm(x: torch.Tensor) -> torch.Tensor:  # (B L H D) -> (B L H)
            return x.abs().mean(dim=-1)

        norms = [branch_norm(t) for t in (ms_out, delta_out, v_direct)]

        gate_features = torch.cat(
            [
                hidden_states,
                rearrange(norms[0], "b l h -> b l (h)"),
                rearrange(norms[1], "b l h -> b l (h)"),
                rearrange(norms[2], "b l h -> b l (h)"),
            ],
            dim=-1,
        )

        fusion_logits = self.fusion_gate_mlp(gate_features)  # (B L H*3)
        fusion_logits = rearrange(
            fusion_logits, "b l (h c) -> b l h c", h=self.num_heads, c=3
        )
        fusion_w = torch.softmax(fusion_logits, dim=-1)

        o = (
            fusion_w[..., 0:1] * ms_out
            + fusion_w[..., 1:2] * delta_out
            + fusion_w[..., 2:3] * v_direct
        )

        # --------------------------------------------------------------
        # Cache update
        # --------------------------------------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # output projection / norm ------------------------------------
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # re-insert padding if removed --------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
