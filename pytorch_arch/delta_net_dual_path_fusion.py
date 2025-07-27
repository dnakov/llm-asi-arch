# -*- coding: utf-8 -*-
"""
DeltaNet – Dual-Path Fusion with Adaptive Mixing Gate (DeltaNet-DPF)
===================================================================
This evolution *combines* the best performing ideas observed across
previous experimental variants:

1. **Adaptive Mixing Gate (AMG)**
   After the chunk-wise **delta rule** we *adaptively* mix the recurrent
   output with the *instantaneous* token value vector on a **per-token, per-head**
   basis.  This stabilises optimisation and improves local reasoning
   (validated in *delta_net_adaptive_mix_gate*).

2. **Dilated Convolutional Memory with *Additive* Residual Fusion**
   We keep the depth-wise causal dilated convolution branch but *replace* the
   convex combination used in DCIG with **additive residual fusion**
   (cf. DCCG):

       out = delta_out + gate · conv_out ,   gate ∈ (0,1)

   where the gate is *decoupled* (learned from current hidden state) and its
   bias is initialised to **−1.0 ⇒ σ(−1) ≈ 0.27** so the convolutional path
   participates *right from the start* – resolving the over-suppression issue
   identified in DCIG.

3. **Safer Convolution Weight Init**
   The dilated depth-wise convolution is now Kaiming-initialised so that the
   branch produces non-zero signals at initialisation (zero-init in DCIG
   delayed learning).

All additional computation is **O(N)** and batch-agnostic.  Public
interfaces, class-name, and signatures remain *unchanged*.  New features are
enabled by default with sensible parameters.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Utility helpers (keep minimal; **no** @torch.compile here)
# -----------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (returns strictly positive values)."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so the last-dim sum equals 1."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Core *chunk-wise* delta rule kernel (unchanged – linear time, causal)
# -----------------------------------------------------------------------------

@torch.compile
def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
):
    """Baseline Delta rule (O(N) with causal masking)."""
    b, h, L, d_k = q.shape
    d_v = v.shape[-1]

    # Pad sequence length to multiple of chunk_size
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & weighting
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks : [B,H,N,C,D]
    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )

    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn.to(torch.bfloat16)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)

    strict_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
#  Main DeltaNet Module (Dual-Path Fusion)
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class DeltaNet(nn.Module):
    """DeltaNet with *Adaptive Mixing* & *Additive Dilated-Conv Fusion*."""

    def __init__(
        self,
        mode: str = "chunk1",
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
        # ---- Dilated convolutional fusion ----
        use_dilated_conv: bool = True,
        dilated_kernel_size: int = 3,
        dilation: int | None = None,
        # ---- Adaptive mixing gate between delta & token value ----
        use_mix_gate: bool = True,
        **kwargs,  # retain extensibility
    ) -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_mix_gate = use_mix_gate

        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

        # Dimensional resolutions ------------------------------------------------
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

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        assert self.value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        # ------------------------------------------------------------------
        # Linear projections (Q, K, V)
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Adaptive mixing gate projection (per-token, per-head scalar)
        if self.use_mix_gate:
            self.mix_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Beta (forget) projection
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------------------------------------------------------------
        # Short convolutional enhancement (local receptive field)
        # ------------------------------------------------------------------
        if self.use_short_conv:
            activation = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=activation)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=activation)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance – disabling is unsupported in this evolution.")

        # ------------------------------------------------------------------
        # Output Normalisation / optional gating
        # ------------------------------------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ------------------------------------------------------------------
        # Dilated convolutional memory path
        # ------------------------------------------------------------------
        self.use_dilated_conv = use_dilated_conv
        if self.use_dilated_conv:
            self.dilation = dilation if dilation is not None else 2 ** ((self.layer_idx or 0) % 4)
            self.dilated_kernel_size = dilated_kernel_size
            self.dilated_conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=self.dilated_kernel_size,
                groups=hidden_size,
                bias=False,
                dilation=self.dilation,
            )
            # Kaiming init → provides signal at t=0 (better than zeros)
            nn.init.kaiming_uniform_(self.dilated_conv.weight, a=math.sqrt(5))

            # Decoupled gate – lower bias (≈ −1) so conv contributes early
            self.dilated_gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.constant_(self.dilated_gate_proj.bias, -1.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,T,D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: "Unpack[Dict]",
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        # ---- 0. Basic validations ----
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be (B,L) 0/1 padding mask"

        batch_size, seq_len, _ = hidden_states.shape

        # Retrieve previous state (if any)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---- 1. Linear projections + optional short-conv ----
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # Save token-local value for adaptive mixing (after head split)
        v_token = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        # ---- 2. Head split & activations ----
        q, k = map(lambda x: rearrange(x, "b t (h d) -> b t h d", d=self.head_k_dim), (q, k))

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)

        if self.qk_norm == "sum":
            q = sum_norm(q)
            k = sum_norm(k)

        # ---- 3. Beta gate ----
        if self.use_beta:
            beta = torch.sigmoid(self.b_proj(hidden_states))  # [B,T,H]
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- 4. Delta rule core (chunk-wise) ----
        q_d = rearrange(q, "b t h d -> b h t d")
        k_d = rearrange(k, "b t h d -> b h t d")
        v_d = rearrange(v_token, "b t h d -> b h t d")
        beta_d = rearrange(beta, "b t h -> b h t")

        delta_out, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d, chunk_size=32)
        delta_out = rearrange(delta_out, "b h t d -> b t h d")  # B,T,H,Dv

        # ---- 5. Update cache ----
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ---- 6. Adaptive Mixing Gate (delta vs instantaneous value) ----
        if self.use_mix_gate:
            mix_gate = torch.sigmoid(self.mix_proj(hidden_states))  # [B,T,H]
            mix_gate = rearrange(mix_gate, "b t h -> b t h 1")
            delta_out = mix_gate * delta_out + (1.0 - mix_gate) * v_token

        # ---- 7. Output normalisation / gating (per-head) ----
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b t (h d) -> b t h d", d=self.head_v_dim)
            delta_out = self.o_norm(delta_out, g)
        else:
            delta_out = self.o_norm(delta_out)

        # Merge heads
        delta_out = rearrange(delta_out, "b t h d -> b t (h d)")  # [B,T,D_model]
        delta_out = self.o_proj(delta_out)

        # ---- 8. Dilated convolution branch + additive fusion ----
        if self.use_dilated_conv and attention_mask is None:
            conv_in = rearrange(delta_out, "b t c -> b c t")
            # causal left pad so conv is strictly causal
            pad_len = self.dilation * (self.dilated_kernel_size - 1)
            conv_in = F.pad(conv_in, (pad_len, 0))
            conv_out = self.dilated_conv(conv_in)
            conv_out = rearrange(conv_out, "b c t -> b t c")

            gate = torch.sigmoid(self.dilated_gate_proj(hidden_states))  # [B,T,C]
            # additive residual fusion (delta_out already contains main signal)
            delta_out = delta_out + gate * conv_out

        # ---- 9. Re-pad if we removed padding earlier ----
        if attention_mask is not None:
            delta_out = pad_input(delta_out.squeeze(0), indices, batch_size, seq_len)

        return delta_out, None, past_key_values
