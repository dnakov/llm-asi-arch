# -*- coding: utf-8 -*-
"""
DeltaNet – EMA Blend v2 with Per-Head / Per-Token Mix-Gating
===========================================================
This evolution upgrades the earlier *delta_net_ema_blend* architecture by
replacing the *global scalar* fusion gate with a **fine-grained, per-head and
per-token gate**.  The new gate is produced directly from the current hidden
state via a lightweight Linear projection (optionally followed by the existing
`ShortConvolution`), yielding a tensor **m ∈ [0,1]** of shape *(B, L, H)*.  The
final output is

    out = (1 − m) · delta_out  +  m · ema_out

This granularity allows each head and each position to adaptively decide how
much it relies on *fast associative* (Delta) versus *smooth long-term* (EMA)
memory, resolving the interference observed on precision-critical tasks such
as ARC-Challenge and WinoGrande in the scalar-gated version.

All additional parameters are tiny (one bias per head plus a weight matrix of
shape *(hidden_size, num_heads)*) and the computational overhead is
negligible.  Complexity remains **O(N)** and fully batch-agnostic.

Implementation notes
--------------------
• The original EMA scan kernel is kept unchanged to guarantee numerical
  equivalence and because it is already `@torch.compile`-optimised.
• The old scalar parameter `self.ema_mix_logit` is **deprecated** but retained
  (frozen) for checkpoint compatibility.
• A new attribute `self.mix_proj` and `self.mix_bias` are introduced and
  enabled by default (`use_head_gate=True`).
• All shapes are handled via *einops.rearrange*; no `.view()`/`.reshape()` is
  used.
• Public interface (class name, `forward` signature, kwargs) is unchanged.

"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

###############################################################################
# Helper functions (identical to previous public release)                     #
###############################################################################

def elu_p1(x: torch.Tensor) -> torch.Tensor:  # Shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)


@torch.compile  # type: ignore[misc]
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Original DeltaNet chunk-wise recurrence (unchanged)."""
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))

    padded_len = l + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn.to(torch.bfloat16)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn_i @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S


@torch.compile  # type: ignore[misc]
def ema_rule_chunkwise(
    v: torch.Tensor,  # (B H L D_v)
    gamma: torch.Tensor,  # (B H L)
    init_state: Optional[torch.Tensor] = None,  # (B H D_v)
):
    """Efficient EMA scan over sequence (O(N))."""
    b, h, l, d_v = v.shape
    ema_out = torch.empty_like(v)
    state = torch.zeros((b, h, d_v), dtype=v.dtype, device=v.device) if init_state is None else init_state

    for t in range(l):  # Python loop but compiled + tiny, runs fast
        g_t = gamma[:, :, t].unsqueeze(-1)  # (B H 1)
        state = g_t * state + (1.0 - g_t) * v[:, :, t]
        ema_out[:, :, t] = state
    return ema_out, state


if TYPE_CHECKING:  # pragma: no cover – static type-checking only
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

###############################################################################
#                          DeltaNet Main Module                               #
###############################################################################

class DeltaNet(nn.Module):  # class name must stay fixed
    """DeltaNet with EMA long-term memory and **fine-grained mix-gating**."""

    def __init__(
        self,
        *,
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
        # === NEW parameters ===
        use_ema: bool = True,
        use_head_gate: bool = True,
        head_gate_init_bias: float = -2.0,  # favour delta initially (sigmoid≈0.12)
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert qk_activation in ["silu", "relu", "elu", "identity"]
        assert qk_norm in ["l2", "sum"]

        # Hidden / derived dims --------------------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.use_beta = use_beta
        self.use_ema = use_ema
        self.use_gate = use_gate
        self.use_head_gate = use_head_gate
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0, "key dim must divide num_heads"
        assert self.value_dim % num_heads == 0, "value dim must divide num_heads"

        # Linear projections ------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # EMA-specific projections ------------------------------------------------
        self.dec_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Deprecated scalar gate (kept for checkpoint compatibility, frozen)
        self.register_parameter("ema_mix_logit", nn.Parameter(torch.zeros(1), requires_grad=False))

        # New fine-grained mix gate ----------------------------------------------
        if self.use_head_gate:
            self.mix_proj = nn.Linear(hidden_size, num_heads, bias=False)
            self.mix_bias = nn.Parameter(torch.full((num_heads,), head_gate_init_bias))
        else:
            self.mix_proj, self.mix_bias = None, None

        # Optional short convolution pre-processing ------------------------------
        if use_short_conv:
            activation = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=activation)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=activation)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
            if self.use_head_gate:
                self.mix_conv1d = ShortConvolution(num_heads, kernel_size=conv_size, activation="silu")
            else:
                self.mix_conv1d = None
        else:
            raise UserWarning("ShortConvolution is crucial; do not disable it unless absolutely necessary.")

        # Output normalisation / gating -----------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # -------------------------------------------------------------------------
    # Forward pass                                                             
    # -------------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "Only 2-D padding masks supported"

        batch_size, seq_len, _ = hidden_states.shape

        # Load cached state (if any)
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Unpadding for efficiency (same as original implementation) ------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # 1. Projections (+ short conv) ----------------------------------------
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            if self.use_head_gate:
                mix_inp, _ = self.mix_conv1d(self.mix_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens) if self.mix_conv1d is not None else (self.mix_proj(hidden_states), None)
        else:  # should never execute per constraints
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.use_head_gate:
                mix_inp = self.mix_proj(hidden_states)

        # 2. Non-linearities on q/k (+ optional normalisation) ------------------
        if self.qk_activation == "silu":
            q, k = F.silu(q), F.silu(k)
        elif self.qk_activation == "relu":
            q, k = F.relu(q), F.relu(k)
        elif self.qk_activation == "elu":
            q, k = elu_p1(q), elu_p1(k)
        # identity: no op

        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # 3. Beta gate ----------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # 4. Prepare for delta rule -------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")

        o_d, recurrent_state = delta_rule_chunkwise(q=q_d, k=k_d, v=v_d, beta=beta_d)
        o_d = rearrange(o_d, "b h l d -> b l h d")

        # 5. EMA path ----------------------------------------------------------
        if self.use_ema:
            gamma = self.dec_proj(hidden_states).sigmoid()  # (B L H)
            gamma_d = rearrange(gamma, "b l h -> b h l")
            ema_state_prev = last_state.get("ema_state") if last_state is not None else None
            v_for_ema = rearrange(v, "b l h d -> b h l d")
            ema_out, ema_state = ema_rule_chunkwise(v_for_ema, gamma_d, ema_state_prev)
            ema_out = rearrange(ema_out, "b h l d -> b l h d")
        else:
            ema_out, ema_state = None, None

        # 6. Mix gating --------------------------------------------------------
        if self.use_ema:
            if self.use_head_gate:
                # mix_inp shape: (B L H); add bias per head then sigmoid
                mix_logits = mix_inp + self.mix_bias  # broadcast bias over seq & batch
                mix = torch.sigmoid(mix_logits)  # (B L H)
            else:
                mix = torch.sigmoid(self.ema_mix_logit) * torch.ones_like(o_d[..., 0])  # broadcast scalar
            mix_e = mix.unsqueeze(-1)  # (B L H 1)
            o = (1.0 - mix_e) * o_d + mix_e * ema_out  # blend
        else:
            o = o_d

        # 7. Cache update ------------------------------------------------------
        if past_key_values is not None:
            if isinstance(past_key_values, dict):
                past_key_values["recurrent_state"] = recurrent_state
                past_key_values["conv_state"] = (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
                past_key_values["ema_state"] = ema_state if self.use_ema else None
                past_key_values["layer_idx"] = self.layer_idx
                past_key_values["offset"] = seq_len
            elif hasattr(past_key_values, "update") and isinstance(past_key_values, dict) is False:
                # Only call update() if past_key_values is not a dict.
                past_key_values.update({
                    'recurrent_state': recurrent_state,
                    'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                    'ema_state': ema_state if self.use_ema else None,
                    'layer_idx': self.layer_idx,
                    'offset': seq_len,
                })

        # 8. Output norm & proj ----------------------------------------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # 9. Re-pad if we unpadded -------------------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
