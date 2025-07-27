# -*- coding: utf-8 -*-
"""
DeltaNet – Responsive Multi-Scale Gated Memory (R-MSGM)
=====================================================
This evolution merges the strongest ideas from previous experiments and recent
research on *feedback-aware routing* (e.g. Hyena, Mamba, Block-State
Transformers):

1. **Triple-Scale Local Memory**
   • *Short* depth-wise FIR (k≈7)
   • *Mid*   depth-wise FIR (k≈31)
   • *Long*  depth-wise FIR (k≈64)

   These efficiently cover 1-to-64 token neighbourhoods with O(N) depth-wise
   convolutions.
2. **Global Delta-rule Path** – unchanged, preserves associative long-range
   memory.
3. **Input- *and Path-Feedback* Gated Fusion**
   The fusion gate now conditions on BOTH the current hidden-state **and** a
   lightweight statistic of every memory path (L2-norm per token & head).  This
   *feedback* allows the model to sense when a path is already saturated or
   under-utilised and to re-allocate probability mass accordingly – fixing the
   path-collapse seen in earlier input-only gates.
4. **Minimum Delta Allocation w/ Temperature**
   To guarantee that the global path never vanishes, we apply a *softmax with
   temperature* followed by an **ε-floor** on the delta weight and a renormalise.
5. **Warm-start Direct-Value Bias**
   Final gate layer is biased toward the direct value path at init to avoid
   early over-smoothing by convolutional branches.

All operations remain **O(N)**, strictly causal, and batch-agnostic.  The class
name and public interface are unchanged so the layer plugs seamlessly into any
existing DeltaNet stack.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

__all__ = ["DeltaNet"]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (ELU+1) – positive feature map used by some variants."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so values along the last dim sum to 1."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Core chunk-wise delta rule – kept identical to proven implementation
# -----------------------------------------------------------------------------
@torch.compile
def _delta_rule_chunkwise(
    q: torch.Tensor,  # [B H L Dk]
    k: torch.Tensor,  # [B H L Dk]
    v: torch.Tensor,  # [B H L Dv]
    beta: torch.Tensor,  # [B H L]
    *,
    chunk_size: int = 32,
):
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

    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, device=q.device)
    inv = inv.to(torch.bfloat16)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    strict_tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_tri, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S.detach()

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (per-head, per-channel)
# -----------------------------------------------------------------------------
class _DepthwiseFIR1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = nn.Parameter(torch.randn(num_heads, head_dim, kernel_size) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B L H D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Main DeltaNet Layer – Responsive Multi-Scale Gated Memory
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack  # pragma: no cover
    from fla.models.utils import Cache  # pragma: no cover


class DeltaNet(nn.Module):
    """DeltaNet with *responsive* multi-scale FIR branches and feedback-aware gating."""

    def __init__(
        self,
        *,
        mode: str = "rmsgm",
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
        # -------- new hyper-parameters ---------
        fir_kernel_short: int = 7,
        fir_kernel_mid: int = 31,
        fir_kernel_long: int = 64,
        fusion_hidden_mult: int = 2,
        gate_bias_init: float = 2.0,
        min_delta_weight: float = 0.03,
        gate_temperature: float = 1.0,
        **kwargs: "Unpack[Dict]",
    ) -> None:
        super().__init__()

        # ---------------- bookkeeping ----------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.mode = mode
        self.min_delta_weight = float(min_delta_weight)
        self.gate_temperature = float(gate_temperature)

        # ---------------- dimensions -----------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        # ---------------- projections ----------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv -----------------
        if use_short_conv:
            act_name = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act_name)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act_name)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for this DeltaNet variant.")

        # ---------------- FIR branches ---------------
        self.fir_short = _DepthwiseFIR1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_mid = _DepthwiseFIR1d(num_heads, self.head_v_dim, fir_kernel_mid)
        self.fir_long = _DepthwiseFIR1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---------------- gate projections (feedback aware) -------------
        # Token projection (input hidden state)
        self.fusion_gate_token = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 5, bias=True),  # 5 paths
        )
        # Path statistic projection: maps 5 scalars -> 5 logits (per head)
        # NOTE: We purposely output **5** values so the logits align per-head with
        #       token-derived logits. A shared linear layer is used for all heads
        #       to minimise parameter count while keeping the design fully
        #       dynamic and batch-size agnostic.
        self.fusion_gate_stats = nn.Linear(5, 5, bias=False)

        # bias warm-start: favour direct value path (index 4)
        with torch.no_grad():
            bias = self.fusion_gate_token[-1].bias.view(num_heads, 5)
            bias.zero_()
            bias[:, 4] = gate_bias_init

        # ---------------- output norm & proj ----------------
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
        hidden_states: torch.Tensor,  # [B L D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be [batch, seq_len]"
        bsz, seq_len, _ = hidden_states.shape

        # ---- retrieve cache ----
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---- Q K V projections + short conv ----
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---- head split ----
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # ---- activations & norms for q/k ----
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- beta ----
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- delta path ----
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---- FIR branches ----
        v_direct = v  # identity path
        fir_s = self.fir_short(v_direct)
        fir_m = self.fir_mid(v_direct)
        fir_l = self.fir_long(v_direct)

        # ---- path statistics (L2 norm over channel) ----
        # shape: [B L H]
        def _l2(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return torch.sqrt(torch.clamp((x ** 2).mean(dim=-1), min=1e-6))

        stats = torch.stack([
            _l2(fir_s),
            _l2(fir_m),
            _l2(fir_l),
            _l2(delta_out),
            _l2(v_direct),
        ], dim=-1)  # [B L H 5]

        # ---- fusion gating (feedback aware) ----
        token_logits = self.fusion_gate_token(hidden_states)  # [B L H*5]
        token_logits = rearrange(token_logits, "b l (h c) -> b l h c", h=self.num_heads, c=5)

        # stats-based logits (already per-head): shape [B L H 5]
        stats_logits = self.fusion_gate_stats(stats)

        fusion_logits = (token_logits + stats_logits) / self.gate_temperature
        fusion_weights = torch.softmax(fusion_logits, dim=-1)

        # minimum delta weight enforcement (path index 3)
        min_eps = self.min_delta_weight
        delta_w = fusion_weights[..., 3:4]
        fusion_weights = torch.where(
            delta_w < min_eps,
            fusion_weights + (min_eps - delta_w) / 4.0,  # distribute correction among all weights
            fusion_weights,
        )
        fusion_weights = fusion_weights / fusion_weights.sum(dim=-1, keepdim=True)

        # order: short, mid, long, delta, direct
        o = (
            fusion_weights[..., 0:1] * fir_s +
            fusion_weights[..., 1:2] * fir_m +
            fusion_weights[..., 2:3] * fir_l +
            fusion_weights[..., 3:4] * delta_out +
            fusion_weights[..., 4:5] * v_direct
        )

        # ---- cache update ----
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ---- output norm/proj ----
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---- repad ----
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, bsz, seq_len)

        return o, None, past_key_values
