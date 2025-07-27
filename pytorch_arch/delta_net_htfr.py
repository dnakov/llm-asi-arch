# -*- coding: utf-8 -*-
"""
DeltaNet – Hierarchical Temperature-and-Floor Regularised Gating (DeltaNet-HTFR)
================================================================================
This evolution unifies the strongest empirical findings from previous DeltaNet
variants into a *single* architecture that simultaneously:

1.  Maintains *dual-scale* causal FIR convolutions for rich local context
    modelling (short + long kernels, **identity-initialised** with small noise).
2.  Integrates a *global* recurrent **Δ-rule** path for unlimited context
    propagation while preserving **O(N)** complexity via chunkwise scan.
3.  Employs a **three-way hierarchical fusion gate** with *learnable per-head
    temperature* **and** a small **ε-floor** at **all stages** to prevent early
    collapse and gradient starvation.
4.  Adds an always-on **entropy regularisation loss** that discourages overly
    sharp gating distributions and promotes balanced path utilisation.

The class name and public interface remain **DeltaNet**; all changes are
internal and enabled by default, ensuring seamless drop-in compatibility.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

################################################################################
# Helper utilities                                                             #
################################################################################

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # shifted ELU keeps >0
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # row-sum normalisation
    return (x / x.sum(-1, keepdim=True)).to(x)

################################################################################
# Core chunk-wise Δ-rule implementation (unchanged, O(N·d))                    #
################################################################################

@torch.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,D_k)
    k: torch.Tensor,  # (B,H,L,D_k)
    v: torch.Tensor,  # (B,H,L,D_v)
    beta: torch.Tensor,  # (B,H,L)
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

    # Normalisations ----------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into chunks ----------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)
    inv = inv.to(torch.bfloat16)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    excl_mask = torch.triu(tri_mask, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(excl_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

################################################################################
# Depth-wise causal FIR convolution                                            #
################################################################################

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal 1-D FIR convolution with identity init."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_size: int,
        noise_std: float = 1e-2,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # causal identity (current timestep)
        if noise_std > 0:
            filt.add_(torch.randn_like(filt) * noise_std)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

################################################################################
# Optional typing imports -----------------------------------------------------#
################################################################################
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache

################################################################################
# Main DeltaNet class                                                          #
################################################################################

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with hierarchical temperature- & ε-floor regularised gating."""

    def __init__(
        self,
        # ===== baseline args =====
        mode: str = "htfr",  # hierarchical temperature-floor regularised
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
        # ===== new hyper-parameters =====
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 64,
        fusion_hidden_mult: int = 2,
        gate_epsilon: float = 0.05,  # ε-floor for *all* gates
        gate_temp_init: float = 1.0,  # initial temperature (per-head, log-space param)
        entropy_reg_weight: float = 0.01,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        # ---------------- basic bookkeeping ----------------
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.gate_eps = gate_epsilon
        self.entropy_reg_weight = entropy_reg_weight

        # ---------------- dimensions ---------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---------------- projections --------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv --------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet-HTFR.")

        # ---------------- FIR branches -------------
        self.local_fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_short_kernel)
        self.local_fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_long_kernel)

        # ---------------- hierarchical gates -------
        fused_in_dim = hidden_size + self.head_v_dim * num_heads * 4  # hidden + all path outputs
        self.stage1_mlp = nn.Sequential(
            nn.Linear(fused_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 2, bias=True),
        )
        local_in_dim = hidden_size + self.head_v_dim * num_heads * 2
        self.stage2_local_mlp = nn.Sequential(
            nn.Linear(local_in_dim, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * 2, bias=True),
        )
        global_in_dim = hidden_size + self.head_v_dim * num_heads * 2
        self.stage2_global_mlp = nn.Sequential(
            nn.Linear(global_in_dim, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads * 2, bias=True),
        )

        # Warm-start bias favouring *direct value* path (index 1 of global gate)
        with torch.no_grad():
            if self.stage2_global_mlp[-1].bias is not None:
                self.stage2_global_mlp[-1].bias.zero_()
                self.stage2_global_mlp[-1].bias[num_heads:] = 4.0  # direct value branch bias

        # Per-head temperatures (log-param) – shared across all gates
        log_temp = math.log(gate_temp_init)
        self.log_temp_stage1 = nn.Parameter(torch.full((num_heads, 1), log_temp))
        self.log_temp_stage2_local = nn.Parameter(torch.full((num_heads, 1), log_temp))
        self.log_temp_stage2_global = nn.Parameter(torch.full((num_heads, 1), log_temp))

        # ---------------- output norm/proj ----------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    ############################################################################
    # forward                                                                  #
    ############################################################################

    # pylint: disable=too-many-statements,too-many-branches,too-many-locals
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compat
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be (batch, seq_len)"

        B_orig, L_orig, _ = hidden_states.shape

        # ------------- unpadding for variable length -------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ------------- linear projections + short conv -----------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ------------- head split & activations -------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        v_direct = v  # identity/value path

        # ------------- β for Δ-rule -------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------- Δ-rule path --------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d, chunk_size=32)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ------------- FIR branches -------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ------------- Hierarchical gating ------------------------
        # Stage-1: local (short+long) vs global (delta+direct)
        stage1_in = torch.cat([
            hidden_states,
            rearrange(local_short, "b l h d -> b l (h d)"),
            rearrange(local_long, "b l h d -> b l (h d)"),
            rearrange(delta_out, "b l h d -> b l (h d)"),
            rearrange(v_direct, "b l h d -> b l (h d)"),
        ], dim=-1)
        logits1 = self.stage1_mlp(stage1_in)  # (B,L,H*2)
        logits1 = rearrange(logits1, "b l (h s) -> b l h s", h=self.num_heads, s=2)
        temp1 = torch.exp(self.log_temp_stage1).unsqueeze(0).unsqueeze(0)  # (1,1,H,1)
        w1 = torch.softmax(logits1 * temp1, dim=-1)
        w1 = w1 * (1.0 - 2 * self.gate_eps) + self.gate_eps  # ε-floor

        # Stage-2 local: short vs long
        stage2_local_in = torch.cat([
            hidden_states,
            rearrange(local_short, "b l h d -> b l (h d)"),
            rearrange(local_long, "b l h d -> b l (h d)"),
        ], dim=-1)
        logits2l = self.stage2_local_mlp(stage2_local_in)
        logits2l = rearrange(logits2l, "b l (h s) -> b l h s", h=self.num_heads, s=2)
        temp2l = torch.exp(self.log_temp_stage2_local).unsqueeze(0).unsqueeze(0)
        w2l = torch.softmax(logits2l * temp2l, dim=-1)
        w2l = w2l * (1.0 - 2 * self.gate_eps) + self.gate_eps

        # Stage-2 global: delta vs direct
        stage2_global_in = torch.cat([
            hidden_states,
            rearrange(delta_out, "b l h d -> b l (h d)"),
            rearrange(v_direct, "b l h d -> b l (h d)"),
        ], dim=-1)
        logits2g = self.stage2_global_mlp(stage2_global_in)
        logits2g = rearrange(logits2g, "b l (h s) -> b l h s", h=self.num_heads, s=2)
        temp2g = torch.exp(self.log_temp_stage2_global).unsqueeze(0).unsqueeze(0)
        w2g = torch.softmax(logits2g * temp2g, dim=-1)
        w2g = w2g * (1.0 - 2 * self.gate_eps) + self.gate_eps

        # Compose outputs --------------------------------------------------
        local_comb = w2l[..., 0:1] * local_short + w2l[..., 1:2] * local_long
        global_comb = w2g[..., 0:1] * delta_out + w2g[..., 1:2] * v_direct
        out = w1[..., 0:1] * local_comb + w1[..., 1:2] * global_comb

        # ------------- cache update --------------------------------------
        if use_cache and past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state_new,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_orig,
            )

        # ------------- output norm & projection ---------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # ------------- repad if necessary --------------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, B_orig, L_orig)

        # ------------- entropy regularisation ----------------------------
        # Compute average negative entropy across all gates
        probs = torch.cat([w1.flatten(-2), w2l.flatten(-2), w2g.flatten(-2)], dim=-1)  # (..., H*2*3)
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        reg_loss = self.entropy_reg_weight * (-entropy)  # maximise entropy => minimise negative

        return out, reg_loss, past_key_values
