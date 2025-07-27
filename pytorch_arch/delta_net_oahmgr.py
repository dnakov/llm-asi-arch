# -*- coding: utf-8 -*-
"""
DeltaNet – Output-Aware Hybrid Memory Gated Normalised Routing (DeltaNet-OAHMGR)
=====================================================================================
A next-generation memory integration architecture synthesizing output-statistics-aware fusion, dynamic hybrid gating, Dirac+noise-initialised multi-scale FIR, per-head adaptive path exploration, and robust variance/path-starvation controls.

(This file has been patched by the automated Code Checker to fix
critical runtime shape mismatches while preserving all architectural
innovations.  The original design intent and computational efficiency
remain unchanged.)
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


def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)


# DIRAC+NOISE FIR convolution for robust path learning
class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, dirac_eps: float = 0.02):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0
        filt += dirac_eps * torch.randn_like(filt)
        self.filters = nn.Parameter(filt)

    def forward(self, x):  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


@torch.compile
def delta_rule_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 32):
    """Chunk-wise causal delta-rule path (identical to original implementation)."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_seq = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_seq) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, : i] += (attn[..., i, :, None].clone() * attn[..., :, : i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, device=q.device, dtype=attn.dtype)
    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S


if TYPE_CHECKING:
    from fla.models.utils import Cache


class DeltaNet(nn.Module):
    """DeltaNet with Output-Aware Hybrid Memory Gated Routing."""

    def __init__(
        self,
        mode: str = "oahmgr",
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
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),
        conv_residual_init: float = -2.0,
        prob_floor: float = 0.005,
        alpha_static_res: float = 0.3,  # always-on static fraction
        dirac_eps: float = 0.02,  # Noise for FIR init
        **kwargs,
    ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        self.mode = mode
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
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.prob_floor = prob_floor
        self.alpha_static_res = alpha_static_res
        self.dirac_eps = dirac_eps

        # === Dimension calculations ===
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # === Projection layers ===
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # === Short convolutional enrichment ===
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # === Multi-scale Dirac+noise FIR ===
        self.local_fir_long = DepthwiseFIRConv1d(
            num_heads=self.num_heads,
            head_dim=self.head_v_dim,
            kernel_size=fir_kernel_size_long,
            dirac_eps=dirac_eps,
        )
        self.local_fir_short = DepthwiseFIRConv1d(
            num_heads=self.num_heads,
            head_dim=self.head_v_dim,
            kernel_size=fir_kernel_size_short,
            dirac_eps=dirac_eps,
        )

        # === Dynamic residual conv path ===
        self.conv_residual_logit = nn.Parameter(torch.full((num_heads,), conv_residual_init))  # static
        self.res_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        with torch.no_grad():
            self.res_gate_proj.bias.fill_(-1.0)  # slightly negative, not severe

        # === Fusion gate (MLP) ===
        # Each _per_head_stats() produces **4** scalars per head. We later concatenate
        # stats from 4 branches, giving 16 dims for *input* or *output* stats.
        self.stat_dim = 4  # single-branch statistics dimension (mean, var, abs-mean, l2)
        fusion_gate_in_dim = hidden_size + (self.stat_dim * 4) * 2  # input+output (16 each)
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(fusion_gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias[:] = torch.tensor(gate_bias_init)

        # === Per-head softplus temperature (tau >= 0.3) ===
        self.logit_temperature = nn.Parameter(torch.full((num_heads,), gate_logit_init))

        # === Output normalisation ===
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # Helper: per-head statistics
    # ---------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)  # (..., 4)

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API compatibility
        **kwargs,
    ):
        # === Attention mask handling (unpad) ===
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        batch_size, seq_len_full, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len_full:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        seq_len = hidden_states.shape[1]

        # === Q/K/V + short conv ===
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state", None) is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q_in, conv_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k_in, conv_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v_in, conv_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        q = rearrange(q_in, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_in, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_in, "b l (h d) -> b l h d", d=self.head_v_dim)

        # === Activation / normalisation for q,k ===
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # === Beta scaling for delta path ===
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # === Global (delta-rule) path ===
        delta_out_t, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_t, "b h l d -> b l h d")

        # === Local FIRs ===
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # === Per-head statistics (INPUT) ===
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_input = torch.cat([stats_short, stats_long, stats_delta, stats_value], dim=-1)  # (..., 16)

        # === Candidate branches ===
        candidates = [local_short, local_long, delta_out, v_direct]

        # ================================================================
        # 1) Pre-fusion pass to obtain *candidate-output statistics*.
        # ------------------------------------------------------------
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (..., H, hidden)

        # Dynamically gated residual local path (static + dynamic)
        res_gate_dyn = torch.sigmoid(self.res_gate_proj(hidden_states)).clamp(min=1e-4, max=1 - 1e-4)
        static_scale = torch.sigmoid(self.conv_residual_logit)[None, None, :, None]
        conv_res_scale_combined = self.alpha_static_res + (1.0 - self.alpha_static_res) * static_scale * res_gate_dyn.unsqueeze(-1)

        # Build fusion-gate input **FOR STAT PASS**.
        # We do *not* yet have output statistics, so we pad with zeros so that the
        # dimensionality matches the full gate MLP expectation.
        zeros_stats = torch.zeros_like(stats_input)
        fusion_gate_in_stat = torch.cat([hs_exp, stats_input, zeros_stats], dim=-1)  # (..., hidden + 32)
        gate_in_flat_stat = rearrange(fusion_gate_in_stat, "b l h d -> (b l h) d")
        gate_logits_flat_stat = self.fusion_gate_mlp(gate_in_flat_stat)

        # === Temperature scaling ===
        temperature_heads = F.softplus(self.logit_temperature).clamp(min=0.3).to(gate_logits_flat_stat.dtype)
        temp = rearrange(temperature_heads, "h -> 1 1 h 1")

        fusion_logits_stat = rearrange(
            gate_logits_flat_stat,
            "(b l h) c -> b l h c",
            b=hs_exp.shape[0],
            l=hs_exp.shape[1],
            h=self.num_heads,
        )
        fusion_logits_stat = fusion_logits_stat / temp
        fusion_weights_stat = torch.softmax(fusion_logits_stat, dim=-1)
        fusion_o_stat = sum(fusion_weights_stat[..., i : i + 1] * c for i, c in enumerate(candidates))

        # === Output-aware statistics (from candidate outputs) ===
        stats_output = [self._per_head_stats(x) for x in [local_short, local_long, delta_out, v_direct, fusion_o_stat]]
        stats_output_concat = torch.cat(stats_output[:4], dim=-1)  # (..., 16) – exclude fusion_o_stat itself

        # ================================================================
        # 2) Main fusion gate (input + output stats).
        # ------------------------------------------------------------
        fusion_gate_in = torch.cat([hs_exp, stats_input, stats_output_concat], dim=-1)  # (..., hidden + 32)
        gate_in_flat = rearrange(fusion_gate_in, "b l h d -> (b l h) d")
        gate_logits_flat = self.fusion_gate_mlp(gate_in_flat)

        fusion_logits = rearrange(
            gate_logits_flat,
            "(b l h) c -> b l h c",
            b=hs_exp.shape[0],
            l=hs_exp.shape[1],
            h=self.num_heads,
        )
        fusion_logits = fusion_logits / temp
        fusion_weights = torch.softmax(fusion_logits, dim=-1)

        # === Epsilon floor ===
        if self.prob_floor > 0.0:
            fusion_weights = torch.clamp(fusion_weights, min=self.prob_floor)
            fusion_weights_sum = fusion_weights.sum(-1, keepdim=True).clamp(min=4 * self.prob_floor + 1e-6)
            fusion_weights = fusion_weights / fusion_weights_sum

        o = sum(fusion_weights[..., i : i + 1] * c for i, c in enumerate(candidates))

        # === Add hybrid always-on residual local path ===
        o = o + conv_res_scale_combined * local_short

        # === Cache update ===
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # === Output projection / (gated) normalisation ===
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # === Re-pad if we had removed padding ===
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len_full)

        return o, None, past_key_values
