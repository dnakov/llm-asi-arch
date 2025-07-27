# -*- coding: utf-8 -*-
"""
DeltaNet – Content-Aware Gated Fusion **with Fixed Minimum-Floor** (CAGF-MF)
============================================================================
Identifier: delta_net_cagf_mf

This version contains a **bug-fix** for the masking logic when padded batches
are converted into a single un-padded sequence.  The original implementation
concatenated all *valid* tokens across the batch dimension and then applied the
causal Δ-rule **without re-segmenting the sequences**.  Consequently, tokens of
later samples could attend to (and receive gradients from) earlier samples –
a form of *cross-batch information leakage*.

To preserve strict per-sample causality **and** batch-size independence we now
keep the standard padded `[B,L,D]` representation throughout the forward path
(Δ-rule and FIR convolutions).  Unpadding is therefore no longer necessary and
has been removed.  The change is minimal and retains all architectural
innovations while guaranteeing correctness.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data  # kept for API-compat (not used)
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # shifted ELU keeps >0
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalisation along last dimension."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity init) ----------------------------
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise 1-D causal FIR convolution with identity init."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filters = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            filters[..., -1] = 1.0  # identity (current timestep)
        self.filters = nn.Parameter(filters)  # (H, D, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged, still @torch.compile) -------------------
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
# pylint: disable=too-many-locals, too-many-statements

def _delta_rule_chunkwise(
    q: torch.Tensor,  # [B,H,L,Dk]
    k: torch.Tensor,  # [B,H,L,Dk]
    v: torch.Tensor,  # [B,H,L,Dv]
    beta: torch.Tensor,  # [B,H,L]
    *,
    chunk_size: int = 32,
):
    """Efficient O(N) associative Δ-rule using chunked causal computation."""
    b, h, L, d_k = q.shape

    # --- optional padding so that L % chunk_size == 0 -----------------------
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # --- normalisation & gating -------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # --- chunk reshape -----------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, : i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, : i].clone()
        ).sum(-2)
    eye = torch.eye(chunk_size, dtype=attn_inv.dtype, device=attn_inv.device)
    attn_inv = attn_inv + eye

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    strict_tri = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_tri, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Optional typing helpers ------------------------------------------------------
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

# -----------------------------------------------------------------------------
# Main DeltaNet layer – CAGF with Minimum-Floor -------------------------------
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet layer with Content-Aware Gated Fusion **and fixed min-floor**."""

    def __init__(
        self,
        *,
        mode: str = "cagf_mf",
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
        # --- FIR kernel sizes ---------------------------------------------
        fir_kernel_size_short: int = 5,
        fir_kernel_size_long: int = 64,
        # --- Gate network --------------------------------------------------
        fusion_hidden_mult: int = 2,
        base_floor: float = 0.05,
        # temperature init for per-head scaling (τ ≈ 1.0)
        gate_log_temp_init: float = 0.0,
        # path-specific bias init (short, long, delta, value)
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 0.5, 1.5),
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        if d_model is not None:
            hidden_size = d_model

        # ------------------- basic bookkeeping ----------------------------
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.base_floor = float(base_floor)
        assert 0.0 < self.base_floor < 0.25, "base_floor must be in (0, 0.25)"

        # ------------------- dimensions -----------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("key/value dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ------------------- projections ----------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------------- optional short conv --------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ------------------- FIR convolutions -----------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)

        # ------------------- Gate MLP -------------------------------------
        # Stats: mean, var, abs-mean, L2 for 4 branches = 16 dims
        self.stat_dim = 16
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias[:] = torch.tensor(gate_bias_init)

        # per-head temperature (learnable, positive)
        self.log_temp = nn.Parameter(gate_log_temp_init * torch.ones(num_heads, 1))

        # ------------------- Output normalisation / projection ------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Per-head statistics helper (mean, var, abs-mean, l2) --------------
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)  # [...,4]

    # ------------------------------------------------------------------
    # forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compatibility
        **kwargs,
    ):
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
            # We *keep* the padded representation to avoid cross-sample leakage.

        B, L_in, _ = hidden_states.shape

        # ------------- retrieve cache -----------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_q = conv_k = conv_v = None
        if self.use_short_conv and last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # We deliberately pass `cu_seqlens=None` (padded path) to maintain
        # one-to-one correspondence between batch samples and their sequences.
        cu_seqlens = None

        # ------------- projections + short conv -------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ------------- head reshape ------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ------------- activations / norms ------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ------------- beta --------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------- Δ-rule global path -------------------------------
        delta_out_d, recurrent_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # ------------- FIR local paths ----------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------- per-head statistics ------------------------------
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = torch.cat([stats_short, stats_long, stats_delta, stats_value], dim=-1)  # [B,L,H,16]

        # ------------- gate input & logits ------------------------------
        hs_exp = hidden_states.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [B,L,H,D]
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)
        gate_logits_flat = self.fusion_gate_mlp(rearrange(gate_in, "b l h d -> (b l h) d"))
        gate_logits = rearrange(
            gate_logits_flat,
            "(b l h) c -> b l h c",
            b=gate_in.shape[0],
            l=gate_in.shape[1],
            h=self.num_heads,
        )  # [B,L,H,4]

        # temperature scaling -------------------------------------------
        temp = torch.exp(self.log_temp).clamp(0.1, 10.0)  # [H,1]
        gate_logits = gate_logits / temp.view(1, 1, self.num_heads, 1)

        soft_w = torch.softmax(gate_logits, dim=-1)  # [B,L,H,4]

        # ------------- fixed minimum floor ------------------------------
        eps = self.base_floor
        fusion_weights = eps + (1.0 - 4.0 * eps) * soft_w  # convex, ≥ eps

        # ------------- fuse branches -----------------------------------
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # ------------- cache update ------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ------------- output norm & projection -------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # No re-padding necessary – we never un-padded.
        return o, None, past_key_values
