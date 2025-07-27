# -*- coding: utf-8 -*-
"""
DeltaNet – Synergetic Local–Global Fusion (delta_net_syngf)
===========================================================
This evolutionary **DeltaNet** variant fuses the most successful ideas of the
AFT → CAGF → DynFuse line while explicitly fixing the residual shortcomings
identified in their evaluation:

1. Per-Head *Statistics-Aware* Gating (CAGF strength)
   • Each head receives a 16-dim vector describing every branch
     (mean/var/abs-mean/ℓ2) enabling informed routing and avoiding premature
     path collapse.

2. *Temperature-Bound* Softmax  (prevents over-sharpening)
   • A learnable temperature τₕ is constrained to **τ ≥ 0.5** using
     `τ = 0.5 + softplus(·)`, guaranteeing minimum entropy that protected
     BoolQ & SWDE in prior studies, while still allowing sharpening.

3. *Partial* Decaying Local Floor  (DynFuse lesson)
   • Minimum probability ε(t) for the two convolutional (local) paths decays
     **exponentially** from `floor_init = 0.05` to a *non-zero* `floor_final =
     0.02`.  This preserves a thin but essential local capacity at convergence,
     solving the late-stage lexical regression seen when ε→0.

4. *Adaptive Residual Bypass*  (new)
   • A per-head learnable residual αₕ (init 0.1) is **scaled online** by the
     *current* lack of local allocation:

         ᾱ₍b,l,h₎ = αₕ · (1 − w_local_total)

     so residual leakage is high only when the gate under-allocates local
     paths, reducing output blur once the gate learns to exploit them.

5. Stronger Entropy Regulariser
   • Coefficient raised to 0.05 to further guard against path collapse during
     the long decay window.

The implementation retains strict **O(N)** complexity, uses chunk-wise Δ-rule
kernels, respect all API/signature constraints, and is fully batch-agnostic via
`einops.rearrange`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (+1) keeping outputs positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise last dimension to L1-sum == 1."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity initialisation)
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity weight on current step
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule (identical maths; kept @torch.compile for speed)
# -----------------------------------------------------------------------------


@torch.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
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
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=inv.device)
    inv = inv.to(torch.bfloat16)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    tri_strict = torch.triu(tri, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        att_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + att_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S


# -----------------------------------------------------------------------------
# Optional typing helper
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore


# -----------------------------------------------------------------------------
# Main **DeltaNet** implementation – Synergetic Fusion
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):  # noqa: D401
    """DeltaNet layer with statistics-aware gate, bounded temperature, partial
    decaying floor, and adaptive residual bypass (identifier: *syngf*)."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        # --------------------- core API ---------------------
        mode: str = "syngf",
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
        # --------------------- FIR kernels ------------------
        fir_kernel_short: int = 5,
        fir_kernel_long: int = 64,
        # --------------------- gating -----------------------
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        # bounded temperature init (τ≈1)
        gate_log_temp_init: float = math.log(math.expm1(0.5)),
        # partial floor schedule
        floor_init: float = 0.05,
        floor_final: float = 0.02,
        floor_decay: float = 10_000.0,
        # residual bypass
        conv_residual_init: float = 0.1,
        # entropy reg
        entropy_target: float = 1.1,
        entropy_coeff: float = 0.05,
        **kwargs: Dict,
    ) -> None:  # noqa: D401
        super().__init__()

        # ----------- basic setup ----------------------------
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

        # ----------- dims -----------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads")

        # ----------- projections ----------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ----------- short convs ----------------------------
        if not use_short_conv:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        # ----------- FIR branches ---------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ----------- gating network -------------------------
        self.stat_dim = 16  # mean/var/abs-mean/l2 × 4 branches
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),  # logits per path
        )
        with torch.no_grad():
            self.fusion_gate_mlp[-1].bias[:] = torch.tensor(gate_bias_init)

        # per-head log-temperature (ensure τ>=0.5)
        self.log_temp = nn.Parameter(torch.full((num_heads,), gate_log_temp_init))

        # per-head residual bypass parameter (sigmoid)
        self.conv_residual_logit = nn.Parameter(torch.full((num_heads,), math.log(conv_residual_init / (1 - conv_residual_init))))

        # ----------- output norm / proj ---------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ----------- floor schedule & entropy ---------------
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay = float(floor_decay)
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)
        self.reg_loss: Optional[torch.Tensor] = None

    # -------------------------------------------------------
    # helpers
    # -------------------------------------------------------
    @staticmethod
    def _per_head_stats(x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D) -> (B,L,H,4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    def _current_floor(self) -> float:
        t = float(self._step.item())
        return self.floor_final + (self.floor_init - self.floor_final) * math.exp(-t / self.floor_decay)

    # -------------------------------------------------------
    # forward
    # -------------------------------------------------------
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, L_in, _ = hidden_states.shape

        # optional unpadding for variable-length sequences
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # retrieve previous conv state
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # projections + short conv
        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # head reshape
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        # activations / norms on Q,K
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # β for Δ-rule
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule path
        delta_out_d, recur_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # local FIR paths
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # stats for gating
        stats = torch.cat([
            self._per_head_stats(local_short),
            self._per_head_stats(local_long),
            self._per_head_stats(delta_out),
            self._per_head_stats(v_direct),
        ], dim=-1)  # (B,L,H,16)
        hs_exp = hidden_states.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)
        gate_in = torch.cat([hs_exp, stats], dim=-1)  # (B,L,H,D+16)
        gate_in_flat = rearrange(gate_in, "b l h d -> (b l h) d")
        logits_flat = self.fusion_gate_mlp(gate_in_flat)
        logits = rearrange(logits_flat, "(b l h) p -> b l h p", b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads)

        # temperature scaling (τ>=0.5)
        temp = 0.5 + F.softplus(self.log_temp)  # (H,)
        logits = logits / temp.view(1, 1, -1, 1)

        fusion_weights = torch.softmax(logits, dim=-1)  # (B,L,H,4)

        # partial decaying local floor
        eps_now = self._current_floor()
        if eps_now > 0:
            scale = 1.0 - 2 * eps_now
            fusion_weights = fusion_weights * scale
            fusion_weights[..., 0] = fusion_weights[..., 0] + eps_now  # short
            fusion_weights[..., 1] = fusion_weights[..., 1] + eps_now  # long
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)

        # entropy regularisation
        entropy = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(-1).mean()
        self.reg_loss = self.entropy_coeff * torch.relu(self.entropy_target - entropy)

        # weighted fusion
        o = (
            fusion_weights[..., 0:1] * local_short +
            fusion_weights[..., 1:2] * local_long +
            fusion_weights[..., 2:3] * delta_out +
            fusion_weights[..., 3:4] * v_direct
        )

        # adaptive residual bypass
        alpha = torch.sigmoid(self.conv_residual_logit).view(1, 1, self.num_heads, 1)  # (1,1,H,1)
        local_total = fusion_weights[..., 0:2].sum(-1, keepdim=True)  # (B,L,H,1)
        alpha_scaled = alpha * (1.0 - local_total)
        o = o + alpha_scaled * 0.5 * (local_short + local_long)

        # cache update
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # output norm / proj
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # re-pad if needed
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B_orig, L_in)

        # step++
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
