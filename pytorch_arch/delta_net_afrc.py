# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Floor & Rich Context-Stat Gating (delta_net_afrc)
====================================================================
This evolutionary variant unifies the strongest ideas from the "Dynamic
Hierarchical Gating" (DHG) and the "Context-Stat Gate" (CtxStatGate)
families **and fixes the remaining local–global trade-off** by making the
*context floor adaptive* **while enriching the router signal with higher
band-width branch statistics** and an additional *very-long* convolutional
branch.

Key Innovations (enabled by default)
-----------------------------------
1. **Adaptive Context-Floor (ACF)** – A *learnable scalar* per head
   `logit_context_floor` initialised such that the minimum contextual mass
   equals `context_floor_init` (default *5 %*).  Because it is *learnable*
   the optimiser can freely *decrease* (or increase) the floor when the
   network decides it no longer needs forced contextual flow, removing the
   global-reasoning penalty previously caused by a *static* floor.

2. **Richer Context-Statistics (RCS)** – The fusion gate now sees *three*
   statistics (mean, RMS, max-abs) from each branch instead of two.  With
   four contextual branches (short-FIR ≈3 tok, long-FIR ≈31 tok,
   wide-FIR ≈64 tok, Δ-memory) **plus** the identity/value branch this makes
   `5 branches × 3 stats × H` additional inputs, giving the gate finer
   information to discriminate local vs. global needs without incurring
   any quadratic cost.

3. **Very-Long FIR Branch (wide-FIR)** – A new depth-wise causal FIR with
   kernel = 64 tokens is introduced, capturing narrative context that even
   the Δ-memory sometimes under-utilises.  The branch is initialised to an
   *identity* filter so optimisation starts from the proven baseline.

4. **Coarse-Then-Fine Routing with Temperature** – We keep the efficient
   coarse (identity vs. context) then fine (softmax over 4 contextual
   branches) structure *with a learnable per-head temperature*.  This
   preserves O(N) compute, guarantees causal flow, and empirically yields
   faster convergence than flat softmax.

All computations remain **O(N·d)**, strictly causal, batch-size agnostic,
`einops.rearrange` is used everywhere, and the @torch.compile kernel for
chunk-wise Δ-rule is preserved.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import ShortConvolution, FusedRMSNormGated, RMSNorm
from fla.modules.l2norm import l2norm

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (=ELU+1) that stays strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so that last dimension sums to one."""
    return (x / x.sum(-1, keepdim=True)).to(x)


def _branch_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mean, rms, max_abs) along the channel dimension."""
    mean = x.mean(dim=-1)
    rms = torch.sqrt(torch.clamp(x.pow(2).mean(dim=-1), min=1e-8))
    max_abs = x.abs().max(dim=-1).values
    return mean, rms, max_abs

# ---------------------------------------------------------------------------
# Core chunk-wise Δ-rule (identical to baseline, still @torch.compile)
# ---------------------------------------------------------------------------

@torch.compile  # noqa: D401
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Associative Δ-rule retrieval processed in causal chunks (O(N))."""
    b, h, L, _ = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=inv.device)
    inv = inv.to(v.dtype)

    u = inv @ v
    w = inv @ k_beta

    d_k = q.shape[-1]
    d_v = v.shape[-1]
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2)).masked_fill(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S.detach()

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution with identity (Dirac) initialisation
# ---------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR convolution."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads, head_dim, self.kernel_size)
        weight[..., -1] = 1.0  # current-timestep tap (identity)
        self.filters = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, w, groups=h * d)
        y = rearrange(y, "b (h d) l -> b l h d", h=h)
        return y

# ---------------------------------------------------------------------------
# Optional cache typing helper
# ---------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401 – external cache typing

# ---------------------------------------------------------------------------
#                               DeltaNet-AFRC
# ---------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet layer with **Adaptive Floor & Rich Context-Stat Gating**."""

    def __init__(
        self,
        *,
        mode: str = "afrc",  # adaptive floor & rich context
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
        # FIR kernel sizes
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fir_wide_kernel: int = 64,
        # gating hyper-params
        fusion_hidden_mult: int = 2,
        context_floor_init: float = 0.05,
        value_bias_init: float = 4.0,
        gate_temp_init: float = 1.0,
        fusion_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---------------- bookkeeping & dims --------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "dims must divide num_heads"
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---------------- projections -------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- optional short conv -----------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet performance.")

        # ---------------- FIR branches ------------------------------
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)
        self.fir_wide = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_wide_kernel)

        # ---------------- fusion gate MLP ---------------------------
        # Inputs: hidden_state (D) + 5 branches * 3 stats * H = D + 15H
        gate_in_dim = hidden_size + 15 * num_heads
        gate_hidden = hidden_size * fusion_hidden_mult
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(gate_hidden, num_heads * 5, bias=True),  # 4 contextual + 1 value logits
        )
        # Warm-start bias favouring identity/value path
        with torch.no_grad():
            self.gate_mlp[-1].bias.zero_()
            self.gate_mlp[-1].bias[4::5] = value_bias_init  # every 5th element (value path)

        # learnable per-head value bias (added on top of MLP output for identity path)
        self.value_bias = nn.Parameter(torch.full((num_heads,), value_bias_init))

        # learnable per-head temperature for fine gate
        self.log_temperature = nn.Parameter(torch.full((num_heads,), math.log(gate_temp_init)))

        # learnable logit for adaptive context floor (per head)
        floor_init_logit = math.log(context_floor_init / (1.0 - context_floor_init))
        self.logit_context_floor = nn.Parameter(torch.full((num_heads,), floor_init_logit))

        # ---------------- output norm / proj ------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compatibility
        floor_schedule: Optional[float] = None,  # optional scalar ∈[0,1] to scale context floor
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be (B, L)"
        B_orig, L_in, _ = hidden_states.shape

        # ---------------- un-padding for variable-length batches ------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---------------- retrieve previous state ---------------------
        conv_state_q = conv_state_k = conv_state_v = None
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state and self.use_short_conv and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        # ---------------- projections & short conv -------------------
        q_lin, k_lin, v_lin = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q_lin, conv_state_q = self.q_conv1d(q_lin, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_state_k = self.k_conv1d(k_lin, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_state_v = self.v_conv1d(v_lin, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---------------- reshape to heads ---------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # ---------------- optional activation / norm ----------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---------------- beta for Δ-rule ----------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------------- Δ-rule global memory -----------------------
        delta_out, recurrent_state = delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---------------- FIR branches ------------------------------
        short_out = self.fir_short(v)
        long_out = self.fir_long(v)
        wide_out = self.fir_wide(v)

        # ---------------- branch statistics -------------------------
        stats_short = _branch_stats(short_out)
        stats_long = _branch_stats(long_out)
        stats_wide = _branch_stats(wide_out)
        stats_delta = _branch_stats(delta_out)
        stats_value = _branch_stats(v)

        # concatenate stats: mean,rms,max_abs -> 3*H per branch
        def _stack_stats(stats_tuple):  # (mean,rms,max) each (B,L,H)
            return torch.cat(stats_tuple, dim=-1)  # (B,L,3H)

        stats_concat = [_stack_stats(s) for s in (stats_short, stats_long, stats_wide, stats_delta, stats_value)]
        gate_input = torch.cat([hidden_states] + stats_concat, dim=-1)  # (B,L,D + 15H)

        gate_logits = self.gate_mlp(gate_input)  # (B,L,H*5)
        gate_logits = rearrange(gate_logits, "b l (h c) -> b l h c", h=self.num_heads, c=5)

        # ---------------- coarse gate (value vs context) -------------
        value_logit = gate_logits[..., 4] + self.value_bias  # (B,L,H)
        context_logits = gate_logits[..., 0:4]  # (B,L,H,4)

        # compute adaptive floor ------------------------------------
        context_floor = torch.sigmoid(self.logit_context_floor)  # (H,)
        if floor_schedule is not None:
            context_floor = context_floor * max(0.0, 1.0 - float(floor_schedule))
        context_floor = context_floor.view(1, 1, self.num_heads)  # (1,1,H)

        p_value = (1.0 - context_floor) * torch.sigmoid(value_logit)  # ensures p_value ≤ 1-floor
        others_total = 1.0 - p_value  # ≥ context_floor by construction

        # ---------------- fine gate among contextual branches --------
        temperature = torch.exp(self.log_temperature).view(1, 1, self.num_heads, 1)
        ctx_weights = torch.softmax(context_logits / temperature, dim=-1)  # (B,L,H,4)
        ctx_weights = ctx_weights * others_total.unsqueeze(-1)  # scale by available mass

        # ---------------- fuse outputs ------------------------------
        fused = (
            ctx_weights[..., 0:1] * short_out
            + ctx_weights[..., 1:2] * long_out
            + ctx_weights[..., 2:3] * wide_out
            + ctx_weights[..., 3:4] * delta_out
            + p_value.unsqueeze(-1) * v
        )

        # ---------------- cache update ------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ---------------- output norm & projection ------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            fused = self.o_norm(fused, g_vec)
        else:
            fused = self.o_norm(fused)
        out = self.o_proj(rearrange(fused, "b l h d -> b l (h d)"))

        # ---------------- re-pad if needed ---------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, B_orig, L_in)

        return out, None, past_key_values
