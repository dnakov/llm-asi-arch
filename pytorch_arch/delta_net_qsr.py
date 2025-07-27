# -*- coding: utf-8 -*-
"""
DeltaNet – Query & Summary Routing (DeltaNet-QSR)
================================================
Identifier: *delta_net_qsr*

This evolution introduces a *parameter-efficient* **query-and-summary gate**
that replaces the expensive flatten-everything approach of OAGMS.  Each
candidate memory stream is represented by a *single scalar per head and
position* (mean over channel) so the router input size is reduced from
`H·D·S` → `4 + H_q` (where `H_q` is a small hidden projection of the query),
saving both parameters and compute while retaining output awareness.

Key elements
------------
1. **Query-conditioned Summary Gate**
   •  For every token we concatenate a *low-rank projection* of the current
      hidden state (the “query”) with *per-stream head summaries* (mean over
      the channel dimension).  This gives the router both semantic context
      and a glimpse of what each branch has produced **without flattening the
      full tensors**.
   •  The gate is a lightweight 2-layer MLP shared across heads, followed by
      per-head temperatures, an ε-floor schedule, and entropy regularisation
      (all adapted from OAGMS).

2. **Multi-Scale Local Convolutions with Identity Path**
   •  We keep the proven depth-wise causal FIR family – now with kernels
      (1, 3, 7, 15) giving micro, short, mid and longer local context within
      the same O(N) framework.

3. **Preserved Strengths, Fewer Parameters**
   •  Chunk-wise Δ-rule global memory, identity initialisation, grouped
      schedule helpers, optional gating RMSNorm, strict causality and
      batch-agnostic operations are all retained.
   •  Router input is now *O(H)* instead of *O(H·D)* so the parameter cost of
      the fusion MLP is reduced by ~98 % for typical dimensions (D≈1024).

The layer remains sub-quadratic (O(N·d)), fully torch.compile-able and drop-in
compatible (class name `DeltaNet`, unchanged `forward` signature).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input  # noqa: F401 – kept for backward compat
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm


# -----------------------------------------------------------------------------
# Helper activations & normalisers --------------------------------------------
# -----------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Shifted ELU so the output is strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """L1 normalisation on the last dimension."""
    return (x / x.sum(-1, keepdim=True)).to(x)


# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac-initialised multi-scale) ------------
# -----------------------------------------------------------------------------


class _DepthwiseMultiScaleFIR(nn.Module):
    """Per-head depth-wise causal FIR filters for an arbitrary set of kernels.

    Each filter is Dirac-initialised (identity) so early training behaviour is
    unchanged.  Complexity: O(N·d) – one 1-D convolution per branch.
    """

    def __init__(self, *, num_heads: int, head_dim: int, kernel_sizes: Tuple[int, ...]):
        super().__init__()
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.num_heads = num_heads
        self.head_dim = head_dim
        channels = num_heads * head_dim
        self.filters = nn.ParameterList()
        for k in self.kernel_sizes:
            w = torch.zeros(channels, 1, k)
            with torch.no_grad():
                w[:, 0, -1] = 1.0  # causal identity (Dirac delta at last tap)
            self.filters.append(nn.Parameter(w))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_ch = rearrange(x, "b l h d -> b (h d) l")  # (B, C, L)
        outs: List[torch.Tensor] = []
        for filt, k in zip(self.filters, self.kernel_sizes):
            x_pad = F.pad(x_ch, (k - 1, 0))  # causal left-pad
            y = F.conv1d(x_pad, filt, groups=h * d)
            outs.append(rearrange(y, "b (h d) l -> b l h d", h=h))
        return outs


# -----------------------------------------------------------------------------
# Causal chunk-wise Δ-rule (unchanged numerics) --------------------------------
# -----------------------------------------------------------------------------


@torch.compile  # type: ignore[misc]
# pylint: disable=too-many-locals,too-many-statements
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Associative Δ-rule memory retrieval with fixed chunk size (causal, O(N))."""
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

    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(tri, 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)
    inv = inv.to(torch.bfloat16).to(q.dtype)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    for blk in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, blk], k[:, :, blk]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        out[:, :, blk] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S


# -----------------------------------------------------------------------------
# Optional type stub (not executed at runtime) ---------------------------------
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401 – for typing only


# -----------------------------------------------------------------------------
# Main **DeltaNet** implementation – Query & Summary Routing variant
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):  # noqa: D401 – class name must remain exactly "DeltaNet"
    """DeltaNet layer with *Query-conditioned Summary Router* (QSR)."""

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    def __init__(
        self,
        *,
        mode: str = "qsr",  # identifier string
        d_model: int | None = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # optional components -------------------------------------------------
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
        # multi-scale kernel sizes -------------------------------------------
        ms_kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15),
        # gating & regularisation -------------------------------------------
        gate_query_proj: int = 128,
        fusion_hidden_mult: float = 1.0,
        floor_start: float = 0.02,
        floor_end: float = 0.0,
        floor_decay_steps: int = 2000,
        entropy_coeff_start: float = 0.02,
        entropy_coeff_end: float = 0.0,
        entropy_decay_steps: int = 2000,
        # temperature per head ----------------------------------------------
        temp_init: float = 1.0,
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        # ---------- bookkeeping -------------------------------------------
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
        self.ms_kernel_sizes = ms_kernel_sizes

        # ---------- schedules ---------------------------------------------
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_coeff_start = float(entropy_coeff_start)
        self.entropy_coeff_end = float(entropy_coeff_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

        # ---------- dimensions --------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")

        # ---------- projections -------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------- short convs (mandatory) --------------------------------
        if not self.use_short_conv:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)

        # ---------- multi-scale FIR memory ---------------------------------
        self.local_fir = _DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)

        # ---------- fusion gate components ---------------------------------
        # Query projection (shared across heads)
        self.q_gate_proj = nn.Linear(hidden_size, gate_query_proj, bias=True)
        # MLP shared across heads; input dim = gate_query_proj + num_streams (scalar summaries per stream)
        self.num_streams = self.num_scales + 2  # FIR branches + delta + direct value
        gate_in_dim = gate_query_proj + self.num_streams  # per head
        gate_hidden_dim = max(8, int(gate_in_dim * fusion_hidden_mult))
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, self.num_streams, bias=True),
        )
        with torch.no_grad():
            self.fusion_gate[-1].bias.zero_()
            # favour direct value path (last index)
            self.fusion_gate[-1].bias[-1] = 1.0

        # temperature per head (softplus parameterisation)
        self.log_temp = nn.Parameter(torch.full((num_heads,), math.log(temp_init)))

        # ---------- output normalisation & projection ----------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------
    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        r = t / max(1.0, self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start) * r

    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_coeff_end
        r = t / max(1.0, self.entropy_decay_steps)
        return self.entropy_coeff_start + (self.entropy_coeff_end - self.entropy_coeff_start) * r

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-branches, too-many-statements
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compatibility
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:  # type: ignore[name-defined]
        # NOTE: The earlier version performed *global unpadding* by flattening the
        #       batch dimension into the sequence dimension (B,L,D) → (1,\sum L,D).
        #       While efficient, this *mixed tokens from different batch examples*,
        #       leading to cross-sample information leakage inside the causal core
        #       (Δ-rule, local FIRs, etc.).
        #
        #       The fix below *removes that unpadding path* so each example stays
        #       isolated.  ShortConvolution and other components operate directly
        #       on the padded (B,L,D) tensors which is fully supported and keeps
        #       the computational complexity unchanged (O(N·d)).
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"

        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)

        # ------------------------------------------------------------------
        # retrieve cache ----------------------------------------------------
        # ------------------------------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ------------------------------------------------------------------
        # projections + optional short conv ---------------------------------
        # ------------------------------------------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ------------------------------------------------------------------
        # head split & activation ------------------------------------------
        # ------------------------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ------------------------------------------------------------------
        # beta coefficients -------------------------------------------------
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # delta-rule (global path) -----------------------------------------
        # ------------------------------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v_direct, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out_d, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # ------------------------------------------------------------------
        # local FIR branches ------------------------------------------------
        # ------------------------------------------------------------------
        conv_branches = self.local_fir(v_direct)  # list length = num_scales

        # ------------------------------------------------------------------
        # assemble streams --------------------------------------------------
        # ------------------------------------------------------------------
        streams: List[torch.Tensor] = conv_branches + [delta_out, v_direct]
        # summaries: mean over feature dimension -> (B,L,H)
        summaries = [s.mean(-1) for s in streams]
        summaries_stack = torch.stack(summaries, dim=-1)  # (B,L,H,S)

        # ------------------------------------------------------------------
        # query projection --------------------------------------------------
        # ------------------------------------------------------------------
        q_proj_gate = self.q_gate_proj(hidden_states)  # (B,L,Q)
        q_proj_gate = q_proj_gate.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)  # (B,L,H,Q)

        gate_in = torch.cat([q_proj_gate, summaries_stack], dim=-1)
        gate_in_flat = rearrange(gate_in, "b l h d -> (b l h) d")
        fusion_logits_flat = self.fusion_gate(gate_in_flat)  # (B*L*H, S)
        fusion_logits = rearrange(
            fusion_logits_flat,
            "(b l h) s -> b l h s",
            b=gate_in.shape[0],
            l=gate_in.shape[1],
            h=self.num_heads,
            s=self.num_streams,
        )

        # temperature scaling
        temp = (F.softplus(self.log_temp) + 1e-4).view(1, 1, -1, 1)
        fusion_logits = fusion_logits / temp

        fusion_probs = torch.softmax(fusion_logits, dim=-1)

        # ------------------------------------------------------------------
        # ε-floor & renormalise --------------------------------------------
        # ------------------------------------------------------------------
        eps_val = self._current_floor()
        if eps_val > 0.0:
            fusion_probs = torch.clamp(fusion_probs, min=eps_val)
            fusion_probs = fusion_probs / fusion_probs.sum(-1, keepdim=True)

        # ------------------------------------------------------------------
        # entropy regularisation -------------------------------------------
        # ------------------------------------------------------------------
        reg_loss = None
        if self.training:
            coeff = self._current_entropy_coeff()
            if coeff > 0.0:
                ent = -(fusion_probs * (fusion_probs + 1e-8).log()).sum(-1).mean()
                if torch.isnan(ent) or torch.isinf(ent):
                    ent = torch.zeros_like(ent)
                reg_loss = coeff * ent

        # ------------------------------------------------------------------
        # final mixture -----------------------------------------------------
        # ------------------------------------------------------------------
        streams_stacked = torch.stack(streams, dim=-2)  # (B,L,H,S,D)
        o = (streams_stacked * fusion_probs.unsqueeze(-1)).sum(-2)  # (B,L,H,D)

        # ------------------------------------------------------------------
        # cache update ------------------------------------------------------
        # ------------------------------------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ------------------------------------------------------------------
        # output norm & projection -----------------------------------------
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ------------------------------------------------------------------
        # step++ -----------------------------------------------------------
        # ------------------------------------------------------------------
        self._step += 1  # type: ignore[operator]

        # Return signature preserved ---------------------------------------
        return o, reg_loss, past_key_values
