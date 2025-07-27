# -*- coding: utf-8 -*-
"""
DeltaNet – Parallel Softplus Adaptive Fusion Gated Memory (PSAFG)
================================================================
Identifier (architecture name): delta_net_psafg

This evolutionary variant **addresses the over-suppression / path-collapse
problem** observed in `delta_net_dmghm` by replacing the *competitive* softmax
fusion with a **normalised additive gate** that is *output-aware*:

1. **Identity-Inclusive Multi-Scale Local Memory**  
   • Adds an *identity* kernel (k = 1) to the depth-wise FIR pyramid
     giving kernel set **(1, 3, 7, 15, 31)** by default.  This preserves
     ultra-local signals helpful for fine-grained extraction benchmarks
     (e.g. SWDE) without having to rely solely on the direct value path.

2. **Output-Aware Gate Features**  
   • For every token & head we concatenate *per-path statistics* – the
     mean absolute activation of each stream – to the hidden state before
     it is processed by the gate MLP.  This provides immediate feedback
     about the usefulness of each memory path, enabling the router to
     decide based on *what each path actually produced* (not just the
     input token embedding).

3. **Parallel Softplus Fusion with Normalised Amplitude**  
   • The MLP outputs *unnormalised* positive scalars `w_i ≥ 0` via
     `softplus`.  A small ε-floor (default 0.02) guarantees gradient flow
     to all paths.  The fused output is

         y = Σ w_i·path_i  /  Σ w_i

     which keeps the overall activation scale roughly constant regardless
     of how many paths are active, fixing the scale-explosion observed in
     `delta_net_psfr` while *avoiding* the hard competition of softmax.

4. **Bias-Initialised Value Dominance**  
   • As in DMGHM, the gate bias is initialised such that the *direct value*
     path dominates early training, ensuring stable optimisation and
     preventing premature over-smoothing from the large FIR kernels.

All changes respect O(N) complexity, strict causality, batch-size
independence, and keep the public API unchanged.  The class name remains
`DeltaNet`, making this a drop-in replacement.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Utility helpers (torch.compile-safe)
# -----------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (+1) – used in several DeltaNet variants."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so that the last dimension sums to 1."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Core Delta-rule kernel (unchanged)
# -----------------------------------------------------------------------------

@torch.compile  # keep high-performance compilation
def _delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Chunk-wise associative Δ-rule (identical to proven baseline)."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # L2 norm and scaling -----------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into fixed chunks ----------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    tri = torch.triu(torch.ones_like(eye, dtype=torch.bool), 0)
    tri_strict = torch.triu(torch.ones_like(eye, dtype=torch.bool), 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (
            inv[..., i, :, None].clone() * inv[..., :, :i].clone()
        ).sum(-2)
    inv = inv + eye

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Multi-scale depth-wise FIR block
# -----------------------------------------------------------------------------

class _DepthwiseMultiScaleFIR(nn.Module):
    """Parallel depth-wise causal convolutions with identity initialisation."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15, 31),
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.total_channels = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.filters: nn.ParameterList = nn.ParameterList()
        for k in kernel_sizes:
            filt = nn.Parameter(torch.zeros(self.total_channels, 1, k))
            with torch.no_grad():
                filt[:, 0, -1] = 1.0  # identity / Dirac init
            self.filters.append(filt)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # x: [B,L,H,D]
        x_ch = rearrange(x, "b l h d -> b (h d) l")  # [B, C, L]
        outs: List[torch.Tensor] = []
        for filt, k in zip(self.filters, self.kernel_sizes):
            x_pad = F.pad(x_ch, (k - 1, 0))
            y = F.conv1d(x_pad, weight=filt, groups=self.total_channels)
            outs.append(rearrange(y, "b (h d) l -> b l h d", h=self.num_heads))
        return outs

# -----------------------------------------------------------------------------
# Main DeltaNet block with Parallel Softplus Adaptive Fusion
# -----------------------------------------------------------------------------

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack  # pragma: no cover
    from fla.models.utils import Cache  # pragma: no cover

class DeltaNet(nn.Module):
    """DeltaNet with Parallel Softplus Adaptive Fusion Gated Memory (PSAFG)."""

    def __init__(
        self,
        *,
        mode: str = "psafg",
        d_model: int | None = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # ---- feature flags ----
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
        # ---- multi-scale args ----
        ms_kernel_sizes: Tuple[int, ...] = (1, 3, 7, 15, 31),
        # ---- fusion gate args ----
        fusion_hidden_mult: int = 2,
        gate_eps_floor: float = 0.02,
        gate_bias_init: float = 0.5,
        **kwargs: "Unpack[Dict]",
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model

        # ---------------- bookkeeping ----------------
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.gate_eps_floor = gate_eps_floor

        # ---------------- dimensions -----------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---------------- linear projections ----------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv enhancement ------
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")

        # ---------------- local FIR pyramid ----------
        self.local_fir = _DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)

        # ---------------- fusion gate MLP ------------
        # input: hidden + per-path stats (H*num_streams)
        self.num_streams = self.num_scales + 2  # conv branches + delta + direct value
        gate_in_dim = hidden_size + num_heads * self.num_streams
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_size * fusion_hidden_mult),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * self.num_streams),
        )

        # bias init – favour direct value path early
        with torch.no_grad():
            bias = self.fusion_gate_mlp[-1].bias  # shape (H*streams)
            bias.fill_(gate_bias_init)
            bias.view(num_heads, self.num_streams)[:, -1] += 1.0  # boost value path

        # ---------------- output norm / projection ---
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: "Unpack[Dict]",
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B0, L0, _ = hidden_states.shape

        # ---- optional unpadding for variable seq lengths -------------
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L0:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---- retrieve prior cache state ------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ---- projections + short conv --------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---- head reshape -------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # ---- activations / norms ------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- beta coefficients (Δ-rule) ------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones((*hidden_states.shape[:2], self.num_heads), device=hidden_states.device, dtype=hidden_states.dtype)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- global Δ-rule memory -----------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---- local FIR branches -------------------------------------
        conv_branches = self.local_fir(v)  # list length = num_scales

        # ---- assemble streams & compute path stats ------------------
        streams: List[torch.Tensor] = conv_branches + [delta_out, v]
        # per-path mean|x| stats for gate feature
        stats = [s.abs().mean(dim=-1) for s in streams]  # each [B,L,H]
        path_stats = torch.cat(stats, dim=-1)  # [B,L,H*streams]

        # ---- gate input construction --------------------------------
        gate_inp = torch.cat(
            [hidden_states, rearrange(path_stats, "b l (h s) -> b l (h s)", h=self.num_heads, s=self.num_streams)],
            dim=-1,
        )  # shape [B,L,hidden + H*streams]

        gate_logits = self.fusion_gate_mlp(gate_inp)  # [B,L,H*streams]
        gate_logits = rearrange(gate_logits, "b l (h s) -> b l h s", h=self.num_heads, s=self.num_streams)

        # positive weights via softplus --------------------------------
        weights = F.softplus(gate_logits) + self.gate_eps_floor  # ensure ≥ ε
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalise

        # ---- fuse streams -------------------------------------------
        streams_stacked = torch.stack(streams, dim=-2)  # [B,L,H,streams,D]
        o = (streams_stacked * weights.unsqueeze(-1)).sum(dim=-2)  # [B,L,H,D]

        # ---- cache update -------------------------------------------
        if past_key_values is not None and use_cache and hasattr(past_key_values, "update"):
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L0,
            )

        # ---- output norm / projection -------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = self.o_proj(rearrange(o, "b l h d -> b l (h d)"))

        # ---- re-pad if we unpadded earlier --------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B0, L0)

        return o, None, past_key_values
