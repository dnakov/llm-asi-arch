# -*- coding: utf-8 -*-
"""
DeltaNet – Multi-Scale FIR with Reserve Gating (MS-RG)
=====================================================
Identifier: delta_net_ms_resgate

Core innovations
----------------
1. Multi-Scale Depth-Wise FIR local memory (kernels 3, 7, 15, 31)
   – Same efficient depth-wise causal convolutions proven in previous variants.
   – Identity initialisation (Dirac at last tap) keeps signal path intact at
     start-up.

2. Reserve Gating  ✱ NEW ✱
   – Per-head, per-token gate that *guarantees* a minimum allocation
     (\epsilon) to the **Δ-rule global path**, preventing the starvation that
     hurt long-range reasoning in earlier multi-scale models.
   – Remaining probability mass is distributed across *local* convolutional
     branches and the *direct value* path via a standard softmax.
   – Gate input combines the token’s hidden state with cheap branch statistics
     (mean-|·|) offering outcome awareness without expensive feature maps.
   – Learnable per-head temperature sharpens or diffuses routing as needed.

3. Strict causality & O(N) complexity
   – All convolutions are causal via left-padding.
   – Global memory uses the established chunk-wise Δ-rule kernel (O(N)).
   – No operation introduces quadratic complexity.

Public API, class name (`DeltaNet`) and forward signature remain unchanged.
All new features are **enabled by default** and require no config changes.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import ShortConvolution, RMSNorm, FusedRMSNormGated
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------
# -----------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (+1) — keeps values positive, useful for kernels."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalisation along the last dimension."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged, kept @torch.compile for speed) ----------
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
# pylint: disable=too-many-locals, too-many-statements
def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
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

    # normalise & apply β gate --------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into blocks -------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i = q[:, :, idx]
        k_i = k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Multi-Scale depth-wise FIR convolution ---------------------------------------
# -----------------------------------------------------------------------------

class DepthwiseMultiScaleFIR(nn.Module):
    """Parallel depth-wise causal convolutions with different kernel sizes.

    Kernels are identity-initialised (Dirac delta) to keep the main information
    path intact at start-up.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        total_channels = num_heads * head_dim
        self.filters: nn.ParameterList = nn.ParameterList()
        for k in kernel_sizes:
            filt = nn.Parameter(torch.zeros(total_channels, 1, k))
            with torch.no_grad():
                filt[:, 0, -1] = 1.0  # identity at last (causal) tap
                filt.add_(torch.randn_like(filt) * init_std)
            self.filters.append(filt)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_flat = rearrange(x, "b l h d -> b (h d) l")  # (B,C,L)
        outs: List[torch.Tensor] = []
        for k, filt in zip(self.kernel_sizes, self.filters):
            x_pad = F.pad(x_flat, (k - 1, 0))
            y = F.conv1d(x_pad, weight=filt, groups=h * d)
            outs.append(rearrange(y, "b (h d) l -> b l h d", h=h))
        return outs  # list[(B,L,H,D)]

# -----------------------------------------------------------------------------
# Optional type hints ----------------------------------------------------------
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401

# -----------------------------------------------------------------------------
# Main DeltaNet layer ----------------------------------------------------------
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with Multi-Scale FIR local memory and Reserve Gating."""

    def __init__(
        self,
        *,
        mode: str = "ms_resgate",
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
        # --- new hyper-parameters ----------------------------------
        ms_kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
        gate_hidden_mult: int = 2,
        delta_floor: float = 0.05,  # minimum allocation to Δ-path
        temp_init: float = 1.0,
        **kwargs: "Dict",  # ignore extra args for compatibility
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        # store basic config ----------------------------------------
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.delta_floor = float(delta_floor)

        # dimensions -----------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Dims must divide num_heads"
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # projections ----------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # short convolutions ---------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")

        # multi-scale FIR ------------------------------------------
        self.local_fir = DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)

        # reserve gate ---------------------------------------------
        # total paths: local scales + direct value + delta
        self.n_local_paths = self.num_scales + 1  # conv branches + direct value
        self.n_paths = self.n_local_paths + 1  # + delta

        gate_in_dim = hidden_size + num_heads * self.n_paths  # hidden + stats per head
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_size * gate_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * gate_hidden_mult, num_heads * self.n_paths, bias=True),
        )
        # temperature per head (positive via softplus)
        self.gate_log_temp = nn.Parameter(torch.ones(num_heads) * temp_init)
        # bias to encourage Δ path early (index n_local_paths)
        with torch.no_grad():
            b = self.fusion_gate_mlp[-1].bias.view(num_heads, self.n_paths)
            b[:, self.n_local_paths] += 2.0  # favour delta initially

        # statistic scaling parameter per path
        self.alpha_stat = nn.Parameter(torch.ones(self.n_paths) * 0.1)

        # output norm / projection ---------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,T,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # unused but kept for API
        **kwargs: "Dict",  # type: ignore[misc]
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:  # type: ignore[name-defined]
        # ---------- optional unpadding --------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, T_orig, _ = hidden_states.shape
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -T_orig:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # previous state ----------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ---------- projections + short conv --------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---------- head split & activation ---------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ---------- β gate for Δ-rule --------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------- Δ-rule global memory -----------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---------- Multi-scale FIR local branches -------------------
        conv_branches = self.local_fir(v)  # list length = num_scales

        # ---------- Direct value path --------------------------------
        value_out = v

        # ---------- Assemble branch list -----------------------------
        branches: List[torch.Tensor] = conv_branches + [value_out, delta_out]
        # order: locals..., value, delta  (len = n_paths)

        # ---------- branch statistics (mean |·|) ----------------------
        branch_stats = torch.stack([b_.abs().mean(dim=-1) for b_ in branches], dim=-1)  # (B,L,H,P)

        # ---------- fusion gate computation --------------------------
        gate_input = torch.cat(
            [hidden_states, rearrange(branch_stats, "b l h p -> b l (h p)")], dim=-1
        )  # (B,L,D + H*P)

        gate_logits = self.fusion_gate_mlp(gate_input)  # (B,L,H*P)
        gate_logits = rearrange(gate_logits, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)

        # add stat contribution --------------------------------------
        gate_logits = gate_logits + self.alpha_stat.view(1, 1, 1, self.n_paths) * branch_stats

        # temperature -------------------------------------------------
        temp = F.softplus(self.gate_log_temp) + 1e-4  # (H,)
        gate_logits = gate_logits * temp.view(1, 1, self.num_heads, 1)

        # split logits ------------------------------------------------
        logits_local = gate_logits[..., : self.n_local_paths]  # (B,L,H,local)
        logit_delta = gate_logits[..., self.n_local_paths]  # (B,L,H)

        # reserve gating ---------------------------------------------
        w_delta = torch.sigmoid(logit_delta) * (1.0 - self.delta_floor) + self.delta_floor  # (B,L,H)
        w_delta_exp = w_delta.unsqueeze(-1)  # (B,L,H,1)

        w_local = F.softmax(logits_local, dim=-1) * (1.0 - w_delta_exp)  # (B,L,H,local)

        weights = torch.cat([w_local, w_delta_exp], dim=-1)  # (B,L,H,P)

        # ---------- Fuse outputs ------------------------------------
        # stack *before* the last dimension so that the new "P" dimension
        # aligns with `weights` (B,L,H,P,D)
        stacked = torch.stack(branches, dim=-2)  # (B,L,H,P,D)
        out = (weights.unsqueeze(-1) * stacked).sum(dim=-2)  # (B,L,H,D)

        # ---------- cache update ------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state_new,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=T_orig,
            )

        # ---------- output projection -------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # ---------- re-pad if unpadded -------------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, B_orig, T_orig)

        return out, None, past_key_values
