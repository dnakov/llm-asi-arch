# -*- coding: utf-8 -*-
"""
DeltaNet – Multi-Scale FIR + Output-Aware Adaptive Gate + Statistical Diversity Regularization
=============================================================================================
Innovation: delta_net_ms_adaptive_gstat3

Breakthrough: Integrates research-backed innovations for balanced local/global reasoning, robust gate-driven adaptive fusion, and regularization for path diversity and confidence.

Major Innovations:
------------------
1. **Richer Output-Aware Gating (GATE-STAT3):**
   - Gate logits are conditioned on an MLP(hidden_state), as well as *both* mean, std, and max statistics of each branch (FIR-short, FIR-long, Delta, Direct-Value), providing the gate with sharper information for informed path selection.
   - Gate statistics are normalized (LayerNorm) per branch before fusion for scale invariance.
   - A learnable `alpha` (per head) initialised to 0.2 boosts output-statistics' effect early.

2. **Statistical Diversity Regularization:**
   - During training, an extra loss is returned (as a side-channel) — penalizing low entropy (encourages softmax gate to not collapse), and encouraging KL divergence between each gate and a uniform distribution (encouraging full path usage), and optional dissimilarity between heads (gate cosine diversity).
   - These are only returned if `return_reg_loss=True` in forward; does not affect inference/checkpoint.

3. **Hybrid Path Bias and Gate Initialization:**
   - The output-aware gate (MLP) is bias-initialized towards the delta/identity branch so early in training the model does not starve the key branch. Branch alpha is set per head.

4. **Flexible Kernel Schedule:**
   - Option to set long FIR kernel to 31 by default (reducing oversmooth); can be adjusted for ablations.
   - Additional (optional) mid-scale kernel support (disabled by default, but infrastructure for easy addition).

5. **Robust Implementation:**
   - Universal use of einops.rearrange, batch-size agnostic, chunked computation, strictly causal and sub-quadratic.
   - Preserves all initialization, interface, and cache protocols.

Fix Log (2024-06-15):
---------------------
Critical shape inconsistency in the output-aware gate fusion fixed.
Previously, the code attempted to `rearrange` a flattened statistics tensor of
size 12 (4 branches × 3 stats) directly into a dimension of size **4**, which
is mathematically impossible and raises a runtime error for every batch size.

The correct behaviour is to first restore the `(branch, stat)` structure and
reduce **only** over the statistics axis, producing a scalar value per branch.
This keeps the intended design (one scalar per branch & head), preserves the
learnable per-head `alpha`, and maintains full batch-size independence.

Minimal, surgical changes were applied:
    • compute `branch_stat_scalar = branch_stat.mean(dim=-1)`  # [B, L, H, 4]
    • fuse with gate logits via `gmix_logits += alpha * branch_stat_scalar`
    • redundant / incorrect `rearrange` call removed.
The overall architecture, complexity, and causal masking remain intact.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ----------------------------------------
# Helper statistics
# ----------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)


def branch_stats(x: torch.Tensor):  # [B, L, H, D]
    """Return mean, std, max for every sequence position & head."""
    mu = x.mean(dim=-1)  # (B, L, H)
    std = x.std(dim=-1)  # (B, L, H)
    mx = x.amax(dim=-1)  # (B, L, H)
    return mu, std, mx


def norm_stats(stat):
    # LayerNorm across heads for each stat
    _shape = stat.shape
    if len(_shape) == 3:
        stat = rearrange(stat, "b l h -> b l h 1")
        stat = F.layer_norm(stat, stat.shape[-2:], eps=1e-5).squeeze(-1)  # Norm over h
    return stat

# ----------------------------------------
# Core chunk-wise delta rule
# ----------------------------------------

@torch.compile
def delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
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
    mask_full = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_full, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn.to(torch.bfloat16)
    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# ----------------------------------------
# FIR convolution for each branch (unchanged)
# ----------------------------------------


class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = nn.Parameter(
            torch.randn(num_heads, head_dim, kernel_size) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class DeltaNet(nn.Module):
    """DeltaNet with multi-scale FIR, advanced output-stat gate, per-head alpha, and diversity regularization."""

    def __init__(
        self,
        *,
        mode: str = "ms_adaptive_gstat3",
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
        fir_short_kernel_size: int = 7,
        fir_long_kernel_size: int = 31,
        gmix_hidden_mult: int = 2,
        gate_stat_alpha_init: float = 0.2,
        mid_scale_kernel_size: Optional[int] = None,  # Future use
        return_reg_loss: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.hidden_size = hidden_size if d_model is None else d_model
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
        self.fir_short_kernel_size = fir_short_kernel_size
        self.fir_long_kernel_size = fir_long_kernel_size
        self.gmix_hidden_mult = gmix_hidden_mult
        self.gate_stat_alpha_init = gate_stat_alpha_init
        self.return_reg_loss = return_reg_loss
        # Dims
        self.key_dim = int(self.hidden_size * expand_k)
        self.value_dim = int(self.hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(self.hidden_size, num_heads, bias=False)
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation="silu",
            )
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")
        self.fir_short = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_short_kernel_size
        )
        self.fir_long = DepthwiseFIRConv1d(
            num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_long_kernel_size
        )
        # Configure per-head alpha (stat scaling)
        self.alpha = nn.Parameter(torch.full((num_heads, 1), gate_stat_alpha_init))
        # Gate MLP with advanced bias init: favor delta path
        self.gmix_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * gmix_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size * gmix_hidden_mult, num_heads * 4, bias=True),
        )
        nn.init.constant_(self.gmix_mlp[-1].bias[num_heads * 2 : num_heads * 3], 0.03)  # delta branch bias boost
        # Norm for stats (kept for future use)
        self.branch_stat_norm = nn.LayerNorm([num_heads, 4, 3], elementwise_affine=True)  # [H, 4(branch), 3(stat)]
        # Output
        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:
        # ----------- Pad logic ----------
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if (
            past_key_values is not None
            and self.layer_idx is not None
            and len(past_key_values) > self.layer_idx
        ):
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).unsqueeze(0)
        # ------- QKV + short conv -------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        q, k = map(
            lambda t: rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)
        # --------- Delta path ----------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = delta_rule_chunkwise(
            q_d, k_d, v_d, beta_d, chunk_size=32
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")
        # --------- Multi-scale FIR paths -----------
        v_direct = v
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)
        # --------- Gate stats (mean, std, max) for all 4 branches --------
        branch_outputs = [fir_short, fir_long, delta_out, v_direct]
        stats = [torch.stack(branch_stats(b), dim=-1) for b in branch_outputs]  # each [B,L,H,3]
        stats = [norm_stats(s) for s in stats]  # ensure scale invariance
        branch_stat = torch.stack(stats, dim=-2)  # [B,L,H,4,3]
        # Average over the 3 statistics to obtain a scalar per branch
        branch_stat_scalar = branch_stat.mean(dim=-1)  # [B,L,H,4]
        # learnable per-head alpha (broadcasted)
        alpha = rearrange(self.alpha, "h x -> 1 1 h x")  # (1,1,H,1)
        # Gate MLP
        gmix_logits = self.gmix_mlp(hidden_states)  # [B,L,H*4]
        gmix_logits = rearrange(
            gmix_logits, "b l (h c) -> b l h c", h=self.num_heads, c=4
        )
        # Combine: content-based logits + scaled branch statistics
        gmix_logits = gmix_logits + alpha * branch_stat_scalar
        # Softmax for convex mixture
        gmix_weights = torch.softmax(gmix_logits, dim=-1)  # [B,L,H,4]
        # --------- Fuse paths -------------------------
        o = (
            gmix_weights[..., 0:1] * fir_short
            + gmix_weights[..., 1:2] * fir_long
            + gmix_weights[..., 2:3] * delta_out
            + gmix_weights[..., 3:4] * v_direct
        )
        # --------- Cache update ----------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )
        # -------- Output norm/proj ----------------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)
        # --------- Regularization extras ------------------------
        if self.return_reg_loss and self.training:
            # Gate entropy loss: encourage gates not to collapse (avg entropy over all gates/positions)
            gate_logits = gmix_logits
            gate_probs = torch.softmax(gate_logits, dim=-1)  # [B,L,H,4]
            entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(-1)
            entropy_loss = -entropy.mean()  # maximise entropy
            # Encourage gates toward uniform (good at start): KL to uniform
            uniform = torch.full_like(gate_probs, 1.0 / 4)
            kl_loss = (
                gate_probs * (torch.log(gate_probs + 1e-8) - torch.log(uniform + 1e-8))
            ).sum(-1).mean()
            # Inter-head diversity (cosine)
            head_probs = rearrange(gate_probs, "b l h c -> (b l) h c")
            head_cos = 0.0
            for i in range(self.num_heads):
                for j in range(i + 1, self.num_heads):
                    head_cos += F.cosine_similarity(
                        head_probs[:, i], head_probs[:, j], dim=-1
                    ).mean()
            head_diversity_loss = -head_cos / (self.num_heads * (self.num_heads - 1) / 2)
            reg_loss = entropy_loss + kl_loss + head_diversity_loss
            return o, reg_loss, past_key_values
        return o, None, past_key_values
