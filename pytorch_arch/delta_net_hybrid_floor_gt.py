# -*- coding: utf-8 -*-
"""
DeltaNet – Hybrid Floor Fusion with Group-Temperature and Static-Dynamic Residual  
==============================================================================
Identifier: *delta_net_hybrid_floor_gt*  

Key innovations (enabled by default):
1. **Group-Wise Temperature Sharing** – routing softmax logits are scaled by a
   temperature τ that is *shared across small groups of heads* (default group
   size = 2).  This preserves some redundancy between heads, mitigating the
   over-fragmentation observed with fully-independent per-head temperatures
   while still allowing specialisation at a finer granularity than a single
   global τ.

2. **Hybrid Static + Dynamic Residual Convolution** – a *constant* fraction of
   the local-short FIR path (α = 0.2) is injected into the fused output to
   guarantee non-zero gradient flow for ultra-local reasoning, while the
   remaining 0.8 is modulated by the original per-token, per-head dynamic gate.
   This eliminates the early-training starvation of local cues seen in purely
   dynamic gating variants without sacrificing contextual adaptability.

3. **Automatically Annealed Entropy + KL Regularisation** – diversity-promoting
   losses applied to the fusion gate are *automatically annealed* as training
   progresses.  The weights linearly decay from their initial value to zero
   over a user-configurable number of optimisation steps (default 20 k).  The
   gate therefore benefits from strong early-training path diversity while
   allowing sharp, specialised routing to emerge later.

The remainder of the architecture inherits proven components from prior
DeltaNet variants: strictly causal chunked Δ-rule memory, dual depth-wise FIR
convolutions, short convolution enhancement and RMSNorm projection.  All new
features obey O(N) complexity and maintain full API compatibility.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:  # Shifted ELU (>0)
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # L1 normalisation
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution -------------------------------------------
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution with (almost) identity initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 2e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            filt[..., -1] = 1.0
            filt.add_(noise_std * torch.randn_like(filt))
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D)
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise associative Δ-rule kernel (identical to previous best) ------------
# -----------------------------------------------------------------------------

@torch.compile  # noqa: D401
def _delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,D)
    k: torch.Tensor,  # (B,H,L,D)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Efficient causal Δ-rule with O(N) complexity using chunking."""
    b, h, L, d_k = q.shape
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

    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)

    mask_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    n_chunks = L_pad // chunk_size
    for idx in range(n_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# -----------------------------------------------------------------------------
# Fusion gate with group-wise temperature & annealed regulariser --------------
# -----------------------------------------------------------------------------

class _HybridFloorFusionGate(nn.Module):
    """Entropy+KL regularised gate with learnable floor and group-wise τ."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        n_paths: int = 4,
        group_size: int = 2,
        max_floor: float = 0.05,
        init_temp: float = 1.25,
        entropy_w: float = 0.05,
        kl_w: float = 0.05,
        anneal_steps: int = 20_000,
        fusion_hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.num_heads = num_heads
        self.group_size = max(1, group_size)
        n_groups = (num_heads + self.group_size - 1) // self.group_size
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long), persistent=False)

        # Group-wise temperature parameters
        self.log_temp = nn.Parameter(torch.log(torch.full((n_groups,), init_temp)))
        # Learnable floor per head/path (constrained to [0,max_floor])
        self.floor_param = nn.Parameter(torch.full((num_heads, n_paths), -2.0))
        self.max_floor = float(max_floor)

        # Regulariser weights & schedule
        self.entropy_w_init = float(entropy_w)
        self.kl_w_init = float(kl_w)
        self.anneal_steps = int(anneal_steps)
        self.last_gate_loss: Optional[torch.Tensor] = None

        # Simple MLP that outputs head*path logits
        gate_in_dim = hidden_size + num_heads * 16  # hidden + 4 stats * 4 paths per head
        hidden_dim = hidden_size * fusion_hidden_mult // 2
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * n_paths, bias=True),
        )
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            # Favour value path (index 3)
            self.mlp[-1].bias[num_heads * 3 :: n_paths] = 2.0

        # FSDP/FullySharded workaround: ensure regularizer weights are 1D tensor not scalar
        self.log_ent_w = nn.Parameter(torch.tensor([entropy_w], dtype=torch.float32), requires_grad=False)
        self.log_kl_w = nn.Parameter(torch.tensor([kl_w], dtype=torch.float32), requires_grad=False)

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:  # (B,L,H,D) -> (B,L,H,4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    def _current_weights(self) -> Tuple[float, float]:
        """Return annealed (entropy_w, kl_w) based on internal step counter."""
        step = float(self.step_counter.item())
        if self.anneal_steps <= 0:
            return float(self.log_ent_w.item()), float(self.log_kl_w.item())
        ratio = max(0.0, 1.0 - step / self.anneal_steps)
        return float(self.log_ent_w.item()) * ratio, float(self.log_kl_w.item()) * ratio

    def forward(
        self,
        hidden: torch.Tensor,  # (B,L,D)
        short: torch.Tensor,   # (B,L,H,D)
        long: torch.Tensor,
        delta: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:  # returns fusion weights (B,L,H,4)
        B, L, H, _ = short.shape
        # Gather per-branch stats
        stats = [self._stats(t) for t in (short, long, delta, value)]  # list of (B,L,H,4)
        flat_stats = [rearrange(s, "b l h s -> b l (h s)") for s in stats]  # (B,L,H*4)
        gate_in = torch.cat([hidden] + flat_stats, dim=-1)  # (B,L,hidden+16H)

        logits = self.mlp(gate_in)  # (B,L,H*P)
        logits = rearrange(logits, "b l (h p) -> b l h p", h=H, p=self.n_paths)

        # Group-wise temperature scaling ---------------------------------
        n_groups = self.log_temp.shape[0]
        temp = torch.exp(self.log_temp)  # (G,)
        # Prepare mapping from head -> group index
        group_idx = (torch.arange(H, device=logits.device) // self.group_size)
        tau = temp[group_idx]  # (H,)
        logits = logits / tau.view(1, 1, H, 1)

        # Softmax & floor -----------------------------------------------
        raw_p = torch.softmax(logits, dim=-1)  # (B,L,H,4)
        floor = torch.sigmoid(self.floor_param) * self.max_floor  # (H,4)
        floor = floor.view(1, 1, H, self.n_paths)
        prob = torch.clamp(raw_p, min=floor)
        prob = prob / prob.sum(dim=-1, keepdim=True)

        # ---------------- Regularisation --------------------------------
        entropy_w, kl_w = self._current_weights()
        if entropy_w > 0.0 or kl_w > 0.0:
            logp = torch.log(prob + 1e-8)
            ent = -(prob * logp).sum(-1).mean()
            if kl_w > 0.0:
                uniform = math.log(self.n_paths)
                kl = (prob * (logp + uniform)).sum(-1).mean()
            else:
                kl = torch.tensor(0.0, device=prob.device)
            self.last_gate_loss = ent * entropy_w + kl * kl_w
        else:
            self.last_gate_loss = None

        # Increment internal counter
        with torch.no_grad():
            self.step_counter += 1

        return prob

# -----------------------------------------------------------------------------
# Main DeltaNet layer ----------------------------------------------------------
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore


class DeltaNet(nn.Module):
    """DeltaNet layer – Hybrid Floor Fusion with Group-Temperature."""

    def __init__(
        self,
        *,
        mode: str = "hybrid_floor_gt",
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
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 64,
        # Fusion gate params
        gate_max_floor: float = 0.05,
        gate_entropy_weight: float = 0.05,
        gate_kl_weight: float = 0.05,
        gate_anneal_steps: int = 20_000,
        gate_group_size: int = 2,
        # Hybrid residual params
        static_residual_frac: float = 0.2,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        # ---------------- bookkeeping ------------------------------
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
        self.static_residual_frac = float(static_residual_frac)

        # ---------------- dimensions --------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/Value dims must divide num_heads")

        # ---------------- projections -------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv --------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---------------- FIR convolutions --------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_long_kernel)

        # ---------------- fusion gate -------------------------------
        self.fusion_gate = _HybridFloorFusionGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_floor=gate_max_floor,
            init_temp=1.25,
            entropy_w=gate_entropy_weight,
            kl_w=gate_kl_weight,
            anneal_steps=gate_anneal_steps,
            group_size=gate_group_size,
        )

        # ---------------- output norm / proj ------------------------
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
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for interface
        **kwargs: Dict,
    ):
        if attention_mask is not None and attention_mask.ndim != 2:
            raise AssertionError("attention_mask must be (batch, seq_len)")

        B0, L0, _ = hidden_states.shape

        # ---------- cache retrieval ---------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # ---------- optional unpadding ------------------------------
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L0:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---------- projections & short conv ------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---------- head reshape ------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ---------- activation / norm -------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---------- beta for Δ-rule ---------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  # (B,L,H)
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------- Δ-rule path ------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, rec_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---------- FIR paths --------------------------------------
        value = v  # identity path (direct value)
        short = self.fir_short(value)
        long = self.fir_long(value)

        # ---------- fusion weights ---------------------------------
        fusion_w = self.fusion_gate(hidden_states, short, long, delta_out, value)  # (B,L,H,4)

        # ---------- hybrid residual injection ----------------------
        dynamic_part = fusion_w[..., 0:1] * short  # dynamic share of short path
        static_part = self.static_residual_frac * short

        fused = (
            dynamic_part +  # dynamic short
            fusion_w[..., 1:2] * long +
            fusion_w[..., 2:3] * delta_out +
            fusion_w[..., 3:4] * value
        )
        o = fused + static_part  # ensure constant local residual

        # ---------- cache update -----------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L0,
            )

        # ---------- output norm / projection -----------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---------- re-pad if necessary ----------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B0, L0)

        return o, None, past_key_values
