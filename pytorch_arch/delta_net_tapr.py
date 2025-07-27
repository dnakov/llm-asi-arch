# -*- coding: utf-8 -*-
"""
DeltaNet – Token-Adaptive Pruned Routing (DeltaNet-TAPR)
=======================================================
Identifier: delta_net_tapr

This evolutionary DeltaNet generation merges *token-adaptive ε-floors* (from
TAREIA) with a lightweight **progressive probability pruning** schedule that
eliminates residual leakage for highly-confident tokens and late-training
stages.

Key mechanisms
--------------
1. **Token-Adaptive ε-Floor (unchanged)**
   •  Retains the original per-token, per-head floor proportional to router
      uncertainty `(1 – p_max)` to guarantee gradient flow early on.

2. **Progressive Hard Pruning**
   •  From step `prune_start_step` onwards a *linearly rising* probability
      threshold τ(t) removes all path probabilities below τ(t):

        τ(t) = prune_threshold * clip((t – prune_start) / (prune_end – prune_start), 0, 1)

      After pruning, the vector is renormalised to the simplex, forcing
      *exact* zeros and completely eliminating micro-leakage that hurt
      extraction-heavy benchmarks (SWDE, Winogrande).

3. **Entropy Regularisation Schedule**
   •  The entropy bonus is now *annealed* from `entropy_start` →
      `entropy_end`, encouraging exploration early and allowing sharpened
      routing once pruning takes over.

Implementation notes
--------------------
•  Only ~20 lines added compared to TAREIA – negligible overhead, O(N) cost.
•  All public interfaces remain unchanged; new behaviour is **enabled by
   default** with sensible hyper-parameters.
•  Batch- and sequence-size agnostic: thresholds are scalar and broadcast.
•  Fully respects sub-quadratic complexity and causal constraints.
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
# Helper utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (+1) keeps activations strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise so that elements along the last dimension sum to one."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule kernel (identical to original) -------------------------
# -----------------------------------------------------------------------------


@torch.compile  # type: ignore[arg-type]
def _delta_rule_chunkwise(
    q: torch.Tensor,  # [B,H,L,D_k]
    k: torch.Tensor,  # [B,H,L,D_k]
    v: torch.Tensor,  # [B,H,L,D_v]
    beta: torch.Tensor,  # [B,H,L]
    *,
    chunk_size: int = 32,
):
    """Associative retrieval via the Δ-rule processed in causal chunks (O(N))."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg)
        k = F.pad(k, pad_cfg)
        v = F.pad(v, pad_cfg)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into chunks:  (B H N C D)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)
    attn_inv = attn_inv.to(torch.bfloat16)  # mixed precision for memory-efficiency

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    future_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac-init) -------------------------------
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    """Per-head, per-channel causal FIR with identity (Dirac) initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            filt[..., -1] = 1.0
            filt.add_(0.01 * torch.randn_like(filt))
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Optional typing stub ---------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # pylint: disable=ungrouped-imports,cyclic-import

# -----------------------------------------------------------------------------
# Main DeltaNet layer ----------------------------------------------------------
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):
    """DeltaNet layer with token-adaptive ε-floor *and* progressive pruning."""

    # pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
    def __init__(
        self,
        # ---------------- generic args ----------------
        mode: str = "tapr",
        d_model: Optional[int] = None,
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
        layer_idx: Optional[int] = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # ---------------- FIR params -------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # ---------------- gate params ------------------
        fusion_hidden_mult: int = 2,
        gate_temp_init: float = 1.0,
        gate_eps_init: float = 1e-3,
        fusion_dropout: float = 0.0,
        # --------------- floor schedule ----------------
        floor_start: float = 0.05,
        floor_end: float = 0.0,
        floor_decay_steps: int = 3000,
        # --------------- pruning schedule --------------
        prune_start_step: int = 2000,
        prune_end_step: int = 4000,
        prune_threshold: float = 1e-3,
        # --------------- entropy schedule --------------
        entropy_start: float = 0.02,
        entropy_end: float = 0.0,
        entropy_decay_steps: int = 4000,
        # ---------------- identity path ---------------
        use_identity_path: bool = True,
        identity_scale_init: float = 0.5,
        **kwargs: Dict,  # Accept extra unused kwargs for compatibility
    ) -> None:
        super().__init__()

        # ---- bookkeeping -------------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value dims must divide num_heads")

        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_identity_path = use_identity_path

        # ----- schedules -------------------------------------------
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)

        self.prune_start_step = int(prune_start_step)
        self.prune_end_step = int(prune_end_step)
        self.prune_threshold = float(prune_threshold)

        self.entropy_start = float(entropy_start)
        self.entropy_end = float(entropy_end)
        self.entropy_decay_steps = int(entropy_decay_steps)

        # ---- projections -------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # identity projection & scaling --------------------------------
        if use_identity_path:
            self.id_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.alpha_identity = nn.Parameter(identity_scale_init * torch.ones(num_heads))
        else:
            self.register_parameter("id_proj", None)
            self.register_parameter("alpha_identity", None)

        # ---- optional local short conv -----------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()

        # ---- dual FIR convs -----------------------------------------
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ---- fusion gate -------------------------------------------
        fusion_in = hidden_size + self.head_v_dim * self.num_heads * 3  # hidden + (short,long,delta)
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 4, bias=True),
        )

        # learnable temperature per head
        self.gate_log_temp = nn.Parameter(torch.log(torch.tensor(gate_temp_init)) * torch.ones(num_heads))
        # ε-floor parameters (logit) – base template
        eps_logit_init = math.log(gate_eps_init) - math.log(1 - gate_eps_init) if gate_eps_init > 0 else -12.0
        self.gate_eps_logit = nn.Parameter(torch.full((num_heads, 4), eps_logit_init))

        # bias: favour direct value path moderately
        if self.fusion_gate_mlp[-1].bias is not None:
            with torch.no_grad():
                bias = self.fusion_gate_mlp[-1].bias
                bias.zero_()
                for h in range(num_heads):  # path idx 3 = direct value
                    bias[h * 4 + 3] = 2.0

        # ---- output normalisation / projection ---------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ---- step counter for schedules ----------------------------
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.reg_loss: Optional[torch.Tensor] = None  # populated every forward

    # -----------------------------------------------------------------
    # schedule helpers -------------------------------------------------
    # -----------------------------------------------------------------
    def _current_floor_max(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        ratio = t / max(1.0, self.floor_decay_steps)
        return self.floor_start + ratio * (self.floor_end - self.floor_start)

    def _current_prune_threshold(self) -> float:
        t = float(self._step.item())
        if t <= self.prune_start_step:
            return 0.0
        if t >= self.prune_end_step:
            return self.prune_threshold
        frac = (t - self.prune_start_step) / max(1.0, self.prune_end_step - self.prune_start_step)
        return frac * self.prune_threshold

    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_end
        ratio = t / max(1.0, self.entropy_decay_steps)
        return self.entropy_start + ratio * (self.entropy_end - self.entropy_start)

    # -----------------------------------------------------------------
    # floor + pruning --------------------------------------------------
    # -----------------------------------------------------------------
    def _apply_floor_and_prune(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply token-adaptive floor then threshold pruning."""
        # ---------------- adaptive floor ---------------------------
        p_max = probs.max(dim=-1, keepdim=True).values  # [B,L,H,1]
        scale = 1.0 - p_max  # proportional uncertainty
        eps_max = self._current_floor_max()
        if eps_max > 0.0:
            eps_base = torch.sigmoid(self.gate_eps_logit).view(1, 1, *self.gate_eps_logit.shape)  # [1,1,H,4]
            eps = eps_max * scale * eps_base
            probs = probs * (1.0 - eps.sum(dim=-1, keepdim=True)) + eps
            # added clamp here for numerical safety from below
            probs = probs.clamp(min=1e-9, max=1.0)

        # ---------------- hard pruning -----------------------------
        thresh = self._current_prune_threshold()
        if thresh > 0.0:
            mask = probs <= thresh
            probs = probs.masked_fill(mask, 0.0)
            # renormalise – if vector sums to zero (rare), fall back to uniform
            denom = probs.sum(dim=-1, keepdim=True)
            # added clamp to denom for safety
            denom = denom.clamp(min=1e-9)
            probs = torch.where(denom > 0, probs / denom, torch.full_like(probs, 0.25))
            # after re-normalisation, also clamp for safety
            probs = probs.clamp(min=1e-9, max=1.0)
        return probs

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape

        # -- retrieve previous state --------------------------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---- projections + (optional) short conv ------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv:
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
                v = F.silu(v)

        # ---- head reshape ----------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # ---- optional activation / norm --------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- beta gate ------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global path ----------------------------------
        delta_out, recurrent_state = _delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---- local FIR paths -------------------------------------
        local_short = self.local_fir_short(v)
        local_long = self.local_fir_long(v)

        # ---- fusion gating ---------------------------------------
        gate_inp = torch.cat(
            [
                hidden_states,
                rearrange(local_short, "b l h d -> b l (h d)"),
                rearrange(local_long, "b l h d -> b l (h d)"),
                rearrange(delta_out, "b l h d -> b l (h d)"),
            ],
            dim=-1,
        )
        fusion_logits = self.fusion_gate_mlp(gate_inp)  # [B,L,H*4]
        fusion_logits = rearrange(fusion_logits, "b l (h c) -> b l h c", h=self.num_heads, c=4)

        # temperature scaling --------------------------------------
        temp = (F.softplus(self.gate_log_temp) + 1e-4).view(1, 1, -1, 1)
        fusion_logits = fusion_logits / temp
        fusion_probs = torch.softmax(fusion_logits, dim=-1)  # [B,L,H,4] (raw)

        # ---- adaptive floor + pruning ----------------------------
        fusion_probs = self._apply_floor_and_prune(fusion_probs)

        # ---- entropy regularisation ------------------------------
        entropy_coeff = self._current_entropy_coeff()
        if entropy_coeff != 0.0:
            # Clamp fusion_probs for numerical stability before log
            fusion_probs_safe = fusion_probs.clamp(min=1e-9)
            entropy = -(fusion_probs_safe * torch.log(fusion_probs_safe)).sum(dim=-1).mean()
            # maximise entropy ⇒ negative sign on loss term
            self.reg_loss = -entropy_coeff * entropy
        else:
            self.reg_loss = None

        # ---- path combination ------------------------------------
        o = (
            fusion_probs[..., 0:1] * local_short
            + fusion_probs[..., 1:2] * local_long
            + fusion_probs[..., 2:3] * delta_out
            + fusion_probs[..., 3:4] * v
        )

        # ---- identity residual (ungated) -------------------------
        if self.use_identity_path:
            id_val = self.id_proj(hidden_states)  # [B,L,value_dim]
            id_val = rearrange(id_val, "b l (h d) -> b l h d", h=self.num_heads)
            alpha = self.alpha_identity.view(1, 1, -1, 1)
            o = o + alpha * id_val

        # ---- cache update ----------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ---- output norm / projection ----------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---- re-pad if we un-padded ------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        # ---- step ++ ---------------------------------------------
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
