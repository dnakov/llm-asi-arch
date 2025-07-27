# -*- coding: utf-8 -*-
"""
DeltaNet – Annealed Entropy-KL Fusion with Temperature Floor (delta_net_annealed_eklf)
===================================================================================
This evolutionary variant builds directly on **delta_net_entropy_kl_floor_gate** and
addresses the two residual weaknesses identified in the experimental evidence:

1. *Over-Regularisation in Late Training* – the fixed-strength Entropy+KL loss keeps
   all paths active but begins to **impede sharp, single-path routing** required by
   selective inference tasks (Winogrande, PIQA).  We therefore **anneal** the
   regulariser **per call** through a user-supplied `reg_schedule∈[0,1]` scalar that
   typically represents *training progress* (0 ⇒ start, 1 ⇒ end).  By default the
   schedule is `0`, preserving baseline behaviour.  Entropy / KL weights decay as

       w_eff = w_init * (1 − reg_schedule)  ,  clamped to a minimum of 10 % of
       the initial value so as not to collapse path diversity entirely.

2. *Unbounded Path Temperatures* – earlier per-head temperatures could shrink to
   extremely small values, creating brittle, near-binary routing that hurt span
   tasks.  We replace simple `exp(log_temp)` with a **softplus-with-offset**
   parameterisation that **guarantees τ ≥ τ_min (default = 0.25)** while still
   allowing arbitrarily large temperatures.

3. *Structural Minimum Floor* – even with learnable floors the optimiser could
   drive all context paths arbitrarily close to zero.  A **hard minimum floor
   (`hard_floor`) is now enforced** on *every* path to guarantee at least a
   residual flow of information (< 1 % of probability mass by default).  The
   learnable floor (via sigmoid) allocates only the *excess* above this hard
   base, preserving flexibility without starvation.

All public APIs are preserved; the only new inputs are optional:
    • forward(..., reg_schedule: float | None = None)

The implementation keeps O(N) complexity, strict causality, and full batch
agnosticism.  It re-uses the proven chunkwise Δ-rule kernel and causal FIR
branches from previous variants.

IMPORTANT
---------
The original implementation *unpadded* the input sequences and concatenated them
into a single long sequence when an `attention_mask` was provided.  Whilst this
is a common optimisation for Flash-/xformers-style attention kernels that can
rely on `cu_seqlens`, our custom **_delta_rule_chunkwise** kernel does *not*
consume `cu_seqlens` and therefore cannot distinguish sequence boundaries.  As a
result, tokens from one sequence could (legitimately) interact with *earlier*
tokens of another sequence – an information leak across the batch dimension.
Although still causal in the temporal sense, this violates the independence of
parallel samples and must be fixed.

The fix is minimal: we simply keep the original padded [B, L, D] layout whenever
we invoke **_delta_rule_chunkwise**.  The small amount of extra compute from the
(potential) padding is negligible compared to the correctness benefit and does
not alter the innovative architecture in any way.
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

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # positive ELU
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # sum-normalise last dim
    s = x.sum(-1, keepdim=True)
    s = s + 1e-6  # Prevent division by zero
    return (x / s).to(x)

# -----------------------------------------------------------------------------
# Depth-wise causal FIR conv (Dirac initialisation)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity kernel
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule (unchanged)
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

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    att_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        att_inv[..., i, :i] += (att_inv[..., i, :, None].clone() * att_inv[..., :, :i].clone()).sum(-2)
    eye = torch.eye(chunk_size, dtype=att_inv.dtype, device=q.device)
    att_inv = att_inv + eye

    u = att_inv @ v
    w = att_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    strict_mask = torch.triu(tri_mask, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local_attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + local_attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Entropy + KL gated fusion with temperature floor & annealing
# -----------------------------------------------------------------------------

class _AnnealedEKLGate(nn.Module):
    """Fusion gate with annealed entropy/KL regularisation and temperature floor."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        hard_floor: float = 0.005,
        learnable_floor_max: float = 0.07,
        init_entropy_w: float = 0.04,
        init_kl_w: float = 0.04,
        tau_min: float = 0.25,
        mlp_hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = 4
        self.tau_min = float(tau_min)
        self.hard_floor = float(hard_floor)
        self.learnable_floor_max = float(learnable_floor_max)
        # --------------------------------------------------------------
        self.log_temp_param = nn.Parameter(torch.zeros(num_heads, self.n_paths))  # unconstrained
        # learnable extra floor (sigmoid) per head/path
        self.floor_param = nn.Parameter(torch.full((num_heads, self.n_paths), -2.0))
        # --------------------------------------------------------------
        gate_in_dim = hidden_size + 16 * num_heads  # 4 stats * 4 paths * H
        hidden_dim = hidden_size * mlp_hidden_mult // 2
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * self.n_paths, bias=True),
        )
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            # favour value path initially
            self.mlp[-1].bias[self.n_paths - 1 :: self.n_paths] = 2.0
        # initial weights for regularisation
        self.reg_w_entropy_init = float(init_entropy_w)
        self.reg_w_kl_init = float(init_kl_w)
        # holders for logging
        self.last_gate_loss: Optional[torch.Tensor] = None

    @staticmethod
    def _stats(t: torch.Tensor) -> torch.Tensor:  # [B,L,H,D] -> [B,L,H,4]
        mean = t.mean(-1, keepdim=True)
        var = t.var(-1, unbiased=False, keepdim=True)
        abs_m = t.abs().mean(-1, keepdim=True)
        l2 = t.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_m, l2], dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,  # [B,L,D]
        path_short: torch.Tensor,
        path_long: torch.Tensor,
        path_delta: torch.Tensor,
        path_value: torch.Tensor,
        *,
        reg_schedule: float = 0.0,  # 0=start, 1=end
    ) -> torch.Tensor:  # returns weights [B,L,H,4]
        # --------------------------------------------------------------
        # Compile stats -------------------------------------------------
        stats = [
            self._stats(p) for p in (path_short, path_long, path_delta, path_value)
        ]  # each [B,L,H,4]
        stats_flat = [rearrange(s, "b l h s -> b l (h s)") for s in stats]
        gate_in = torch.cat([hidden] + stats_flat, dim=-1)  # [B,L, hidden+16H]
        logits = self.mlp(gate_in)  # [B,L, H*4]
        logits = rearrange(logits, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)
        # temperature with softplus to guarantee tau>=tau_min
        tau = F.softplus(self.log_temp_param) + self.tau_min  # [H,4]
        logits = logits / tau[None, None, :, :]
        probs = torch.softmax(logits, dim=-1)  # [B,L,H,4]
        # Floors --------------------------------------------------------
        learnable_floor = torch.sigmoid(self.floor_param) * self.learnable_floor_max  # [H,4]
        floor_total = self.hard_floor + learnable_floor  # ensure ≥ hard_floor
        floor_total = floor_total.clamp(max=0.25)  # safety
        floor_total = floor_total[None, None, :, :]
        # Numerically stable residual: guarantee sum(floor_total) < 1 by renorm
        sum_floor = floor_total.sum(-1, keepdim=True).clamp(max=0.99)
        norm_floor_total = floor_total / sum_floor * 0.99
        # Blend in with main weights
        # Add a small epsilon to probs for safety
        clipped = torch.maximum(probs, norm_floor_total + 1e-9)  # element-wise max
        weights = clipped / (clipped.sum(-1, keepdim=True) + 1e-8)  # <<-- FIX: add epsilon for numerical stability
        # --------------------------------------------------------------
        # Regularisation (annealed)-------------------------------------
        ent_weight = self.reg_w_entropy_init * (1.0 - reg_schedule) * 0.9 + self.reg_w_entropy_init * 0.1
        kl_weight = self.reg_w_kl_init * (1.0 - reg_schedule) * 0.9 + self.reg_w_kl_init * 0.1
        if self.training and (ent_weight > 0 or kl_weight > 0):
            logw = torch.log(weights + 1e-8)
            entropy = -(weights * logw).sum(-1).mean()
            uniform = torch.full_like(weights, 1.0 / self.n_paths)
            kl = (weights * (logw - math.log(1.0 / self.n_paths))).sum(-1).mean()
            self.last_gate_loss = ent_weight * entropy + kl_weight * kl
        else:
            self.last_gate_loss = None
        return weights

# -----------------------------------------------------------------------------
# Optional typing helper
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

# -----------------------------------------------------------------------------
# Main DeltaNet layer – Annealed EKL Fusion
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with annealed Entropy-KL fusion gate, temperature floor, and hard path floor."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mode: str = "annealed_eklf",
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
        # FIR kernels
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 63,
        # Fusion gate params
        gate_hard_floor: float = 0.005,
        gate_learnable_floor_max: float = 0.07,
        gate_entropy_w: float = 0.04,
        gate_kl_w: float = 0.04,
        gate_tau_min: float = 0.25,
        gate_mlp_hidden_mult: int = 2,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
        # ---------------- basic fields ----------------------------------
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

        # ---------------- dimensions ------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")

        # ---------------- projections -----------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---------------- short conv ------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory.")

        # ---------------- FIR branches ----------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ---------------- Fusion gate -----------------------------------
        self.fusion_gate = _AnnealedEKLGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            hard_floor=gate_hard_floor,
            learnable_floor_max=gate_learnable_floor_max,
            init_entropy_w=gate_entropy_w,
            init_kl_w=gate_kl_w,
            tau_min=gate_tau_min,
            mlp_hidden_mult=gate_mlp_hidden_mult,
        )

        # ---------------- output norm/proj ------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-locals
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore  # noqa: F821
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # kept for API compat
        reg_schedule: float = 0.0,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:  # noqa: F821
        # ------------------------------------------------------------------
        # Basic checks & shapes
        # ------------------------------------------------------------------
        if attention_mask is not None and attention_mask.ndim != 2:
            raise AssertionError("attention_mask must be [batch, seq_len]")
        B, L_in, _ = hidden_states.shape

        # ------------------------------------------------------------------
        # NOTE [Batch-mixing fix]
        # ------------------------------------------------------------------
        # The earlier implementation removed padding by concatenating all valid
        # tokens into a single long sequence (batch=1).  Because our custom
        # _delta_rule_chunkwise kernel has *no* notion of sequence boundaries,
        # this led to information leakage across different samples inside the
        # same batch.  We therefore keep the original padded layout.  The
        # optional get_unpad_data / cu_seqlens pathway is left intact for other
        # kernels (e.g. flash attention) but *disabled* here.
        # ------------------------------------------------------------------
        indices = None  # keeps type consistency for later conditionals
        cu_seqlens = None  # ShortConvolution still accepts None

        # ------------------------------------------------------------------
        # Retrieve cache ----------------------------------------------------
        # ------------------------------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # ------------------------------------------------------------------
        # Projections + (optional) convolution -----------------------------
        # ------------------------------------------------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ------------------------------------------------------------------
        # Reshape to heads --------------------------------------------------
        # ------------------------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ------------------------------------------------------------------
        # Activation / normalisation ---------------------------------------
        # ------------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ------------------------------------------------------------------
        # Beta ----------------------------------------------------------------
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # Δ-rule global path (chunkwise, causal) ----------------------------
        # ------------------------------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ------------------------------------------------------------------
        # FIR paths ----------------------------------------------------------
        # ------------------------------------------------------------------
        value_path = v  # identity
        short_path = self.fir_short(value_path)
        long_path = self.fir_long(value_path)

        # ------------------------------------------------------------------
        # Fusion gate --------------------------------------------------------
        # ------------------------------------------------------------------
        fusion_w = self.fusion_gate(
            hidden_states,
            short_path,
            long_path,
            delta_out,
            value_path,
            reg_schedule=float(reg_schedule),
        )  # [B,L,H,4]

        # ------------------------------------------------------------------
        # Final mix ----------------------------------------------------------
        # ------------------------------------------------------------------
        o = (
            fusion_w[..., 0:1] * short_path
            + fusion_w[..., 1:2] * long_path
            + fusion_w[..., 2:3] * delta_out
            + fusion_w[..., 3:4] * value_path
        )

        # ------------------------------------------------------------------
        # Cache update -------------------------------------------------------
        # ------------------------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_in,
            )

        # ------------------------------------------------------------------
        # Output norm / proj -------------------------------------------------
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ------------------------------------------------------------------
        # (No) re-pad step ---------------------------------------------------
        # ------------------------------------------------------------------
        # We did not unpad, so the tensor already has shape [B, L, D].  The
        # old pad_input() pathway is therefore unnecessary and safely skipped.
        # ------------------------------------------------------------------

        return o, None, past_key_values
