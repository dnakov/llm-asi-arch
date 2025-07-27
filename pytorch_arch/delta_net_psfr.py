# -*- coding: utf-8 -*-
"""
DeltaNet – Parallel Sigmoid Fusion with Retention (PSF-R)
=========================================================
Identifier: delta_net_psfr

Key Innovations
---------------
1. **Parallel (non-competitive) Sigmoid Fusion**
   Each memory path (Short-FIR, Long-FIR, Δ-rule, Value) receives an *independent*
   gating weight in the range **[ε, 1]**.  This removes the probability–simplex
   budget that previously forced an unavoidable trade-off between local and
   global context capacity.  The gates are produced per-token *and* per-head by
   a lightweight MLP that consumes the hidden state **plus per-path norm
   statistics**.  The design draws on the *Parallel-MoE* literature as well as
   findings from ReGLA and Block-State Transformers showing that additive
   fusion unlocks simultaneous gains on local and global benchmarks.

2. **Identity-Preserving Depth-wise FIR Memory**
   Two causal depth-wise FIR branches provide short-range *(kernel=3)* and
   long-range *(kernel=63)* local context.  Both are **Dirac-initialised** so
   they start as an identity mapping, avoiding early oversmoothing.

3. **Per-Head Retention (λ) in the Δ-Kernel**
   Following TransNormer-LLM, a learnable per-head retention factor extends the
   associative Δ-rule with controllable memory horizon.  The parameter is
   constrained to the interval **[λ_min, 1]** to prevent premature forgetting.

4. **Adaptive Temperature & Minimum-Flow ε**
   Gating sharpness is controlled by a learnable per-head temperature.  A
   small, fixed ε (default 0.02) guarantees gradient flow to each path during
   the earliest training steps.

All changes are fully **O(N)**, strictly causal, and batch-size agnostic.  The
public API (`DeltaNet.__init__`, `forward`) is unchanged, making the layer a
plug-and-play replacement for previous variants.
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

# ---------------------------------------------------------------------------
# Utility helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Shifted ELU keeping output strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Normalise so that the last dimension sums to one."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# ---------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac init) ----------------------------
# ---------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR with Dirac initialisation.

    Parameter shape: (H, D, K) where H=num_heads, D=head_dim, K=kernel_size.
    """

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # identity / Dirac
            if noise_std > 0:
                weight.add_(torch.randn_like(weight) * noise_std)
        self.filters = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Chunk-wise Δ-rule with optional per-head retention ------------------------
# ---------------------------------------------------------------------------


@torch.compile  # noqa: D401
# pylint: disable=too-many-locals,too-many-statements

def _retention_delta_chunkwise(
    q: torch.Tensor,  # [B,H,L,Dk]
    k: torch.Tensor,  # [B,H,L,Dk]
    v: torch.Tensor,  # [B,H,L,Dv]
    beta: torch.Tensor,  # [B,H,L]
    forget: Optional[torch.Tensor] = None,  # [B,H] or None
    *,
    chunk_size: int = 32,
):
    """Efficient O(N) associative Δ-kernel with per-head forgetting."""

    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_spec = (0, 0, 0, pad_len)
        q = F.pad(q, pad_spec)
        k = F.pad(k, pad_spec)
        v = F.pad(v, pad_spec)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Feature normalisation ------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape --------------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    tri_mask = torch.triu(torch.ones_like(eye, dtype=torch.bool), 0)
    strict_mask = torch.triu(torch.ones_like(eye, dtype=torch.bool), 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + eye

    u = inv @ v
    w = inv @ k_beta

    S = q.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)

    lam = None
    if forget is not None:
        lam = forget[..., None, None]  # [B,H,1,1]

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        if lam is None:
            S = S + k_i.transpose(-1, -2) @ u_i
        else:
            S = S * lam + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S.detach()

# ---------------------------------------------------------------------------
# Parallel (additive) sigmoid fusion gate ----------------------------------
# ---------------------------------------------------------------------------


class _ParallelSigmoidGate(nn.Module):
    """Independent sigmoid gates per path with ε-floor and learnable temp."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        hidden_mult: int = 2,
        eps_floor: float = 0.02,
        temp_init: float = 1.0,
        # Bias order: short, long, delta, value
        bias_init: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 3.0),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.eps_floor = eps_floor

        in_dim = hidden_size + num_heads * 4  # hidden + 4 per-head stats (mean|x|)
        hid = hidden_size * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid, bias=True),
            nn.GELU(),
            nn.Linear(hid, num_heads * 4, bias=True),
        )
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor(bias_init * num_heads, dtype=self.mlp[-1].bias.dtype))

        # Learnable per-head temperature (positive)
        self.log_temp = nn.Parameter(torch.log(torch.full((num_heads,), temp_init)))

        # Stats placeholders for logging
        self.last_entropy: Optional[float] = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # [B,L,in_dim]
        b, l, _ = feat.shape
        h = self.num_heads

        logits = rearrange(self.mlp(feat), "b l (h c) -> b l h c", h=h, c=4)
        temp = torch.exp(self.log_temp).view(1, 1, h, 1)
        logits = logits / temp

        sig = torch.sigmoid(logits)  # [B,L,H,4] in (0,1)
        p = self.eps_floor + (1.0 - self.eps_floor) * sig  # ensure ≥ ε

        # entropy for logging
        with torch.no_grad():
            ent = -(p * torch.log(p + 1e-8)).sum(-1).mean().item()
            self.last_entropy = ent

        return p  # [B,L,H,4]

# ---------------------------------------------------------------------------
# Typing helper -------------------------------------------------------------
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # type: ignore

# ---------------------------------------------------------------------------
# Main DeltaNet implementation ---------------------------------------------
# ---------------------------------------------------------------------------


class DeltaNet(nn.Module):  # noqa: D401
    """DeltaNet layer with Parallel Sigmoid Fusion and Retention Δ-kernel."""

    def __init__(
        self,
        # ---- base params ---------------------------------------------
        mode: str = "psfr",
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
        # ---- retention params ----------------------------------------
        use_retention: bool = True,
        retention_min: float = 0.6,
        retention_init: float = 1.0,
        # ---- FIR kernels --------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 63,
        # ---- fusion gate params ------------------------------------
        fusion_hidden_mult: int = 2,
        fusion_eps_floor: float = 0.02,
        fusion_temp_init: float = 1.0,
        # -------------------------------------------------------------
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.mode = mode
        self.use_retention = use_retention
        self.retention_min = retention_min

        # ---- dimensional bookkeeping --------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("key/value dims must divide num_heads")

        # ---- linear projections -------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- retention λ per head -----------------------------------
        if use_retention:
            ratio = (retention_init - retention_min) / (1.0 - retention_min)
            ratio = float(max(min(ratio, 1 - 1e-4), 1e-4))
            init_logit = math.log(ratio) - math.log(1 - ratio)
            self.retention_param = nn.Parameter(init_logit * torch.ones(num_heads))
        else:
            self.register_parameter("retention_param", None)

        # ---- mandatory short convolution ----------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- FIR memory branches ------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ---- fusion gate -------------------------------------------
        self.fusion_gate = _ParallelSigmoidGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            hidden_mult=fusion_hidden_mult,
            eps_floor=fusion_eps_floor,
            temp_init=fusion_temp_init,
        )

        # ---- output norm / projection ------------------------------
        if use_gate:
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
        output_attentions: Optional[bool] = False,  # kept for API comp
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be [batch, seq_len]"
        B_orig, L_orig, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ---- projections + short conv ----------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---- head split --------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # ---- activations / norms ---------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- beta gate ---------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- retention λ --------------------------------------------------
        if self.use_retention:
            lam = self.retention_min + (1.0 - self.retention_min) * torch.sigmoid(self.retention_param)
            lam = lam.unsqueeze(0).expand(q.shape[0], -1)  # [B,H]
        else:
            lam = None

        # ---- Δ-kernel ----------------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v_direct, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recur_new = _retention_delta_chunkwise(q_d, k_d, v_d, beta_d, forget=lam)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---- FIR memory branches ----------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ---- gate feature construction ----------------------------------
        def _norm(t: torch.Tensor) -> torch.Tensor:
            return t.abs().mean(dim=-1)  # [B,L,H]

        gate_feat = torch.cat(
            [
                hidden_states,
                rearrange(_norm(local_short), "b l h -> b l (h)"),
                rearrange(_norm(local_long), "b l h -> b l (h)"),
                rearrange(_norm(delta_out), "b l h -> b l (h)"),
                rearrange(_norm(v_direct), "b l h -> b l (h)"),
            ],
            dim=-1,
        )

        # ---- fusion ------------------------------------------------------
        weights = self.fusion_gate(gate_feat)  # [B,L,H,4]
        w_short, w_long, w_delta, w_value = torch.split(weights, 1, dim=-1)
        fused = (
            w_short * local_short +
            w_long * local_long +
            w_delta * delta_out +
            w_value * v_direct
        )

        # ---- cache update -----------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recur_new,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_orig,
            )

        # ---- output norm / projection -----------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            fused = self.o_norm(fused, g_vec)
        else:
            fused = self.o_norm(fused)
        out = self.o_proj(rearrange(fused, "b l h d -> b l (h d)"))

        # ---- re-pad if needed -------------------------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, B_orig, L_orig)

        return out, None, past_key_values
