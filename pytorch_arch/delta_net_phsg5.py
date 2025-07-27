# -*- coding: utf-8 -*-
"""
DeltaNet – Per-Head Simplex Gating with Multi-Scale Local Memory (PHSG-5way)
============================================================================
Identifier: delta_net_phsg5

(See original file header for detailed motivation and description.)

FIX NOTE
--------
The previous implementation performed *global un-padding* by concatenating all
tokens from **every** sequence in the batch into a single long sequence:

    hidden_states = index_first_axis(...).unsqueeze(0)  # -> batch = 1

Subsequent sequential operations (short FIRs, Δ-rule, etc.) therefore mixed
information **across different samples in the batch** – later tokens of sample
*B₁* could "see" earlier tokens of sample *B₀*.  This violates the fundamental
independence assumption between batch elements and constitutes a *causality /
mask correctness* error according to the checking policy.

While token-level un-padding is an effective optimisation, it must be paired
with sequence-boundary aware kernels (e.g. via *cu_seqlens* support) for **all**
stateful paths.  `delta_rule_chunkwise` currently has no such support, so the
safest fix is to **disable global un-padding** for now and operate on the
original `(B,L,·)` tensors.  This preserves correctness at the cost of a small
amount of extra FLOPs, without touching the innovative architecture.

Key changes
~~~~~~~~~~~
1. Removed global un-padding and the corresponding re-padding at the end of
   `forward`.  The `attention_mask` is still checked for shape but is no longer
   used to reshape the batch.
2. `cu_seqlens` is set to `None` for the internal short convolutions – these
   kernels gracefully fall back to standard convs when the argument is absent.
3. All remaining logic and parameters are unchanged, so the model's behaviour
   (apart from the fixed leakage) is identical.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input  # noqa: F401 – kept for future use
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ============================================================================
# Helper utilities
# ============================================================================

def elu_p1(x: torch.Tensor) -> torch.Tensor:  # shifted ELU so output >0
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:  # L1 normalise last dim
    return (x / x.sum(-1, keepdim=True)).to(x)

# ============================================================================
# Depth-wise causal FIR convolution (identity initialisation)
# ============================================================================
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution with δ-kernel initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # (H, D, K)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity at time-step 0 (causal)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")  # groups=h*d
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # left pad – causal
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ============================================================================
# Causal chunk-wise Δ-rule kernel (unchanged, proven baseline)
# ============================================================================
@torch.compile  # type: ignore[misc]
def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal associative Δ-rule evaluated in fixed-size chunks (O(N·d))."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalisation & beta scaling
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk view -> (B,H,N,C,D)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# ============================================================================
# Per-Head Linear Gate (no inter-head mixing)
# ============================================================================
class PerHeadGate(nn.Module):
    """Per-head linear projection producing logits for *n_paths* branches.

    Weight: (H, out, in) so each head is completely independent.
    """

    def __init__(self, hidden_size: int, num_heads: int, n_paths: int):
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = n_paths
        weight = torch.zeros(num_heads, n_paths, hidden_size)
        # kaiming-like init per head
        bound = 1.0 / math.sqrt(hidden_size)
        weight.uniform_(-bound, bound)
        self.weight = nn.Parameter(weight)  # (H, P, D)
        self.bias = nn.Parameter(torch.zeros(num_heads, n_paths))  # (H, P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,D)
        # logits: (B,L,H,P)
        logits = torch.einsum("b l d, h p d -> b l h p", x, self.weight) + self.bias
        return logits

# ============================================================================
# Optional cache typing
# ============================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401

# ============================================================================
# Main DeltaNet Layer (PHSG-5way)
# ============================================================================
class DeltaNet(nn.Module):  # noqa: D401 – name mandated by framework
    """DeltaNet with Per-Head 5-Way Simplex Gating and Multi-Scale Local FIRs."""

    def __init__(
        self,
        *,
        mode: str = "phsg5",
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
        fir_kernel_short: int = 3,
        fir_kernel_mid: int = 15,
        fir_kernel_long: int = 63,
        # Gating parameters
        gate_eps_init: float = 0.02,
        gate_temp_init: float = 1.0,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
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
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---- dimensions ----
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ---- projections ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- optional short convolutions ----
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet stability.")

        # ---- multi-scale FIR branches ----
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_mid = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_mid)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---- per-head simplex gate ----
        self.n_paths = 5  # short, mid, long, delta, value
        self.gate_linear = PerHeadGate(hidden_size, num_heads, self.n_paths)
        # learnable temperature per head
        self.log_temp = nn.Parameter(torch.full((num_heads, 1), math.log(gate_temp_init)))
        # learnable ε-floor per head (clamped in forward)
        self.eps_param = nn.Parameter(torch.full((num_heads, 1), gate_eps_init))

        # ---- output normalisation / projection ----
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _apply_temperature_and_floor(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply per-head temperature and ε-floor to logits then return probs."""
        # logits: (B,L,H,P)
        temp = torch.exp(self.log_temp).view(1, 1, -1, 1)  # (1,1,H,1)
        probs = torch.softmax(logits / temp, dim=-1)
        eps = torch.clamp(self.eps_param, 0.0, 0.2).view(1, 1, -1, 1)
        k = self.n_paths
        probs = probs * (1.0 - k * eps) + eps  # ensure ≥eps & sum-to-1
        return probs

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,  # unused, kept for API
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        # ------------------------------------------------------------------
        # 1. Basic checks & setup
        # ------------------------------------------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
            # The current implementation does *not* perform global un-padding –
            # this avoids cross-batch information leakage.  The mask can still
            # be used by downstream components (not needed inside this layer).
        B, L, _ = hidden_states.shape

        # --- retrieve previous cache (if any) ---
        last_state: Optional[Dict] = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ------------------------------------------------------------------
        # 2. QKV projections + optional short-conv (no un-padding)
        # ------------------------------------------------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        # ShortConvolution kernels accept `cu_seqlens=None` and will default to
        # regular depth-wise 1-D convolutions, which is correct when we keep the
        # batch dimension intact.
        q, conv_state_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        k, conv_state_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        v, conv_state_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=None,
        )

        # ------------------------------------------------------------------
        # 3. Head split
        # ------------------------------------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ------------------------------------------------------------------
        # 4. Activations / normalisation on Q/K
        # ------------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ------------------------------------------------------------------
        # 5. Beta coefficients for Δ-rule
        # ------------------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  # (B,L,H)
        else:
            beta = torch.ones((*hidden_states.shape[:2], self.num_heads), dtype=q.dtype, device=q.device)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # 6. Δ-rule path (causal, chunk-wise)
        # ------------------------------------------------------------------
        delta_out, recurrent_state = delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ------------------------------------------------------------------
        # 7. Multi-scale FIR local memories
        # ------------------------------------------------------------------
        local_short = self.fir_short(v_direct)
        local_mid = self.fir_mid(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------------------------------------------------------------
        # 8. Per-head simplex gating
        # ------------------------------------------------------------------
        gate_logits = self.gate_linear(hidden_states)  # (B,L,H,P)
        fusion_weights = self._apply_temperature_and_floor(gate_logits)  # (B,L,H,P)

        # split weights
        w_short = fusion_weights[..., 0:1]
        w_mid = fusion_weights[..., 1:2]
        w_long = fusion_weights[..., 2:3]
        w_delta = fusion_weights[..., 3:4]
        w_value = fusion_weights[..., 4:5]

        o = (
            w_short * local_short
            + w_mid * local_mid
            + w_long * local_long
            + w_delta * delta_out
            + w_value * v_direct
        )

        # ------------------------------------------------------------------
        # 9. Cache update
        # ------------------------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=L,
            )

        # ------------------------------------------------------------------
        # 10. Output projection & norm
        # ------------------------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # No re-padding needed – batch structure preserved.
        return o, None, past_key_values
