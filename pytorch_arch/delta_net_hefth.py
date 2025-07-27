# -*- coding: utf-8 -*-
"""
DeltaNet – Hybrid Epsilon-Floor Fusion with Talking-Heads Mixing (DeltaNet-HEFTH)
===============================================================================
Identifier: delta_net_hefth

This architecture combines the strongest empirical findings from earlier
DeltaNet variants while rectifying their core weaknesses:

1.  Scheduled ε-floor on the fusion gate
    • Guarantees every path (short-FIR, long-FIR, Δ-memory, value) keeps a
      minimum mixing probability early in training – preventing gradient
      starvation – but linearly decays that floor to **0** over a configurable
      window (``epsilon_anneal_steps``).  This resolves the gate-collapse
      issue that harmed global tasks once the per-head temperature sharpened.

2.  Length-conditioned local-path dampening
    • A smooth scaling factor ``s_local = 1 / (1 + (L / length_scale)**2)``
      down-weights convolutional (short/long) paths on very long sequences,
      mitigating the *local context swamp* that previously devastated
      narrative reasoning (e.g. Lambada).

3.  Talking-Heads cross-head mixer
    • A lightweight, learnable head-mixing matrix (initialised to identity)
      applied after path fusion lets heads exchange information, fixing the
      lack of cross-head communication that hurt ARC/HellaSwag.
      Complexity is O(H²) per token (H ≈ 4) – negligible vs. O(N).

4.  Simplified, efficient implementation
    • The code starts from the proven **MSDAF-HT** backbone, modifying only
      the fusion gate and adding the mixer.  All public APIs, tensor contracts
      and O(N) complexity are preserved.

Default settings enable **all** new features – no config changes required.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ---------------------------------------------------------------------------
# Helper functions (torch.compile-safe)
# ---------------------------------------------------------------------------

def _elu_p1(x: torch.Tensor) -> torch.Tensor:  # shifted ELU (+1)
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # row-sum = 1
    return (x / x.sum(-1, keepdim=True)).to(x)

# ---------------------------------------------------------------------------
# Depth-wise causal FIR conv (unchanged numerics)
# ---------------------------------------------------------------------------
class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.randn(num_heads, head_dim, self.kernel_size) * 0.02
        self.filters = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Causal chunk-wise Δ-rule (identical numerics – kept under @torch.compile)
# ---------------------------------------------------------------------------
@torch.compile
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

    tri_inc = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    att_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_inc, 0)
    for i in range(1, chunk_size):
        att_inv[..., i, :i] += (
            att_inv[..., i, :, None].clone() * att_inv[..., :, :i].clone()
        ).sum(-2)
    att_inv = att_inv + torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    att_inv = att_inv.to(torch.bfloat16)

    u = att_inv @ v
    w = att_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    tri_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        att_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + att_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# ---------------------------------------------------------------------------
# Optional typing imports
# ---------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401

# ---------------------------------------------------------------------------
# Main DeltaNet layer – Hybrid ε-floor Fusion + Talking-Heads
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """DeltaNet with scheduled ε-floor fusion and talking-heads mixing."""

    def __init__(
        self,
        mode: str = "hefth",
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
        # FIR kernel sizes -------------------------------------------------
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # Fusion gate params ---------------------------------------------
        fusion_hidden_mult: int = 2,
        epsilon_floor_init: float = 0.05,
        epsilon_anneal_steps: int = 2000,
        # Talking-heads mixer --------------------------------------------
        enable_head_mixer: bool = True,
        # Length-condition scaling ---------------------------------------
        length_scale: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model

        # Store params ----------------------------------------------------
        self.mode = mode
        self.hidden_size = hidden_size
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
        self.length_scale = float(length_scale)
        self.enable_head_mixer = enable_head_mixer

        # Dimensions ------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")

        # Linear projections ---------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Beta projection -------------------------------------------------
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Short convolutions ---------------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory – do not disable.")

        # FIR convs -------------------------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long)

        # Statistics helper ----------------------------------------------
        def _stats(t: torch.Tensor) -> torch.Tensor:  # mean, abs-mean, rms, l2
            m = t.mean(dim=-1, keepdim=True)
            a = t.abs().mean(dim=-1, keepdim=True)
            rms = torch.sqrt((t ** 2).mean(dim=-1, keepdim=True) + 1e-6)
            l2n = t.norm(dim=-1, keepdim=True)
            return torch.cat([m, a, rms, l2n], dim=-1)
        self._stats = _stats  # type: ignore

        # Fusion gate -----------------------------------------------------
        stats_per_branch = 4  # we aggregate across D -> only 4 scalars per head
        fusion_in_dim = hidden_size + stats_per_branch * num_heads * 4  # 4 branches
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * 4, bias=True),
        )
        # Bias initialisation – favour value path
        with torch.no_grad():
            bias = self.fusion_gate_mlp[-1].bias.view(num_heads, 4)
            bias[:, 3] = 1.5  # value
            bias[:, 2] = 0.2  # delta
            bias[:, 1] = -0.5  # long
            bias[:, 0] = -1.0  # short

        # Learnable per-head log-temperature -----------------------------
        self.gate_log_tau = nn.Parameter(torch.zeros(num_heads))

        # ε-floor scheduling ---------------------------------------------
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.epsilon_floor_init = float(epsilon_floor_init)
        self.epsilon_anneal_steps = int(epsilon_anneal_steps)

        # Talking-heads mixer --------------------------------------------
        if enable_head_mixer:
            mix = torch.eye(num_heads)
            self.head_mix = nn.Parameter(mix)  # shape (H,H)
        else:
            self.head_mix = None

        # Output norm & projection ---------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _current_epsilon(self) -> float:
        step = float(self._step.item())
        if step >= self.epsilon_anneal_steps or self.epsilon_floor_init == 0.0:
            return 0.0
        return self.epsilon_floor_init * (1.0 - step / self.epsilon_anneal_steps)

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape

        # Unpad (for packed KV) -----------------------------------------
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # Retrieve cache -------------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # QKV projections + short conv ----------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # Head split -----------------------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # Activations ----------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # Beta -----------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule ---------------------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v_direct, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out_d, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # FIR paths ------------------------------------------------------
        fir_short = self.fir_short(v_direct)
        fir_long = self.fir_long(v_direct)

        # Length-condition scaling for local paths ----------------------
        seq_scale = 1.0 / (1.0 + (seq_len / self.length_scale) ** 2)
        fir_short = fir_short * seq_scale
        fir_long = fir_long * seq_scale

        # Stats for gate -------------------------------------------------
        stats_concat = torch.cat([
            self._stats(fir_short),
            self._stats(fir_long),
            self._stats(delta_out),
            self._stats(v_direct),
        ], dim=-1)  # (B,L,H, 4*4)
        stats_flat = rearrange(stats_concat, "b l h s -> b l (h s)")
        gate_in = torch.cat([hidden_states, stats_flat], dim=-1)

        # Fusion gate ----------------------------------------------------
        fusion_logits = self.fusion_gate_mlp(gate_in)  # (B,L,H*4)
        fusion_logits = rearrange(fusion_logits, "b l (h c) -> b l h c", h=self.num_heads, c=4)
        tau = torch.exp(self.gate_log_tau)[None, None, :, None]
        fusion_logits = fusion_logits / tau
        fusion_w = torch.softmax(fusion_logits, dim=-1)

        # Apply ε-floor ---------------------------------------------------
        eps = self._current_epsilon()
        if eps > 0.0:
            fusion_w = fusion_w * (1.0 - 4 * eps) + eps

        # Fuse -----------------------------------------------------------
        o = (
            fusion_w[..., 0:1] * fir_short +
            fusion_w[..., 1:2] * fir_long +
            fusion_w[..., 2:3] * delta_out +
            fusion_w[..., 3:4] * v_direct
        )  # (B,L,H,D)

        # Talking-heads mixer -------------------------------------------
        if self.head_mix is not None:
            o = torch.einsum("b l h d, h g -> b l g d", o, self.head_mix)

        # Cache update ---------------------------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # Output norm / projection --------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # Re-pad if necessary -------------------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        # Increment step counter ----------------------------------------
        self._step += 1

        return o, None, past_key_values
