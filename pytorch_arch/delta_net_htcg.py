# -*- coding: utf-8 -*-
"""
DeltaNet – Hierarchical Temperature-Controlled Gating (delta_net_htcg)
=====================================================================
This evolutionary DeltaNet variant introduces a *two-stage hierarchical gate* with a
learnable temperature per head.  The design directly targets the precision/recall
trade-off observed in previous experiments:

1. **Stage-1 (Global vs Local) Sigmoid Gate**
   • A lightweight projection produces per-token, per-head logits.  A sigmoid maps
     this to a probability **w_global ∈ (0,1)** allocated to the *Delta* (global)
     path, while **w_local = 1−w_global** is routed to a *local fusion* mixture.
   • A small positive bias initialises the gate toward the *Delta* path to preserve
     long-range reasoning from the first steps, fixing the bug highlighted in
     earlier analyses.

2. **Stage-2 (Local Path Softmax) Temperature Gate**
   • The remaining mass *w_local* is distributed across the three local branches
     – (Value, Local-Short FIR, Local-Long FIR) via a temperature-controlled
     softmax.  Each head owns its **learnable temperature τ_h > 0** (realised via
     soft-plus).  Lower τ sharpens selection, higher τ smooths blending, allowing
     the model to adaptively control gate entropy during training, thereby
     recovering local precision without sacrificing flexibility.

3. **Identity-initialised FIR Kernels** with small noise ensure that the two FIR
   branches start as near-copies of the Value path yet remain decorrelated enough
   for early learning – balancing signal preservation and gradient richness.

4. **All other mechanics (chunk-wise Delta kernel, ShortConv pre-conditioning,
   causal masking, cache handling) are inherited unchanged**, guaranteeing
   sub-quadratic O(N) complexity, strict causality, and interface compatibility.

The entire implementation respects the developer constraints:
• class name *DeltaNet* and forward signature unchanged
• einops.rearrange used for every reshape/view
• batch-size agnostic – no hard-coded dimensions
• @torch.compile retained on the heavy kernels only

"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, TYPE_CHECKING
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (ELU + 1) – keeps activations positive while smooth."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Row-wise sum normalisation (probability over last dim)."""
    return (x / x.sum(-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Delta rule – unchanged from previous variants (except for dtype fix)
# -----------------------------------------------------------------------------

@torch.compile  # type: ignore[misc]
def delta_rule_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, chunk_size: int = 32):
    """Original O(N) Delta kernel with chunk-wise inversion – keeps causality.

    NOTE: A small dtype mismatch bug has been fixed.  The inversion buffer
    `attn_inv` is now cast to `q.dtype` (which matches the rest of the tensors)
    instead of being hard-coded to `torch.bfloat16`.  This preserves the desired
    memory/performance characteristics when the model is run in bfloat16/float16
    **and** prevents runtime errors stemming from mixed precision matmuls.
    """
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # reshape into chunks of length `chunk_size`
    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

    # build causal masks (shared across batch/head for efficiency)
    mask_upper_tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)

    # (H, C, C) – strictly lower-triangular inverse term
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_upper_tri, 0)

    # recursive inverse update (causal)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)

    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)

    # ------------------------------------------------------------------
    # DTYPE FIX – keep everything in the same precision as the incoming tensors
    # ------------------------------------------------------------------
    attn_inv = attn_inv.to(q.dtype)

    # perform chunk-wise solves
    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution with identity initialisation + noise
# -----------------------------------------------------------------------------

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR convolution (1-D)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31, noise_std: float = 1e-3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Parameter shape: (H, D, K)
        self.filters = nn.Parameter(torch.zeros(num_heads, head_dim, self.kernel_size))
        with torch.no_grad():
            # Identity kernel – last tap = 1
            self.filters[..., -1] = 1.0
            if noise_std > 0:
                self.filters.add_(noise_std * torch.randn_like(self.filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H, D)
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Main DeltaNet class – Hierarchical Temperature Controlled Gate
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # local alias for typing only

class DeltaNet(nn.Module):
    """DeltaNet layer with hierarchical two-stage gating and learnable temperature."""

    def __init__(
        self,
        *,
        mode: str = "htcg",  # hierarchical temperature-controlled gating
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
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # Gating hyper-params
        global_gate_bias: float = 0.5,  # favour delta path a bit at init (sigmoid(~0.62))
        value_path_bias: float = 2.0,   # bias inside local softmax toward value path
        temp_init: float = 1.0,         # initial temperature τ
        **kwargs,  # absorb extra
    ) -> None:
        super().__init__()
        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}
        if d_model is not None:
            hidden_size = d_model

        # ------------- Basic attributes -------------
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.allow_neg_eigval = allow_neg_eigval

        # ------------- Dimensions -------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # ------------- Projections -------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------- Short convolutions -------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet-HTCG.")

        # ------------- FIR branches -------------
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel)

        # ------------- Hierarchical gate parameters -------------
        # Stage-1 (global vs local) sigmoid gate
        self.global_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        nn.init.constant_(self.global_gate_proj.bias, global_gate_bias)
        # Stage-2 (local 3-way softmax) logits proj
        self.local_gate_proj = nn.Linear(hidden_size, num_heads * 3, bias=True)
        with torch.no_grad():
            # Bias order: [value, local_short, local_long]
            bias = self.local_gate_proj.bias.view(num_heads, 3)
            bias[:, 0] = value_path_bias  # favour value path early
            bias[:, 1:].zero_()
        # Per-head learnable log temperature ( >0 after softplus )
        self.log_temp = nn.Parameter(torch.full((num_heads,), math.log(math.e * temp_init)))

        # ------------- Output norm/gate -------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [batch, seq_len]"
        B, L_in, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # Remove padding for variable length batches
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # ------------- Q/K/V projections (with short conv) -------------
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ------------- Head split -------------
        q, k = map(lambda t: rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # ------------- Activation & normalisation on Q/K -------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ------------- Beta scaling factor -------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  # (B,L,H)
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------- Delta path -------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        recurrent_state_prev = last_state.get("recurrent_state") if last_state else None
        delta_out_d, recurrent_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # ------------- Local paths -------------
        v_direct = v  # (b l h d)
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ------------- Stage-1 gate (global vs local) -------------
        global_logit = self.global_gate_proj(hidden_states)  # (B,L,H)
        w_global = torch.sigmoid(global_logit)  # (B,L,H)
        w_local = 1.0 - w_global  # (B,L,H)
        w_global = rearrange(w_global, "b l h -> b l h 1")  # for broadcasting
        w_local = rearrange(w_local, "b l h -> b l h 1")

        # ------------- Stage-2 gate (local 3-way softmax) -------------
        local_logits = self.local_gate_proj(hidden_states)  # (B,L,H*3)
        local_logits = rearrange(local_logits, "b l (h p) -> b l h p", h=self.num_heads, p=3)
        # Temperature scaling per head
        tau = F.softplus(self.log_temp).to(local_logits)  # (H)
        local_logits = local_logits / rearrange(tau, "h -> 1 1 h 1")
        local_weights = F.softmax(local_logits, dim=-1)  # (B,L,H,3)
        local_weights = rearrange(local_weights, "b l h p -> b l h p 1")

        # ------------- Combine outputs -------------
        # Stack local paths in same order as weights: [value, short, long]
        local_stack = torch.stack([v_direct, local_short, local_long], dim=3)  # (B,L,H,3,D)
        local_out = (local_weights * local_stack).sum(dim=3)  # (B,L,H,D)

        o = w_global * delta_out + w_local * local_out  # (B,L,H,D)

        # ------------- Cache update -------------
        if use_cache and past_key_values is not None and self.layer_idx is not None:
            layer_state = {
                "recurrent_state": recurrent_state,
                "conv_state": (conv_state_q, conv_state_k, conv_state_v),
                "layer_idx": self.layer_idx,
                "offset": L_in,
            }
            if hasattr(past_key_values, "__setitem__"):
                past_key_values[self.layer_idx] = layer_state
            elif hasattr(past_key_values, "update"):
                past_key_values.update(layer_state)

        # ------------- Output norm & projection -------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # Re-pad if removed padding earlier
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B, L_in)

        return o, None, past_key_values
