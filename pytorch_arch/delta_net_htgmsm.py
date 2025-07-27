# -*- coding: utf-8 -*-
"""
DeltaNet – Hierarchical Two-Stage Gated Multi-Scale Memory (HTG-MSM)
===================================================================
Identifier: delta_net_htgmsm

Core innovations (implemented in this file)
------------------------------------------
1. **Hierarchical Two-Stage Gating (HTG)**
   •  First stage chooses *Local* vs *Global* memory groups with a per-token,
      per-head softmax (coarse gate).
   •  Second stage distributes each group’s probability mass across its
      internal paths with another softmax (fine gates).
   •  Paths:  ─ Local  : {Direct-Value 𝑉, Short-EMA 𝑬ₛ}
              ─ Global : {Delta        Δ, Long-EMA  𝑬ₗ}
   •  This reduces gate entropy (only 2+2 logits instead of one flat 4-way
      softmax) and makes it easier for the model to focus on a single group
      before specialising within it – directly addressing the *path dilution*
      bottleneck identified in experimental evidence.

2. **Per-Head Learnable Temperatures** for both stages enabling adaptive gate
   sharpness without manual scheduling.

3. **Bias Initialisation**
   •  Coarse gate biased towards the *Local* group (identity/value) to protect
      optimisation in early training.
   •  Fine-Local gate biased towards direct value   (𝑉).
   •  Fine-Global gate biased towards delta path    (Δ).
   These biases follow research on curriculum gating and correct the warm-start
   bug highlighted in previous variants.

4. **Dual-Scale EMA** with carefully chosen *a-priori* timescales:
   •  Short-EMA:  γ ≈ 0.05  (fast – captures recent context)
   •  Long-EMA :  γ ≈ 0.95  (slow – keeps long-term memory)
   Biases on the decay projection layers are set accordingly so the network
   starts with meaningful, non-destructive initialisation as recommended by
   Hyena/S4 literature.

5. **Fully O(N) causal computation**
   •  Re-uses the proven `delta_rule_chunkwise` kernel for the Δ path.
   •  Implements chunk-wise EMA for both scales.
   •  All operations are element-wise or chunk-wise linear – no quadratic
     softmax attention anywhere.

6. **Universal einops usage & Batch Agnosticism** – all reshapes via
   `einops.rearrange`, dimensions inferred from runtime tensors, never from
   config constants.

The class name and `forward` signature are unchanged, ensuring drop-in
compatibility with existing training/evaluation pipelines.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

################################################################################
# Helper functions                                                             #
################################################################################

def _elu_p1(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 (RetNet / Hyena convention – keeps positives)."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalise along last dim (used as optional q/k normalisation)."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

################################################################################
# O(N) chunk-wise kernels (Δ-rule & EMA)                                       #
################################################################################

@torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Fast associative Δ-rule – identical to prior proven implementation."""
    b, h, l, d_k = q.shape
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    l_pad = l + pad_len

    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
                           (q, k, v, k_beta))

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)
    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(l_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(strict_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :l]
    return o, S


@torch.compile
def ema_rule_chunkwise(v: torch.Tensor,  # (b h l d_v)
                        gamma: torch.Tensor,  # (b h l)
                        init_state: Optional[torch.Tensor] = None):
    """Chunk-wise causal EMA (stateful) – O(N d)."""
    b, h, l, d_v = v.shape
    ema_out = torch.empty_like(v)
    state = torch.zeros((b, h, d_v), dtype=v.dtype, device=v.device) if init_state is None else init_state
    for t in range(l):
        g_t = gamma[:, :, t].unsqueeze(-1)
        state = g_t * state + (1.0 - g_t) * v[:, :, t]
        ema_out[:, :, t] = state
    return ema_out, state

################################################################################
# Hierarchical two-stage gate                                                  #
################################################################################

class HierarchicalGate(nn.Module):
    """Per-token, per-head hierarchical gate producing weights for 4 paths.

    Stage-1 (coarse): Local vs Global  → probabilities p_L, p_G.
    Stage-2 (fine)  : within each group (2 paths each) producing q_V, q_Es
                      and r_Δ, r_El respectively.
    Final weights   : [V, Es, Δ, El] = [p_L*q_V, p_L*q_Es, p_G*r_Δ, p_G*r_El]
    """

    def __init__(self, hidden_size: int, num_heads: int, temp_init: float = 1.0):
        super().__init__()
        self.num_heads = num_heads

        # Shared trunk MLP (lightweight)
        hid = max(8, hidden_size // 2)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, hid),
            nn.SiLU(),
        )
        # Output projections
        self.coarse_proj = nn.Linear(hid, num_heads * 2)   # Local / Global
        self.local_proj  = nn.Linear(hid, num_heads * 2)    # V / Es
        self.global_proj = nn.Linear(hid, num_heads * 2)    # Δ / El

        # Bias initialisation following curriculum insights
        nn.init.constant_(self.coarse_proj.bias, 1.0)   # favour *Local* initially
        # local fine-gate bias: favour V
        bias_local = torch.zeros(num_heads * 2)
        bias_local[::2] = 1.0  # path-0 (V) has +1
        self.local_proj.bias.data.copy_(bias_local)
        # global fine-gate bias: favour Δ
        bias_global = torch.zeros(num_heads * 2)
        bias_global[::2] = 1.0  # path-0 (Δ) has +1
        self.global_proj.bias.data.copy_(bias_global)

        # Learnable per-head temperature (>0) for both stages
        self.log_temp_coarse = nn.Parameter(torch.log(torch.tensor(temp_init)) * torch.ones(num_heads))
        self.log_temp_fine   = nn.Parameter(torch.log(torch.tensor(temp_init)) * torch.ones(num_heads))

    def _softmax_h(self, logits: torch.Tensor, temp: torch.Tensor):
        # logits: (b l h k), temp:(h,) – broadcast along (b,l)
        logits = logits / temp.view(1, 1, -1, 1)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return gate weights with shape (b, l, h, 4) in order [V, Es, Δ, El]."""
        b, l, _ = x.shape
        h = self.num_heads
        z = self.trunk(x)  # (b, l, hid)

        # ---- Stage-1: coarse Local/Global ----
        coarse_logits = rearrange(self.coarse_proj(z), "b l (h k) -> b l h k", h=h, k=2)
        temp_c = F.softplus(self.log_temp_coarse) + 1e-4
        pg = self._softmax_h(coarse_logits, temp_c)  # (b l h 2)
        p_local, p_global = pg[..., 0:1], pg[..., 1:2]  # keep last dim size=1 for broadcasting

        # ---- Stage-2: fine gates ----
        local_logits = rearrange(self.local_proj(z),  "b l (h k) -> b l h k", h=h, k=2)
        global_logits = rearrange(self.global_proj(z), "b l (h k) -> b l h k", h=h, k=2)
        temp_f = F.softplus(self.log_temp_fine) + 1e-4
        q = self._softmax_h(local_logits,  temp_f)  # (b l h 2)
        r = self._softmax_h(global_logits, temp_f)  # (b l h 2)

        # Combine hierarchically
        w_v  = p_local * q[..., 0:1]   # (b l h 1)
        w_es = p_local * q[..., 1:2]
        w_delta = p_global * r[..., 0:1]
        w_el   = p_global * r[..., 1:2]

        weights = torch.cat([w_v, w_es, w_delta, w_el], dim=-1)  # (b l h 4)
        return weights  # Already sums to 1 per token/head

################################################################################
# Main DeltaNet class                                                          #
################################################################################

class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet layer with Hierarchical Two-Stage Gated Multi-Scale Memory."""

    def __init__(
        self,
        *,
        mode: str = "htgmsm",
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
        gate_temp_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---------------- Book-keeping ----------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.mode = mode
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
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # --------------- Dimensions ---------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0

        # --------------- Linear projections ---------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # EMA decay projections – two distinct scales
        # NOTE: bias=True is REQUIRED here because we set biases to specific
        # values (≈logit of 0.05 / 0.95). Setting bias=False would have caused
        # an AttributeError when trying to access `.bias` and, more critically,
        # would remove the intended warm-start behaviour.
        self.dec_proj_short = nn.Linear(hidden_size, num_heads, bias=True)
        self.dec_proj_long  = nn.Linear(hidden_size, num_heads, bias=True)
        # Bias init: sigmoid(bias) ≈ γ ; want γ_s≈0.05 , γ_l≈0.95
        self.dec_proj_short.bias.data.fill_(-2.9444)  # sigmoid ≈ 0.05
        self.dec_proj_long.bias.data.fill_(2.9444)    # sigmoid ≈ 0.95

        # Hierarchical gate
        self.h_gate = HierarchicalGate(hidden_size, num_heads, temp_init=gate_temp_init)

        # Short convolution (mandatory as per requirements)
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu")
        else:
            raise UserWarning("ShortConvolution is crucial; do not disable it.")

        # Output normalisation & projection
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
    ) -> Tuple[torch.Tensor, None, Optional[Dict]]:
        # -------- Input unpadding (optional) --------
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask must be (batch, seq_len)."
        batch_size, seq_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # -------- Linear projections + optional conv --------
        if self.use_short_conv:
            cs_q = cs_k = cs_v = None
            if last_state is not None and last_state.get("conv_state") is not None:
                cs_q, cs_k, cs_v = last_state["conv_state"]
            q, cs_q = self.q_conv1d(self.q_proj(hidden_states), cache=cs_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, cs_k = self.k_conv1d(self.k_proj(hidden_states), cache=cs_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, cs_v = self.v_conv1d(self.v_proj(hidden_states), cache=cs_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:  # never reached per design
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        # -------- Head reshape + activations --------
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # -------- β scaling for Δ path --------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -------- Compute Δ path (chunk-wise) --------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        rec_prev = last_state.get("recurrent_state") if last_state else None
        delta_out_d, rec_state = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out_d, "b h l d -> b l h d")

        # -------- EMA paths --------
        # Short EMA
        gamma_short = self.dec_proj_short(hidden_states).sigmoid()  # (b l h)
        gamma_s_d = rearrange(gamma_short, "b l h -> b h l")
        ema_s_prev = last_state.get("ema_state_s") if last_state else None
        ema_s_d, ema_s_state = ema_rule_chunkwise(v_d, gamma_s_d, ema_s_prev)
        ema_s = rearrange(ema_s_d, "b h l d -> b l h d")
        # Long EMA
        gamma_long = self.dec_proj_long(hidden_states).sigmoid()
        gamma_l_d = rearrange(gamma_long, "b l h -> b h l")
        ema_l_prev = last_state.get("ema_state_l") if last_state else None
        ema_l_d, ema_l_state = ema_rule_chunkwise(v_d, gamma_l_d, ema_l_prev)
        ema_l = rearrange(ema_l_d, "b h l d -> b l h d")

        # -------- Hierarchical gating --------
        weights = self.h_gate(hidden_states)  # (b l h 4)
        w_v, w_es, w_delta, w_el = weights.unbind(dim=-1)
        # add channel dim for broadcasting
        w_v = w_v.unsqueeze(-1)
        w_es = w_es.unsqueeze(-1)
        w_delta = w_delta.unsqueeze(-1)
        w_el = w_el.unsqueeze(-1)

        o = w_v * v + w_es * ema_s + w_delta * delta_out + w_el * ema_l  # (b l h d)

        # -------- Cache update --------
        if past_key_values is not None and use_cache:
            layer_state = {
                "recurrent_state": rec_state,
                "conv_state": (cs_q, cs_k, cs_v) if self.use_short_conv else None,
                "ema_state_s": ema_s_state,
                "ema_state_l": ema_l_state,
                "layer_idx": self.layer_idx,
                "offset": seq_len,
            }
            if hasattr(past_key_values, "__setitem__") and self.layer_idx is not None:
                past_key_values[self.layer_idx] = layer_state
            else:
                past_key_values.update(layer_state)

        # -------- Output norm / projection --------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # -------- Re-padding if needed --------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, None, past_key_values
