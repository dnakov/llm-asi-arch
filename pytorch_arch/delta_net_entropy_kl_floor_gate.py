# -*- coding: utf-8 -*-
"""
DeltaNet – Entropic Floor+KL Regularized Output-Stat Gating & Monotonic Long-Horizon Memory
=========================================================================================
Identifier: delta_net_entropy_kl_floor_gate

(Original header remains unchanged)
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# ---------------------------------------------------------------------------
# Depthwise causal FIR convolution (Dirac+noise)
# ---------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 3, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = nn.Parameter(torch.zeros(num_heads, head_dim, self.kernel_size))
        with torch.no_grad():
            self.filters[..., -1] = 1.0
            self.filters.add_(noise_std * torch.randn_like(self.filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ---------------------------------------------------------------------------
# Monotonic per-head forgetting: λ in [λ_min, 1], sigmoid parameterization
# ---------------------------------------------------------------------------

def _monotonic_lambda(forget_param: torch.Tensor, lambda_min=0.5) -> torch.Tensor:
    """Parameterize λ ∈ [λ_min, 1] monotonically via sigmoid/logit."""
    return lambda_min + (1.0 - lambda_min) * torch.sigmoid(forget_param)

# ---------------------------------------------------------------------------
# Causal chunkwise delta rule with monotonic per-head λ
# ---------------------------------------------------------------------------


@torch.compile
def _delta_chunk_monotonic(q, k, v, beta, lam, chunk_size: int = 32):
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
    chunk_num = L_pad // chunk_size
    mask_ = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)
    u = attn @ v
    w = attn @ k_beta
    S = q.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(chunk_num):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        lam_bh = lam[:, :, None, None] if lam is not None else 1.0
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o_inter = q_i @ S
        o[:, :, idx] = o_inter + attn_local @ u_i
        S = S * lam_bh + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# ---------------------------------------------------------------------------
# Entropy+KL-regularized output-stat fusion gate with learnable per-path floor
# ---------------------------------------------------------------------------


class _EntropyKLFusionGate(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        fusion_hidden_mult: int = 2,
        max_floor: float = 0.075,
        temp_init: float = 1.25,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_floor = max_floor
        self.n_paths = 4
        # Learnable per-head temp
        self.log_temp = nn.Parameter(torch.log(torch.full((num_heads,), temp_init)))
        # Per-head,path learnable logit, bias favoring value
        self.floor_param = nn.Parameter(torch.full((num_heads, self.n_paths), -2.0))
        # ------------------------------------------------------------------
        # INPUT DIMENSION FIX:
        # The gating network receives the hidden vector [hidden_size] plus
        # for each of the 4 paths the concatenated statistics
        # (mean, var, max, l2) per head → 4 statistics * num_heads values.
        # Hence, additional features = 4 (stats) * 4 (paths) * num_heads.
        # The previous implementation mistakenly multiplied by head_dim.
        # ------------------------------------------------------------------
        gate_in = hidden_size + 4 * self.n_paths * num_heads  # = hidden + 16 * H
        self.mlp = nn.Sequential(
            nn.Linear(gate_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * self.n_paths, bias=True),
        )
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            # Favor value (path index 3) at start for every head
            self.mlp[-1].bias[num_heads * 3 :: self.n_paths] = 2.0
        self.last_entropy = None
        self.last_kl = None
        self.last_gate_loss = None

    def forward(
        self,
        hidden,
        short,
        long,
        delta,
        value,
        entropy_weight=0.04,
        kl_weight=0.04,
    ):
        # Gather output statistics per branch [mean, var, max, l2-norm]
        def stats(t):
            # [B,L,H,D]
            m = t.mean(dim=-1, keepdim=True)  # [B,L,H,1]
            v = t.var(dim=-1, unbiased=False, keepdim=True)
            mx = t.amax(dim=-1, keepdim=True)
            l2 = t.norm(dim=-1, keepdim=True)
            return [m, v, mx, l2]

        cat_stats = [torch.cat(stats(b), dim=-1) for b in [short, long, delta, value]]  # [B,L,H,4]
        # Flatten across heads/stats → never across batch/seq
        flat_stats = [rearrange(cs, "b l h s -> b l (h s)") for cs in cat_stats]
        gate_in = torch.cat([hidden] + flat_stats, dim=-1)  # [B,L,hidden+16H]
        logits = self.mlp(gate_in)  # [B,L,H*P]
        logits = rearrange(logits, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)
        temp = torch.exp(self.log_temp)[None, None, :, None]
        logits = logits / temp
        raw_p = torch.softmax(logits, dim=-1)
        floor = torch.sigmoid(self.floor_param) * self.max_floor  # [H,P]
        floor = floor[None, None, :, :]
        clipped = torch.clamp(raw_p, min=floor)
        p = clipped / clipped.sum(dim=-1, keepdim=True)
        # Calculate entropy & KL for regularization (logged, not back-proped)
        with torch.no_grad():
            entropy = -(p * torch.log(p + 1e-8)).sum(-1).mean().item()
            self.last_entropy = entropy
            uniform = torch.full_like(p, 1.0 / self.n_paths)
            kl = (p * (torch.log(p + 1e-8) - torch.log(uniform))).sum(-1).mean().item()
            self.last_kl = kl
        # Differentiable loss to be consumed by the main model
        logp = torch.log(p + 1e-8)
        entropy_loss = -(p * logp).sum(-1).mean()
        kl_loss = (p * (logp - torch.log(torch.full_like(p, 1.0 / self.n_paths)))).sum(-1).mean()
        self.last_gate_loss = entropy_weight * entropy_loss + kl_weight * kl_loss
        return p

# ---------------------------------------------------------------------------
# Main DeltaNet
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from fla.models.utils import Cache


class DeltaNet(nn.Module):
    """DeltaNet with Entropy+KL-regularized gating and monotonic memory decay."""

    def __init__(
        self,
        # Baseline & legacy parameters
        mode: str = "entropy_kl_floor_gate",
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
        # Newer params
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 63,
        fir_noise_std: float = 7e-3,
        fusion_hidden_mult: int = 2,
        fusion_max_floor: float = 0.075,
        fusion_temp_init: float = 1.25,
        gate_entropy_weight: float = 0.04,
        gate_kl_weight: float = 0.04,
        use_forget_gate: bool = True,
        forget_min: float = 0.55,
        forget_init: float = 1.0,
        **kwargs: Dict,
    ):
        super().__init__()
        if d_model is not None:
            hidden_size = d_model
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
        # dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Per-head monotonic forgetting parameterized by sigmoid
        if use_forget_gate:
            ratio = (forget_init - forget_min) / (1.0 - forget_min)
            ratio = float(max(min(ratio, 1 - 1e-4), 1e-4))
            init_logit = torch.logit(torch.tensor(ratio))
            self.forget_param = nn.Parameter(init_logit * torch.ones(num_heads))
        else:
            self.register_parameter("forget_param", None)
        # Short-conv projections
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for robust DeltaNet performance.")
        # Dual FIR branches
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel, noise_std=fir_noise_std)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel, noise_std=fir_noise_std)
        # Gating
        self.fusion_gate = _EntropyKLFusionGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            fusion_hidden_mult=fusion_hidden_mult,
            max_floor=fusion_max_floor,
            temp_init=fusion_temp_init,
        )
        self.gate_entropy_weight = gate_entropy_weight
        self.gate_kl_weight = gate_kl_weight
        # Output norm/project
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.last_gate_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        B, L, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0
        if hasattr(self, "forget_param") and self.forget_param is not None:
            lam = _monotonic_lambda(self.forget_param, lambda_min=0.55).view(1, self.num_heads)
            lam = lam.expand(q.shape[0], -1)
        else:
            lam = None
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, rec_state = _delta_chunk_monotonic(q_d, k_d, v_d, beta_d, lam)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")
        value = v
        short = self.fir_short(value)
        long = self.fir_long(value)
        fusion_w = self.fusion_gate(
            hidden_states,
            short,
            long,
            delta_out,
            value,
            entropy_weight=self.gate_entropy_weight,
            kl_weight=self.gate_kl_weight,
        )  # [B,L,H,4]
        o = (
            fusion_w[..., 0:1] * short
            + fusion_w[..., 1:2] * long
            + fusion_w[..., 2:3] * delta_out
            + fusion_w[..., 3:4] * value
        )
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L,
            )
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, B, L)
        # Expose entropy+KL-regularized loss for training aggregation
        self.last_gate_loss = self.fusion_gate.last_gate_loss
        return o, None, past_key_values
