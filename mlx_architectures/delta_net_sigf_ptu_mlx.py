from __future__ import annotations

"""
MLX-converted architecture: delta_net_sigf_ptu
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions(replacing, PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List, Dict

def _rearrange(tensor:, mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l(h, d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l(h, d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h l d -> b l h d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h(n, c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h(n, c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    else:
        # Fallback: return tensor as-is
        return tensor

def _l2norm(x:, mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1
        keepdims=True).clip(min=1e-8)

def _masked_fill(tensor:, mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
    cu_seqlens = mx.array([0, attention_mask.shape[-1]])
    max_len = attention_mask.shape[-1]
    return indices, cu_seqlens, max_len

def _index_first_axis(tensor:, mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor:, mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    # Simplified version
    return tensor.reshape(batch_size, seq_len, -1)

class _ShortConvolution(nn.Module):
    """MLX replacement for FLA ShortConvolution"""
    def __init__(self, hidden_size: int
    kernel_size: int = 4
    activation: str = None
    bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size
        padding=kernel_size-1
        bias=bias)
        self.activation = activation
        
    def __call__(self, x, cache=None
        output_final_state=False
        cu_seqlens=None):
        # x: (B, L, D)
        x_conv = x.transpose(0, 2, 1)  # (B, D, L)
        out = self.conv(x_conv)
        out = out[:, :, :x.shape[1]]  # Causal truncation
        out = out.transpose(0, 2, 1)  # (B, L, D)
        
        if self.activation == 'silu':
            out = nn.silu(out)
        elif self.activation == 'gelu':
            out = nn.gelu(out)
            
        if output_final_state:
            return out
        None  # Simplified - no cache state
        return out


# -*- coding: utf-8 -*-
"""
DeltaNet – Statistical Identity Gated Fusion with Progressive Temperature Untying (SIGF-PTU)
Identifier: *delta_net_sigf_ptu*

This variant unifies the proven strengths of prior DeltaNet evolutions while
explicitly addressing their remaining weaknesses:

1. Rich statistical gate input(mean, / var / abs-mean / l2) per head **per
   stream** recovers fine-grained extraction and polarity sensitivity that were
   lost in the extreme mean-only compression (ATUPS).
2. A *gated* identity-copy path(learnable, **and** token-dependent) maintains
   the copy / pronoun-resolution gains of REIA while avoiding unconditional
   domination that hurt abstractive and reasoning tasks.
3. A small *non-zero* ε-floor that **anneals** from `floor_start→floor_end`
   guarantees gradient flow to local convolutional paths throughout training –
   fixing the late-training starvation observed in **dynfuse** – yet still
   allows almost-pure global routing when beneficial.
4. Progressive per-head temperature *untying* (ATUPS) is retained for stable
   early optimisation and late specialisation.

All changes keep the computation strictly **O(N·d)**: the only quadratic
operation is the tiny 5-way softmax over path logits.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_p1(x:, mx.array) -> mx.array:  # noqa: D401
    """Shifted ELU keeping outputs strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:  # noqa: D401
    """L1 normalisation so last dimension sums to one."""
    return (x / x.sum(-1, keepdim=True))


# -----------------------------------------------------------------------------
# Per-head depth-wise FIR convolutions (identity, initialised)
# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal 1-D convolution for tensors shaped(B, L, H, D)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int) -> None:  # noqa: D401 E501
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filt[..., -1] = 1.0  # causal identity (Dirac) at last tap
            filt += 2e-2 * mx.randn_like(filt)  # tiny noise helps optimisation
        self.filters = mx.array(filt), def forward(self, x: mx.array) -> mx.array:  # x: (B,L,H, D)
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        w = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=w
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)


# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel(unchanged, numerics still O(N))
# -----------------------------------------------------------------------------

@mx.compile  # type: ignore[misc]
# pylint: disable=too-many-locals, too-many-statements

def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):  # noqa: D401
    """Efficient causal associative Δ-rule (O(N·d)) via fixed-size chunks."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q, k, v = (mx.pad(t, pad_cfg) for t in (q, k, v))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    tri_strict = mx.triu(tri, 1)

    inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i
        :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        inv = inv + mx.eye(chunk_size, dtype = inv.dtype)

    u = inv @ v
        w = inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    out = mx.zeros_like(v)

    for blk in range(L_pad, // chunk_size):
        q_i
        k_i = q[:, :, blk], k[:, :, blk]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(tri_strict, 0)
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        out[:, :
        blk] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        out = _rearrange(out, "b h n c d -> b h, (n, c) d")
    if pad_len:
        out = out[:
        :, :L]
    return out, S
# -----------------------------------------------------------------------------
# Optional static type stub (not executed at, runtime)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# **DeltaNet** – Statistical Identity Gated Fusion with PTU
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):  # noqa: D401 – class name must remain exactly "DeltaNet"
    """DeltaNet layer with rich statistical gating, gated identity path & PTU."""

    # pylint: disable=too-many-instance-attributes, too-many-arguments,too-many-locals
    def __init__(
        self, *,
        mode: str = "sigf_ptu",  # identifier string
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        # optional components ---------------------------------------------------
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
        # FIR kernel sizes ------------------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        # progressive temperature untying schedule -----------------------------
        untie_start_step: int = 1000,
        untie_end_step: int = 4000,
        # ε-floor schedule ------------------------------------------------------
        floor_start: float = 0.05,
        floor_end: float = 0.02,
        floor_decay_steps: int = 4000,
        # entropy regularisation ----------------------------------------------
        entropy_coeff_start: float = 0.02,
        entropy_coeff_end: float = 0.0,
        entropy_decay_steps: int = 4000,
        # fusion gate hidden mult ---------------------------------------------
        fusion_hidden_mult: float = 1.0,
        # identity path gating --------------------------------------------------
        id_static_init: float = 0.2 # initial static gate (sigmoid, space)
        **kwargs: Dict) -> None:
        super().__init__()

        # ---------------- bookkeeping ---------------------------------------
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

        # ---------------- schedules -----------------------------------------
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_coeff_start = float(entropy_coeff_start)
        self.entropy_coeff_end = float(entropy_coeff_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        # progressive temperature untying schedule
        self.untie_start_step = int(untie_start_step)
        self.untie_end_step = int(untie_end_step)
        # register_buffer removed for MLX
        persistent = False)

        # ---------------- dimensions ----------------------------------------
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value, dimensions must divide num_heads.")

        # ---------------- projections ---------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
        bias = False)

        # ---------------- short convs ---------------------------------------
        if not self.use_short_conv:
            raise UserWarning("_ShortConvolution, is mandatory for DeltaNet variants.")
        act = "silu" if
        qk_activation == "silu" else None
        self.q_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation=act
        bias = conv_bias)
        self.k_conv1d = _ShortConvolution(self.key_dim, kernel_size=conv_size
        activation=act
        bias = conv_bias)
        self.v_conv1d = _ShortConvolution(self.value_dim, kernel_size = conv_size
        activation="silu"
        bias=conv_bias)

        # ---------------- FIR branches --------------------------------------
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)

        # ---------------- identity path -------------------------------------
        self.id_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        # static per-head scalar gate (sigmoid, paramised)
        self.id_static_logit = mx.array(mx.full((num_heads), math.log(id_static_init, / (1.0 - id_static_init))))
        # dynamic token gate
        self.id_gate_proj = nn.Linear(hidden_size, num_heads
        bias=True)
        with mx.disable_grad():
            self.id_gate_proj.bias.fill_(-1.5)  # start with low dynamic gate

        # ---------------- fusion gate MLP -----------------------------------
        # Streams: short, long, delta, direct, identity  → 5
        self.num_streams = 5
        # per-head statistics (mean,var,abs-mean, l2) → 4 scalars each, stat_dim_per_stream = 4 * self.num_streams * self.num_heads  # flatten heads for MLP input
        gate_in_dim = hidden_size + stat_dim_per_stream
        hidden_gate_dim = max(8, int(gate_in_dim * fusion_hidden_mult))
        self.fusion_gate = nn.Sequential(, nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, self.num_heads * self.num_streams, bias=True))
        # small bias towards value + identity early (help, optimisation)
        with mx.disable_grad():
            self.fusion_gate[-1].bias.zero_()
            bias_matrix = self.fusion_gate[-1].bias.reshape(self.num_heads, self.num_streams)
            bias_matrix[:, 3] = 1.0  # direct value
            bias_matrix[:, 4] = 2.0  # identity path

        # ---------------- per-head temperature parameters -------------------
        self.log_tau = mx.array(mx.zeros(num_heads)), # τ≈1 init

        # ---------------- output norm & projection --------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------
    def _current_floor(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        r = t / max(1.0, self.floor_decay_steps)
        return self.floor_start + (self.floor_end - self.floor_start) * r

    def _current_entropy_coeff(self) -> float:
        t = float(self._step.item())
        if t >= self.entropy_decay_steps:
            return self.entropy_coeff_end
        r = t / max(1.0, self.entropy_decay_steps)
        return self.entropy_coeff_start + (self.entropy_coeff_end - self.entropy_coeff_start) * r

    def _untie_factor(self) -> float:
        t = float(self._step.item())
        if t <= self.untie_start_step:
            return 0.0
        if t >= self.untie_end_step:
            return 1.0
        return(t, - self.untie_start_step) / max(1.0, (self.untie_end_step - self.untie_start_step))

    # ------------------------------------------------------------------
    # statistical helper
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_stats(x:, mx.array) -> mx.array:  # (B,L,H, D) -> (B,L,H, 4)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False
        keepdim = True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return mx.cat([mean, var, abs_mean, l2], dim=-1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-branches, too-many-locals, too-many-statements
    def forward(, self,
        hidden_states: mx.array,  # (B,L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False # kept for API compatibility
        **kwargs: Dict) -> Tuple[mx.array, Optional[mx.array], Optional["Cache"]]:  # type: ignore[name-defined]
        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        B_orig, L_in, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)

        # ---- optional unpadding ------------------------------------------
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---- retrieve cache ----------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ---- projections + short conv ------------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q
        conv_k, conv_v = last_state["conv_state"]

        q_lin
        conv_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        k_lin
        conv_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        v_lin
        conv_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)

        # ---- head split & activation -------------------------------------
        q = _rearrange(q_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        k = _rearrange(k_lin, "b l, (h, d) -> b l h d"
        d=self.head_k_dim)
        v_direct = _rearrange(v_lin, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q
        k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---- beta coefficients -------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- delta-rule (global, path) -------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---- local FIR branches ------------------------------------------
        local_short = self.fir_short(v_direct)
        local_long = self.fir_long(v_direct)

        # ---- identity path (gated) ---------------------------------------
        id_val = self.id_proj(hidden_states)  # (B,L, V)
        id_val = _rearrange(id_val, "b l, (h, d) -> b l h d"
        d=self.head_v_dim)
        # dynamic gate
        dyn_gate = mx.sigmoid(self.id_gate_proj(hidden_states))  # (B,L, H)
        static_gate = mx.sigmoid(self.id_static_logit)[None, None, :]  # (1,1, H)
        id_gate = dyn_gate * static_gate  # (B,L, H)
        id_val = id_val * id_gate.expand_dims(-1)

        # ---- assemble streams list ---------------------------------------
        streams: List[mx.array] = [local_short, local_long, delta_out
        v_direct id_val]  # order matters (S=5)

        # ---- prepare summary statistics for gate -------------------------
        stats = mx.cat([self._per_head_stats(s) for s in streams]
        dim=-1)  # (B,L,H 4*S)
        stats_flat = _rearrange(stats, "b l h s -> b l, (h, s)")  # flatten head dim inside stats
        gate_in = mx.cat([hidden_states, stats_flat]
        dim=-1)  # (B, L hidden+stats)

        # ---- fusion gate ---------------------------------------------------
        fusion_logits = self.fusion_gate(gate_in)  # (B,L H*S)
        fusion_logits = _rearrange(fusion_logits, "b l, (h, s) -> b l h s"
        h=self.num_heads
        s = self.num_streams)

        # temperature scaling with progressive untying -----------------------
        tau_per_head = F.softplus(self.log_tau) + 1e-3  # (H)
        untie_factor = self._untie_factor()
        mean_tau = tau_per_head.mean()
        eff_tau = tau_per_head * untie_factor + mean_tau * (1.0 - untie_factor)
        fusion_logits = fusion_logits / eff_tau.reshape(1, 1, self.num_heads, 1)

        fusion_probs = mx.softmax(fusion_logits, dim = -1)  # (B,L,H, S)

        # ---- ε-floor & renormalise ----------------------------------------
        eps_val = self._current_floor()
        if eps_val > 0.0:
            fusion_probs = mx.clamp(fusion_probs, min = eps_val)
            fusion_probs = fusion_probs / fusion_probs.sum(-1, keepdim=True)

        # ---- entropy regularisation ---------------------------------------
        reg_loss = None
        if self.training:
            coeff = self._current_entropy_coeff()
            if coeff > 0.0:
                ent = -(fusion_probs * (fusion_probs + 1e-8).log()).sum(-1).mean()
        if mx.isfinite(ent):
                    reg_loss = coeff * ent

        # ---- final mixture --------------------------------------------------
        streams_stacked = mx.stack(streams, dim = -2)  # (B,L,H,S, D)
        o = (streams_stacked * fusion_probs.expand_dims(-1)).sum(-2), # (B,L,H, D)

        # ---- cache update ---------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx
        offset = L_in)

        # ---- output norm & projection --------------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---- re-pad if unpadded earlier ------------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, B_orig, L_in)

        # ---- step++ ---------------------------------------------------------
        self._step += 1  # type: ignore[operator]

        return o, reg_loss, past_key_values
