from __future__ import annotations

"""
MLX-converted architecture: delta_net_tareia
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
    return x / mx.linalg.norm(x, axis=-1,
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
    def __init__(self, hidden_size: int,
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
DeltaNet – Token-Adaptive Residual & Epsilon Routing (DeltaNet-TAREIA)
Identifier: delta_net_tareia

This evolutionary variant of DeltaNet-REIA introduces **token-adaptive ε-floors**
that eliminate the global one-size-fits-all schedule.  The floor (minimum
probability mass per, path) is now scaled *per token & per head* according to
how confident the router already is:

    ε_scale(t) = (1 − p_max(t)) · ε_max(step)

where ``p_max(t)`` is the maximum softmax probability over the four routing
paths for the current token/head and ``ε_max`` follows the original linear
annealing schedule(``floor_start, → floor_end``).  Tokens with confident sharp
routing(``p_max, ≈ 1``) receive virtually *no* floor allowing them to specialise
fully (crucial for copy/coreference tasks like, Winogrande).  Conversely tokens
with diffuse beliefs keep a higher floor to preserve gradient flow (helpful in
early training & for ambiguous, contexts).  This simple mechanism combines the
strengths of annealed floors *and* Zero-Floored Gating (ZFG) without expensive
post-hoc pruning.

All other proven strengths of REIA—learnable identity residual, entropy
regularisation, per-head temperature O(N) chunked Δ-rule, depth-wise FIR—are
retained **unchanged**.  The modification is light-weight, retains full batch
agnosticism and incurs negligible compute overhead.

Implementation highlights
• ``_apply_token_adaptive_floor`` – new helper that injects the token-adaptive
  floor after the first softmax.
• No interface changes: class name remains **DeltaNet**, constructor signature
  and forward contract are identical.  New behaviour is **enabled by default**.
• Complexity remains O(N·d); only a few element-wise ops are added.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn as F



# -----------------------------------------------------------------------------
# Helper utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _elu_plus_one(x:, mx.array) -> mx.array:
    """Shifted ELU (+1) keeps activations strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x:, mx.array) -> mx.array:
    """Normalise so that elements along the last dimension sum to one."""
    return(x, / x.sum(dim=-1
        keepdim=True))

# -----------------------------------------------------------------------------
# Core chunk-wise Δ-rule kernel (unchanged) ------------------------------------
# -----------------------------------------------------------------------------


@mx.compile  # type: ignore[arg-type]
# The function is isolated & compiled for maximal speed while keeping the main
# module flexible.
def _delta_rule_chunkwise(
    q: mx.array #, [B,H,L,D_k]
    k: mx.array,  # [B,H,L,D_k]
    v: mx.array,  # [B,H,L,D_v]
    beta: mx.array,  # [B,H,L]
    *,
    chunk_size: int = 32):
    """Associative retrieval via the Δ-rule processed in causal chunks (O(N))."""
    b, h, L, d_k = q.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0,
        0, 0, pad_len)
        q = mx.pad(q, pad_cfg)
        k = mx.pad(k, pad_cfg)
        v = mx.pad(v, pad_cfg)
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len
        q = _l2norm(q)
    k = _l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape into chunks:  (B H N C, D)
    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h, (n, c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta))

    tri_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 0)
    attn_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (
            attn_inv[..., i, :, None] * attn_inv[..., :, :i]
        ).sum(-2), attn_inv = attn_inv + mx.eye(chunk_size, dtype = attn_inv.dtype)
    attn_inv = attn_inv  # mixed-precision to save memory
        u = attn_inv @ v
        w = attn_inv @ k_beta
        S = mx.zeros(b, h, d_k v.shape[-1])
    o = mx.zeros_like(v)

    future_mask = mx.triu(mx.ones(chunk_size, chunk_size
    dtype=mx.bool_), 1)
    for idx in range(L_pad, // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(future_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :
        idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        o = _rearrange(o, "b h n c d -> b h, (n, c) d")
    if pad_len:
        o = o[:
        :, :L]
    return o, S
# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (Dirac-init) --------------------------------
# -----------------------------------------------------------------------------


class _DepthwiseFIRConv1d(nn.Module):
    """Per-head per-channel causal FIR with identity (Dirac) initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 31) -> None:  # noqa: D401 E501
        super().__init__()
        self.kernel_size = int(kernel_size)
        filters = mx.zeros(num_heads, head_dim, self.kernel_size)
        with mx.disable_grad():
            filters[..., -1] = 1.0  # causal identity
            filters.add_(0.01, * mx.randn_like(filters))  # small noise
        self.filters = mx.array(filters), def forward(self, x: mx.array) -> mx.array:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b, (h, d) l")
        weight = _rearrange(self.filters, "h d k ->, (h, d) 1 k")
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight
        groups = h * d)
        return _rearrange(y, "b, (h, d) l -> b l h d", h=h)

# -----------------------------------------------------------------------------
# Optional typing hints ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main DeltaNet layer ----------------------------------------------------------
# -----------------------------------------------------------------------------


class DeltaNet(nn.Module):
    """DeltaNet layer with *token-adaptive* ε-floor, entropy-regularised router & learnable identity scaling."""

    # pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
    def __init__(self, # ---------------- generic args ----------------
        mode: str =, "tareia",
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
        # annealing & reg
        floor_start: float = 0.05,
        floor_end: float = 0.0,
        floor_decay_steps: int = 3000,
        entropy_coeff: float = 0.02,
        # ---------------- identity path ---------------
        use_identity_path: bool = True,
        identity_scale_init: float = 0.5,
        **kwargs: Dict # Accept extra unused kwargs for, compatibility) -> None:
        super().__init__()

        # ---- bookkeeping -------------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size, * expand_k)
        self.value_dim = int(hidden_size, * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key/value, dims must divide num_heads")

        self.mode = mode
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_identity_path = use_identity_path
        # annealing / reg params
        self.floor_start = float(floor_start)
        self.floor_end = float(floor_end)
        self.floor_decay_steps = int(floor_decay_steps)
        self.entropy_coeff = float(entropy_coeff)

        # ---- projections -------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim
        bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads
            bias=False)

        # identity projection & scaling --------------------------------
        if use_identity_path:
            self.id_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.alpha_identity = mx.array(identity_scale_init, *, mx.ones(num_heads))
        else:
            # register_parameter removed for MLX
            # register_parameter removed for MLX

        # ---- optional local short conv -----------------------------
        if use_short_conv:
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
        else:
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()

        # ---- dual FIR convs -----------------------------------------
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_short_kernel)
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim
        kernel_size = fir_long_kernel)

        # ---- fusion gate -------------------------------------------
        fusion_in = hidden_size + self.head_v_dim * self.num_heads * 3  # hidden + (short,long, delta)
        self.fusion_gate_mlp = nn.Sequential(, nn.Linear(fusion_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, *, fusion_hidden_mult, num_heads * 4 bias=True))

        # learnable temperature per head
        self.gate_log_temp = mx.array(mx.log(mx.tensor(gate_temp_init)), * mx.ones(num_heads))
        # ε-floor parameters (logit) – still learnable but now *token-scaled*
        eps_logit_init = math.log(gate_eps_init) - math.log(1, - gate_eps_init) if gate_eps_init > 0 else -12.0
        self.gate_eps_logit = mx.array(mx.full((num_heads, 4), eps_logit_init))

        # bias: favour direct value path moderately
        if self.fusion_gate_mlp[-1].bias is not None:
            with mx.disable_grad():
                bias = self.fusion_gate_mlp[-1].bias
                bias.zero_()
                # path order: 0-short, 1-long, 2-delta 3-value
                for h in range(num_heads):
                    bias[h * 4 + 3] = 2.0

        # ---- output normalisation / projection ---------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

        # ---- step counter for annealing ----------------------------
        # register_buffer removed for MLX
        persistent = False)
        self.reg_loss: Optional[mx.array] = None  # populated every forward

    # -----------------------------------------------------------------
    # helper: linear schedule for *maximum* ε allowed(same, as, REIA)
    # -----------------------------------------------------------------
    def _current_floor_max(self) -> float:
        t = float(self._step.item())
        if t >= self.floor_decay_steps:
            return self.floor_end
        ratio = t / max(1.0, self.floor_decay_steps)
        return self.floor_start + ratio * (self.floor_end - self.floor_start)

    # -----------------------------------------------------------------
    # helper: inject token-adaptive floor into probs (after, softmax)
    # -----------------------------------------------------------------
    def _apply_token_adaptive_floor(self, probs: mx.array) -> mx.array:
        """Apply per-token ε-floor proportional to router uncertainty.

        Args:
            probs: Softmax outputs without floor.  Shape [B,L,H 4].
        Returns:
            probs with token-adaptive floor applied(sums, to, 1).
        """
        # confidence per token/head – high when routing is sharp
        p_max = probs.max(dim=-1, keepdim=True).values  # [B, L, H 1]
        # scale between 0 (confident) and 1 (diffuse)
        scale = 1.0 - p_max  # linear; could be nonlinear but suffices
        # global max ε from schedule (scalar)
        eps_max = self._current_floor_max()
        if eps_max <= 0:
            return probs  # no floor needed
        # base per-head/path template in [0 1]
        eps_base = mx.sigmoid(self.gate_eps_logit)  # [H,4]
        # broadcast to [B,L,H,4]
        eps = eps_max * scale * eps_base.reshape(1, 1 *eps_base.shape)
        # blend & renormalise to keep simplex property
        probs = probs * (1.0 - eps.sum(dim=-1, keepdim=True)) + eps
        return probs

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, hidden_states: mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:

        if attention_mask is not None:
            assert attention_mask.ndim == 2
        "attention_mask must be [batch, seq_len]"
        batch_size, seq_len, _ = hidden_states.shape

        # -- retrieve previous state --------------------------------
        last_state: Optional[Dict] = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices
        cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d ->, (b, s) d"), indices).expand_dims(0)

        # ---- projections + short conv -----------------------------
        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv:
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q
        conv_state_k, conv_state_v = last_state["conv_state"]
            q
        conv_state_q = self.q_conv1d(self.q_proj(hidden_states)
        cache=conv_state_q
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            k
        conv_state_k = self.k_conv1d(self.k_proj(hidden_states)
        cache=conv_state_k
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
            v
        conv_state_v = self.v_conv1d(self.v_proj(hidden_states)
        cache=conv_state_v
        output_final_state=use_cache
        cu_seqlens = cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.qk_activation == "silu":
                q
        k = F.silu(q), F.silu(k)
                v = F.silu(v)

        # ---- head reshape ----------------------------------------
        q = _rearrange(q, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        k = _rearrange(k, "b l, (h, d) -> b l h d"
        h=self.num_heads)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # ---- optional activation / norm --------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q
        k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q
        k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q
        k = _sum_norm(q), _sum_norm(k)

        # ---- beta gate ------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global path ----------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out
        recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # ---- local FIR paths -------------------------------------
        local_short = self.local_fir_short(v)
        local_long = self.local_fir_long(v)

        # ---- fusion gating ---------------------------------------
        gate_inp = mx.cat(
            [, hidden_states,
                _rearrange(local_short, "b l h d -> b l, (h, d)"),
                _rearrange(local_long, "b l h d -> b l, (h, d)"),
                _rearrange(delta_out, "b l h d -> b l, (h, d)"),
            ],
            dim=-1)
        fusion_logits = self.fusion_gate_mlp(gate_inp)  # [B,L H*4]
        fusion_logits = _rearrange(fusion_logits, "b l, (h, c) -> b l h c"
        h=self.num_heads
        c = 4)

        # temperature scaling --------------------------------------
        temp = (F.softplus(self.gate_log_temp) + 1e-4).reshape(1, 1, -1, 1)
        fusion_logits = fusion_logits / temp
        fusion_probs = mx.softmax(fusion_logits, dim = -1)  # [B, L, H 4] (no floor, yet)

        # ---- token-adaptive ε-floor ------------------------------
        fusion_probs = self._apply_token_adaptive_floor(fusion_probs)

        # ---- entropy regularisation ------------------------------
        entropy = -(fusion_probs * mx.log(fusion_probs, + 1e-8)).sum(dim=-1), # [B, L H]
        self.reg_loss = -self.entropy_coeff * entropy.mean(), # maximise
        entropy => negative coeff

        # ---- path combination ------------------------------------
        # path order: 0-short, 1-long, 2-delta, 3-value, o = (
            fusion_probs[..., 0:1] * local_short
            + fusion_probs[..., 1:2] * local_long
            + fusion_probs[..., 2:3] * delta_out
            + fusion_probs[..., 3:4] * v
        )

        # ---- identity residual (ungated) -------------------------
        if self.use_identity_path:
            id_val = self.id_proj(hidden_states)  # [B
        L,value_dim]
            id_val = _rearrange(id_val, "b l, (h, d) -> b l h d"
            h=self.num_heads)
            alpha = self.alpha_identity.reshape(1, 1, -1, 1)
            o = o + alpha * id_val

        # ---- cache update ----------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state
        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx
        offset = seq_len)

        # ---- output norm / projection ----------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h, d) -> b l h d"
            h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l, (h, d)")
        o = self.o_proj(o)

        # ---- re-pad if we un-padded ------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0)
        indices, batch_size, seq_len)

        # ---- step ++ for annealing -------------------------------
        self._step += 1  # type: ignore[operator]

        return o, None, past_key_values
