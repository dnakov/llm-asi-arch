"""
MLX-converted architecture: delta_net_adgr
Auto-converted from PyTorch to MLX format
"""

# MLX Utility Functions (replacing PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """Simple einops rearrange replacement for common patterns"""
    if "b l (h d) -> b l h d" in pattern:
        h = kwargs.get('h', kwargs.get('d', 1))
        b, l, hd = tensor.shape
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif "b l h d -> b l (h d)" in pattern:
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif "b l h d -> b h l d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h l d -> b l h d" in pattern:
        return tensor.transpose(0, 2, 1, 3)
    elif "b h (n c) d -> b h n c d" in pattern:
        c = kwargs.get('c', 1)
        b, h, nc, d = tensor.shape
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif "b h n c d -> b h (n c) d" in pattern:
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    else:
        # Fallback: return tensor as-is
        return tensor

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)

def _masked_fill(tensor: mx.array, mask: mx.array, value: float) -> mx.array:
    """Masked fill operation"""
    return mx.where(mask, value, tensor)

def _get_unpad_data(attention_mask):
    """Simple unpad data extraction (placeholder)"""
    # Simplified version - just return indices for non-masked positions
    indices = mx.where(attention_mask.flatten())[0]
    cu_seqlens = mx.array([0, attention_mask.shape[-1]])
    max_len = attention_mask.shape[-1]
    return indices, cu_seqlens, max_len

def _index_first_axis(tensor: mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor: mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    # Simplified version
    return tensor.reshape(batch_size, seq_len, -1)

class _ShortConvolution(nn.Module):
    """MLX replacement for FLA ShortConvolution"""
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)
        self.activation = activation
        
    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
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
            return out, None  # Simplified - no cache state
        return out

# -*- coding: utf-8 -*-
"""
DeltaNet – Adaptive Diversity-Gated Multi-Scale Routing (DeltaNet-ADGR)
=======================================================================
Breakthrough architecture integrating theoretical and empirical advances to resolve the fundamental local-global tradeoff and diversity-collapse bottlenecks of previous DeltaNet experiments, particularly the limitations of hard-coded copy-path bias and excessive uniform sharpening.

Core Innovations:
-----------------
1. **Learnable Adaptive Copy/Value Bias (Per-Head)**: Replaces the static +4.0 bias with a learnable, per-head bias parameter. The bias starts at +1.75 but is optimized during training, allowing the gating network to adaptively favor copy/local fidelity or relax for global context as needed (as per AFT/LRPE-d/Hyena guidance).

2. **KL (Entropy-Diversity) Path Regularization**: During forward, a KL-divergence loss from the fusion softmax weights to a uniform distribution is computed per token, per head, and returned as a reg_loss (entropy_reg_weight * KL). This directly penalizes gate collapse and nudges the model to maintain path usage diversity, while allowing specialization where beneficial. The reg_loss is returned by the forward pass for external use.

3. **Dynamic Annealed Entropy Floor**: Rather than strict or fixed epsilon floors, a small trainable parameter (with a minimum, e.g., 0.005) is added per path, per head. This ensures that the router never fully collapses traffic on any path but allows the degree of mixture to be tuned (cf. MoE/TransNormerLLM best practice).

4. **All Else Preserved**: Dual FIR (short/long), chunked delta O(N) path, per-head temperature softmax, fully batch-size agnostic einops, causal chunking, efficient _ShortConvolution. All other working improvements are strictly retained.

Technical Features:
--------------------
- Efficient @mx.compile for delta kernel
- Gating MLP remains output-aware (includes all three non-copy outputs in its input)
- Interface, __init__ signature, and class name are fully preserved
- Sensible defaults; no config changes required
- Reg_loss is always returned (forward returns o, reg_loss, cache)
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.core as mx.nn as nn
import mlx.core as mx.nn.functional as F


# -----------------------------------------------------------------------------

def _elu_p1(x: mx.Tensor) -> mx.Tensor:
    return (F.elu(x, 1.0, False) + 1.0)


def _sum_norm(x: mx.Tensor) -> mx.Tensor:
    return (x / x.sum(-1, keepdim=True))

# -----------------------------------------------------------------------------

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 64, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filters = mx.array(mx.zeros(num_heads, head_dim, self.kernel_size))
        with mx.no_grad():
            self.filters[..., -1] = 1.0
            self.filters.add_(noise_std * mx.randn_like(self.filters))

    def forward(self, x: mx.Tensor) -> mx.Tensor:
        # x : (b, l, h, d)
        b, l, h, d = x.shape
        # Ensure weight dtype follows input dtype for AMP compatibility
        weight = self.filters
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        weight = _rearrange(weight, "h d k -> (h d) 1 k")
        # Pre-pad on the left to maintain causality
        x_pad = mx.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return _rearrange(y, "b (h d) l -> b l h d"h=h)

# -----------------------------------------------------------------------------

@mx.compile
def _delta_rule_chunkwise(q, k, v, beta, *, chunk_size: int = 32):
    """Chunk-wise implementation of the DeltaNet global path.
    Ensures O(N*chunk_size) complexity and strict causality."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    # ------------------------------------------------------------------
    # Padding so that L is divisible by chunk_size
    # ------------------------------------------------------------------
    if pad_len:
        q = mx.pad(q, (0, 0, 0, pad_len))
        k = mx.pad(k, (0, 0, 0, pad_len))
        v = mx.pad(v, (0, 0, 0, pad_len))
        beta = mx.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # ------------------------------------------------------------------
    # Normalisation & weighting
    # ------------------------------------------------------------------
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: _rearrange(t, "b h (n c) d -> b h n c d"c=chunk_size),
        (q, k, v, k_beta),
    )

    mask_tri_inc = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 0)
    # Inverse attention kernel (lower-triangular)
    att_inv = -(k_beta @ k.transpose(-1, -2))._masked_fill(mask_tri_inc, 0)
    for i in range(1, chunk_size):
        att_inv[..., i, :i] += (
            att_inv[..., i, :, None].clone() * att_inv[..., :, :i].clone()
        ).sum(-2)
    att_inv = att_inv + mx.eye(chunk_size, dtype=q.dtype, q.device)

    # ------------------------------------------------------------------
    # IMPORTANT: Keep dtype consistent with q/k/v to avoid runtime errors
    # ------------------------------------------------------------------
    att_inv = att_inv

    u = att_inv @ v
    w = att_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = mx.zeros_like(v)

    # Strictly causal mask inside each chunk
    mask_future = mx.triu(mx.ones(chunk_size, chunk_size, dtype=mx.bool, q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2))._masked_fill(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet with Adaptive Diversity-Gated Routing (ADGR).

    The implementation preserves all innovative components while ensuring:
      • Strict causal masking
      • O(N) chunk-wise global path computation
      • Full batch/sequence-length agnosticism
    """

    def __init__(
        self,
        mode: str = "adgr",
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
        fir_kernel_size_long: int = 31,
        fir_kernel_size_short: int = 3,
        fusion_hidden_mult: int = 2,
        copy_bias_init: float = 1.75,
        temp_init: float = 1.0,
        temp_min: float = 0.5,
        gate_entropy_reg_weight: float = 0.01,
        min_path_eps: float = 0.005,
        **kwargs,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Hyper-parameters
        # ------------------------------------------------------------------
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
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.fir_kernel_size_short = fir_kernel_size_short
        self.fir_kernel_size_long = fir_kernel_size_long
        self.fusion_hidden_mult = fusion_hidden_mult
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.gate_entropy_reg_weight = gate_entropy_reg_weight
        self.min_path_eps = min_path_eps

        # ------------------------------------------------------------------
        # Derived dimensions
        # ------------------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("Key/Value dimensions must divide num_heads.")

        # ------------------------------------------------------------------
        # Linear projections
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ------------------------------------------------------------------
        # Short convolution paths (mandatory)
        # ------------------------------------------------------------------
        if self.use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = _ShortConvolution(
                self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias
            )
            self.k_conv1d = _ShortConvolution(
                self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias
            )
            self.v_conv1d = _ShortConvolution(
                self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias
            )
        else:
            raise UserWarning("_ShortConvolution is mandatory.")

        # ------------------------------------------------------------------
        # FIR (long / short) local memory branches
        # ------------------------------------------------------------------
        self.local_fir_long = _DepthwiseFIRConv1d(
            num_heads, self.head_v_dim, kernel_size=fir_kernel_size_long
        )
        self.local_fir_short = _DepthwiseFIRConv1d(
            num_heads, self.head_v_dim, kernel_size=fir_kernel_size_short
        )

        # ------------------------------------------------------------------
        # Gating network parameters
        # ------------------------------------------------------------------
        self.copy_path_bias = nn.Parameter(
            mx.full((num_heads,), copy_bias_init, dtype=mx.float32)
        )  # per-head learnable bias for copy/value path

        # Per-path, per-head min epsilon (learnable, bounded ≥ min_path_eps)
        self.path_min_logit = mx.array(mx.zeros(num_heads, 4))
        self._min_eps = float(min_path_eps)

        # Per-head temperature (log-space)
        self.gate_log_tau = mx.array(mx.log(mx.ones(num_heads) * temp_init))

        # ------------------------------------------------------------------
        # Fusion gate MLP (two-layer, GELU)
        # ------------------------------------------------------------------
        gate_in_dim = hidden_size + 3 * self.value_dim  # concat [hidden | short | long | delta]
        fusion_hidden_dim = fusion_hidden_mult * self.num_heads * 4
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, self.num_heads * 4, bias=True),
        )

        # ------------------------------------------------------------------
        # Output projection & (optional) gating
        # ------------------------------------------------------------------
        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Holder for last reg loss
        self.last_reg_loss: Optional[mx.Tensor] = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: mx.Tensor,
        attention_mask: Optional[mx.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.Tensor, Optional[mx.Tensor], Optional["Cache"]]:
        # ------------------------------------------------------------------
        # Input handling & (optional) unpadding for variable-length batches
        # ------------------------------------------------------------------
        if attention_mask is not None:
            assert (
                attention_mask.ndim == 2
            ), "attention_mask must be [batch, seq_len]"

        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -seq_len:])
            hidden_states = _index_first_axis(
                _rearrange(hidden_states, "b s d -> (b s) d"), indices
            ).expand_dims(0)

        # ------------------------------------------------------------------
        # Linear projections + optional short convolutions
        # ------------------------------------------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(
            self.q_proj(hidden_states),
            cache=conv_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_k = self.k_conv1d(
            self.k_proj(hidden_states),
            cache=conv_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_v = self.v_conv1d(
            self.v_proj(hidden_states),
            cache=conv_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # Split into heads --------------------------------------------------
        q, k = map(
            lambda t: _rearrange(t, "... (h d) -> ... h d"d=self.head_k_dim), (q, k)
        )
        v = _rearrange(v, "... (h d) -> ... h d"d=self.head_v_dim)

        # Optional activation & norm on Q/K --------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        v_direct = v  # copy/value path

        # Beta (eigen-value) gate for delta path ---------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ------------------------------------------------------------------
        # Delta rule (global) path  – O(N)
        # ------------------------------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = _rearrange(delta_out, "b h l d -> b l h d")

        # ------------------------------------------------------------------
        # Local FIR memories (short & long)
        # ------------------------------------------------------------------
        fir_short = self.local_fir_short(v_direct)
        fir_long = self.local_fir_long(v_direct)

        # ------------------------------------------------------------------
        # Gating (fusion) network
        #   Input  : concat[hidden | short | long | delta]
        #   Output : per-head 4-way routing weights (softmax)
        # ------------------------------------------------------------------
        gate_in = mx.cat(
            [
                hidden_states,
                _rearrange(fir_short, "b l h d -> b l (h d)"),
                _rearrange(fir_long, "b l h d -> b l (h d)"),
                _rearrange(delta_out, "b l h d -> b l (h d)"),
            ],
            dim=-1,
        )

        fusion_logits = self.fusion_gate_mlp(gate_in)  # (B,L, H*4)
        fusion_logits = _rearrange(fusion_logits, "b l (h c) -> b l h c"h=self.num_heads, c=4)

        # Learnable per-head bias on copy/value path (path index = 3)
        fusion_logits[..., 3] = fusion_logits[..., 3] + self.copy_path_bias.reshape(1, 1, -1)

        # Minimum path epsilon (learnable, bounded)
        min_path_eps = self._min_eps + (1 - self._min_eps) * mx.sigmoid(
            self.path_min_logit
        )  # (H,4)

        # Temperature scaling (per-head)
        tau = F.softplus(self.gate_log_tau) + self.temp_min  # (H,)
        fusion_logits = fusion_logits / tau.reshape(1, 1, -1, 1)

        # Softmax with path floor to prevent collapse ----------------------
        fusion_weights = mx.softmax(fusion_logits, dim=-1)  # (B,L,H,4)
        fusion_weights = fusion_weights * (
            1.0 - min_path_eps.sum(dim=-1)[None, None, :, None]
        ) + min_path_eps[None, None, :, :]
        fusion_weights = fusion_weights / fusion_weights.sum(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        # KL-diversity regularisation (w.r.t uniform distribution)
        # ------------------------------------------------------------------
        uniform = mx.full_like(fusion_weights, 1.0 / 4)
        kl = (
            fusion_weights
            * (fusion_weights.clamp(min=1e-8).log() - uniform.log())
        ).sum(-1)  # (B,L,H)
        kl_reg = kl.mean()
        reg_loss = self.gate_entropy_reg_weight * kl_reg
        self.last_reg_loss = reg_loss

        # ------------------------------------------------------------------
        # Weighted fusion of the four paths
        # ------------------------------------------------------------------
        o = (
            fusion_weights[..., 0:1] * fir_short
            + fusion_weights[..., 1:2] * fir_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )

        # ------------------------------------------------------------------
        # Cache update for decoding (if requested)
        # ------------------------------------------------------------------
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        # ------------------------------------------------------------------
        # Output projection & (optional) gating
        # ------------------------------------------------------------------
        if self.use_gate:
            g = _rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d"d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # Re-pad to original batch structure (if unpadded earlier)
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, seq_len)

        return o, reg_loss, past_key_values