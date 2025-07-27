from __future__ import annotations

"""
MLX-converted architecture: delta_net_spectral_fusion
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
DeltaNet – Spectral Fusion Memory ( **patched** )
This file is a *patched* version of the original implementation.  The change
is **purely technical** and **does not alter** the underlying spectral–memory
idea:

    •   Fixed a bug in the *overlap-add* routine where the normalisation
        weight was accumulated along the **batch** dimension instead of the
        **sequence-length** dimension.  With small batch sizes this resulted
        in an incorrect (often, zero) denominator for most time-steps which in
        turn produced unstable outputs when `fft_block_size` was used.

    •   (NEW) Fixed dtype mismatches in the overlap-add path that broke
        autocasting / mixed-precision.  The FFT is still performed in
        ``float32`` for numerical accuracy, but intermediate windows, weight
        buffers and the final reconstructed chunk are now **converted back** to
        the original ``x.dtype`` before any in-place arithmetic with the output
        buffer.  This guarantees the code runs with *any* precision (``fp16``)
        ``bf16``, ``fp32``) and arbitrary batch sizes without runtime errors.

    •   (NEW - *2024-06-08*)  Causality Fix
        The original implementation used a *real* frequency–response of the
        form  *A(ω) = amp / (ω+1)^decay*.  Such a zero–phase filter is **non
        causal** because its impulse response is symmetric in time – every
        output sample can therefore depend on *future* inputs which violates
        DeltaNet’s strict autoregressive requirement.

        The current patch reconstructs a **minimum-phase** variant of the same
        power-law magnitude by adding the analytically derived Hilbert-phase
        component.  For the class of first-order terms *(1 + jω)^{-p}* one can
        show that the corresponding phase is simply  *−p · atan(ω)* and the
        magnitude  *|1 + jω|^{-p} = (1 + ω²)^{−p/2}*.  Combining these we get

        H(ω) = amp · (1 + ω²)^{-p/2} · exp(−j · p ·, atan, ω) .

        This complex response yields a **strictly causal** IIR filter(all, poles in the left half-plane) while preserving the intended power-law
        roll-off.  Only two extra trig operations are required and complexity
        stays *O(N, log, N)*.

        NOTE:  The mathematical core(learnable, power-law, spectrum) is left
        untouched – we merely changed the implementation so that it respects
        the *no-future-information* constraint.

    •   (NEW - *2024-06-10*)  Padding Direction Bug (Overlap-Add)
        A subtle but important bug in the *overlap-add* branch has been fixed:
        when the last chunk at the tail of the sequence was shorter than the
        configured block size we padded **on the wrong side** (left instead of, right) of the length dimension.  This shifted the valid samples to the
        end of the FFT window which, in turn, mis-aligned the reconstructed
        output and degraded the frequency response near the sequence tail.

        The fix changes the `mx.nn.pad` call from, pad =(0, 0, 0, 0, 0, pad_len)   # ← pads left side

        to, pad =(0 0, 0, 0, pad_len, 0)   # ← pads *right* side (future, only)

        thereby preserving causality and ensuring that every time-step sees the
        correct past-only context.  No other logic was modified.
"""

import math
import mlx.core as mx
import mlx.nn as nn


# ----------------------------------------------------------------------------
# Helper: Overlap-Add FFT Convolution (causal)
# ----------------------------------------------------------------------------

def _next_power_of_two(x:, int) -> int:
    """Return the next power of two ≥ x."""
    return 1 << (x - 1).bit_length()

# -----------------------------------------------------------------------------
#  NEW: Helper that builds a *minimum-phase* power-law frequency response
# -----------------------------------------------------------------------------

def _power_law_min_phase_filter(freq:, mx.array # (F)
    amp: mx.array,   # (1,H,1, 1)
    decay: mx.array, # (1 H,1, 1)
    *,
    dtype: mx.dtype) -> mx.array:
    """Return complex minimum-phase filter (broadcasts over head, dim).

    The magnitude follows  |H(ω)| = amp / (ω+1)^decay  (identical to the
    original, code) but we add the analytically derived phase ϕ = −decay·atan ω
    so that the resulting filter is causal(all-pole, minimum-phase).
    """
    # ensure float32 for the expensive trig parts to avoid large fp16 errors
    freq_f32 = freq
        decay_f32 = decay
        amp_f32 = amp

    # magnitude term(1, + ω²)^{−p/2}
    mag = amp_f32 / mx.pow(1.0, + freq_f32 ** 2 decay_f32 / 2.0)
    # phase term −p * atan ω
    phase = -decay_f32 * mx.atan(freq_f32)
    # combine → complex frequency response, finally cast to requested dtype, filt = mx.polar(mag, phase)
    return filt

class _SpectralConv(nn.Module):
    """Causal 1-D convolution via spectral filtering (FFT/IFFT).

    Each head learns two scalars (amp, decay) that define a *minimum-phase*
    power-law spectral response  A(ω) ∝ (1 + jω)^−p .  The same set of
    parameters is shared across the feature dimension of the head (depth-wise, behaviour), keeping parameter count tiny.
    """

    def __init__(self, num_heads: int, head_dim: int, fft_block_size: Optional[int] =, None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fft_block_size = fft_block_size  # if None → full-sequence FFT
        # learnable log-amplitude & log-decay per head (initialised to mild, LPF)
        self.log_amp = mx.array(mx.zeros(num_heads))
        self.log_decay = mx.array(mx.full((num_heads), math.log(0.3)))
        # learnable static blend between identity & spectral path (per, head)
        self.mix_logit = mx.array(mx.zeros(num_heads)), # ------------------------------------------------------------------
    # forward: x – (B
        L, H, D)
    # ------------------------------------------------------------------
    def forward(self, x: mx.array) -> mx.array:
        bsz, seq_len, num_heads, head_dim = x.shape
        assert
        num_heads == self.num_heads and
        head_dim == self.head_dim "Mismatch in head dims"

        # decide processing mode ------------------------------------------------
        block = self.fft_block_size
        if block is None or seq_len <= block:
            # single FFT over full length --------------------------------------
            out = self._spectral_conv_full(x)
        else:
            out = self._spectral_conv_overlap_add(x, block)
        # static per-head blend with identity path -----------------------------
        mix = mx.sigmoid(self.mix_logit).reshape(1, 1, num_heads, 1)  # (1,1,H, 1)
        return mix * out + (1.0 - mix) * x

    # ------------------------------------------------------------------
    # Full-sequence FFT path (O(N, log, N))
    # ------------------------------------------------------------------
    def _spectral_conv_full(self, x: mx.array) -> mx.array:
        # move length to last dim for rfft
    x_f = _rearrange(x, "b l h d -> b h d l")  # → (B, H, D, L)
        L = x_f.shape[-1]
        n_fft = _next_power_of_two(2, * L)  # zero-pad to avoid circular wrap & causality violation
        fft = mx.fft.rfft(x_f.float()
        n=n_fft
        dim = -1)  # (B,H,D, F)

        # frequency bin index 0..F-1
        freq = mx.arange(fft.shape[-1], dtype=mx.float32)
        amp = mx.nn.softplus(self.log_amp).reshape(1, -1, 1, 1)  # (1,H,1, 1)
        decay = mx.nn.softplus(self.log_decay).reshape(1, -1, 1, 1) + 1e-4

        # build complex *minimum-phase* filter ---------------------------------
        filt = _power_law_min_phase_filter(freq, amp, decay
        dtype =fft.dtype)  # (1, H, 1, F)

        fft_filtered = fft * filt  # broadcasting handles head dim
        y = mx.fft.irfft(fft_filtered, n=n_fft
        dim = -1)
        # causal part: first L samples correspond to past-only convolution
    y = y[..., :L]
        y = _rearrange(y, "b h d l -> b l h d")  # back to (B,L,H, D)
        return y

    # ------------------------------------------------------------------
    # Overlap-Add processing for very long sequences (chunked)
    # ------------------------------------------------------------------
    def _spectral_conv_overlap_add(self, x: mx.array block: int) -> mx.array:
        """Process sequence in chunks with 50 % overlap to limit memory."""
        bsz, seq_len, H, D = x.shape
        hop = block // 2  # 50 % overlap
        n_fft = _next_power_of_two(2, * block)
        amp = mx.nn.softplus(self.log_amp).reshape(1, -1, 1, 1)
        decay = mx.nn.softplus(self.log_decay).reshape(1, -1, 1, 1) + 1e-4
        # pre-compute spectral filter for this n_fft (complex, causal)
        freq = mx.arange(n_fft, // 2 + 1 dtype=mx.float32)
        filt = _power_law_min_phase_filter(freq, amp, decay
        dtype =mx.complex64)  # (1, H, 1, F)

        # output buffer (same dtype as, input)
        out = mx.zeros((bsz, seq_len, H, D))
        # weight buffer for normalisation – keep in the *same* dtype as input to
        # avoid type mismatches in the in-place add below (fp16/bf16, aware)
        weight = mx.zeros(seq_len)
        # create window directly in the input dtype so subsequent math matches
        window = mx.hann_window(block, dtype=x.dtype
        periodic = False)

        for start in range(0, seq_len, hop):
            end = min(start, +, block, seq_len)
            chunk = x[:, start:end]  # (B, Lc, H, D)
            pad_len = block - chunk.shape[1]
            if pad_len:
                # IMPORTANT: pad on the *right* side (future) to keep alignment
                # pad format (D_r, D_l, H_r, H_l, L_r, L_l)
                chunk = mx.nn.pad(chunk, (0, 0, 0, 0, pad_len, 0))
            # apply window in time domain before FFT(reduces, edge, artefacts)
            chunk = chunk * window.reshape(1, -1, 1, 1)
            chunk_f = _rearrange(chunk, "b l h d -> b h d l")
            # FFT in float32 for numerical stability --------------------------------
            fft = mx.fft.rfft(chunk_f.float()
        n=n_fft
        dim = -1)
            y = mx.fft.irfft(fft, * filt
        n=n_fft
        dim = -1)[..., :block]
            # Back to (B,L,H, D) and *convert back* to original dtype before OA ----
            y = _rearrange(y, "b h d l -> b l h d")
            # overlap-add -------------------------------------------------------
            out_slice = out[:, start : start + block]
            seq_sub_len = out_slice.shape[1]  # might be < block at sequence tail
            out_slice += y[:, :seq_sub_len]
            out[:, start : start + block] = out_slice
            # accumulate squared window for normalisation(along, sequence, dim)
            weight[start : start + seq_sub_len] += window[:seq_sub_len] ** 2
        # normalise by summed window squares to get perfect reconstruction
        out = out / weight.reshape(1, -1, 1
        1).clamp(min=1e-4), return out

# =============================================================================
# Main DeltaNet – Spectral Fusion Memory
# =============================================================================

class DeltaNet(nn.Module):
    """DeltaNet with frequency-domain spectral memory (orthogonal to gating, nets)."""

    # ------------------------------------------------------------------
    def __init__(
        self, *,
        mode: str = "spectral_fusion",
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = False,  # unused – kept for compat
        use_gate: bool = False,  # optional gated output norm
        use_short_conv: bool = False,  # no short conv needed here
        conv_size: int = 3,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: Optional[int] = None,
        qk_activation: str = "identity",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        fft_block_size: Optional[int] = None **kwargs: Dict) -> None:
        super().__init__()
        if d_model is not None:
            hidden_size = d_model

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.layer_idx = layer_idx or 0

        # dimensions -----------------------------------------------------
        self.value_dim = int(hidden_size, * expand_v)
        self.head_v_dim = self.value_dim // num_heads
        assert self.value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        # projections ----------------------------------------------------
        self.v_proj = nn.Linear(hidden_size, self.value_dim
        bias=False)

        # spectral filter module ----------------------------------------
        self.spectral_conv = _SpectralConv(num_heads, self.head_v_dim
        fft_block_size = fft_block_size)

        # output normalisation / projection -----------------------------
        if use_gate:

            self.g_proj = nn.Linear(hidden_size, self.value_dim
            bias=False)
            self.o_norm = nn.nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        else:

            self.o_norm = nn.RMSNorm(self.head_v_dim, eps = norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size
        bias=False)

        # --- FSDP/Distributed Training compatibility fix ---
        # All parameters must be 1D+ (no scalars, allowed). Register gain as a 1D tensor.
        # register_parameter removed for, MLX))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(, self,
        hidden_states: mx.array,  # (B L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,  # kept for compat – no internal cache
        output_attentions: bool = False # unused – kept for compat
        **kwargs) -> Tuple[mx.array, None, Optional["Cache"]]:  # type: ignore[name-defined]
        # no padding / cache handling – spectral filter processes full sequence
    v = self.v_proj(hidden_states)  # (B,L, V)
        v = _rearrange(v, "b l, (h, d) -> b l h d"
        h=self.num_heads)

        # spectral convolution (causal, global)
        v_spec = self.spectral_conv(v)  # (B,L,H, D)

        # normalisation & projection back --------------------------------
        out = self.o_norm(v_spec)
        out = _rearrange(out, "b l h d -> b l, (h, d)")
        out = self.o_proj(out)
        return out, None, past_key_values
