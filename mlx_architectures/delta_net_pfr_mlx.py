# -*- coding: utf-8 -*-
"""
DeltaNet – Persistent-Floor Dynamic Fusion with Per-Head Residual (delta_net_pfr) - MLX Implementation
=====================================================================================================
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict
import mlx.core as mx
import mlx.nn as nn

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """MLX tensor reshaping and transposition utility"""
    if pattern == "b l h d -> b (h d) l":
        b, l, h, d = tensor.shape
        return tensor.transpose(0, 2, 3, 1).reshape(b, h * d, l)
    elif pattern == "h d k -> (h d) 1 k":
        h, d, k = tensor.shape
        return tensor.reshape(h * d, 1, k)
    elif pattern == "b (h d) l -> b l h d":
        b, hd, l = tensor.shape
        h = kwargs.get('h', hd // kwargs.get('d', 1))
        d = hd // h
        return tensor.reshape(b, h, d, l).transpose(0, 3, 1, 2)
    elif pattern == "... (h d) -> ... h d":
        *dims, hd = tensor.shape
        d = kwargs.get('d')
        h = hd // d
        return tensor.reshape(*dims, h, d)
    elif pattern == "b s d -> (b s) d":
        b, s, d = tensor.shape
        return tensor.reshape(b * s, d)
    elif pattern == "b l h d -> b h l d":
        return tensor.transpose(0, 2, 1, 3)
    elif pattern == "b h l d -> b l h d":
        return tensor.transpose(0, 2, 1, 3)
    elif pattern == "b l h d -> b l (h d)":
        b, l, h, d = tensor.shape
        return tensor.reshape(b, l, h * d)
    elif pattern == "b h (n c) d -> b h n c d":
        b, h, nc, d = tensor.shape
        c = kwargs.get('c')
        n = nc // c
        return tensor.reshape(b, h, n, c, d)
    elif pattern == "b h n c d -> b h (n c) d":
        b, h, n, c, d = tensor.shape
        return tensor.reshape(b, h, n * c, d)
    elif pattern == "b l h -> b h l":
        return tensor.transpose(0, 2, 1)
    elif pattern == "b l (h d) -> b l h d":
        b, l, hd = tensor.shape
        d = kwargs.get('d')
        h = hd // d
        return tensor.reshape(b, l, h, d)
    elif pattern == "(b l h) d -> b l h d":
        blh, d = tensor.shape
        b = kwargs.get('b')
        l = kwargs.get('l')
        h = kwargs.get('h')
        return tensor.reshape(b, l, h, d)
    elif pattern == "(b l h) c -> b l h c":
        blh, c = tensor.shape
        b = kwargs.get('b')
        l = kwargs.get('l')
        h = kwargs.get('h')
        return tensor.reshape(b, l, h, c)
    elif pattern == "b l h d -> (b l h) d":
        b, l, h, d = tensor.shape
        return tensor.reshape(b * l * h, d)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1, keepdims=True)

def _elu_p1(x: mx.array) -> mx.array:
    """ELU + 1"""
    return nn.elu(x) + 1.0

def _sum_norm(x: mx.array) -> mx.array:
    """Sum normalization"""
    return x / mx.sum(x, axis=-1, keepdims=True)

def _get_unpad_data(attention_mask: mx.array):
    """Get unpadding data from attention mask"""
    seqlens = mx.sum(attention_mask, axis=1)
    indices = mx.arange(attention_mask.shape[0] * attention_mask.shape[1])
    cu_seqlens = mx.concatenate([mx.array([0]), mx.cumsum(seqlens)])
    return indices, cu_seqlens, seqlens.max()

def _index_first_axis(tensor: mx.array, indices: mx.array) -> mx.array:
    """Index first axis"""
    return tensor[indices]

def _pad_input(tensor: mx.array, indices: mx.array, batch_size: int, seq_len: int) -> mx.array:
    """Pad input back to original shape"""
    return tensor.reshape(batch_size, seq_len, -1)

class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal left padding (O(N))."""
    
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        
        # Identity-like initialization (weight on first step, matching PyTorch version)
        filt = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Set first element to 1.0 (current step), not last
        mask = mx.zeros_like(filt)
        mask = mx.where(mx.arange(filt.shape[-1]) == 0, 1.0, 0.0)  # First element = 1
        mask = mx.broadcast_to(mask, filt.shape)
        filt = mx.where(mask, 1.0, filt)
        # Add small noise
        filt = filt + 0.02 * mx.random.normal(filt.shape)
        self.filters = filt

    def __call__(self, x: mx.array) -> mx.array:
        b, l, h, d = x.shape
        weight = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        x_pad = mx.pad(x_f, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])  # causal left pad
        
        # Implement grouped convolution manually
        y_list = []
        for j in range(l):
            x_slice = x_pad[:, :, j:j + self.kernel_size]  # (b, h*d, kernel_size)
            conv_result = mx.sum(x_slice * weight[:, 0, :], axis=2)  # (b, h*d)
            y_list.append(conv_result)
        y = mx.stack(y_list, axis=2)  # (b, h*d, l)
        
        return _rearrange(y, "b (h d) l -> b l h d", h=h)

def _delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Efficient O(N) associative Δ-rule with strict causality."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    
    L_pad = L + pad_len
    
    # Normalize q/k; scale v & k by β
    q = _l2norm(q)
    k = _l2norm(k)
    v = v * mx.expand_dims(beta, -1)
    k_beta = k * mx.expand_dims(beta, -1)
    
    # Reshape into chunks
    q = _rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size)
    k = _rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = _rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    k_beta = _rearrange(k_beta, "b h (n c) d -> b h n c d", c=chunk_size)
    
    tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0).astype(mx.bool_)
    tri_strict = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)
    
    # Compute inverse using iterative method (approximating the PyTorch version)
    inv = -(k_beta @ k.transpose(0, 1, 2, 4, 3))
    inv = mx.where(tri, 0, inv)
    
    # Simplified iterative approximation
    for i in range(1, min(chunk_size, 5)):  # Limit iterations for performance
        if i < chunk_size:
            # Create slices for matrix operations
            inv_slice = inv[..., i, :]
            inv_partial = inv[..., :, :i]
            if inv_partial.shape[-1] > 0:
                correction = mx.sum(inv_slice[..., None] * inv_partial, axis=-2)
                inv = inv.at[..., i, :i].add(correction[..., :i])
    
    inv = inv + mx.eye(chunk_size)
    
    u = inv @ v
    w = inv @ k_beta
    
    S = mx.zeros((b, h, d_k, v.shape[-1]))
    out_chunks = []
    
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(0, 1, 3, 2))
        attn_local = mx.where(tri_strict, 0, attn_local)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out_chunks.append(q_i @ S + attn_local @ u_i)
        S = S + k_i.transpose(0, 1, 3, 2) @ u_i
    
    # Stack all chunks
    out = mx.stack(out_chunks, axis=2)  # (b, h, n_chunks, c, d)
    out = _rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len > 0:
        out = out[:, :, :L]
    
    return out, S

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x / mx.sqrt(variance + self.eps)
        return self.weight * x

class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x / mx.sqrt(variance + self.eps)
        return self.weight * x * gate

class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)

    def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # MLX Conv1d expects (batch, length, in_channels), x is already in this format
        y = self.conv(x)
        y = y[:, :x.shape[1], :]  # Trim to original sequence length
        
        if self.activation == "silu":
            y = nn.silu(y)
        
        final_state = None if not output_final_state else y[:, -self.kernel_size+1:]
        return y, final_state

class DeltaNet(nn.Module):
    """DeltaNet layer with *persistent local-floor* & *per-head residual bypass*."""
    
    def __init__(
        self,
        # -------- core API (unchanged) ----------------------------------
        mode: str = "pfr",  # persistent-floor residual variant id
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
        # -------- FIR kernels -------------------------------------------
        fir_kernel_size_long: int = 64,
        fir_kernel_size_short: int = 5,
        # -------- Gating network ----------------------------------------
        fusion_hidden_mult: int = 2,
        gate_bias_init: Tuple[float, float, float, float] = (-0.5, -0.5, 1.0, 3.0),
        gate_logit_init: float = math.log(math.expm1(0.7)),
        # -------- Decaying floor schedule -------------------------------
        floor_init: float = 0.08,
        floor_final: float = 0.02,  # <- persistent non-zero floor (was 0.0)
        floor_decay: float = 10_000.0,
        # -------- Conv residual bypass ----------------------------------
        conv_residual_init: float = 0.1,  # α initial in sigmoid space
        # -------- Entropy regularisation --------------------------------
        entropy_target: float = 1.0,
        entropy_coeff: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        
        # -------- bookkeeping ------------------------------------------
        self.mode = mode
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        
        # -------- dimensions -------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        
        # -------- projections ------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # -------- short convs ------------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")
        
        # -------- Dual FIR convolutions --------------------------------
        self.local_fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)
        self.local_fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)
        
        # -------- Content-aware gating ---------------------------------
        self.stat_dim = 16  # per-branch stats (4 branches × 4 stats)
        gate_in_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        # Initialize bias
        bias_tensor = mx.array(list(gate_bias_init))
        self.fusion_gate_mlp.layers[-1].bias = bias_tensor
        
        # learnable temperature (scalar) --------------------------------
        self.logit_temperature = mx.array([gate_logit_init])
        
        # -------- Per-head residual bypass -----------------------------
        init_logit = math.log(conv_residual_init / (1 - conv_residual_init))
        self.conv_residual_logit = mx.full((num_heads,), init_logit)
        
        # -------- Output norm / projection ----------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        # -------- Decaying floor schedule -----------------------------
        self._step = mx.array([0], dtype=mx.int32)
        self.floor_init = float(floor_init)
        self.floor_final = float(floor_final)
        self.floor_decay = float(floor_decay)
        
        # -------- Entropy regularisation ------------------------------
        self.entropy_target = float(entropy_target)
        self.entropy_coeff = float(entropy_coeff)
        self.reg_loss = None

    ###############################################################
    # Statistic helpers                                            #
    ###############################################################
    
    @staticmethod
    def _per_head_stats(x: mx.array) -> mx.array:  # (B,L,H,D) → (B,L,H,4)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        abs_mean = mx.mean(mx.abs(x), axis=-1, keepdims=True)
        l2 = mx.linalg.norm(x, axis=-1, keepdims=True)
        return mx.concatenate([mean, var, abs_mean, l2], axis=-1)

    ###############################################################
    # Forward                                                      #
    ###############################################################

    def __call__(
        self,
        hidden_states: mx.array,  # (B, L, D)
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mx.array, None, Optional[dict]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B, L_in, _ = hidden_states.shape

        # ---------- optional unpadding for variable-length batches ----
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices)
            hidden_states = mx.expand_dims(hidden_states, axis=0)

        # ---------- retrieve previous conv state ----------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # ---------- projections + short conv --------------------------
        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # reshape to heads --------------------------------------------
        q = _rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = _rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = _rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # Q,K activations / norms -------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_p1(q), _elu_p1(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # β for Δ-rule -------------------------------------------------
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---------- Δ-rule global memory ------------------------------
        delta_out_d, recurrent_state = _delta_rule_chunkwise(
            q=_rearrange(q, "b l h d -> b h l d"),
            k=_rearrange(k, "b l h d -> b h l d"),
            v=_rearrange(v_direct, "b l h d -> b h l d"),
            beta=_rearrange(beta, "b l h -> b h l"),
        )
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---------- Local FIR paths -----------------------------------
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)

        # ---------- Content-aware gating -------------------------------
        stats_vec = mx.concatenate(
            [
                self._per_head_stats(local_short),
                self._per_head_stats(local_long),
                self._per_head_stats(delta_out),
                self._per_head_stats(v_direct),
            ],
            axis=-1,
        )  # (B,L,H,16)
        hs_exp = mx.expand_dims(hidden_states, axis=-2)
        hs_exp = mx.broadcast_to(hs_exp, (hs_exp.shape[0], hs_exp.shape[1], self.num_heads, hs_exp.shape[-1]))
        gate_in = mx.concatenate([hs_exp, stats_vec], axis=-1)  # (B,L,H,D+16)
        gate_logits = self.fusion_gate_mlp(_rearrange(gate_in, "b l h d -> (b l h) d", b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads))

        # temperature scaling -----------------------------------------
        temp = nn.softplus(self.logit_temperature) + 1e-4
        gate_logits = gate_logits / temp
        fusion_logits = _rearrange(gate_logits, "(b l h) c -> b l h c", b=gate_in.shape[0], l=gate_in.shape[1], h=self.num_heads)
        fusion_weights = nn.softmax(fusion_logits, axis=-1)  # (B,L,H,4)

        # ---------- Persistent local-floor enforcement ---------------
        eps_now = self.floor_final + (self.floor_init - self.floor_final) * math.exp(-float(self._step.item()) / self.floor_decay)
        if eps_now > 0.0:
            scale = 1.0 - 2 * eps_now
            fusion_weights = fusion_weights * scale
            # Use index assignment properly for MLX
            fusion_weights_list = [fusion_weights[..., i] for i in range(4)]
            fusion_weights_list[0] = fusion_weights_list[0] + eps_now  # short
            fusion_weights_list[1] = fusion_weights_list[1] + eps_now  # long
            fusion_weights = mx.stack(fusion_weights_list, axis=-1)
            fusion_weights = fusion_weights / mx.sum(fusion_weights, axis=-1, keepdims=True)

        # ---------- Entropy regularisation ---------------------------
        entropy = -mx.sum(fusion_weights * mx.log(fusion_weights + 1e-8), axis=-1).mean()
        self.reg_loss = self.entropy_coeff * mx.maximum(0.0, self.entropy_target - entropy)

        # ---------- Branch fusion ------------------------------------
        o = (
            mx.expand_dims(fusion_weights[..., 0], -1) * local_short
            + mx.expand_dims(fusion_weights[..., 1], -1) * local_long
            + mx.expand_dims(fusion_weights[..., 2], -1) * delta_out
            + mx.expand_dims(fusion_weights[..., 3], -1) * v_direct
        )

        # add per-head residual bypass --------------------------------
        alpha = nn.sigmoid(self.conv_residual_logit).reshape(1, 1, self.num_heads, 1)
        o = o + alpha * 0.5 * (local_short + local_long)

        # ---------- Cache update -------------------------------------
        if past_key_values is not None and self.layer_idx is not None and use_cache:
            if past_key_values is None:
                past_key_values = {}
            past_key_values[self.layer_idx] = {
                "recurrent_state": recurrent_state,
                "conv_state": (conv_q, conv_k, conv_v),
            }

        # ---------- Output norm / projection -------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---------- Re-pad if we unpadded -----------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B, L_in)

        # ---------- increment step counter ---------------------------
        self._step = self._step + 1

        return o, None, past_key_values
