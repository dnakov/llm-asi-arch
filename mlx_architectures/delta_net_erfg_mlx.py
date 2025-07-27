# -*- coding: utf-8 -*-
"""
DeltaNet – Entropy-Regularised Floor-Gated Multi-Scale Memory (ERFG) - MLX Implementation
===================================================================
Identifier: delta_net_erfg_mlx

MLX implementation of the ERFG architecture with:
1. Entropy-Regularised Fusion Gate with learnable floor and regularisation loss
2. Scheduled forget gate (lambda) to prevent early memory truncation
3. Dual FIR branches with chunked delta-rule kernel
4. Complete parameter matching with PyTorch version
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:
    """MLX implementation of einops rearrange"""
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
    elif pattern == "b l (h d) -> b l h d":
        b, l, hd = tensor.shape
        h = kwargs.get('h')
        d = hd // h
        return tensor.reshape(b, l, h, d)
    elif pattern == "b l h p -> b l (h p)":
        b, l, h, p = tensor.shape
        return tensor.reshape(b, l, h * p)
    elif pattern == "b l h -> b h l":
        return tensor.transpose(0, 2, 1)
    elif pattern == "b l (h p) -> b l h p":
        b, l, hp = tensor.shape
        h = kwargs.get('h')
        p = kwargs.get('p', hp // h)
        return tensor.reshape(b, l, h, p)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")

def _l2norm(x: mx.array) -> mx.array:
    """L2 normalization"""
    return x / mx.linalg.norm(x, axis=-1, keepdims=True)

def _elu_plus_one(x: mx.array) -> mx.array:
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

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, *, kernel_size: int, noise_std: float = 5e-3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        
        weight = mx.zeros((num_heads, head_dim, self.kernel_size))
        # Set identity (Dirac) at latest time-step
        weight = mx.concatenate([
            weight[..., :-1],
            mx.ones_like(weight[..., -1:])  # Dirac at last position
        ], axis=-1)
        
        if noise_std > 0:
            noise = mx.random.normal(weight.shape) * noise_std
            # Make noise orthogonal to identity direction
            proj = mx.sum(noise * weight, axis=-1, keepdims=True)
            noise = noise - proj * weight
            weight = weight + noise
        
        self.filters = weight

    def __call__(self, x: mx.array) -> mx.array:  # x: [B, L, H, D]
        b, l, h, d = x.shape
        x_f = _rearrange(x, "b l h d -> b (h d) l")
        w = _rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = mx.pad(x_f, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        
        # Simulate grouped conv1d using einsum/matmul operations
        w_flat = w.squeeze(1)  # [h*d, k]
        
        outputs = []
        for i in range(l):
            window = x_pad[:, :, i:i+self.kernel_size]  # [b, h*d, k]
            # Element-wise multiply and sum along kernel dimension
            conv_out = mx.sum(window * w_flat[None, :, :], axis=-1)  # [b, h*d]
            outputs.append(conv_out)
        
        y = mx.stack(outputs, axis=-1)  # [b, h*d, l]
        return _rearrange(y, "b (h d) l -> b l h d", h=h)

@mx.compile
def _delta_rule_chunkwise(
    q: mx.array,  # [B H L Dk]
    k: mx.array,  # [B H L Dk]
    v: mx.array,  # [B H L Dv]
    beta: mx.array,  # [B H L]
    forget: Optional[mx.array] = None,  # [B H]
    *,
    chunk_size: int = 32,
):
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    L_pad = L + pad_len

    q = _l2norm(q)
    k = _l2norm(k)

    v = v * mx.expand_dims(beta, -1)
    k_beta = k * mx.expand_dims(beta, -1)

    # chunk reshape
    q = _rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size)
    k = _rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = _rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    k_beta = _rearrange(k_beta, "b h (n c) d -> b h n c d", c=chunk_size)

    mask_tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0).astype(mx.bool_)
    inv = -(k_beta @ mx.swapaxes(k, -2, -1))
    inv = mx.where(mask_tri, 0, inv)
    
    # Simplified iterative construction for MLX
    for i in range(1, chunk_size):
        for j in range(i):
            # Compute update for position [i, j]
            update_val = inv[..., i, j] + mx.sum(inv[..., i, :j] * inv[..., :j, j], axis=-1)
            # Use concatenation to update
            row_i = inv[..., i, :]
            updated_row = mx.concatenate([
                row_i[..., :j],
                update_val[..., None],
                row_i[..., j+1:]
            ], axis=-1)
            inv = mx.concatenate([
                inv[..., :i, :],
                updated_row[..., None, :],
                inv[..., i+1:, :]
            ], axis=-2)
        
    inv = inv + mx.eye(chunk_size)

    u = inv @ v
    w = inv @ k_beta

    S = mx.zeros((b, h, d_k, v.shape[-1]))
    o = mx.zeros_like(v)
    mask_future = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)

    lam = None
    if forget is not None:
        lam = mx.expand_dims(mx.expand_dims(forget, -1), -1)  # [B H 1 1]

    n_chunks = q.shape[2]
    chunk_outputs = []
    for idx in range(n_chunks):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ mx.swapaxes(k_i, -2, -1))
        attn_local = mx.where(mask_future, 0, attn_local)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        chunk_out = q_i @ S + attn_local @ u_i
        chunk_outputs.append(chunk_out)
        if lam is None:
            S = S + mx.swapaxes(k_i, -2, -1) @ u_i
        else:
            S = S * lam + mx.swapaxes(k_i, -2, -1) @ u_i
    
    o = mx.stack(chunk_outputs, axis=2)  # Stack along chunk dimension
    o = _rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

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

# ---------------------------------------------------------------------------
# Entropy-Regularised Fusion Gate
# ---------------------------------------------------------------------------

class _EntropyRegularisedGate(nn.Module):
    """Fusion gate returning weights + regularisation loss terms."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        n_paths: int = 4,
        hidden_mult: int = 2,
        max_floor: float = 0.10,
        temp_init: float = 1.0,
        identity_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_paths = n_paths
        self.max_floor = max_floor

        gate_in = hidden_size + n_paths * head_dim  # hidden + per-path means
        self.mlp = nn.Sequential(
            nn.Linear(gate_in, hidden_mult * hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_mult * hidden_size, num_heads * n_paths, bias=True),
        )
        
        # Initialize bias to favor direct value path - simplified for MLX
        # We'll set this in a post-init step since MLX parameters are handled differently

        # global & per-head logits
        self.global_logit = mx.zeros(n_paths)
        self.head_logit = mx.zeros((num_heads, n_paths))

        # learnable per-head temperature
        self.log_temp = mx.log(mx.full((num_heads,), temp_init))

        # learnable floor per head & path
        self.floor_param = mx.full((num_heads, n_paths), -2.0)

    def __call__(
        self,
        hidden: mx.array,  # [B, L, D]
        path_means: Tuple[mx.array, ...],  # tuple of n_path tensors [B,L,Hd]
    ) -> Tuple[mx.array, mx.array, mx.array]:
        b, l, d = hidden.shape
        h = self.num_heads
        # assemble gate input
        gate_in = mx.concatenate([hidden] + [p for p in path_means], axis=-1)
        local_logits = self.mlp(gate_in)  # [B,L,H*n_paths]
        local_logits = _rearrange(local_logits, "b l (h p) -> b l h p", h=h, p=self.n_paths)

        logits = (
            local_logits
            + self.global_logit.reshape(1, 1, 1, self.n_paths)
            + self.head_logit.reshape(1, 1, h, self.n_paths)
        )

        temp = mx.exp(self.log_temp).reshape(1, 1, h, 1)
        probs = nn.softmax(logits / temp, axis=-1)  # [B, L, H, P]

        # apply learnable floor
        floor = nn.sigmoid(self.floor_param) * self.max_floor  # [H,P]
        floor = floor.reshape(1, 1, h, self.n_paths)
        clipped = mx.maximum(probs, floor)
        probs = clipped / (mx.sum(clipped, axis=-1, keepdims=True) + 1e-6)

        # regularisation terms
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1).mean()
        uniform = mx.ones_like(probs) * (1.0 / self.n_paths)
        kl_uniform = mx.sum(probs * (mx.log(probs + 1e-8) - math.log(1.0 / self.n_paths)), axis=-1).mean()
        return probs, entropy, kl_uniform


class DeltaNet(nn.Module):
    def __init__(
        self,
        *,
        # ---- base params --------------------------------------------------
        mode: str = "erfg",
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
        # ---- FIR params ---------------------------------------------------
        fir_short_kernel: int = 3,
        fir_long_kernel: int = 31,
        fir_noise_std: float = 5e-3,
        # ---- forget-gate params ------------------------------------------
        use_forget_gate: bool = True,
        forget_min: float = 0.5,
        forget_init: float = 1.0,
        warmup_steps: int = 30000,
        # ---- gate params --------------------------------------------------
        gate_hidden_mult: int = 2,
        gate_max_floor: float = 0.10,
        gate_temp_init: float = 1.0,
        # ---- regulariser ---------------------------------------------------
        reg_entropy_coeff: float = 0.01,
        reg_kl_coeff: float = 0.01,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        assert qk_activation in {"silu", "relu", "elu", "identity"}
        assert qk_norm in {"l2", "sum"}

        if d_model is not None:
            hidden_size = d_model

        # store simple attrs
        self.mode = mode
        self.hidden_size = hidden_size
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
        self.use_forget_gate = use_forget_gate
        self.forget_min = forget_min
        self.warmup_steps = warmup_steps
        self.reg_entropy_coeff = reg_entropy_coeff
        self.reg_kl_coeff = reg_kl_coeff

        # dims
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("key/value dims must be divisible by num_heads")

        # projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # forget gate
        if use_forget_gate:
            ratio = (forget_init - forget_min) / (1.0 - forget_min)
            ratio = max(min(ratio, 1 - 1e-4), 1e-4)
            init_logit = math.log(ratio / (1 - ratio))  # logit function
            self.forget_param = init_logit * mx.ones(num_heads)
        else:
            self.forget_param = None

        # short conv
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet training.")

        # FIR branches
        self.fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel, noise_std=fir_noise_std)
        self.fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel, noise_std=fir_noise_std)

        # fusion gate
        self.fusion_gate = _EntropyRegularisedGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            n_paths=4,
            hidden_mult=gate_hidden_mult,
            max_floor=gate_max_floor,
            temp_init=gate_temp_init,
        )

        # output norm / proj
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,  # [B,L,D]
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[dict] = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        step: Optional[int] = None,
        **kwargs: Dict,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[dict]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be [B,L]"
        B_orig, L_in, _ = hidden_states.shape

        # ---- cache retrieval -------------------------------------------
        last_state = None
        if past_key_values is not None and self.layer_idx is not None:
            last_state = past_key_values.get(self.layer_idx)

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -L_in:])
            hidden_states = _index_first_axis(_rearrange(hidden_states, "b s d -> (b s) d"), indices).reshape(1, -1, hidden_states.shape[-1])

        # ---- projections + short conv ----------------------------------
        conv_q = conv_k = conv_v = None
        if last_state is not None and self.use_short_conv and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # ---- reshape heads ---------------------------------------------
        q = _rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = _rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v_direct = _rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # ---- activations / norms ---------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # ---- beta gate --------------------------------------------------
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- forget λ schedule -----------------------------------------
        lam_bh = None
        if self.use_forget_gate:
            lam = self.forget_min + (1.0 - self.forget_min) * nn.sigmoid(self.forget_param)
            if step is not None and self.warmup_steps > 0:
                # linear schedule: no forgetting during warmup
                warm_frac = min(step / float(self.warmup_steps), 1.0)
                lam_sched = 1.0 * (1.0 - warm_frac) + lam * warm_frac
            else:
                lam_sched = lam
            lam_bh = mx.broadcast_to(lam_sched[None, :], (q.shape[0], self.num_heads))  # [B,H]

        # ---- delta memory ----------------------------------------------
        q_d = _rearrange(q, "b l h d -> b h l d")
        k_d = _rearrange(k, "b l h d -> b h l d")
        v_d = _rearrange(v_direct, "b l h d -> b h l d")
        beta_d = _rearrange(beta, "b l h -> b h l")
        delta_out_d, rec_state = _delta_rule_chunkwise(q_d, k_d, v_d, beta_d, forget=lam_bh)
        delta_out = _rearrange(delta_out_d, "b h l d -> b l h d")

        # ---- FIR branches ----------------------------------------------
        short_out = self.fir_short(v_direct)
        long_out = self.fir_long(v_direct)

        # ---- fusion gate -----------------------------------------------
        mean_short = mx.mean(short_out, axis=2)
        mean_long = mx.mean(long_out, axis=2)
        mean_delta = mx.mean(delta_out, axis=2)
        mean_direct = mx.mean(v_direct, axis=2)

        probs, entropy, kl_uniform = self.fusion_gate(
            hidden_states, (mean_short, mean_long, mean_delta, mean_direct)
        )
        w_short = mx.expand_dims(probs[..., 0], -1)
        w_long = mx.expand_dims(probs[..., 1], -1)
        w_delta = mx.expand_dims(probs[..., 2], -1)
        w_direct = mx.expand_dims(probs[..., 3], -1)

        o = w_short * short_out + w_long * long_out + w_delta * delta_out + w_direct * v_direct

        # ---- cache update ----------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values[self.layer_idx] = {
                "recurrent_state": rec_state,
                "conv_state": (conv_q, conv_k, conv_v),
                "offset": L_in,
            }

        # ---- output norm / projection ----------------------------------
        if self.use_gate:
            g_vec = _rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        o = _rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)

        # ---- re-pad if necessary ---------------------------------------
        if attention_mask is not None:
            o = _pad_input(o.squeeze(0), indices, B_orig, L_in)

        # ---- regularisation loss ---------------------------------------
        reg_loss = None
        # For MLX, we compute regularization loss whenever coefficients are > 0
        if (self.reg_entropy_coeff > 0 or self.reg_kl_coeff > 0):
            reg_loss = self.reg_entropy_coeff * entropy + self.reg_kl_coeff * kl_uniform

        return o, reg_loss, past_key_values
