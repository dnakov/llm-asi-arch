# PyTorch to MLX Conversion Instructions

This document provides comprehensive guidelines for converting PyTorch neural network architectures to MLX format for Apple Silicon optimization.

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Framework Migration Patterns](#framework-migration-patterns)
3. [Critical Component Replacements](#critical-component-replacements)
4. [Mathematical Operations Translation](#mathematical-operations-translation)
5. [Architecture-Specific Patterns](#architecture-specific-patterns)
6. [Step-by-Step Conversion Process](#step-by-step-conversion-process)
7. [Example Conversion](#example-conversion)
8. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
9. [Validation and Testing](#validation-and-testing)

## Quick Reference

### Import Changes
```python
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F

# MLX
import mlx.core as mx
import mlx.nn as nn
# Note: MLX doesn't have separate functional module
```

### Basic Type Mapping
| PyTorch | MLX |
|---------|-----|
| `torch.Tensor` | `mx.array` |
| `torch.nn.Module` | `mlx.nn.Module` |
| `torch.compile` | Remove (MLX auto-optimizes) |
| `torch.device` | Not needed (MLX handles automatically) |
| `.to(device)` | Remove |
| `.cuda()` | Remove |

## Framework Migration Patterns

### 1. Module Definition
```python
# PyTorch
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 128)
    
    def forward(self, x):
        return self.linear(x)

# MLX
class MyModule(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 128)
    
    def __call__(self, x):  # Note: __call__ instead of forward
        return self.linear(x)
```

### 2. Parameter Initialization
```python
# PyTorch
self.weight = nn.Parameter(torch.randn(256, 128))
with torch.no_grad():
    self.weight.fill_(0.1)

# MLX
self.weight = mx.random.normal((256, 128))  # Direct array assignment
# For manual initialization:
self.weight = mx.full((256, 128), 0.1)
```

### 3. Tensor Operations
```python
# PyTorch
x = torch.zeros(4, 256)
y = torch.ones_like(x)
z = torch.cat([x, y], dim=1)

# MLX
x = mx.zeros((4, 256))
y = mx.ones_like(x)
z = mx.concatenate([x, y], axis=1)  # axis instead of dim
```

## Critical Component Replacements

### 1. Activation Functions
```python
# PyTorch -> MLX
F.relu(x)           -> nn.relu(x)
F.silu(x)           -> nn.silu(x)
F.gelu(x)           -> nn.gelu(x)
F.elu(x, 1.0) + 1.0 -> nn.elu(x) + 1.0
F.sigmoid(x)        -> nn.sigmoid(x)
F.softmax(x, dim=-1) -> nn.softmax(x, axis=-1)
F.softplus(x)       -> nn.softplus(x)
```

### 2. Normalization Layers
```python
# PyTorch RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# MLX RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)) * self.weight
```

### 3. Convolution Operations
```python
# PyTorch depth-wise convolution
def forward(self, x):  # (B, L, H, D)
    b, l, h, d = x.shape
    x_f = rearrange(x, "b l h d -> b (h d) l")
    weight = rearrange(self.filters, "h d k -> (h d) 1 k")
    x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
    y = F.conv1d(x_pad, weight=weight, groups=h * d)
    return rearrange(y, "b (h d) l -> b l h d", h=h)

# MLX depth-wise convolution (manual implementation)
def __call__(self, x: mx.array) -> mx.array:  # (B, L, H, D)
    b, l, h, d = x.shape
    x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0), (0, 0)])
    
    # Process each head separately
    output_list = []
    for head_idx in range(h):
        head_filter = self.filters[head_idx]  # (D, K)
        head_outputs = []
        
        for pos in range(l):
            window = x_padded[:, pos:pos + self.kernel_size, head_idx, :]  # (B, K, D)
            if window.shape[1] == self.kernel_size:
                conv_out = mx.sum(window * head_filter[None, :, :].transpose(0, 2, 1), axis=1)
                head_outputs.append(conv_out)
            else:
                head_outputs.append(mx.zeros((b, d)))
        
        head_result = mx.stack(head_outputs, axis=1)  # (B, L, D)
        output_list.append(head_result)
    
    output = mx.stack(output_list, axis=2)  # (B, L, H, D)
    return output
```

### 4. Attention and Matrix Operations
```python
# PyTorch
attn = torch.matmul(q, k.transpose(-1, -2))
attn = attn.masked_fill(mask, float('-inf'))
attn = F.softmax(attn, dim=-1)

# MLX
attn = mx.matmul(q, k.transpose(0, 1, 3, 2))  # Specify all axes
attn = mx.where(mask, float('-inf'), attn)
attn = nn.softmax(attn, axis=-1)
```

## Mathematical Operations Translation

### 1. Linear Algebra
```python
# PyTorch -> MLX
torch.linalg.norm(x, dim=-1)     -> mx.linalg.norm(x, axis=-1)
torch.eye(n)                     -> mx.eye(n)
torch.triu(x, diagonal=1)        -> mx.triu(x, k=1)
torch.inverse(x)                 -> mx.linalg.inv(x)
x.transpose(-1, -2)              -> x.transpose(0, 1, 3, 2)  # Specify all dims
```

### 2. Statistical Operations
```python
# PyTorch -> MLX
x.mean(dim=-1, keepdim=True)     -> mx.mean(x, axis=-1, keepdims=True)
x.var(dim=-1, unbiased=False)    -> mx.var(x, axis=-1, keepdims=True)
x.sum(dim=-1, keepdim=True)      -> mx.sum(x, axis=-1, keepdims=True)
x.max(dim=-1)                    -> mx.max(x, axis=-1)
x.amax(dim=-1)                   -> mx.max(x, axis=-1)
```

### 3. Shape Manipulation
```python
# PyTorch -> MLX
torch.cat([x, y], dim=1)         -> mx.concatenate([x, y], axis=1)
torch.stack([x, y], dim=0)       -> mx.stack([x, y], axis=0)
x.reshape(b, -1)                 -> x.reshape(b, -1)
x.view(b, -1)                    -> x.reshape(b, -1)
x.unsqueeze(-1)                  -> mx.expand_dims(x, axis=-1)
x.squeeze(0)                     -> mx.squeeze(x, axis=0)
```

### 4. Tensor Creation
```python
# PyTorch -> MLX
torch.zeros(4, 256)              -> mx.zeros((4, 256))
torch.ones_like(x)               -> mx.ones_like(x)
torch.randn(256, 128)            -> mx.random.normal((256, 128))
torch.full((4, 256), 0.1)        -> mx.full((4, 256), 0.1)
```

## Architecture-Specific Patterns

### 1. FLA Module Dependencies
Remove or replace FLA-specific imports:
```python
# Remove these imports:
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# Replace with MLX implementations or remove functionality
```

### 2. Einops Replacement
```python
# PyTorch with einops
from einops import rearrange
x = rearrange(x, "b l (h d) -> b l h d", h=num_heads)

# MLX native operations
x = x.reshape(b, l, num_heads, d)
```

### 3. Device Management
```python
# Remove all device-related code:
device=x.device          # Remove
.to(device)             # Remove
.cuda()                 # Remove
dtype=torch.bool        # Replace with mx.bool_
```

### 4. Cache and State Management
```python
# PyTorch cache handling
if past_key_values is not None and len(past_key_values) > self.layer_idx:
    last_state = past_key_values[self.layer_idx]

# MLX cache handling (simplified)
last_state = None
if past_key_values is not None and self.layer_idx is not None:
    last_state = past_key_values.get(self.layer_idx, {})
```

## Step-by-Step Conversion Process

### Step 1: Update Imports
1. Replace `torch` imports with `mlx.core` and `mlx.nn`
2. Remove `einops` if present
3. Remove FLA module imports
4. Remove device-related imports

### Step 2: Class Definition
1. Change `forward` method to `__call__`
2. Update parameter initialization to use MLX arrays
3. Replace `nn.Parameter` with direct array assignment

### Step 3: Replace Operations
1. Update all tensor operations (see mapping tables above)
2. Replace activation functions
3. Convert normalization layers
4. Update matrix operations

### Step 4: Handle Special Cases
1. Remove `@torch.compile` decorators
2. Replace complex convolution operations
3. Update attention mechanisms
4. Fix dtype specifications

### Step 5: Simplify Architecture
1. Remove unpadding/repadding logic (MLX handles this)
2. Simplify cache management
3. Remove device placement code

## Example Conversion

Let's convert `delta_net_entropy_kl_floor_gate.py` as a complete example:

### Original PyTorch Structure:
```python
# Key components to convert:
1. _DepthwiseFIRConv1d class
2. _delta_chunk_monotonic function  
3. _EntropyKLFusionGate class
4. Main DeltaNet class
```

### Converted MLX Version:
```python
# -*- coding: utf-8 -*-
"""
DeltaNet – Entropic Floor+KL Regularized Output-Stat Gating & Monotonic Long-Horizon Memory - MLX Implementation
=========================================================================================
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict
import mlx.core as mx
import mlx.nn as nn

# Utility functions
def elu_p1(x: mx.array) -> mx.array:
    return nn.elu(x) + 1.0

def sum_norm(x: mx.array) -> mx.array:
    return x / mx.sum(x, axis=-1, keepdims=True)

def l2norm(x: mx.array) -> mx.array:
    return x / mx.linalg.norm(x, axis=-1, keepdims=True)

# Depthwise causal FIR convolution
class DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 3, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize with Dirac + noise
        filters = mx.zeros((num_heads, head_dim, kernel_size))
        # Set last element to 1.0 (Dirac)
        filters = filters.at[..., -1].set(1.0)
        # Add noise
        filters = filters + noise_std * mx.random.normal(filters.shape)
        self.filters = filters

    def __call__(self, x: mx.array) -> mx.array:  # (B, L, H, D)
        b, l, h, d = x.shape
        x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0), (0, 0)])
        
        output_list = []
        for head_idx in range(h):
            head_filter = self.filters[head_idx]  # (D, K)
            head_outputs = []
            
            for pos in range(l):
                window = x_padded[:, pos:pos + self.kernel_size, head_idx, :]  # (B, K, D)
                if window.shape[1] == self.kernel_size:
                    # Apply convolution: (B, K, D) * (D, K) -> (B, D)
                    conv_out = mx.sum(window * head_filter[None, :, :].transpose(0, 2, 1), axis=1)
                    head_outputs.append(conv_out)
                else:
                    head_outputs.append(mx.zeros((b, d)))
            
            head_result = mx.stack(head_outputs, axis=1)  # (B, L, D)
            output_list.append(head_result)
        
        output = mx.stack(output_list, axis=2)  # (B, L, H, D)
        return output

# Monotonic lambda parameterization
def monotonic_lambda(forget_param: mx.array, lambda_min: float = 0.5) -> mx.array:
    return lambda_min + (1.0 - lambda_min) * nn.sigmoid(forget_param)

# Delta rule with monotonic forgetting
def delta_chunk_monotonic(
    q: mx.array, k: mx.array, v: mx.array, beta: mx.array, 
    lam: Optional[mx.array], chunk_size: int = 32
) -> Tuple[mx.array, mx.array]:
    b, h, L, d_k = q.shape
    
    # Padding
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad_len)])
    L_pad = L + pad_len
    
    # Normalize and scale
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    # Reshape into chunks
    n_chunks = L_pad // chunk_size
    q = q.reshape(b, h, n_chunks, chunk_size, d_k)
    k = k.reshape(b, h, n_chunks, chunk_size, d_k)
    v = v.reshape(b, h, n_chunks, chunk_size, -1)
    k_beta = k_beta.reshape(b, h, n_chunks, chunk_size, d_k)
    
    # Masks
    mask_tri = mx.triu(mx.ones((chunk_size, chunk_size)), k=0).astype(mx.bool_)
    mask_future = mx.triu(mx.ones((chunk_size, chunk_size)), k=1).astype(mx.bool_)
    
    # Process chunks
    S = mx.zeros((b, h, d_k, v.shape[-1]))
    o = mx.zeros_like(v)
    
    for idx in range(n_chunks):
        q_i, k_i, v_i, k_beta_i = q[:, :, idx], k[:, :, idx], v[:, :, idx], k_beta[:, :, idx]
        
        # Attention computation
        attn = -mx.matmul(k_beta_i, k_i.transpose(0, 1, 3, 2))
        attn = mx.where(mask_tri[None, None, :, :], 0, attn)
        
        # Simplified inversion
        identity = mx.eye(chunk_size)[None, None, :, :]
        try:
            attn = mx.linalg.inv(identity - attn)
        except:
            attn = identity
        
        u = mx.matmul(attn, v_i)
        w = mx.matmul(attn, k_beta_i)
        
        # Local attention
        attn_local = mx.matmul(q_i, k_i.transpose(0, 1, 3, 2))
        attn_local = mx.where(mask_future[None, None, :, :], 0, attn_local)
        
        # Update
        u_i = u - mx.matmul(w, S)
        o_inter = mx.matmul(q_i, S)
        
        # Update output at this chunk
        chunk_output = o_inter + mx.matmul(attn_local, u_i)
        output_slices = []
        for j in range(n_chunks):
            if j == idx:
                output_slices.append(chunk_output)
            else:
                output_slices.append(o[:, :, j])
        o = mx.stack(output_slices, axis=2)
        
        # Update state with monotonic forgetting
        if lam is not None:
            lam_bh = lam[:, :, None, None]
            S = S * lam_bh + mx.matmul(k_i.transpose(0, 1, 3, 2), u_i)
        else:
            S = S + mx.matmul(k_i.transpose(0, 1, 3, 2), u_i)
    
    # Reshape output
    o = o.reshape(b, h, L_pad, -1)
    if pad_len:
        o = o[:, :, :L]
    
    return o, S

# Entropy+KL fusion gate
class EntropyKLFusionGate(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int,
        fusion_hidden_mult: int = 2, max_floor: float = 0.075, temp_init: float = 1.25
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_floor = max_floor
        self.n_paths = 4
        
        # Learnable parameters
        self.log_temp = mx.log(mx.full((num_heads,), temp_init))
        self.floor_param = mx.full((num_heads, self.n_paths), -2.0)
        
        # MLP
        gate_in = hidden_size + 4 * self.n_paths * num_heads
        self.mlp = nn.Sequential(
            nn.Linear(gate_in, hidden_size * fusion_hidden_mult, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * fusion_hidden_mult, num_heads * self.n_paths, bias=True),
        )
        
        # Initialize bias to favor value path
        bias = mx.zeros((num_heads * self.n_paths,))
        for i in range(num_heads):
            bias = bias.at[i * self.n_paths + 3].set(2.0)  # Favor value path
        self.mlp.layers[-1].bias = bias
        
        self.last_entropy = None
        self.last_kl = None
        self.last_gate_loss = None

    def __call__(
        self, hidden: mx.array, short: mx.array, long: mx.array, 
        delta: mx.array, value: mx.array,
        entropy_weight: float = 0.04, kl_weight: float = 0.04
    ) -> mx.array:
        
        def stats(t: mx.array) -> mx.array:
            # Calculate [mean, var, max, l2-norm] statistics
            m = mx.mean(t, axis=-1, keepdims=True)
            v = mx.var(t, axis=-1, keepdims=True)
            mx_val = mx.max(t, axis=-1, keepdims=True)
            l2 = mx.linalg.norm(t, axis=-1, keepdims=True)
            return mx.concatenate([m, v, mx_val, l2], axis=-1)
        
        # Gather statistics for all branches
        cat_stats = [stats(branch) for branch in [short, long, delta, value]]
        
        # Flatten statistics across heads
        flat_stats = [stat.reshape(stat.shape[0], stat.shape[1], -1) for stat in cat_stats]
        gate_in = mx.concatenate([hidden] + flat_stats, axis=-1)
        
        # MLP forward pass
        logits = self.mlp(gate_in)
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.num_heads, self.n_paths)
        
        # Temperature scaling
        temp = mx.exp(self.log_temp)[None, None, :, None]
        logits = logits / temp
        
        # Softmax with floor
        raw_p = nn.softmax(logits, axis=-1)
        floor = nn.sigmoid(self.floor_param) * self.max_floor
        floor = floor[None, None, :, :]
        
        clipped = mx.maximum(raw_p, floor)
        p = clipped / mx.sum(clipped, axis=-1, keepdims=True)
        
        # Entropy and KL calculation (for logging)
        entropy = -mx.sum(p * mx.log(p + 1e-8), axis=-1).mean()
        uniform = mx.full_like(p, 1.0 / self.n_paths)
        kl = mx.sum(p * (mx.log(p + 1e-8) - mx.log(uniform)), axis=-1).mean()
        
        # Store for logging
        self.last_entropy = float(entropy)
        self.last_kl = float(kl)
        
        # Calculate loss
        logp = mx.log(p + 1e-8)
        entropy_loss = -mx.sum(p * logp, axis=-1).mean()
        kl_loss = mx.sum(p * (logp - mx.log(uniform)), axis=-1).mean()
        self.last_gate_loss = entropy_weight * entropy_loss + kl_weight * kl_loss
        
        return p

# RMS Normalization
class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)) * self.weight

class FusedRMSNormGated(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        norm_x = (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)) * self.weight
        return norm_x * gate

# Short convolution
class ShortConvolution(nn.Module):
    def __init__(self, dims: int, kernel_size: int, activation: Optional[str] = None, bias: bool = False):
        super().__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = mx.random.normal((dims, kernel_size)) * 0.02
        if bias:
            self.bias = mx.zeros((dims,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array, cache=None, output_final_state=False, cu_seqlens=None):
        b, l, d = x.shape
        x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        
        # Apply convolution
        output_list = []
        for i in range(l):
            window = x_padded[:, i:i + self.kernel_size, :]  # (B, K, D)
            if window.shape[1] == self.kernel_size:
                conv_out = mx.sum(window * self.weight[None, :, :].transpose(0, 2, 1), axis=1)
                if self.bias is not None:
                    conv_out = conv_out + self.bias
                output_list.append(conv_out)
            else:
                output_list.append(mx.zeros((b, d)))
        
        output = mx.stack(output_list, axis=1)
        
        # Apply activation
        if self.activation == "silu":
            output = nn.silu(output)
        elif self.activation == "relu":
            output = nn.relu(output)
        
        final_state = None
        if output_final_state:
            final_state = x[:, -self.kernel_size + 1:, :]
        
        return output, final_state

# Main DeltaNet class
class DeltaNet(nn.Module):
    """DeltaNet with Entropy+KL-regularized gating and monotonic memory decay - MLX Implementation."""

    def __init__(
        self,
        # Core parameters
        mode: str = "entropy_kl_floor_gate",
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
        # Architecture-specific parameters
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
        
        # Dimensions
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
        
        # Monotonic forgetting parameter
        if use_forget_gate:
            ratio = (forget_init - forget_min) / (1.0 - forget_min)
            ratio = max(min(ratio, 1 - 1e-4), 1e-4)
            init_logit = mx.log(mx.array(ratio) / (1.0 - mx.array(ratio)))
            self.forget_param = mx.full((num_heads,), float(init_logit))
        else:
            self.forget_param = None
        
        # Short convolutions
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for robust DeltaNet performance.")
        
        # FIR branches
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_short_kernel, noise_std=fir_noise_std)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, kernel_size=fir_long_kernel, noise_std=fir_noise_std)
        
        # Fusion gate
        self.fusion_gate = EntropyKLFusionGate(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            fusion_hidden_mult=fusion_hidden_mult,
            max_floor=fusion_max_floor,
            temp_init=fusion_temp_init,
        )
        self.gate_entropy_weight = gate_entropy_weight
        self.gate_kl_weight = gate_kl_weight
        
        # Output normalization and projection
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        self.last_gate_loss = None

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict,
    ) -> Tuple[mx.array, None, Optional[Dict]]:
        
        B, L, _ = hidden_states.shape
        
        # Handle cache
        last_state = None
        if past_key_values is not None and self.layer_idx is not None:
            last_state = past_key_values.get(self.layer_idx, {})
        
        # Convolution states
        conv_q = conv_k = conv_v = None
        if last_state is not None:
            conv_states = last_state.get("conv_state", (None, None, None))
            conv_q, conv_k, conv_v = conv_states
        
        # Apply projections and short convolutions
        q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache)
        k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache)
        v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache)
        
        # Reshape to heads
        q = q.reshape(B, L, self.num_heads, self.head_k_dim)
        k = k.reshape(B, L, self.num_heads, self.head_k_dim)
        v = v.reshape(B, L, self.num_heads, self.head_v_dim)
        
        # Apply activations
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = nn.relu(q), nn.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)
        
        # Beta scaling
        if self.use_beta:
            beta = nn.sigmoid(self.b_proj(hidden_states))
        else:
            beta = mx.ones((B, L, self.num_heads))
        
        if self.allow_neg_eigval:
            beta = beta * 2.0
        
        # Monotonic forgetting
        lam = None
        if self.forget_param is not None:
            lam = monotonic_lambda(self.forget_param, lambda_min=0.55)
            lam = mx.broadcast_to(lam[None, :], (B, self.num_heads))
        
        # Delta rule computation
        q_d = q.transpose(0, 2, 1, 3)
        k_d = k.transpose(0, 2, 1, 3)
        v_d = v.transpose(0, 2, 1, 3)
        beta_d = beta.transpose(0, 2, 1)
        
        delta_out, rec_state = delta_chunk_monotonic(q_d, k_d, v_d, beta_d, lam)
        delta_out = delta_out.transpose(0, 2, 1, 3)
        
        # FIR branches
        value = v
        short = self.fir_short(value)
        long = self.fir_long(value)
        
        # Fusion gate
        fusion_w = self.fusion_gate(
            hidden_states, short, long, delta_out, value,
            entropy_weight=self.gate_entropy_weight,
            kl_weight=self.gate_kl_weight,
        )
        
        # Weighted combination
        o = (
            fusion_w[..., 0:1] * short +
            fusion_w[..., 1:2] * long +
            fusion_w[..., 2:3] * delta_out +
            fusion_w[..., 3:4] * value
        )
        
        # Update cache
        if past_key_values is not None and use_cache:
            if past_key_values is None:
                past_key_values = {}
            past_key_values[self.layer_idx] = {
                "recurrent_state": rec_state,
                "conv_state": (conv_q, conv_k, conv_v),
            }
        
        # Output normalization and projection
        if self.use_gate:
            g_vec = self.g_proj(hidden_states).reshape(B, L, self.num_heads, self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)
        
        o = o.reshape(B, L, self.value_dim)
        o = self.o_proj(o)
        
        # Store gate loss for training
        self.last_gate_loss = self.fusion_gate.last_gate_loss
        
        return o, None, past_key_values
```

## Common Pitfalls and Solutions

### 1. Axis vs Dim Parameters
**Problem**: MLX uses `axis` while PyTorch uses `dim`
**Solution**: Replace all `dim=` with `axis=`

### 2. In-place Operations
**Problem**: MLX doesn't support in-place operations like `.fill_()`, `.masked_fill_()`
**Solution**: Use functional equivalents like `mx.where()`, direct assignment

### 3. Device Management
**Problem**: PyTorch device management doesn't exist in MLX
**Solution**: Remove all `.to(device)`, `.cuda()`, `device=` parameters

### 4. Parameter Registration
**Problem**: `nn.Parameter()` doesn't exist in MLX
**Solution**: Assign arrays directly as attributes

### 5. Einops Dependencies
**Problem**: `einops.rearrange()` may not work with MLX
**Solution**: Use native MLX reshape/transpose operations

### 6. Complex Convolutions
**Problem**: MLX lacks some specialized convolution operations
**Solution**: Implement manually using matrix operations and broadcasting

## Validation and Testing

Use the provided `test_conversion.py` framework to validate conversions:

1. **Syntax Check**: Ensure valid Python syntax
2. **Import Check**: Verify all MLX imports work
3. **Shape Compatibility**: Test with sample inputs
4. **Mathematical Equivalence**: Compare outputs where possible
5. **Performance Benchmarking**: Measure MLX vs PyTorch performance

## Best Practices

1. **Start Simple**: Convert basic components first, then complex ones
2. **Test Incrementally**: Validate each component before moving to the next
3. **Preserve Mathematics**: Ensure mathematical correctness is maintained
4. **Simplify When Possible**: Remove unnecessary complexity for MLX
5. **Document Changes**: Note significant modifications for future reference

This guide provides a systematic approach to converting PyTorch neural architectures to MLX format while maintaining mathematical fidelity and leveraging Apple Silicon optimization capabilities.

## Single File Conversion Workflow

For converting a specific PyTorch architecture file to MLX format, follow this streamlined process:

### Prerequisites
1. Ensure MLX is installed: `pip install mlx mlx-lm`
2. Have both testing scripts available:
   - `test_conversion.py` - Basic testing framework
   - `enhanced_test_conversion.py` - Detailed testing with error analysis

### Step-by-Step Single File Conversion

#### 1. Choose Your Source File
```bash
# Example: Converting delta_net_entropy_kl_floor_gate.py
SOURCE_FILE="pytorch_arch/delta_net_entropy_kl_floor_gate.py"
TARGET_FILE="mlx_architectures/delta_net_entropy_kl_floor_gate_mlx.py"
```

#### 2. Create the MLX Version
Follow the conversion patterns in this guide to create the MLX version:

```bash
# Create the MLX file (manually following conversion guidelines)
cp $SOURCE_FILE $TARGET_FILE
# Then edit $TARGET_FILE following the conversion patterns below
```

#### 3. Test the Conversion
Use the enhanced testing framework to validate your conversion:

```bash
# Basic test of single file
python enhanced_test_conversion.py --test-file delta_net_entropy_kl_floor_gate_mlx.py

# Detailed test with verbose output
python enhanced_test_conversion.py --test-file delta_net_entropy_kl_floor_gate_mlx.py --detailed --verbose

# Save detailed results for analysis
python enhanced_test_conversion.py --test-file delta_net_entropy_kl_floor_gate_mlx.py --detailed --save-results my_conversion_results.json
```

#### 4. Interpret Test Results
The enhanced tester provides detailed feedback:

- **✅ Passed**: Component works correctly
- **⚠️ Warning**: Works but has issues (e.g., PyTorch remnants)
- **❌ Failed**: Critical error that needs fixing
- **⏭️ Skipped**: Test skipped due to dependencies

#### 5. Fix Issues Based on Test Output
The enhanced tester provides specific suggestions for each error type:

```bash
# Example output interpretation:
#   ❌ imports: failed - No module named 'fla'
#   └─ Suggestion: Check if all required MLX modules are available and properly imported
#   └─ Error Type: import_error:ImportError

# This tells you to remove FLA dependencies and replace with MLX equivalents
```

### Quick Conversion Checklist

For each file you convert, ensure you:

- [ ] **Remove PyTorch imports**: Replace `torch` with `mlx.core as mx` and `mlx.nn as nn`
- [ ] **Update class methods**: Change `forward` to `__call__`
- [ ] **Convert tensor operations**: Replace `torch.tensor` with `mx.array`, etc.
- [ ] **Fix parameter handling**: Remove `nn.Parameter()`, use direct array assignment
- [ ] **Update activation functions**: Replace `F.relu()` with `nn.relu()`, etc.
- [ ] **Remove device management**: Remove `.cuda()`, `.to(device)`, etc.
- [ ] **Fix axis parameters**: Change `dim=` to `axis=`
- [ ] **Test thoroughly**: Run the enhanced test suite

### Testing Commands Reference

```bash
# Test single file (basic)
python test_conversion.py --test-file YOUR_FILE_mlx.py

# Test single file (enhanced with error analysis)
python enhanced_test_conversion.py --test-file YOUR_FILE_mlx.py --detailed

# Compare with original PyTorch version
python test_conversion.py --compare-original YOUR_ORIGINAL_FILE.py

# Performance benchmark
python enhanced_test_conversion.py --test-file YOUR_FILE_mlx.py --detailed --iterations 20

# Save detailed results for debugging
python enhanced_test_conversion.py --test-file YOUR_FILE_mlx.py --save-results debug_results.json
```

### Common Conversion Issues and Solutions

Based on testing framework output:

**Import Errors (`import_error:ImportError`)**:
- Remove `from fla.layers.utils import ...`
- Remove `from fla.modules import ...`
- Remove `from einops import rearrange`
- Replace with MLX equivalents or manual implementations

**Attribute Errors (`attribute_error:AttributeError`)**:
- Replace `F.relu()` with `nn.relu()`
- Replace `torch.matmul()` with `mx.matmul()`
- Replace `x.transpose(-1, -2)` with `x.transpose(0, 1, 3, 2)`

**PyTorch Remnants (`pytorch_remnant`)**:
- Remove `@torch.compile` decorators
- Remove `.cuda()`, `.to(device)` calls
- Replace `torch.no_grad()` contexts
- Convert `einops.rearrange()` to native reshaping

**MLX Compatibility Issues (`mlx_specific`)**:
- Use `axis=` instead of `dim=` in operations
- Use `keepdims=` instead of `keepdim=`
- Replace in-place operations with functional equivalents

This guide provides a systematic approach to converting PyTorch neural architectures to MLX format while maintaining mathematical fidelity and leveraging Apple Silicon optimization capabilities.