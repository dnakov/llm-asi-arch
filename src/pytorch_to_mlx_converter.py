#!/usr/bin/env python3
"""
PyTorch to MLX Architecture Converter
=====================================

Converts the 106 discovered PyTorch architectures from ASI-Arch to MLX format.
Handles the complex PyTorch patterns and FLA library dependencies.
"""

import json
import re
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class PyTorchToMLXConverter:
    """Converts PyTorch architecture code to MLX format"""
    
    def __init__(self):
        self.conversion_rules = self._init_conversion_rules()
        
    def _init_conversion_rules(self) -> List:
        """Initialize PyTorch -> MLX conversion rules"""
        return [
            # Basic imports - order matters! More comprehensive patterns
            (r'import torch\.nn\.functional as F', 'import mlx.nn as F'),  # MLX doesn't have nn.functional
            (r'import torch\.nn as nn', 'import mlx.nn as nn'),
            (r'import torch\b(?!\.)', 'import mlx.core as mx'),
            (r'from torch import', 'from mlx.core import'),
            (r'\btorch\.', 'mx.'),
            
            # Fix standalone mx imports that might be missing
            (r'^(\s*)mx\.', r'\1mx.', re.MULTILINE),  # Ensure mx. calls have proper import
            
            # Ensure we have the basic import if mx. is used
            # This will be handled in post-processing
            
            # Tensor operations
            (r'\.to\(.*?\)', ''),  # Remove .to() calls
            (r'\.cuda\(\)', ''),   # Remove .cuda() calls
            (r'\.cpu\(\)', ''),    # Remove .cpu() calls
            (r'device=.*?,?\s*', ''),  # Remove device arguments
            (r'dtype=torch\.', 'dtype=mx.'),
            
            # nn.Parameter -> mx.array
            (r'nn\.Parameter\((.*?)\)', r'mx.array(\1)'),
            
            # Tensor creation
            (r'torch\.zeros', 'mx.zeros'),
            (r'torch\.ones', 'mx.ones'),
            (r'torch\.randn', 'mx.random.normal'),
            (r'torch\.rand', 'mx.random.uniform'),
            (r'torch\.empty', 'mx.zeros'),  # MLX doesn't have empty
            (r'torch\.tensor', 'mx.array'),
            (r'torch\.eye', 'mx.eye'),
            (r'torch\.arange', 'mx.arange'),
            (r'torch\.linspace', 'mx.linspace'),
            
            # Tensor methods
            (r'\.sigmoid\(\)', '.sigmoid()'),  # Keep as-is, MLX has this
            (r'\.relu\(\)', '.relu()'),
            (r'\.tanh\(\)', '.tanh()'),
            (r'\.softmax\(', '.softmax('),
            (r'\.log_softmax\(', '.log_softmax('),
            (r'\.gelu\(\)', '.gelu()'),
            (r'\.silu\(\)', '.silu()'),
            
            # Shape operations
            (r'\.view\(', '.reshape('),
            (r'\.unsqueeze\(', '.expand_dims('),
            (r'\.squeeze\(', '.squeeze('),
            (r'\.transpose\(', '.transpose('),
            (r'\.permute\(', '.transpose('),
            
            # Math operations
            (r'\.sum\(', '.sum('),
            (r'\.mean\(', '.mean('),
            (r'\.var\(', '.var('),
            (r'\.std\(', '.std('),
            (r'\.max\(', '.max('),
            (r'\.min\(', '.min('),
            
            # Attention/Linear layers
            (r'nn\.MultiheadAttention', 'nn.MultiHeadAttention'),
            (r'nn\.Linear', 'nn.Linear'),
            (r'nn\.LayerNorm', 'nn.LayerNorm'),
            (r'nn\.RMSNorm', 'nn.RMSNorm'),
            (r'nn\.Embedding', 'nn.Embedding'),
            (r'nn\.Dropout', 'nn.Dropout'),
            
            # Activations
            (r'nn\.ReLU', 'nn.ReLU'),
            (r'nn\.GELU', 'nn.GELU'),
            (r'nn\.SiLU', 'nn.SiLU'),
            (r'nn\.Sigmoid', 'nn.Sigmoid'),
            (r'nn\.Tanh', 'nn.Tanh'),
            
            # Convolutions - MLX has limited conv support
            (r'nn\.Conv1d', 'nn.Conv1d'),
            (r'F\.conv1d', 'F.conv1d'),
            (r'F\.pad', 'mx.pad'),
            
            # Loss functions
            (r'nn\.CrossEntropyLoss', 'nn.losses.cross_entropy'),
            (r'nn\.MSELoss', 'nn.losses.mse_loss'),
            (r'F\.cross_entropy', 'nn.losses.cross_entropy'),
            (r'F\.mse_loss', 'nn.losses.mse_loss'),
            
            # Fix F.functional calls (MLX doesn't have nn.functional)
            (r'from mlx\.nn\.functional import', 'from mlx.nn import'),
            (r'mx\.nn\.functional', 'mx.nn'),
            
            # Optimizers
            (r'torch\.optim\.Adam', 'mx.optimizers.Adam'),
            (r'torch\.optim\.SGD', 'mx.optimizers.SGD'),
            (r'torch\.optim\.AdamW', 'mx.optimizers.AdamW'),
            
            # Special MLX patterns - remove register_buffer/parameter calls
            (r'self\.register_buffer\([^)]+\)', '# register_buffer removed for MLX'),
            (r'self\.register_parameter\([^)]+\)', '# register_parameter removed for MLX'),
            
            # Fix common tensor indexing
            (r'\.at\[(.*?)\]\.set\((.*?)\)', r'[\1] = \2'),  # MLX array assignment
            
            # Remove TYPE_CHECKING imports that cause issues
            (r'from typing import.*TYPE_CHECKING.*', ''),
            (r'if TYPE_CHECKING:.*?(?=\n\S|\nclass|\ndef|\n$)', '', re.DOTALL),
            
            # FLA library conversions (convert to standard MLX equivalents)
            (r'from fla\.layers\.utils import.*', ''),
            (r'from fla\.modules import.*', ''),
            (r'from fla\.modules\..*import.*', ''),
            (r'get_unpad_data', '_get_unpad_data'),
            (r'index_first_axis', '_index_first_axis'),
            (r'pad_input', '_pad_input'),
            (r'FusedRMSNormGated', 'nn.RMSNorm'),  # Fallback to regular RMSNorm
            (r'ShortConvolution', '_ShortConvolution'),
            (r'l2norm', '_l2norm'),
            
            # Einops conversion - fix rearrange syntax
            (r'from einops import rearrange', ''),
            (r'rearrange\((.*?),\s*[\'\"](.*?)[\'\"](?:,\s*(.*?))?\)', r'_rearrange(\1, "\2", \3)'),
            
            # Fix missing commas in rearrange calls  
            (r'_rearrange\((.*?), "([^"]*)"([^,\)])', r'_rearrange(\1, "\2", \3)'),
            (r'_rearrange\((.*?), "([^"]*)", \)', r'_rearrange(\1, "\2")'),
            
            # PyTorch compile decorator
            (r'@torch\.compile', '@mx.compile'),
            
            # Fix boolean tensor issues
            (r'torch\.bool', 'mx.bool_'),
            (r'torch\.float32', 'mx.float32'),
            (r'torch\.int32', 'mx.int32'),
            (r'torch\.int64', 'mx.int64'),
            
            # Fix Python 3.10+ type union syntax for older Python compatibility
            (r'int \| None', 'Optional[int]'),
            (r'str \| None', 'Optional[str]'),
            (r'float \| None', 'Optional[float]'),
            (r'(\w+) \| None', r'Optional[\1]'),
            
            # Grad computation (MLX handles differently)
            (r'torch\.autograd\.grad', 'mx.grad'),
            (r'\.backward\(\)', ''),  # Remove explicit backward calls
            (r'\.requires_grad_\(.*?\)', ''),
            (r'\.grad', ''),  # Remove .grad access
            
            # Memory operations
            (r'torch\.cuda\.empty_cache\(\)', ''),
            (r'\.detach\(\)', ''),  # MLX doesn't need detach
            
            # Advanced indexing
            (r'torch\.triu', 'mx.triu'),
            (r'torch\.tril', 'mx.tril'),
            (r'\.masked_fill\(', '._masked_fill('),  # Custom implementation needed
            (r'\.masked_fill_\(', '._masked_fill('),
            
            # Fix common keyword argument syntax errors - removed lambda
            
            # Fix device parameter removal
            (r', device=[^,\)]+', ''),
            (r'device=[^,\)]+,\s*', ''),
        ]
    
    def _add_mlx_utilities(self) -> str:
        """Add utility functions that MLX doesn't have built-in"""
        return '''
# MLX Utility Functions (replacing PyTorch/FLA dependencies)
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List, Dict

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

'''

    def _clean_architecture_code(self, code: str) -> str:
        """Clean and fix common issues in architecture code"""
        # Remove problematic imports and dependencies
        lines = code.split('\n')
        cleaned_lines = []
        skip_block = False
        
        for line in lines:
            # Skip TYPE_CHECKING blocks
            if 'TYPE_CHECKING' in line:
                skip_block = True
                continue
            if skip_block and (line.startswith('class ') or line.startswith('def ') or (line.strip() and not line.startswith(' '))):
                skip_block = False
            if skip_block:
                continue
                
            # Skip problematic imports
            if any(skip in line for skip in [
                'from fla.', 'import fla.', 'from flash_attn', 
                'import flash_attn', 'from apex', 'import apex'
            ]):
                continue
                
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)
    
    def _fix_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues in converted code"""
        import re
        
        # Move all "from __future__ import" statements to the very beginning
        lines = code.split('\n')
        future_imports = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith('from __future__ import'):
                future_imports.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct code with future imports first
        if future_imports:
            # Remove duplicates while preserving order
            seen = set()
            unique_future = []
            for imp in future_imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_future.append(imp)
            code = '\n'.join(unique_future + [''] + other_lines)
        else:
            code = '\n'.join(other_lines)
        
        # Fix _rearrange calls with missing commas - more targeted
        def fix_rearrange(match):
            content = match.group(1)
            pattern = match.group(2)
            remainder = match.group(3) if len(match.groups()) > 2 else ""
            
            # Clean up the remainder part
            remainder = remainder.strip()
            if remainder.startswith(','):
                remainder = remainder[1:].strip()
            
            if remainder and '=' in remainder:
                # Has keyword arguments
                return f'_rearrange({content}, "{pattern}", {remainder})'
            elif remainder:
                # Has positional arguments without = 
                return f'_rearrange({content}, "{pattern}", {remainder})'
            else:
                # No additional arguments
                return f'_rearrange({content}, "{pattern}")'
        
        # Fix rearrange calls with proper comma placement
        code = re.sub(r'_rearrange\(([^,]+), "([^"]+)"([^)]*)\)', fix_rearrange, code)
        
        # Fix F.elu calls - comprehensive patterns for the specific broken cases
        code = re.sub(r'F\.elu\(x\s+1\.0,\s*False\)', r'F.elu(x, 1.0, False)', code)
        code = re.sub(r'F\.elu\(x\s+1\.\s*,\s*False\)', r'F.elu(x, 1.0, False)', code)
        code = re.sub(r'F\.elu\(([^,\s()]+)\s+([^,\s()]+),\s*([^)]+)\)', r'F.elu(\1, \2, \3)', code)
        code = re.sub(r'F\.elu\(([^,\s()]+)\s+([^,\s()]+)\)', r'F.elu(\1, \2)', code)
        
        # Fix missing commas in other function calls
        code = re.sub(r'F\.softmax\(([^,\s()]+)\s+([^)]+)\)', r'F.softmax(\1, \2)', code)
        code = re.sub(r'\.sum\(([^,\s()]+)\s+([^)]+)\)', r'.sum(\1, \2)', code)
        code = re.sub(r'\.mean\(([^,\s()]+)\s+([^)]+)\)', r'.mean(\1, \2)', code)
        code = re.sub(r'\.var\(([^,\s()]+)\s+([^)]+)\)', r'.var(\1, \2)', code)
        code = re.sub(r'\.amax\(([^,\s()]+)\s+([^)]+)\)', r'.amax(\1, \2)', code)
        code = re.sub(r'\.clamp\(([^,\s()]+)\s+([^)]+)\)', r'.clamp(\1, \2)', code)
        code = re.sub(r'\.pow\(([^,\s()]+)\s+([^)]+)\)', r'.pow(\1, \2)', code)
        code = re.sub(r'\.norm\(([^,\s()]+)\s+([^)]+)\)', r'.norm(\1, \2)', code)
        
        # Fix mx.* function calls with missing commas
        code = re.sub(r'mx\.zeros\(([^,\s()]+)\s*,\s*\)', r'mx.zeros(\1)', code)  # Fix mx.zeros(-1, )
        code = re.sub(r'mx\.ones\(([^,\s()]+)\s+([^)]+)\)', r'mx.ones(\1, \2)', code)
        code = re.sub(r'mx\.full\(([^,\s()]+)\s+([^)]+)\)', r'mx.full(\1, \2)', code)
        code = re.sub(r'mx\.randn\(([^,\s()]+)\s+([^)]+)\)', r'mx.randn(\1, \2)', code)
        code = re.sub(r'mx\.empty\(([^,\s()]+)\s+([^)]+)\)', r'mx.empty(\1, \2)', code)
        code = re.sub(r'mx\.cat\(([^,\s()]+)\s+([^)]+)\)', r'mx.cat(\1, \2)', code)
        code = re.sub(r'mx\.pad\(([^,\s()]+)\s+([^)]+)\)', r'mx.pad(\1, \2)', code)
        code = re.sub(r'mx\.softmax\(([^,\s()]+)\s+([^)]+)\)', r'mx.softmax(\1, \2)', code)
        code = re.sub(r'mx\.triu\(([^,\s()]+)\s+([^)]+)\)', r'mx.triu(\1, \2)', code)
        code = re.sub(r'mx\.eye\(([^,\s()]+)\s+([^)]+)\)', r'mx.eye(\1, \2)', code)
        code = re.sub(r'mx\.sqrt\(([^,\s()]+)\s+([^)]+)\)', r'mx.sqrt(\1, \2)', code)
        
        # Fix constructor calls with missing commas
        code = re.sub(r'nn\.Linear\(([^,\s()]+)\s+([^,\s()]+)\s+([^)]+)\)', r'nn.Linear(\1, \2, \3)', code)
        code = re.sub(r'nn\.Linear\(([^,\s()]+)\s+([^,\s()]+)([^)]*)\)', r'nn.Linear(\1, \2\3)', code)
        code = re.sub(r'nn\.Conv1d\(([^,\s()]+)\s+([^,\s()]+)\s+([^,\s()]+)([^)]*)\)', r'nn.Conv1d(\1, \2, \3\4)', code)
        code = re.sub(r'nn\.RMSNorm\(([^,\s()]+)\s+([^)]+)\)', r'nn.RMSNorm(\1, \2)', code)
        
        # Fix function definition signatures with missing commas in parameters
        code = re.sub(r'def __init__\(([^)]+)\s+([^,)]+)\s*([^)]*)\):', r'def __init__(\1, \2\3):', code)
        code = re.sub(r'def forward\(([^)]+)\s+([^,)]+)\s*([^)]*)\):', r'def forward(\1, \2\3):', code)
        
        # More general fix for any function call with missing comma after first arg
        # Fix pattern: func(arg1 arg2, ...) -> func(arg1, arg2, ...)
        code = re.sub(r'([a-zA-Z_]\w*)\(([a-zA-Z_]\w*)\s+([0-9.]+),', r'\1(\2, \3,', code)
        
        # Fix specific problematic patterns found in the converted files:
        
        # Fix broken function definitions with missing commas
        # Pattern: def __init__(self mode: str = "value", next_param: type = val) 
        code = re.sub(r'def __init__\(self\s+([^,]+):', r'def __init__(self, \1:', code)
        code = re.sub(r'def forward\(self\s+([^,]+):', r'def forward(self, \1:', code)
        
        # Fix missing commas in parameter lists 
        # Pattern: num_heads: int head_dim: int -> num_heads: int, head_dim: int
        code = re.sub(r'([a-zA-Z_]\w*:\s*[a-zA-Z_]\w*(?:\[[^\]]*\])?)\s+([a-zA-Z_]\w*:\s*[a-zA-Z_]\w*)', r'\1, \2', code)
        
        # Fix broken variable assignments with missing commas
        # Pattern: q, k, v k_beta = ... -> q, k, v, k_beta = ...
        code = re.sub(r'([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*=', r'\1, \2 =', code)
        
        # Fix function calls with missing commas between arguments
        # Pattern: mx.zeros(shape, dtype) -> mx.zeros(shape, dtype) 
        code = re.sub(r'mx\.zeros\(([^,)]+)\s+([^)]+)\)', r'mx.zeros(\1, \2)', code)
        code = re.sub(r'mx\.ones\(([^,)]+)\s+([^)]+)\)', r'mx.ones(\1, \2)', code)
        code = re.sub(r'mx\.empty\(([^,)]+)\s+([^)]+)\)', r'mx.empty(\1, \2)', code)
        code = re.sub(r'mx\.array\(([^,)]+)\s+([^)]+)\)', r'mx.array(\1, \2)', code)
        code = re.sub(r'mx\.randn_like\(([^,)]+)\s+([^)]+)\)', r'mx.randn_like(\1, \2)', code)
        
        # Fix nn.init function calls
        code = re.sub(r'nn\.init\.(\w+)\(([^,)]+)\s+([^)]+)\)', r'nn.init.\1(\2, \3)', code)
        
        # Fix broken _rearrange calls - common pattern issues
        # Pattern: _rearrange(tensor "pattern" var=val) -> _rearrange(tensor, "pattern", var=val)
        code = re.sub(r'_rearrange\(([^,\s()]+)\s+"([^"]+)"\s+([^)]+)\)', r'_rearrange(\1, "\2", \3)', code)
        
        # Fix function returns with missing commas
        # Pattern: return o S -> return o, S
        code = re.sub(r'return\s+([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*$', r'return \1, \2', code, flags=re.MULTILINE)
        
        # Fix .at[].set() calls that weren't handled properly
        code = re.sub(r'\.at\[([^\]]+)\]\.set\(([^)]+)\)', r'[\1] = \2', code)
        
        # Fix broken tuple unpacking with missing commas
        # Pattern: q conv_state_q = ... -> q, conv_state_q = ...
        code = re.sub(r'([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*=\s*self\.', r'\1, \2 = self.', code)
        
        # Fix einsum call patterns
        code = re.sub(r'einsum\(([^,)]+)\s+([^,)]+)\s+([^)]+)\)', r'einsum(\1, \2, \3)', code)
        
        # Fix shape tuple syntax issues  
        # Pattern: (b, h, nc, d = tensor.shape -> b, h, nc, d = tensor.shape
        code = re.sub(r'\(([^)]+)\s+=\s+([^.]+)\.shape', r'\1 = \2.shape', code)
        
        # Fix argument list issues in function definitions
        # Pattern: bias: bool = False) -> bias: bool = False)
        code = re.sub(r'([a-zA-Z_]\w*:\s*\w+\s*=\s*[^,)]+)\s+([a-zA-Z_]\w*:\s*\w+)', r'\1, \2', code)
        
        # Fix activation function calls
        code = re.sub(r'F\.silu\(([^,)]+)\s+([^)]+)\)', r'F.silu(\1, \2)', code)
        code = re.sub(r'F\.gelu\(([^,)]+)\s+([^)]+)\)', r'F.gelu(\1, \2)', code)
        code = re.sub(r'F\.relu\(([^,)]+)\s+([^)]+)\)', r'F.relu(\1, \2)', code)
        
        # Fix tensor operation calls
        code = re.sub(r'\.expand_dims\(([^,)]+)\s+([^)]+)\)', r'.expand_dims(\1, \2)', code)
        code = re.sub(r'\.reshape\(([^,)]+)\s+([^)]+)\)', r'.reshape(\1, \2)', code)
        code = re.sub(r'\.transpose\(([^,)]+)\s+([^)]+)\)', r'.transpose(\1, \2)', code)
        
        # Fix missing commas in other function calls
        # Handle F.softmax(tensor dim) -> F.softmax(tensor, dim)
        code = re.sub(r'F\.softmax\(([^,\s]+)\s+([^)]+)\)', r'F.softmax(\1, \2)', code)
        # Handle tensor.sum(axis keepdim) -> tensor.sum(axis, keepdim)
        code = re.sub(r'\.sum\(([^,\s]+)\s+([^)]+)\)', r'.sum(\1, \2)', code)
        code = re.sub(r'\.std\(([^,\s]+)\s+([^)]+)\)', r'.std(\1, \2)', code)
        code = re.sub(r'\.var\(([^,\s]+)\s+([^)]+)\)', r'.var(\1, \2)', code)
        code = re.sub(r'\.mean\(([^,\s]+)\s+([^)]+)\)', r'.mean(\1, \2)', code)
        code = re.sub(r'\.norm\(([^,\s]+)\s+([^)]+)\)', r'.norm(\1, \2)', code)
        code = re.sub(r'\.clamp\(([^,\s]+)\s+([^)]+)\)', r'.clamp(\1, \2)', code)
        
        # Handle constructor parameter issues 
        # Fix def __init__(self, arg: type param: type = value) -> def __init__(self, arg: type, param: type = value)
        code = re.sub(r'(\w+:\s*[a-zA-Z_]\w*)\s+([a-zA-Z_]\w*:\s*[a-zA-Z_]\w*)', r'\1, \2', code)
        
        # Fix bias= arguments issues like bias=False)
        code = re.sub(r'bias=([^,)]+)\s+([^)]+)\)', r'bias=\1, \2)', code)
        
        # Fix mx.array creation syntax
        code = re.sub(r'mx\.array\(([^,\s]+)\s+([^)]+)\)', r'mx.array(\1, \2)', code)
        
        # Fix einsum calls  
        code = re.sub(r'einsum\(\s*([^,]+),\s*([^,]+),\s*([^,]+)\s+([^)]+)\)', r'einsum(\1, \2, \3, \4)', code)
        
        # Fix constructor parameter syntax issues
        code = re.sub(r'def __init__\(([^)]+),\s*([^)]+)\s+([^)]+)\):', r'def __init__(\1, \2, \3):', code)
        code = re.sub(r'([a-zA-Z_]\w*:\s*[a-zA-Z_]\w*)\s+([a-zA-Z_]\w*:\s*[a-zA-Z_]\w*)', r'\1, \2', code)
        
        # Fix device parameter issues - much more targeted
        # Only remove explicit device= arguments, not all arguments
        code = re.sub(r',\s*device=[^,)]+', '', code)  # Remove device=... in middle
        code = re.sub(r'device=[^,)]+,\s*', '', code)  # Remove device=... at start
        code = re.sub(r'\(\s*device=[^,)]+\s*\)', '()', code)  # Remove (device=...) only
        
        # Fix .device attribute access (like q.device, k.device) - remove entirely
        code = re.sub(r',\s*[a-zA-Z_]\w*\.device', '', code)  # Remove ,tensor.device
        code = re.sub(r'[a-zA-Z_]\w*\.device,\s*', '', code)  # Remove tensor.device,
        code = re.sub(r'\(\s*[a-zA-Z_]\w*\.device\s*\)', '()', code)  # Remove (tensor.device)
        
        # Fix .to() calls that weren't caught
        code = re.sub(r'\.to\([^)]+\)', '', code)
        
        # Fix mx.Tensor type annotations
        code = re.sub(r'mx\.Tensor', 'mx.array', code)
        
        # Fix new_zeros calls
        code = re.sub(r'([a-zA-Z_]\w*)\.new_zeros\(', r'mx.zeros(', code)
        code = re.sub(r'([a-zA-Z_]\w*)\.new_ones\(', r'mx.ones(', code)
        
        # Fix clone() calls  
        code = re.sub(r'\.clone\(\)', '', code)
        
        # Fix item() calls
        code = re.sub(r'\.item\(\)', '.item()', code)
        
        # Fix dtype issues - mx.bool vs mx.bool_
        code = re.sub(r'dtype=mx\.bool([^_])', r'dtype=mx.bool_\1', code)
        
        # Fix no_grad context 
        code = re.sub(r'with mx\.no_grad\(\):', 'with mx.disable_grad():', code)
        
        # More targeted fixes for specific problematic patterns
        
        # Fix broken constructor definitions (common pattern issues)
        # Fix things like: def __init__(self, *, arg1, arg2: type = val, arg3, ...)
        # Replace with: def __init__(self, arg1, arg2: type = val, arg3, *, ...)
        def fix_constructor_args(match):
            func_name = match.group(1)
            args_str = match.group(2)
            
            # Don't modify if it's a simple case
            if '*, ' not in args_str or '=' not in args_str:
                return match.group(0)
            
            # Simple fix: just remove the *, for now
            args_str_fixed = args_str.replace('*, ', '')
            return f"def {func_name}({args_str_fixed})"
        
        # Fix broken constructor definitions  
        code = re.sub(r'def (__init__)\(([^)]+)\)', fix_constructor_args, code)
        
        # Fix any remaining import issues
        code = re.sub(r'from mx\.nn as F', 'import mlx.nn as F', code)
        code = re.sub(r'from mx\.nn\.functional as F', 'import mlx.nn as F', code)
        
        # Fix RMSNorm references that might not exist
        code = re.sub(r'([^n])RMSNorm', r'\1nn.RMSNorm', code)
        
        # Fix empty else blocks after register_parameter/register_buffer removal
        # Look for else blocks that only contain register_* comments
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)
            
            # Check if this is an else: line
            if line.strip().endswith('else:'):
                # Look ahead to see if the next non-empty line is a register_* comment
                j = i + 1
                next_content_line = None
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line:  # First non-empty line
                        next_content_line = next_line
                        break
                    j += 1
                
                # If the next content is a register_* comment, we need to add pass
                if (next_content_line and 
                    next_content_line.startswith('# register_') and
                    j < len(lines) - 1):
                    # Check if there's actual code after the comment at the same indent level
                    else_indent = len(line) - len(line.lstrip())
                    expected_block_indent = else_indent + 4
                    
                    # Look for the next line that could be part of the else block
                    k = j + 1
                    has_block_content = False
                    while k < len(lines):
                        check_line = lines[k]
                        if not check_line.strip():  # Skip empty lines
                            k += 1
                            continue
                        check_indent = len(check_line) - len(check_line.lstrip())
                        if check_indent <= else_indent:  # Same or less indent = end of block
                            break
                        if check_indent == expected_block_indent:  # Proper block content
                            has_block_content = True
                            break
                        k += 1
                    
                    # If no block content found, insert pass after the comment
                    if not has_block_content:
                        # Insert pass with proper indentation
                        pass_line = ' ' * expected_block_indent + 'pass'
                        # We'll insert this after we add the comment line
                        fixed_lines.append(lines[j])  # Add the comment
                        fixed_lines.append(pass_line)  # Add pass
                        i = j + 1  # Skip the comment line since we already added it
                        continue
            i += 1
        
        code = '\n'.join(fixed_lines)
        
        # Fix unmatched parentheses in function calls
        # Common pattern: func(arg1, arg2, arg3,) with trailing comma before )
        code = re.sub(r',\s*\)', ')', code)
        
        # Fix specific problematic patterns found in the code
        # Pattern: else, None) should be else None)
        code = re.sub(r'else,\s*([^,)]+)\)', r'else \1)', code)
        
        # Fix more specific comma issues
        # Pattern: ", None)" should be " None)"  
        code = re.sub(r',\s*None\)', ' None)', code)
        
        # Fix _rearrange calls with improper commas in pattern strings
        # Pattern: "(n, c)" should be "(n c)"
        code = re.sub(r'"([^"]*)\(([^,)]+),\s*([^)]+)\)([^"]*)"', r'"\1(\2 \3)\4"', code)
        
        # Fix standalone comma issues in function calls
        code = re.sub(r'(\w+)\s*,\s*([^,)]+)\)', r'\1 \2)', code)
        
        # Fix missing closing parentheses and general parentheses balancing
        lines = code.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Fix specific patterns that cause unmatched parentheses
            
            # Pattern 1: Lines that end with incomplete function calls
            if (stripped.endswith(',') and 
                '(' in stripped and 
                stripped.count('(') > stripped.count(')')):
                # Look ahead to see if next lines have continuation
                j = i + 1
                found_continuation = False
                while j < len(lines) and j < i + 3:  # Look at next 3 lines
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith('#'):
                        if (next_line.startswith((')', 'nn.', 'mx.', 'self.')) and 
                            not '(' in next_line):
                            found_continuation = True
                        break
                    j += 1
                
                if not found_continuation:
                    # This line seems to be missing a closing parenthesis
                    missing_parens = stripped.count('(') - stripped.count(')')
                    line = line.rstrip(',') + ')' * missing_parens
            
            # Pattern 2: Sequential calls that are missing closing parentheses
            if 'nn.Sequential(' in stripped:
                paren_count = stripped.count('(') - stripped.count(')')
                if paren_count > 0 and stripped.endswith(','):
                    # Look ahead for the closing pattern
                    j = i + 1
                    found_close = False
                    while j < len(lines) and j < i + 10:
                        next_line = lines[j].strip()
                        if next_line.startswith(')') or 'nn.' in next_line or 'self.' in next_line:
                            found_close = True
                            break
                        j += 1
                    
                    if not found_close:
                        line = line.rstrip(',') + ')' * paren_count
            
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        # Fix broken function call patterns
        # Pattern: func(arg1, arg2, arg3 arg4) - missing comma
        code = re.sub(r'(\w+)\s+(\w+)\s*\)', r'\1, \2)', code)
        
        # Fix invalid syntax patterns and more specific issues
        lines = code.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Check for common invalid syntax patterns
            # Pattern: standalone operators or malformed expressions
            if re.match(r'^[\+\-\*/=<>!&|]+$', stripped):
                # Skip invalid operator lines
                continue
            
            # Fix specific problematic patterns found in the failing architectures
            
            # Pattern: "rearrange(" without "_" prefix (should be "_rearrange(")
            if 'rearrange(' in line and '_rearrange(' not in line:
                line = line.replace('rearrange(', '_rearrange(')
            
            # Pattern: fix nn.Parameter issues that weren't caught
            if 'nn.Parameter(' in line:
                line = line.replace('nn.Parameter(', 'mx.array(')
            
            # Pattern: Fix .device references that weren't removed
            line = re.sub(r'\.device\b', '', line)
            
            # Pattern: Fix register_buffer that might still exist
            if 'register_buffer(' in line:
                line = '# ' + line  # Comment out the line
            
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        # Fix import statement issues - ensure imports don't chain incorrectly
        # Pattern: import mlx.core as mx.nn as nn (chained as clauses)
        code = re.sub(r'import\s+(\w+(?:\.\w+)*)\s+as\s+(\w+(?:\.\w+)*)\s+as\s+(\w+)', r'import \1 as \3', code)
        
        # Ensure required imports exist if needed symbols are used
        lines = code.split('\n')
        
        # Check what imports we need - be more thorough
        has_mx_import = any('import mlx.core as mx' in line for line in lines)
        has_nn_import = any('import mlx.nn as nn' in line for line in lines)
        has_F_import = any('import mlx.nn as F' in line for line in lines)
        
        # Also check for 'from mx.nn import' patterns
        has_mx_from_import = any('from mx.nn import' in line for line in lines)
        
        # Check for standalone 'mx' usage and specific patterns
        needs_mx = (('mx.' in code or ' mx ' in code or '@mx.compile' in code or 
                    'mx.zeros' in code or 'mx.ones' in code or 'mx.array' in code) 
                   and not has_mx_import and not has_mx_from_import)
        needs_nn = 'nn.' in code and not has_nn_import
        needs_F = 'F.' in code and not has_F_import
        
        # Find where to insert imports (after future imports and docstrings)
        insert_pos = 0
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = not in_docstring
                insert_pos = i + 1
            elif not in_docstring and (stripped.startswith('from __future__') or 
                                     stripped.startswith('#') or 
                                     not stripped):
                insert_pos = i + 1
            else:
                break
        
        # Insert missing imports
        new_imports = []
        if needs_mx:
            new_imports.append('import mlx.core as mx')
        if needs_nn:
            new_imports.append('import mlx.nn as nn')
        if needs_F:
            new_imports.append('import mlx.nn as F')
        
        if new_imports:
            lines = lines[:insert_pos] + new_imports + lines[insert_pos:]
        
        code = '\n'.join(lines)
        
        return code
    
    def convert_architecture(self, pytorch_code: str, arch_name: str) -> str:
        """Convert a single PyTorch architecture to MLX"""
        # Clean the code first
        code = self._clean_architecture_code(pytorch_code)
        
        # Apply conversion rules
        for rule in self.conversion_rules:
            if len(rule) == 3:  # Has flags
                pattern, replacement, flags = rule
                code = re.sub(pattern, replacement, code, flags=flags)
            else:  # No flags
                pattern, replacement = rule
                code = re.sub(pattern, replacement, code)
        
        # Fix syntax issues
        code = self._fix_syntax_issues(code)
        
        # Add MLX utilities at the top
        utilities = self._add_mlx_utilities()
        
        # Extract any from __future__ imports from the converted code first
        future_imports = []
        code_lines = code.split('\n')
        non_future_lines = []
        
        for line in code_lines:
            if line.strip().startswith('from __future__ import'):
                future_imports.append(line)
            else:
                non_future_lines.append(line)
        
        # Remove duplicates from future imports
        if future_imports:
            seen = set()
            unique_future = []
            for imp in future_imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_future.append(imp)
            future_imports = unique_future
        
        # Combine utilities + converted code with future imports at the very top
        header = f'''"""
MLX-converted architecture: {arch_name}
Auto-converted from PyTorch to MLX format
"""
'''
        
        if future_imports:
            return '\n'.join(future_imports) + '\n\n' + header + utilities + '\n'.join(non_future_lines)
        else:
            return header + utilities + '\n'.join(non_future_lines)
    
    def convert_all_architectures(self, json_path: str, output_dir: str) -> Dict[str, str]:
        """Convert all architectures from 106.json to MLX format"""
        with open(json_path, 'r') as f:
            architectures = json.load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        converted = {}
        
        for i, arch in enumerate(architectures):
            name = arch['name']
            pytorch_code = arch['program']
            
            print(f"Converting {i+1}/106: {name}")
            
            try:
                mlx_code = self.convert_architecture(pytorch_code, name)
                
                # Save to file
                filename = f"{name}_mlx.py"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(mlx_code)
                
                converted[name] = {
                    'mlx_code': mlx_code,
                    'filepath': filepath,
                    'original_result': arch['result'],
                    'parameters': arch.get('parameters', 'Unknown'),
                    'score': arch.get('score', 0.0),
                    'parent': arch.get('parent', None),
                    'index': arch.get('index', i)
                }
                
            except Exception as e:
                print(f"Error converting {name}: {e}")
                converted[name] = {'error': str(e)}
        
        return converted

def main():
    """Convert all PyTorch architectures to MLX"""
    converter = PyTorchToMLXConverter()
    
    # Convert all architectures
    results = converter.convert_all_architectures(
        json_path='106.json',
        output_dir='mlx_architectures'
    )
    
    # Save conversion summary
    summary = {
        'total_architectures': len(results),
        'successful_conversions': len([r for r in results.values() if 'error' not in r]),
        'failed_conversions': len([r for r in results.values() if 'error' in r]),
        'results': results
    }
    
    with open('mlx_conversion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Successful: {summary['successful_conversions']}/106")
    print(f"Failed: {summary['failed_conversions']}/106")
    print(f"MLX architectures saved to: mlx_architectures/")

if __name__ == "__main__":
    main()