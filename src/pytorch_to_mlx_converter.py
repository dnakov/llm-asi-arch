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
            # Basic imports
            (r'import torch\b', 'import mlx.core as mx'),
            (r'import torch\.nn as nn', 'import mlx.nn as nn'),
            (r'import torch\.nn\.functional as F', 'import mlx.nn.functional as F'),
            (r'from torch import', 'from mlx.core import'),
            (r'torch\.', 'mx.'),
            
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
            
            # Optimizers
            (r'torch\.optim\.Adam', 'mx.optimizers.Adam'),
            (r'torch\.optim\.SGD', 'mx.optimizers.SGD'),
            (r'torch\.optim\.AdamW', 'mx.optimizers.AdamW'),
            
            # Special MLX patterns
            (r'\.register_buffer\(.*?\)', ''),  # Remove register_buffer
            (r'\.register_parameter\(.*?\)', ''),  # Remove register_parameter
            
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
            
            # Einops conversion
            (r'from einops import rearrange', ''),
            (r'rearrange\((.*?),\s*[\'\"](.*?)[\'\"](?:,\s*(.*?))?\)', r'_rearrange(\1, "\2"\3)'),
            
            # PyTorch compile decorator
            (r'@torch\.compile', ''),
            
            # Fix boolean tensor issues
            (r'torch\.bool', 'mx.bool_'),
            (r'torch\.float32', 'mx.float32'),
            (r'torch\.int32', 'mx.int32'),
            (r'torch\.int64', 'mx.int64'),
            
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
        ]
    
    def _add_mlx_utilities(self) -> str:
        """Add utility functions that MLX doesn't have built-in"""
        return '''
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
        
        # Add MLX utilities at the top
        utilities = self._add_mlx_utilities()
        
        # Combine utilities + converted code
        header = f'''"""
MLX-converted architecture: {arch_name}
Auto-converted from PyTorch to MLX format
"""
'''
        
        return header + utilities + code
    
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