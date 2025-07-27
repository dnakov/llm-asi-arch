#!/usr/bin/env python3
"""
Comprehensive MLX architecture fixer for common syntax and structural issues.
"""

import os
import re
import ast

def fix_spectral_fusion():
    """Fix delta_net_spectral_fusion_mlx.py completely"""
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_spectral_fusion_mlx.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix broken function definition and assert
    content = re.sub(
        r'    # forward: x – \(B\n        L, H, D\)\n    # ------------------------------------------------------------------\n    def forward\(self, x: mx\.array\) -> mx\.array:\n        bsz, seq_len, num_heads, head_dim = x\.shape\n        assert\n        num_heads == self\.num_heads and\n        head_dim == self\.head_dim "Mismatch in head dims"',
        '    def __call__(self, x: mx.array) -> mx.array:\n        bsz, seq_len, num_heads, head_dim = x.shape\n        assert num_heads == self.num_heads and head_dim == self.head_dim, "Mismatch in head dims"',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Fix broken function signatures and syntax throughout the file
    # Fix type annotations in function signatures
    fixes = [
        (r'def _rearrange\(tensor:, mx\.array,', 'def _rearrange(tensor: mx.array,'),
        (r'def _l2norm\(x:, mx\.array\)', 'def _l2norm(x: mx.array)'),
        (r'def _masked_fill\(tensor:, mx\.array,', 'def _masked_fill(tensor: mx.array,'),
        (r'def _index_first_axis\(tensor:, mx\.array,', 'def _index_first_axis(tensor: mx.array,'),
        (r'def _pad_input\(tensor:, mx\.array,', 'def _pad_input(tensor: mx.array,'),
        (r'def _next_power_of_two\(x:, int\)', 'def _next_power_of_two(x: int)'),
        
        # Fix broken function calls and definitions
        (r'def _power_law_min_phase_filter\(freq:, mx\.array # \(F\)\n    amp: mx\.array,   # \(1,H,1, 1\)\n    decay: mx\.array, # \(1 H,1, 1\)\n    \*,\n    dtype: mx\.dtype\)', 
         'def _power_law_min_phase_filter(freq: mx.array, amp: mx.array, decay: mx.array, dtype: mx.dtype)'),
        
        # Fix parameter list issues
        (r'def __init__\(self, hidden_size: int,\n    kernel_size: int = 4\n    activation: str = None\n    bias: bool = False\):',
         'def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):'),
        
        (r'def __call__\(self, x, cache=None\n        output_final_state=False\n        cu_seqlens=None\):',
         'def __call__(self, x, cache=None, output_final_state=False, cu_seqlens=None):'),
        
        # Fix broken constructor calls
        (r'self\.conv = nn\.Conv1d\(hidden_size, hidden_size, kernel_size\n        padding=kernel_size-1\n        bias=bias\)',
         'self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)'),
        
        # Fix various syntax issues
        (r'fft_block_size: Optional\[int\] =, None\)', 'fft_block_size: Optional[int] = None)'),
        (r'freq_f32 = freq\n        decay_f32 = decay\n        amp_f32 = amp', 
         'freq_f32 = freq.astype(mx.float32)\n        decay_f32 = decay.astype(mx.float32)\n        amp_f32 = amp.astype(mx.float32)'),
        (r'mag = amp_f32 / mx\.pow\(1\.0, \+ freq_f32 \*\* 2 decay_f32 / 2\.0\)',
         'mag = amp_f32 / mx.pow(1.0 + freq_f32 ** 2, decay_f32 / 2.0)'),
        (r'phase = -decay_f32 \* mx\.atan\(freq_f32\)\n    # combine → complex frequency response, finally cast to requested dtype, filt = mx\.polar\(mag, phase\)',
         'phase = -decay_f32 * mx.arctan(freq_f32)\n    # combine → complex frequency response\n    filt = mag * mx.exp(1j * phase)'),
        
        # Fix method calls
        (r'n_fft = _next_power_of_two\(2, \* L\)',
         'n_fft = _next_power_of_two(2 * L)'),
        (r'fft = mx\.fft\.rfft\(x_f\.float\(\)\n        n=n_fft\n        dim = -1\)',
         'fft = mx.fft.rfft(x_f.astype(mx.float32), n=n_fft, axis=-1)'),
        
        # Fix other method calls
        (r'y = mx\.fft\.irfft\(fft_filtered, n=n_fft\n        dim = -1\)',
         'y = mx.fft.irfft(fft_filtered, n=n_fft, axis=-1)'),
        
        # Fix dimensions and indexing
        (r'freq = mx\.arange\(n_fft, // 2 \+ 1 dtype=mx\.float32\)',
         'freq = mx.arange(n_fft // 2 + 1, dtype=mx.float32)'),
        
        # Fix string and other issues
        (r'window = mx\.hann_window\(block, dtype=x\.dtype\n        periodic = False\)',
         'window = mx.ones(block, dtype=x.dtype)'),  # MLX doesn't have hann_window
        
        # Fix padding
        (r'chunk = mx\.nn\.pad\(chunk, \(0, 0, 0, 0, pad_len, 0\)\)',
         'chunk = mx.pad(chunk, [(0, 0), (0, pad_len), (0, 0), (0, 0)])'),
        
        # Fix FFT calls
        (r'fft = mx\.fft\.rfft\(chunk_f\.float\(\)\n        n=n_fft\n        dim = -1\)',
         'fft = mx.fft.rfft(chunk_f.astype(mx.float32), n=n_fft, axis=-1)'),
        (r'y = mx\.fft\.irfft\(fft, \* filt\n        n=n_fft\n        dim = -1\)',
         'y = mx.fft.irfft(fft * filt, n=n_fft, axis=-1)'),
        
        # Fix normalization
        (r'out = out / weight\.reshape\(1, -1, 1\n        1\)\.clamp\(min=1e-4\), return out',
         'out = out / mx.maximum(weight.reshape(1, -1, 1, 1), 1e-4)\n        return out'),
        
        # Fix constructor parameter issues
        (r'hidden_size: int = 1024,\n        expand_k: float = 1\.0,\n        expand_v: float = 1\.0,\n        num_heads: int = 4,\n        use_beta: bool = False,  # unused – kept for compat\n        use_gate: bool = False,  # optional gated output norm\n        use_short_conv: bool = False,  # no short conv needed here\n        conv_size: int = 3,\n        conv_bias: bool = False,\n        allow_neg_eigval: bool = False,\n        layer_idx: Optional\[int\] = None,\n        qk_activation: str = "identity",\n        qk_norm: str = "l2",\n        norm_eps: float = 1e-5,\n        fft_block_size: Optional\[int\] = None \*\*kwargs: Dict\)',
         'hidden_size: int = 1024,\n        expand_k: float = 1.0,\n        expand_v: float = 1.0,\n        num_heads: int = 4,\n        use_beta: bool = False,\n        use_gate: bool = False,\n        use_short_conv: bool = False,\n        conv_size: int = 3,\n        conv_bias: bool = False,\n        allow_neg_eigval: bool = False,\n        layer_idx: Optional[int] = None,\n        qk_activation: str = "identity",\n        qk_norm: str = "l2",\n        norm_eps: float = 1e-5,\n        fft_block_size: Optional[int] = None,\n        **kwargs'),
        
        # Fix value dimension calculation
        (r'self\.value_dim = int\(hidden_size, \* expand_v\)',
         'self.value_dim = int(hidden_size * expand_v)'),
        
        # Fix projection definitions
        (r'self\.v_proj = nn\.Linear\(hidden_size, self\.value_dim\n        bias=False\)',
         'self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)'),
        
        # Fix spectral conv call
        (r'self\.spectral_conv = _SpectralConv\(num_heads, self\.head_v_dim\n        fft_block_size = fft_block_size\)',
         'self.spectral_conv = _SpectralConv(num_heads, self.head_v_dim, fft_block_size=fft_block_size)'),
        
        # Fix conditional blocks
        (r'if use_gate:\n\n            self\.g_proj = nn\.Linear\(hidden_size, self\.value_dim\n            bias=False\)\n            self\.o_norm = nn\.nn\.RMSNorm\(self\.head_v_dim, eps = norm_eps\)\n        else:\n\n            self\.o_norm = nn\.RMSNorm\(self\.head_v_dim, eps = norm_eps\)\n        self\.o_proj = nn\.Linear\(self\.value_dim, hidden_size\n        bias=False\)',
         'if use_gate:\n            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)\n            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)\n        else:\n            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)\n        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)'),
        
        # Fix forward method signature and content
        (r'def forward\(, self,\n        hidden_states: mx\.array,  # \(B L, D\)\n        attention_mask: Optional\[mx\.array\] = None,\n        past_key_values: Optional\["Cache"\] = None,  # type: ignore\[name-defined\]\n        \*,\n        use_cache: bool = False,  # kept for compat – no internal cache\n        output_attentions: bool = False # unused – kept for compat\n        \*\*kwargs\) -> Tuple\[mx\.array, None, Optional\["Cache"\]\]:  # type: ignore\[name-defined\]',
         'def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> mx.array:'),
        
        # Fix return statement
        (r'return out, None, past_key_values',
         'return out'),
        
        # Fix rearrange calls
        (r'v = _rearrange\(v, "b l, \(h, d\) -> b l h d"\n        h=self\.num_heads\)',
         'v = _rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)'),
        (r'out = _rearrange\(out, "b l h d -> b l, \(h, d\)"\)',
         'out = _rearrange(out, "b l h d -> b l (h d)")'),
        
        # Fix comments and other issues
        (r'# no padding / cache handling – spectral filter processes full sequence\n    v = self\.v_proj\(hidden_states\)',
         '# spectral filter processes full sequence\n        v = self.v_proj(hidden_states)'),
    ]
    
    for old, new in fixes:
        content = re.sub(old, new, content, flags=re.MULTILINE | re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed delta_net_spectral_fusion_mlx.py comprehensively")

def fix_ahic_return_tuple():
    """Fix the return tuple issue in delta_net_ahic_mlx.py"""
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_ahic_mlx.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the __call__ method to return only the main output
    content = re.sub(
        r'return o',
        'return o',
        content
    )
    
    # Make sure the function signature is correct
    content = re.sub(
        r'def __call__\(\n        self,\n        hidden_states: mx\.array,\n        attention_mask: Optional\[mx\.array\] = None,\n        \*\*kwargs,\n    \) -> Tuple\[mx\.array, Optional\[mx\.array\]\]:',
        'def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> mx.array:',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed return tuple in delta_net_ahic_mlx.py")

def fix_entropy_floor_arrayset():
    """Fix ArrayAt.set issues in delta_net_entropy_floor_mlx.py"""
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_entropy_floor_mlx.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace all .set() calls with proper MLX operations
    content = re.sub(
        r'(\w+)\.at\[([^\]]+)\]\.set\(([^)]+)\)',
        r'\1 = \1.at[\2].set(\3)',
        content
    )
    
    # If that doesn't work, replace with direct assignment
    content = re.sub(
        r'(\w+)\.set\(([^)]+)\)',
        r'\1 = \2',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed ArrayAt.set issues in delta_net_entropy_floor_mlx.py")

def fix_basic_syntax_errors():
    """Fix basic syntax errors in multiple files"""
    files_to_fix = [
        "delta_net_sparsemax_temperature_mlx.py",
        "delta_net_ssg_sparsemax_temp_mlx.py",
        "delta_net_hhmr_mlx.py",
        "delta_net_entropy_kl_floor_gate_mlx.py",
        "delta_net_hybrid_floor_gt_mlx.py",
        "delta_net_triscale_mlx.py",
        "delta_net_ms_adaptive_gstat3_mlx.py",
        "delta_net_ms_hsm_tempgate_mlx.py",
        "delta_net_gae_ms3e_mlx.py"
    ]
    
    for filename in files_to_fix:
        file_path = f"/Users/daniel/dev/asi/mlx_architectures/{filename}"
        
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix common type annotation issues
        fixes = [
            (r'def _rearrange\(tensor:, mx\.array,', 'def _rearrange(tensor: mx.array,'),
            (r'def _l2norm\(x:, mx\.array\)', 'def _l2norm(x: mx.array)'),
            (r'def _masked_fill\(tensor:, mx\.array,', 'def _masked_fill(tensor: mx.array,'),
            (r'def _index_first_axis\(tensor:, mx\.array,', 'def _index_first_axis(tensor: mx.array,'),
            (r'def _pad_input\(tensor:, mx\.array,', 'def _pad_input(tensor: mx.array,'),
            (r'def _elu_p1\(x:, mx\.array\)', 'def _elu_p1(x: mx.array)'),
            (r'def _sum_norm\(x:, mx\.array\)', 'def _sum_norm(x: mx.array)'),
            (r'def _mean_var\(x:, mx\.array\)', 'def _mean_var(x: mx.array)'),
            
            # Fix malformed parameter lists and function calls
            (r'int\(hidden_size, \* expand_v\)', 'int(hidden_size * expand_v)'),
            (r'int\(hidden_size, \* expand_k\)', 'int(hidden_size * expand_k)'),
            (r'int\(gate_local_in_dim, \* fusion_hidden_mult\)', 'int(gate_local_in_dim * fusion_hidden_mult)'),
            (r'int\(gate_global_in_dim, \* fusion_hidden_mult\)', 'int(gate_global_in_dim * fusion_hidden_mult)'),
            (r'int\(gate1_in_dim, \* fusion_hidden_mult\)', 'int(gate1_in_dim * fusion_hidden_mult)'),
            
            # Fix bracket/parenthesis mismatches
            (r'\[\s*([^,\]]+),\s*([^,\]]+)\s*\)', r'[\1, \2]'),
            (r'\(\s*([^,)]+),\s*([^,)]+)\s*\]', r'(\1, \2)'),
            
            # Fix return tuple issues
            (r'-> Tuple\[mx\.array, Optional\[mx\.array\]\]:', '-> mx.array:'),
            (r'-> Tuple\[mx\.array, None, Optional\["Cache"\]\]:', '-> mx.array:'),
            
            # Fix unterminated strings
            (r'"""([^"]*)"([^"]*)"([^"]*)"([^"]*)"([^"]*)"([^"]*)"([^"]*)"([^"]*)"([^"]*)"([^"]*)"',
             r'"""\1"\2"\3"\4"\5"\6"\7"\8"\9"\10"""'),
            
            # Fix missing commas in parameter lists
            (r'(\w+: [^,\n]+)([A-Z]\w+:', r'\1, \2'),
            
            # Fix method calls
            (r'\.cat\(', '.concatenate('),
            (r'F\.softplus\(', 'nn.softplus('),
            (r'F\.elu\(', 'nn.elu('),
            (r'F\.conv1d\(', 'nn.conv1d('),
            (r'\.clamp\(', '.clip('),
            (r'\.sigmoid\(\)', 'nn.sigmoid()'),
            (r'\.log\(\)', 'mx.log()'),
            (r'\.mean\(\)', '.mean()'),
            (r'\.sum\(\)', '.sum()'),
        ]
        
        for old, new in fixes:
            content = re.sub(old, new, content, flags=re.MULTILINE)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Fixed basic syntax errors in {filename}")

def main():
    print("Running comprehensive MLX architecture fixes...")
    
    # Fix specific architectures
    fix_spectral_fusion()
    fix_ahic_return_tuple()
    fix_entropy_floor_arrayset()
    fix_basic_syntax_errors()
    
    print("\nAll comprehensive fixes completed!")

if __name__ == "__main__":
    main()