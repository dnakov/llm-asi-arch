#!/usr/bin/env python3
"""
Final Syntax Fixer for MLX Architectures
Fixes the specific syntax issues without introducing new problems.
"""

import os
import re
import glob

def fix_double_commas(content: str) -> str:
    """Remove double commas introduced by previous fixes."""
    # Fix: axis=-1,, -> axis=-1,
    content = re.sub(r'(axis=-1),\s*,', r'\1,', content)
    
    # Fix: cache=conv_state_q,, -> cache=conv_state_q,
    content = re.sub(r'(cache=[^,\n]+),\s*,', r'\1,', content)
    
    # Fix: kernel_size,,  -> kernel_size,
    content = re.sub(r'(\w+=?[^,\n]*),\s*,', r'\1,', content)
    
    # Fix general double commas
    content = re.sub(r',,+', ',', content)
    
    return content

def fix_missing_parameter_commas(content: str) -> str:
    """Fix missing commas in function parameter lists."""
    
    # Fix _ShortConvolution __init__ parameters
    # Pattern: def __init__(self, hidden_size: int,\n    kernel_size: int = 4\n    activation: str = None
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for function definition lines that need comma fixes
        if 'def __init__(self, hidden_size: int,' in line and i + 2 < len(lines):
            # Check if next lines are parameters without commas
            next_line = lines[i + 1].strip()
            next_next_line = lines[i + 2].strip()
            
            if ('kernel_size: int =' in next_line and 
                'activation: str =' in next_next_line and
                not next_line.endswith(',')):
                
                # Fix the parameters
                fixed_lines.append(line)
                fixed_lines.append(lines[i + 1].rstrip() + ',')
                fixed_lines.append(lines[i + 2].rstrip() + ',')
                i += 3
                continue
        
        # Similar pattern for __call__ method
        if 'def __call__(self, x, cache=None' in line and i + 2 < len(lines):
            next_line = lines[i + 1].strip()
            next_next_line = lines[i + 2].strip()
            
            if ('output_final_state=' in next_line and 
                'cu_seqlens=' in next_next_line and
                not line.endswith(',') and 'cache=None' in line):
                
                # Fix the function definition
                fixed_line = line.replace('cache=None', 'cache=None,')
                fixed_lines.append(fixed_line)
                fixed_lines.append(lines[i + 1].rstrip() + ',')
                i += 2
                continue
                
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_broken_expressions(content: str) -> str:
    """Fix broken expressions that span multiple lines incorrectly."""
    
    # Fix: return(x, / x.sum(dim=-1, -> return x / x.sum(dim=-1,
    content = re.sub(r'return\s*\(\s*x\s*,\s*/\s*x\.sum\(', 'return x / x.sum(', content)
    
    # Fix: q_i\n        k_i = -> q_i, k_i = 
    content = re.sub(r'(\w+)\s*\n\s*(\w+)\s*=\s*([^=\n]+)$', r'\1, \2 = \3', content, flags=re.MULTILINE)
    
    # Fix: S = mx.zeros(b, h, d_k v.shape[-1]) -> S = mx.zeros(b, h, d_k, v.shape[-1])
    content = re.sub(r'mx\.zeros\(([^,)]+),\s*([^,)]+),\s*(\w+)\s+(\w+\.[^)]+)\)', r'mx.zeros(\1, \2, \3, \4)', content)
    
    # Fix: for idx in range(L_pad, // chunk_size): -> for idx in range(L_pad // chunk_size):
    content = re.sub(r'range\(([^,)]+),\s*//\s*([^)]+)\)', r'range(\1 // \2)', content)
    
    # Fix assignment issues - o[:, :]\n        idx = -> o[:, :, idx] =
    content = re.sub(r'(o\[:,\s*:\])\s*\n\s*(idx)\s*=', r'\1, \2] =', content)
    
    # Fix slicing issues - o = o[:]\n        :, :L -> o = o[:, :L]
    content = re.sub(r'(o\s*=\s*o\[:)\]\s*\n\s*(:,\s*:L)', r'\1, \2]', content)
    
    return content

def fix_function_call_continuations(content: str) -> str:
    """Fix function calls that are split across lines incorrectly."""
    
    # Fix Conv1d calls missing commas
    # nn.Conv1d(hidden_size, hidden_size, kernel_size\n        padding=
    content = re.sub(
        r'(nn\.Conv1d\([^)]+?)\s*\n\s+(padding=)',
        r'\1,\n        \2',
        content
    )
    
    # Fix Linear calls missing commas  
    content = re.sub(
        r'(nn\.Linear\([^)]+?)\s*\n\s+(bias=)',
        r'\1,\n        \2',
        content
    )
    
    # Fix _ShortConvolution calls missing commas
    content = re.sub(
        r'(_ShortConvolution\([^)]+?)\s*\n\s+(\w+=)',
        r'\1,\n        \2',
        content
    )
    
    # Fix Sequential calls starting with comma
    content = re.sub(r'nn\.Sequential\(\s*,\s*', 'nn.Sequential(', content)
    
    return content

def fix_specific_patterns(content: str) -> str:
    """Fix specific patterns identified from error messages."""
    
    # Fix: def forward(, self, -> def forward(self,
    content = re.sub(r'def (\w+)\(\s*,\s*self\s*,', r'def \1(self,', content)
    
    # Fix: def __init__(, self, -> def __init__(self,
    content = re.sub(r'def __init__\(\s*,\s*self\s*,', r'def __init__(self,', content)
    
    # Fix: assert(condition, == value) -> assert condition == value
    content = re.sub(r'assert\s*\(\s*([^,)]+)\s*,\s*==\s*(\w+)', r'assert \1 == \2', content)
    
    # Fix: num_heads:, int, -> num_heads: int,
    content = re.sub(r'(\w+):\s*,\s*(\w+)\s*,', r'\1: \2,', content)
    
    # Fix: entropy_anneal_steps: int = 20000) -> None: (missing comma)
    content = re.sub(r'(entropy_anneal_steps:\s*int\s*=\s*\d+)\s*\)', r'\1,', content)
    
    # Fix class member function definitions
    content = re.sub(r'def (\w+)\(\s*,\s*self\s*,', r'def \1(self,', content)
    
    return content

def fix_indent_and_syntax_errors(content: str) -> str:
    """Fix indentation and basic syntax errors."""
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix unexpected indent issues
        if i > 0 and line.strip() and not line[0].isspace():
            prev_line = lines[i-1].strip()
            if prev_line.endswith(':') and not line.strip().startswith(('#', '"""', "'''")):
                # Add proper indentation
                line = '    ' + line
        
        # Fix unterminated string literals by adding missing quotes
        if line.count('"') % 2 == 1 and not line.rstrip().endswith('"'):
            line = line + '"'
        if line.count("'") % 2 == 1 and not line.rstrip().endswith("'"):
            line = line + "'"
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def process_single_file(filepath: str) -> bool:
    """Process a single MLX architecture file."""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply fixes in careful sequence
        content = original_content
        content = fix_double_commas(content)
        content = fix_missing_parameter_commas(content)
        content = fix_broken_expressions(content)
        content = fix_function_call_continuations(content)
        content = fix_specific_patterns(content)
        content = fix_indent_and_syntax_errors(content)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Apply final fixes to all MLX architecture files."""
    
    mlx_dir = "/Users/daniel/dev/asi/mlx_architectures"
    if not os.path.exists(mlx_dir):
        print(f"Directory not found: {mlx_dir}")
        return
    
    # Find all MLX files
    pattern = os.path.join(mlx_dir, "delta_net_*_mlx.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No MLX files found in {mlx_dir}")
        return
    
    print(f"Applying final fixes to {len(files)} MLX architecture files...")
    
    fixed_count = 0
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        if process_single_file(filepath):
            print(f"✓ Fixed: {filename}")
            fixed_count += 1
        else:
            print(f"- No changes: {filename}")
    
    print(f"\nFixed {fixed_count}/{len(files)} files")
    
    # Run test to see results
    print("Running test to verify improvements...")
    os.system('cd /Users/daniel/dev/asi && python test_all_architectures.py')

if __name__ == "__main__":
    main()