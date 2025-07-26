#!/usr/bin/env python3
"""
Aggressive Syntax Error Fixer for MLX Converted Architectures
============================================================

This script specifically targets the exact syntax errors we're seeing in the converted files.
"""

import os
import re
from pathlib import Path

def fix_syntax_errors(code: str) -> str:
    """Apply aggressive regex fixes for the specific syntax errors we're seeing"""
    
    # Fix F.elu calls with missing commas - very specific patterns
    code = re.sub(r'F\.elu\(x\s+1\.0,\s*False\)', r'F.elu(x, 1.0, False)', code)
    code = re.sub(r'F\.elu\(([a-zA-Z_]\w*)\s+([0-9.]+),\s*([^)]+)\)', r'F.elu(\1, \2, \3)', code)
    code = re.sub(r'F\.elu\(([a-zA-Z_]\w*)\s+([0-9.]+)\s+([^)]+)\)', r'F.elu(\1, \2, \3)', code)
    
    # Fix function parameter definitions with missing commas and malformed commas
    # Pattern: noise_std: float =,, 2e-2 -> noise_std: float = 2e-2
    code = re.sub(r'(\w+:\s*\w+)\s*=\s*,\s*,\s*([^,)]+)', r'\1 = \2', code)
    # Pattern: param: type = val, next_param -> param: type = val, next_param
    code = re.sub(r'(\w+:\s*\w+\s*=\s*[^,)]+)\s+(\w+:\s*\w+)', r'\1, \2', code)
    # Pattern: param: type next_param -> param: type, next_param  
    code = re.sub(r'(\w+:\s*\w+)\s+(\w+:\s*\w+)', r'\1, \2', code)
    
    # Fix function calls with missing commas between arguments
    # Pattern: mx.zeros(num_heads, head_dim self.kernel_size) -> mx.zeros(num_heads, head_dim, self.kernel_size)
    code = re.sub(r'mx\.zeros\(([^,)]+),\s*([^,)]+)\s+([^)]+)\)', r'mx.zeros(\1, \2, \3)', code)
    code = re.sub(r'mx\.ones\(([^,)]+),\s*([^,)]+)\s+([^)]+)\)', r'mx.ones(\1, \2, \3)', code)
    code = re.sub(r'mx\.empty\(([^,)]+),\s*([^,)]+)\s+([^)]+)\)', r'mx.empty(\1, \2, \3)', code)
    code = re.sub(r'mx\.array\(([^,)]+),\s*([^,)]+)\s+([^)]+)\)', r'mx.array(\1, \2, \3)', code)
    code = re.sub(r'mx\.full\(([^,)]+),\s*([^,)]+)\s+([^)]+)\)', r'mx.full(\1, \2, \3)', code)
    
    # Fix nn.Linear calls with missing commas
    code = re.sub(r'nn\.Linear\(([^,)]+),\s*([^,)]+)\s+([^)]+)\)', r'nn.Linear(\1, \2, \3)', code)
    
    # Fix variable assignments with missing commas
    # Pattern: q, k, v k_beta = ... -> q, k, v, k_beta = ...
    code = re.sub(r'([a-zA-Z_]\w*),\s*([a-zA-Z_]\w*),\s*([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*=', r'\1, \2, \3, \4 =', code)
    # Pattern: var1 var2 = -> var1, var2 =
    code = re.sub(r'([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*=\s*', r'\1, \2 = ', code)
    
    # Fix return statements with missing commas
    # Pattern: return o S -> return o, S  
    code = re.sub(r'return\s+([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*$', r'return \1, \2', code, flags=re.MULTILINE)
    
    # Fix function calls in assignments with missing commas
    # Pattern: q conv_state_q = func(...) -> q, conv_state_q = func(...)
    code = re.sub(r'([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*\.[a-zA-Z_]\w*\()', r'\1, \2 = \3', code)
    
    # Fix _rearrange calls with missing commas
    code = re.sub(r'_rearrange\(([^,)]+)\s+"([^"]+)"\s+([^)]+)\)', r'_rearrange(\1, "\2", \3)', code)
    
    # Fix shape unpacking with missing assignment operator
    # Pattern: b, h, L d_k = q.shape -> b, h, L, d_k = q.shape
    code = re.sub(r'([a-zA-Z_]\w*),\s*([a-zA-Z_]\w*),\s*([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*=\s*([^.]+\.shape)', r'\1, \2, \3, \4 = \5', code)
    
    # Fix method calls with missing commas
    code = re.sub(r'\.sum\(([^,)]+)\s+([^)]+)\)', r'.sum(\1, \2)', code)
    code = re.sub(r'\.mean\(([^,)]+)\s+([^)]+)\)', r'.mean(\1, \2)', code)
    code = re.sub(r'\.clamp\(([^,)]+)\s+([^)]+)\)', r'.clamp(\1, \2)', code)
    code = re.sub(r'\.reshape\(([^,)]+)\s+([^)]+)\)', r'.reshape(\1, \2)', code)
    code = re.sub(r'\.expand\(([^,)]+)\s+([^)]+)\)', r'.expand(\1, \2)', code)
    
    # Fix multiple variable declarations in one line  
    # Pattern: var1 = expr, var2 = expr -> var1 = expr; var2 = expr
    code = re.sub(r'([a-zA-Z_]\w*\s*=\s*[^,]+),\s*([a-zA-Z_]\w*\s*=)', r'\1\n        \2', code)
    
    # Fix unmatched parentheses in complex expressions
    # Look for lines ending with commas that should be followed by closing parens
    lines = code.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix specific pattern: multiple assignment on one line with missing commas
        # Pattern: a = b, c = d, e = f -> separate into multiple lines
        if '=' in line and line.count('=') > 1 and ',' in line:
            # Split by comma and fix each assignment
            parts = line.split(',')
            base_indent = len(line) - len(line.lstrip())
            indent = ' ' * base_indent
            
            fixed_parts = []
            for part in parts:
                part = part.strip()
                if '=' in part:
                    fixed_parts.append(indent + part)
                else:
                    # This part doesn't have =, probably belongs to previous assignment
                    if fixed_parts:
                        fixed_parts[-1] += ', ' + part
                    else:
                        fixed_parts.append(indent + part)
            
            # Add all but last part as separate lines
            for part in fixed_parts[:-1]:
                fixed_lines.append(part)
            # Add last part 
            if fixed_parts:
                fixed_lines.append(fixed_parts[-1])
        else:
            fixed_lines.append(line)
    
    code = '\n'.join(fixed_lines)
    
    # Fix specific broken patterns that appear in the files
    
    # Pattern: function_call(arg1 arg2, arg3) -> function_call(arg1, arg2, arg3)
    code = re.sub(r'([a-zA-Z_]\w*)\(([^,)]+)\s+([^,)]+),', r'\1(\2, \3,', code)
    
    # Pattern: if condition: statement, next_statement -> if condition: statement; next_statement  
    code = re.sub(r'(\s+if\s+[^:]+:\s*[^,\n]+),\s*([^,\n]+)', r'\1\n        \2', code)
    
    # Fix trailing commas before closing parens
    code = re.sub(r',\s*\)', ')', code)
    
    # Fix double commas
    code = re.sub(r',,+', ',', code)
    
    # Fix space before commas in function calls
    code = re.sub(r'\s+,', ',', code)
    
    return code

def fix_all_architectures():
    """Fix syntax errors in all MLX architecture files"""
    mlx_dir = Path('mlx_architectures')
    if not mlx_dir.exists():
        print("❌ mlx_architectures directory not found")
        return
    
    fixed_count = 0
    total_count = 0
    
    for arch_file in mlx_dir.glob('*_mlx.py'):
        total_count += 1
        print(f"🔧 Fixing {arch_file.name}")
        
        try:
            # Read original file
            with open(arch_file, 'r') as f:
                original_code = f.read()
            
            # Apply fixes
            fixed_code = fix_syntax_errors(original_code)
            
            # Write back if changed
            if fixed_code != original_code:
                with open(arch_file, 'w') as f:
                    f.write(fixed_code)
                fixed_count += 1
                print(f"  ✅ Fixed {arch_file.name}")
            else:
                print(f"  ℹ️  No changes needed for {arch_file.name}")
                
        except Exception as e:
            print(f"  ❌ Error fixing {arch_file.name}: {e}")
    
    print(f"\n📊 Fixed {fixed_count}/{total_count} architecture files")

if __name__ == "__main__":
    fix_all_architectures()