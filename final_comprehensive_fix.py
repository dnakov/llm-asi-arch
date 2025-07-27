#!/usr/bin/env python3
"""
Final comprehensive fix for all remaining syntax errors in MLX architectures.
"""

import os
import re
import sys
from pathlib import Path

def fix_all_remaining_issues(content):
    """Apply all remaining fixes comprehensively"""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix indentation for specific patterns
        if re.match(r'^    [a-zA-Z_]\w*\s*=', line) and i > 0:
            prev_line = lines[i-1]
            # If previous line is an if/elif/else and current line starts with variable assignment
            if any(x in prev_line for x in ['if ', 'elif ', 'else:']):
                line = '        ' + line.strip()  # Proper indentation for block
        
        # Fix function parameters broken across lines
        if ('def ' in line and '(' in line and line.count('(') > line.count(')')):
            # Function definition with parameters spread across lines
            func_lines = [line]
            j = i + 1
            while j < len(lines) and ')' not in lines[j]:
                func_lines.append(lines[j])
                j += 1
            if j < len(lines):
                func_lines.append(lines[j])
            
            # Fix the function definition
            if len(func_lines) > 1:
                # Combine and reformat
                full_def = ' '.join(line.strip() for line in func_lines)
                # Fix missing commas and formatting
                full_def = re.sub(r':\s*int\s+([a-zA-Z_]\w*)', r': int,\n        \1', full_def)
                full_def = re.sub(r':\s*float\s+([a-zA-Z_]\w*)', r': float,\n        \1', full_def)
                full_def = re.sub(r':\s*str\s+([a-zA-Z_]\w*)', r': str,\n        \1', full_def)
                
                fixed_lines.append(full_def)
                i = j
                continue
        
        # Fix function call parameters
        if ('(' in line and line.count('(') > line.count(')') and 
            any(x in line for x in ['_ShortConvolution', 'nn.Linear', 'mx.zeros', 'mx.ones'])):
            # Function call with parameters spread across lines
            call_lines = [line]
            j = i + 1
            paren_count = line.count('(') - line.count(')')
            while j < len(lines) and paren_count > 0:
                call_lines.append(lines[j])
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
            
            # Fix the function call
            if len(call_lines) > 1:
                # Extract indentation from first line
                indent = len(line) - len(line.lstrip())
                base_indent = ' ' * indent
                
                # Reconstruct with proper formatting
                if '_ShortConvolution' in line:
                    # Special handling for _ShortConvolution
                    params = []
                    for call_line in call_lines:
                        call_line = call_line.strip()
                        if 'hidden_size=' in call_line or call_line.startswith('hidden_size'):
                            params.append('hidden_size=self.key_dim')
                        elif 'kernel_size=' in call_line or call_line.startswith('kernel_size'):
                            params.append('kernel_size=conv_size')
                        elif 'activation=' in call_line or call_line.startswith('activation'):
                            if 'silu' in call_line:
                                params.append('activation="silu"')
                            else:
                                params.append('activation=act')
                        elif 'bias=' in call_line or call_line.startswith('bias'):
                            params.append('bias=conv_bias')
                    
                    if params:
                        fixed_lines.append(f"{base_indent}_ShortConvolution(")
                        for param in params[:-1]:
                            fixed_lines.append(f"{base_indent}    {param},")
                        fixed_lines.append(f"{base_indent}    {params[-1]})")
                        i = j - 1
                        continue
                
                # Default handling for other function calls
                combined = ' '.join(call_line.strip() for call_line in call_lines)
                fixed_lines.append(combined)
                i = j - 1
                continue
        
        # Fix variable assignments
        if ('=' in line and not line.strip().startswith('#') and 
            i + 1 < len(lines) and not lines[i+1].strip().startswith('#')):
            next_line = lines[i + 1]
            if (re.match(r'^\s*[a-zA-Z_]\w*', next_line) and 
                '=' not in next_line and ':' not in next_line):
                # Continuation line that should be on same line
                line = line.rstrip() + ' ' + next_line.strip()
                i += 1  # Skip next line
        
        # Fix specific patterns
        line = re.sub(r'\(([0-9.-]+)\s+([0-9.-]+)\)', r'(\1, \2)', line)  # Tuple commas
        line = re.sub(r'mx\.zeros\(([^,\)]+)\s+([^,\)]+)\)', r'mx.zeros(\1, \2)', line)  # mx.zeros args
        line = re.sub(r'int\(([^,]+),\s*\*\s*([^)]+)\)', r'int(\1 * \2)', line)  # int(x, * y)
        line = re.sub(r'mx\.array\(([^,\)]+),\s*\*', r'mx.array(\1 *', line)  # mx.array args
        
        # Fix Sequential calls
        if 'nn.Sequential(' in line and ', nn.' in line:
            line = re.sub(r'nn\.Sequential\(,\s*nn\.', r'nn.Sequential(\n            nn.', line)
        
        # Fix assert statements
        if line.strip().startswith('assert') and line.count('"') == 1:
            if i + 1 < len(lines) and lines[i+1].strip().startswith('"'):
                line = line + ', ' + lines[i+1].strip()
                i += 1  # Skip next line
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_file(file_path):
    """Apply final comprehensive fix to a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = fix_all_remaining_issues(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    mlx_dir = Path("mlx_architectures")
    if not mlx_dir.exists():
        print(f"Directory {mlx_dir} does not exist!")
        return 1
    
    python_files = list(mlx_dir.glob("*.py"))
    if not python_files:
        print(f"No Python files found in {mlx_dir}")
        return 1
    
    print(f"Applying final comprehensive fixes to {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        print(f"Final fix {file_path.name}...", end=" ")
        if fix_file(file_path):
            print("✓ Fixed")
            fixed_count += 1
        else:
            print("✗ No changes")
    
    print(f"\nFinal pass: Fixed {fixed_count}/{len(python_files)} files")
    return 0

if __name__ == "__main__":
    sys.exit(main())