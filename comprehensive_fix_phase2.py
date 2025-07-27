#!/usr/bin/env python3
"""
Phase 2 comprehensive fix for remaining syntax errors in MLX architectures.
"""

import os
import re
import sys
from pathlib import Path

def fix_indentation_errors(content):
    """Fix incorrect indentation patterns"""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix cases where variables are incorrectly indented (like line 197 in abrgf)
        if re.match(r'^\s{8,}[a-zA-Z_]\w*\s*=', line) and i > 0:
            prev_line = lines[i-1]
            if not prev_line.strip().endswith(':') and not 'def ' in prev_line:
                # Remove extra indentation
                line = re.sub(r'^(\s{8,})', '    ', line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_missing_commas_in_params(content):
    """Fix missing commas in function parameters and tuples"""
    # Fix tuples like (0.5 1.5) -> (0.5, 1.5)
    content = re.sub(r'\(([0-9.-]+)\s+([0-9.-]+)\)', r'(\1, \2)', content)
    content = re.sub(r'\(([0-9.-]+),\s*([0-9.-]+)\s+([0-9.-]+)\)', r'(\1, \2, \3)', content)
    content = re.sub(r'\(([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\)', r'(\1, \2, \3, \4)', content)
    
    # Fix function calls missing commas
    content = re.sub(r'mx\.zeros\(([^,\)]+)\s+([^,\)]+)\)', r'mx.zeros(\1, \2)', content)
    content = re.sub(r'mx\.ones\(([^,\)]+)\s+([^,\)]+)', r'mx.ones(\1, \2', content)
    content = re.sub(r'mx\.full\(\(([^)]+)\)\s+([^,)]+)\)', r'mx.full((\1), \2)', content)
    
    return content

def fix_broken_multiline_calls(content):
    """Fix broken multiline function calls and expressions"""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix patterns like "for idx in range(L_pad, // chunk_size):"
        if 'range(' in line and ', //' in line:
            line = re.sub(r'range\(([^,]+),\s*//\s*([^)]+)\)', r'range(\1 // \2)', line)
        
        # Fix patterns like "o[:, :\n        idx]"
        if re.search(r':\s*$', line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^[a-zA-Z_]\w*\]', next_line):
                # Combine the lines
                line = line.rstrip() + next_line
                i += 1  # Skip next line since we combined it
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_malformed_assignments(content):
    """Fix malformed variable assignments"""
    # Fix patterns like "w = attn @ k_beta\n        S = mx.zeros"
    content = re.sub(r'(w\s*=\s*[^=\n]+)\n\s+(S\s*=)', r'\1\n    \2', content)
    
    # Fix Sequential calls with improper formatting
    content = re.sub(r'nn\.Sequential\(,\s*nn\.', r'nn.Sequential(\n            nn.', content)
    
    # Fix mx.array calls
    content = re.sub(r'mx\.array\(([^,\)]+),\s*\*', r'mx.array(\1 *', content)
    
    return content

def fix_string_and_bracket_errors(content):
    """Fix string literal and bracket matching errors"""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix broken assertion messages
        if re.search(r'assert.*\n.*"[^"]*"$', line):
            line = re.sub(r'assert ([^"]+)\n\s*"([^"]*)"', r'assert \1, "\2"', line)
        elif line.strip().startswith('"') and line.count('"') == 1:
            # This is likely a continuation of an assert - fix it
            prev_line_idx = len(fixed_lines) - 1
            if prev_line_idx >= 0 and 'assert' in fixed_lines[prev_line_idx]:
                fixed_lines[prev_line_idx] += ', ' + line.strip()
                continue
        
        # Fix bias_tensor indentation issues
        if 'bias_tensor =' in line and line.startswith(' '):
            line = '        ' + line.strip()
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_specific_patterns(content):
    """Fix specific patterns found in the files"""
    # Fix _ShortConvolution calls with missing commas
    content = re.sub(
        r'_ShortConvolution\(([^,\n]+)\s*\n\s*activation=([^,\n]+)\s*\n\s*bias\s*=([^)]+)\)',
        r'_ShortConvolution(\1,\n                activation=\2,\n                bias=\3)',
        content
    )
    
    # Fix Linear calls with missing commas
    content = re.sub(
        r'nn\.Linear\(([^,\n]+)\s*\n\s*bias\s*=([^)]+)\)',
        r'nn.Linear(\1,\n        bias=\2)',
        content
    )
    
    # Fix "int(x, * y)" patterns
    content = re.sub(r'int\(([^,]+),\s*\*\s*([^)]+)\)', r'int(\1 * \2)', content)
    
    # Fix unmatched brackets in slice operations
    content = re.sub(r'o\[\s*:\s*\n\s*:([^]]+)\]', r'o[:, :\1]', content)
    
    # Fix method chaining issues like ".sum(-2), attn_inv = "
    content = re.sub(r'\.sum\(([^)]+)\),\s*([a-zA-Z_]\w*\s*=)', r'.sum(\1)\n    \2', content)
    
    return content

def fix_file(file_path):
    """Apply all fixes to a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all fixes in sequence
        content = fix_indentation_errors(content)
        content = fix_missing_commas_in_params(content)
        content = fix_broken_multiline_calls(content)
        content = fix_malformed_assignments(content)
        content = fix_string_and_bracket_errors(content)
        content = fix_specific_patterns(content)
        
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
    
    print(f"Applying phase 2 fixes to {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        print(f"Fixing {file_path.name}...", end=" ")
        if fix_file(file_path):
            print("✓ Fixed")
            fixed_count += 1
        else:
            print("✗ No changes")
    
    print(f"\nPhase 2: Fixed {fixed_count}/{len(python_files)} files")
    return 0

if __name__ == "__main__":
    sys.exit(main())