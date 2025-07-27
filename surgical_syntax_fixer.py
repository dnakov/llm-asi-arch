#!/usr/bin/env python3
"""
Surgical syntax fixer for MLX architectures.
Fixes specific, well-identified syntax patterns without corrupting code.
"""

import re
import os
from pathlib import Path

def fix_architecture_file(file_path):
    """Fix syntax errors in a single architecture file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix broken return statements (most critical)
    # Pattern: "d = hd // h return tensor.reshape" -> "d = hd // h\n        return tensor.reshape"
    content = re.sub(
        r'(\w+\s*=\s*[^=\n]+)\s+return\s+',
        r'\1\n        return ',
        content
    )
    
    # 1b. Fix indentation issues from above replacement
    # Pattern: "    d = hd // h\n        return" should all be indented the same
    content = re.sub(
        r'^(\s*)(\w+\s*=\s*[^=\n]+)\n(\s+)return\s+',
        r'\1\2\n\1return ',
        content,
        flags=re.MULTILINE
    )
    
    # 2. Fix missing commas in function definitions
    # Pattern: "activation: str = None bias: bool = False" -> "activation: str = None, bias: bool = False"
    content = re.sub(
        r'(=\s*None)\s+(\w+:\s*\w+)',
        r'\1, \2',
        content
    )
    content = re.sub(
        r'(=\s*False)\s+(\w+:\s*\w+)',
        r'\1, \2',
        content
    )
    content = re.sub(
        r'(=\s*True)\s+(\w+:\s*\w+)',
        r'\1, \2',
        content
    )
    content = re.sub(
        r'(=\s*\d+)\s+(\w+:\s*\w+)',
        r'\1, \2',
        content
    )
    
    # 3. Fix function call signatures with missing commas
    # Pattern: "def __call__(self, x, cache=None output_final_state=False cu_seqlens=None)"
    content = re.sub(
        r'(cache=None)\s+(output_final_state=\w+)\s+(cu_seqlens=None)',
        r'\1, \2, \3',
        content
    )
    
    # 4. Fix mx.zeros calls with missing commas
    # Pattern: "mx.zeros(b, h, d_k v.shape[-1])" -> "mx.zeros(b, h, d_k, v.shape[-1])"
    content = re.sub(
        r'mx\.zeros\(([^,]+),\s*([^,]+),\s*(\w+)\s+([^)]+)\)',
        r'mx.zeros(\1, \2, \3, \4)',
        content
    )
    
    # 5. Fix mx.ones calls with broken dtype
    # Pattern: "mx.ones(chunk_size, chunk_size dtype=mx.bool_)" -> "mx.ones(chunk_size, chunk_size, dtype=mx.bool_)"
    content = re.sub(
        r'mx\.ones\((\w+),\s*(\w+)\s+dtype=',
        r'mx.ones(\1, \2, dtype=',
        content
    )
    
    # 6. Fix duplicate parameter lines (common in these files)
    lines = content.split('\n')
    fixed_lines = []
    prev_line = ""
    
    for line in lines:
        # Skip obvious duplicate parameter lines
        if "bias=False)" in line and ("Linear" in prev_line or "bias" in prev_line):
            continue
        if "cu_seqlens=None):" in line and "cu_seqlens" in prev_line:
            continue
        if "dtype=mx.bool_), " in line and "mx.ones" in prev_line:
            continue
        fixed_lines.append(line)
        prev_line = line
    
    content = '\n'.join(fixed_lines)
    
    # 7. Fix indentation for class attributes
    # Pattern: "self.q_proj = nn.Linear..." not indented properly
    content = re.sub(
        r'^self\.(\w+_proj) = nn\.Linear',
        r'        self.\1 = nn.Linear',
        content,
        flags=re.MULTILINE
    )
    
    # 8. Fix Conv1d calls with missing commas
    content = re.sub(
        r'nn\.Conv1d\(([^,]+),\s*([^,]+),\s*([^,\n]+)\s*\n\s*padding=',
        r'nn.Conv1d(\1, \2, \3,\n        padding=',
        content
    )
    
    # 9. Fix function definition on separate line issue
    content = re.sub(
        r'^def (__init__|__call__|forward)\(',
        r'    def \1(',
        content,
        flags=re.MULTILINE
    )
    
    return content, content != original_content

def main():
    """Fix all architecture files in the mlx_architectures directory."""
    arch_dir = Path("/Users/daniel/dev/asi/mlx_architectures")
    
    if not arch_dir.exists():
        print(f"Directory {arch_dir} does not exist")
        return
    
    files_fixed = 0
    total_files = 0
    
    for file_path in arch_dir.glob("*.py"):
        total_files += 1
        try:
            fixed_content, was_modified = fix_architecture_file(file_path)
            
            if was_modified:
                # Write back the fixed content
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                files_fixed += 1
                print(f"✓ Fixed {file_path.name}")
            else:
                print(f"- No changes needed for {file_path.name}")
        except Exception as e:
            print(f"✗ Error fixing {file_path.name}: {e}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()