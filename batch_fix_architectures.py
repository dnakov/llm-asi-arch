#!/usr/bin/env python3
"""
Batch Architecture Fixer
========================

Applies systematic fixes to all MLX architectures based on common error patterns.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple
import ast

def apply_common_fixes(code: str) -> str:
    """Apply common syntax fixes to architecture code"""
    
    # Fix 1: Type annotations with extra commas
    # Pattern: "tensor:, mx.array" -> "tensor: mx.array"
    code = re.sub(r'(\w+):\s*,\s*(mx\.\w+)', r'\1: \2', code)
    
    # Fix 2: Function parameter definitions missing commas
    # Pattern: "def func(self param:" -> "def func(self, param:"
    code = re.sub(r'def\s+(\w+)\s*\(\s*self\s+([^,)]+)', r'def \1(self, \2', code)
    
    # Fix 3: Broken comment lines with code
    # Pattern: "# comment, actual_code" -> "# comment\n    actual_code"
    code = re.sub(r'#\s*([^,\n]+),\s*(\w+\s*=)', r'# \1\n    \2', code)
    
    # Fix 4: _rearrange calls missing commas
    # Pattern: "_rearrange(tensor "pattern"" -> "_rearrange(tensor, "pattern""
    code = re.sub(r'_rearrange\s*\(\s*([^,\s]+)\s+"([^"]+)"', r'_rearrange(\1, "\2"', code)
    
    # Fix 5: Function calls with missing commas between parameters
    # Pattern: "func(param1 param2," -> "func(param1, param2,"
    code = re.sub(r'([a-zA-Z_]\w*)\s*\(\s*([^,\s()]+)\s+([^,\s()]+)\s*,', r'\1(\2, \3,', code)
    
    # Fix 6: mx.array creation with trailing commas
    # Pattern: "mx.array(data)," -> "mx.array(data)"
    code = re.sub(r'mx\.array\(([^)]+)\)\s*,\s*#', r'mx.array(\1)  #', code)
    
    # Fix 7: F.elu and similar function calls
    # Pattern: "F.elu(x 1.0, False)" -> "F.elu(x, 1.0, False)"
    code = re.sub(r'F\.elu\s*\(\s*([^,\s()]+)\s+([^,\s()]+)\s*,\s*([^)]+)\)', r'F.elu(\1, \2, \3)', code)
    
    # Fix 8: Missing commas in function calls (general pattern)
    # Pattern: "func(arg1 arg2)" -> "func(arg1, arg2)"
    code = re.sub(r'(\w+)\s*\(\s*([^,\s()]+)\s+([^,)]+)\)', r'\1(\2, \3)', code)
    
    # Fix 9: Tuple unpacking issues
    # Pattern: "pad = (0\n        0, 0, val)" -> "pad = (0,\n        0, 0, val)"
    code = re.sub(r'\(\s*(\d+)\s*\n\s*(\d+)', r'(\1,\n        \2', code)
    
    # Fix 10: Remove device references
    code = re.sub(r'\.device\b', '', code)
    code = re.sub(r',\s*device=[^,)]+', '', code)
    code = re.sub(r'device=[^,)]+,\s*', '', code)
    
    # Fix 11: kwargs.get calls
    # Pattern: "kwargs.get('a' kwargs.get('b'" -> "kwargs.get('a', kwargs.get('b'"
    code = re.sub(r"kwargs\.get\s*\(\s*'([^']+)'\s+kwargs\.get\s*\(\s*'([^']+)',\s*([^)]+)\)\)", 
                  r"kwargs.get('\1', kwargs.get('\2', \3))", code)
    
    return code

def fix_architecture_file(filepath: Path) -> Tuple[bool, str]:
    """Fix a single architecture file"""
    try:
        with open(filepath, 'r') as f:
            original_code = f.read()
        
        # Apply fixes
        fixed_code = apply_common_fixes(original_code)
        
        # Test syntax
        try:
            ast.parse(fixed_code)
            syntax_valid = True
            error_msg = ""
        except SyntaxError as e:
            syntax_valid = False
            error_msg = f"Syntax error on line {e.lineno}: {e.msg}"
        
        # Save if improved
        if fixed_code != original_code:
            with open(filepath, 'w') as f:
                f.write(fixed_code)
            return syntax_valid, f"Fixed and saved. {error_msg if not syntax_valid else 'Syntax valid!'}"
        else:
            return syntax_valid, f"No changes made. {error_msg if not syntax_valid else 'Already valid!'}"
    
    except Exception as e:
        return False, f"Error processing file: {e}"

def main():
    """Fix all architecture files"""
    mlx_dir = Path("mlx_architectures")
    arch_files = list(mlx_dir.glob("*_mlx.py"))
    
    print(f"🔧 Batch fixing {len(arch_files)} architecture files...")
    print("=" * 60)
    
    fixed_count = 0
    syntax_valid_count = 0
    
    for filepath in sorted(arch_files):
        name = filepath.stem.replace('_mlx', '')
        syntax_valid, message = fix_architecture_file(filepath)
        
        status = "✅" if syntax_valid else "❌"
        print(f"{status} {name}: {message}")
        
        if "Fixed and saved" in message:
            fixed_count += 1
        if syntax_valid:
            syntax_valid_count += 1
    
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Files processed: {len(arch_files)}")
    print(f"   Files modified: {fixed_count}")
    print(f"   Syntax valid: {syntax_valid_count}/{len(arch_files)} ({syntax_valid_count/len(arch_files)*100:.1f}%)")
    
    if syntax_valid_count == len(arch_files):
        print("🎉 All architectures are now syntax valid!")
    else:
        print(f"⚠️  {len(arch_files) - syntax_valid_count} architectures still need manual fixes")

if __name__ == "__main__":
    main()