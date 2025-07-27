#!/usr/bin/env python3
"""
Fix ONLY the type annotation issue: variable:, type -> variable: type
This is the most common error affecting all 106 files
"""

import os
import re
import sys


def fix_type_annotations_in_file(file_path: str) -> bool:
    """Fix type annotation errors in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Very specific pattern: word followed by :, followed by mx.array
        # This captures the exact error we see
        pattern = r'(\b\w+):\s*,\s+(mx\.array|mx\.Array)'
        
        # Replace with proper syntax
        fixed_content = re.sub(pattern, r'\1: \2', content)
        
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Fix type annotations in all MLX architecture files"""
    mlx_dir = "mlx_architectures"
    
    if not os.path.exists(mlx_dir):
        print(f"Directory {mlx_dir} not found!")
        return 1
    
    files = [f for f in os.listdir(mlx_dir) if f.endswith('_mlx.py')]
    files.sort()
    
    print(f"Fixing type annotations in {len(files)} files...")
    
    fixed_count = 0
    
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(mlx_dir, filename)
        arch_name = filename.replace('_mlx.py', '')
        
        print(f"[{i:3d}/106] {arch_name}...", end=" ")
        
        if fix_type_annotations_in_file(file_path):
            print("✅ Fixed")
            fixed_count += 1
        else:
            print("⚠️  No changes")
    
    print(f"\nFixed type annotations in {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())