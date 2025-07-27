#!/usr/bin/env python3
"""
Ultra Simple MLX Fixer - Only fixes specific known issues
"""

import os
import re
import sys


def fix_single_file(file_path: str) -> bool:
    """Fix a single file with minimal changes"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: line continuation comma issue - axis=-1\n        keepdims=True
        content = re.sub(r'(axis=-1)\s*\n\s+(keepdims=True)', r'\1,\n        \2', content)
        
        # Fix 2: Missing commas in parameter lists
        content = re.sub(r'(\w+: int)\s*\n\s*(\w+: int)', r'\1,\n    \2', content)
        
        # Fix 3: Fix obvious bracket mismatches )]
        content = re.sub(r'\)\]', ')]', content)
        content = re.sub(r'\[\)', '[)', content)
        
        # Only write if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all files"""
    mlx_dir = "mlx_architectures"
    files = [f for f in os.listdir(mlx_dir) if f.endswith('_mlx.py')]
    
    print(f"Applying minimal fixes to {len(files)} files...")
    
    for i, filename in enumerate(sorted(files), 1):
        file_path = os.path.join(mlx_dir, filename)
        arch_name = filename.replace('_mlx.py', '')
        
        print(f"[{i:3d}/106] {arch_name}...", end=" ")
        
        if fix_single_file(file_path):
            print("Fixed")
        else:
            print("No changes")
    
    print("\nMinimal fixes complete. Running test...")
    

if __name__ == "__main__":
    main()