#!/usr/bin/env python3
"""
Targeted fixer for "Pattern not implemented" errors in MLX architectures.

These errors typically occur when einops operations need to be converted to MLX equivalents.
"""

import os
import subprocess
import json
from pathlib import Path

def find_pattern_errors():
    """Find architectures with pattern-related errors"""
    cmd = ["python", "claude_code_mlx_fixer.py", "--test-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    pattern_errors = []
    for line in result.stdout.split('\n'):
        if "❌" in line and ("Pattern" in line and "not implemented" in line):
            arch_name = line.split(': ')[0].replace('❌ ', '')
            pattern_errors.append(arch_name)
    
    return pattern_errors

def create_pattern_fix_prompt(arch_name: str) -> str:
    """Create a focused prompt for fixing pattern/einops issues"""
    
    prompt = f"""Fix the MLX architecture {arch_name} that has a "Pattern not implemented" error.

SPECIFIC ISSUE: The error "Pattern b l h -> b h l not implemented" indicates einops operations that need MLX conversion.

TASK: Fix the MLX implementation in mlx_architectures/{arch_name}_mlx.py by:

1. IDENTIFY einops operations causing the error
2. REPLACE einops with MLX equivalents:
   - einops.rearrange('b l h -> b h l', x) → mx.transpose(x, (0, 2, 1))
   - einops.rearrange('b h l -> b l h', x) → mx.transpose(x, (0, 2, 1))
   - einops.repeat() → mx.broadcast_to() or mx.tile()
   - einops.reduce() → mx.sum(), mx.mean(), etc.

3. COMMON PATTERNS:
   - 'b l h -> b h l' = transpose last two dims: mx.transpose(x, (0, 2, 1))
   - 'b h l -> b l h' = transpose last two dims: mx.transpose(x, (0, 2, 1))
   - 'b l (h d) -> b h l d' = reshape: x.reshape(b, l, h, d)
   - 'b h l d -> b l (h d)' = reshape: x.reshape(b, l, h*d)

4. ENSURE all imports are MLX-compatible:
   - Remove: from einops import rearrange, repeat, reduce
   - Use: mlx.core operations instead

5. TEST the fix by ensuring forward pass works with shape (2, 32, 256)

FOCUS: Make minimal changes to fix the pattern error while maintaining functionality.

Please read the current MLX file and fix the pattern/einops issues."""

    return prompt

def fix_pattern_architecture(arch_name: str) -> bool:
    """Fix a single architecture with pattern errors"""
    print(f"🔧 Fixing pattern error in {arch_name}...")
    
    prompt = create_pattern_fix_prompt(arch_name)
    
    # Use shorter, focused approach
    cmd = [
        'claude', '-p', prompt,
        '--max-turns', '2',  # Fewer turns for focused fix
        '--output-format', 'json',
        '--system-prompt', 
        'You are an expert at converting einops operations to MLX. '
        'Focus only on fixing pattern/einops errors. Work quickly and efficiently.'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print(f"  ✅ Claude completed fix for {arch_name}")
            return True
        else:
            print(f"  ❌ Claude failed for {arch_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Timeout fixing {arch_name}")
        return False

def main():
    """Fix all architectures with pattern errors"""
    print("🎯 Targeted Pattern Error Fixer")
    print("=" * 40)
    
    pattern_errors = find_pattern_errors()
    print(f"Found {len(pattern_errors)} architectures with pattern errors:")
    
    for arch in pattern_errors:
        print(f"  - {arch}")
    
    if not pattern_errors:
        print("🎉 No pattern errors found!")
        return
    
    print(f"\n🚀 Fixing {len(pattern_errors)} pattern errors...")
    
    fixed = 0
    for i, arch_name in enumerate(pattern_errors, 1):
        print(f"\n[{i}/{len(pattern_errors)}] {arch_name}")
        if fix_pattern_architecture(arch_name):
            fixed += 1
    
    print(f"\n📊 Results: {fixed}/{len(pattern_errors)} pattern errors fixed")

if __name__ == "__main__":
    main()