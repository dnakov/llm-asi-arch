#!/usr/bin/env python3
"""
Comprehensive syntax fixer for all MLX architecture files.
Fixes common PyTorch to MLX conversion issues systematically.
"""

import os
import re
import glob
from pathlib import Path

def fix_syntax_errors(content: str) -> str:
    """Fix common syntax errors in MLX architecture files."""
    
    # 1. Fix type annotation syntax: `tensor:, mx.array` -> `tensor: mx.array`
    content = re.sub(r'(\w+):\s*,\s*(mx\.array)', r'\1: \2', content)
    
    # 2. Fix missing commas in function calls and parameters
    # Pattern: word( word -> word(word,
    content = re.sub(r'([a-zA-Z_]\w*)\(\s*([a-zA-Z_]\w*)\s+([a-zA-Z_])', r'\1(\2, \3', content)
    
    # 3. Fix missing commas in kwargs.get calls
    content = re.sub(r"kwargs\.get\('([^']+)'\s+kwargs\.get\('([^']+)', ([^)]+)\)", 
                    r"kwargs.get('\1', kwargs.get('\2', \3)", content)
    
    # 4. Fix F.elu calls with missing commas
    content = re.sub(r'F\.elu\((\w+)\s+([0-9.]+),?\s*(False|True)\)', r'F.elu(\1, \2, \3)', content)
    
    # 5. Fix missing commas in function definitions
    content = re.sub(r'def (\w+)\(self\s+([^)]+)\)', r'def \1(self, \2)', content)
    
    # 6. Fix transpose calls with missing commas
    content = re.sub(r'\.transpose\((-?\d+)\s+(-?\d+)\)', r'.transpose(\1, \2)', content)
    
    # 7. Fix MLX specific parameter issues
    content = re.sub(r'keepdim=True', 'keepdims=True', content)
    content = re.sub(r'axis=(-?\d+)\s+keepdims=True', r'axis=\1, keepdims=True', content)
    
    # 8. Fix constructor calls with missing commas
    content = re.sub(r'nn\.Linear\((\w+)\s+(\w+)\)', r'nn.Linear(\1, \2)', content)
    content = re.sub(r'nn\.Conv1d\((\w+),\s*(\w+),\s*(\w+)\s+(\w+)', r'nn.Conv1d(\1, \2, \3, \4', content)
    
    # 9. Fix broken parameter lists in function definitions
    content = re.sub(r'def __init__\(,\s*self,', r'def __init__(self,', content)
    content = re.sub(r'def forward\(,\s*self,', r'def forward(self,', content)
    
    # 10. Fix unmatched parentheses in function calls
    content = re.sub(r'(\w+)\.reshape\(([^)]+)\s+([^)]+)\)', r'\1.reshape(\2, \3)', content)
    
    # 11. Fix missing commas in _rearrange calls
    content = re.sub(r'_rearrange\((\w+)\s+"([^"]+)"\s+([^)]+)\)', r'_rearrange(\1, "\2", \3)', content)
    
    # 12. Fix broken Tuple type annotations
    content = re.sub(r'Tuple\[mx\.array\s+mx\.array\]', r'Tuple[mx.array, mx.array]', content)
    
    # 13. Fix broken variable assignments with commas
    content = re.sub(r'(\w+)\s+var\s*=\s*([^,\n]+)', r'\1, var = \2', content)
    
    # 14. Fix broken unbiased parameter calls
    content = re.sub(r'\.var\((-?\d+)\s+unbiased=False\)', r'.var(\1, unbiased=False)', content)
    
    # 15. Fix broken kernel_sizes parameter in function definitions
    content = re.sub(r'kernel_sizes:\s*Tuple\[int,\s*\.\.\.\]', r'kernel_sizes: Tuple[int, ...]', content)
    
    # 16. Fix missing commas in Sequential definitions
    content = re.sub(r'nn\.Sequential\(,\s*', r'nn.Sequential(', content)
    
    # 17. Fix broken bias parameter syntax
    content = re.sub(r'bias\s*=\s*conv_bias\)', r'bias=conv_bias)', content)
    
    # 18. Fix missing commas in max function calls
    content = re.sub(r'max\((\d+)\s+int\(', r'max(\1, int(', content)
    
    # 19. Fix broken comma placement in multi-line statements
    content = re.sub(r'(\w+)\s*=\s*([^,\n]+),\s*def\s+', r'\1 = \2\n    def ', content)
    
    # 20. Fix broken mx.array constructor calls
    content = re.sub(r'mx\.array\(mx\.zeros\(([^)]+)\)\),\s*def\s+', r'mx.array(mx.zeros(\1))\n    def ', content)
    
    # 21. Fix missing commas in eps parameter
    content = re.sub(r'eps\s*=\s*norm_eps\)', r'eps=norm_eps)', content)
    
    # 22. Fix broken RMSNorm calls
    content = re.sub(r'nn\.nn\.RMSNorm', r'nn.RMSNorm', content)
    
    # 23. Fix missing commas in max calls with floats
    content = re.sub(r'max\(([0-9.]+)\s+([^)]+)\)', r'max(\1, \2)', content)
    
    # 24. Fix broken expand_dims calls in multi-line
    content = re.sub(r'(\w+)\.expand_dims\(0\)\s+last_state', r'\1.expand_dims(0)\n        last_state', content)
    
    # 25. Fix comma-separated assignments on same line
    content = re.sub(r'(\w+)\s+(\w+)\s*=\s*([^,\n]+),\s*([^,\n]+)', r'\1, \2 = \3, \4', content)
    
    # 26. Fix broken comment continuation
    content = re.sub(r'# Simplified version - just return indices for non-masked positions,\s*indices', 
                    r'# Simplified version - just return indices for non-masked positions\n    indices', content)
    
    # 27. Fix broken string literals and multi-line issues
    content = re.sub(r'return out\n\s*None\s*#', r'return out, None  #', content)
    
    # 28. Fix broken mx.cat calls
    content = re.sub(r'mx\.cat\(\[\s*,\s*([^]]+)\]', r'mx.cat([\1]', content)
    
    # 29. Fix broken annotation in forward function
    content = re.sub(r'def forward\(\s*,\s*self,', r'def forward(self,', content)
    
    # 30. Fix missing commas in complex expressions
    content = re.sub(r'(\w+)\s*=\s*([^,\n]+)\s+([^,\n=]+)\s*=\s*', r'\1 = \2, \3 = ', content)
    
    return content

def process_file(file_path: str) -> bool:
    """Process a single MLX architecture file and fix syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        fixed_content = fix_syntax_errors(original_content)
        
        # Only write if content actually changed
        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ Fixed: {os.path.basename(file_path)}")
            return True
        else:
            print(f"⚪ No changes needed: {os.path.basename(file_path)}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all MLX architecture files."""
    mlx_arch_dir = "mlx_architectures"
    
    if not os.path.exists(mlx_arch_dir):
        print(f"❌ Directory {mlx_arch_dir} not found!")
        return
    
    # Find all MLX architecture files
    pattern = os.path.join(mlx_arch_dir, "delta_net_*_mlx.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No MLX architecture files found in {mlx_arch_dir}")
        return
    
    print(f"🔧 Found {len(files)} MLX architecture files to process...")
    
    fixed_count = 0
    for file_path in sorted(files):
        if process_file(file_path):
            fixed_count += 1
    
    print(f"\n✅ Summary: Fixed {fixed_count}/{len(files)} files")
    print(f"📊 Success rate: {(fixed_count/len(files)*100):.1f}%")

if __name__ == "__main__":
    main()