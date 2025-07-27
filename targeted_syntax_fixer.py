#!/usr/bin/env python3
"""
Targeted syntax fixer for specific remaining issues
"""

import re
import os
import glob

def fix_targeted_syntax(code):
    """Apply targeted fixes for specific syntax patterns"""
    
    # Fix 1: Broken function calls across lines
    # return x / mx.linalg.norm(x, axis=-1\n        keepdims=True).clip(min=1e-8)
    code = re.sub(r'mx\.linalg\.norm\(([^)]+),\s*axis=([^)]+)\s*\n\s*keepdims=([^)]+)\)', 
                  r'mx.linalg.norm(\1, axis=\2,\n        keepdims=\3)', code)
    
    # Fix 2: Function definition issues
    # @mx.compile\ndef delta_rule_chunkwise(q, k, v, beta chunk_size: int = 32,
    code = re.sub(r'@mx\.compile\s*\n\s*def\s+(\w+)\(([^)]+)\s+(\w+):\s*int\s*=\s*(\d+),?', 
                  r'@mx.compile\ndef \1(\2,\n    \3: int = \4):', code)
    
    # Fix 3: Missing commas in function definitions more comprehensively
    # def delta_rule_chunkwise(q, k, v, beta chunk_size: int = 32,
    code = re.sub(r'def\s+\w+\([^)]*\w+\s+(\w+):\s*int\s*=\s*\d+', 
                  r'def \g<0>, \1: int = \g<0>', code)
    
    # Fix 4: Fix function definitions where parameters are missing commas
    # Look for patterns like: def func(param1 param2: type = value
    code = re.sub(r'(\w+)\s+(\w+):\s*([\w\[\]]+)(?:\s*=\s*[^,)]+)?(?=\s*[,)])', 
                  r'\1,\n    \2: \3', code)
    
    # Fix 5: Simple missing commas in function calls
    # Fix .sigmoid(), -> .sigmoid()
    code = re.sub(r'\.sigmoid\(\)\s*,\s*(?=\n|\r)', r'.sigmoid()', code)
    
    # Fix 6: Class definition issues
    # def __init__(, self, -> def __init__(self,
    code = re.sub(r'def\s+__init__\(\s*,\s*self\s*,', r'def __init__(self,', code)
    
    # Fix 7: String with commas issues
    # return(x, / x.sum(dim=-1, -> return x / x.sum(dim=-1,
    code = re.sub(r'return\s*\(\s*x\s*,\s*/', r'return x /', code)
    
    # Fix 8: Broken function parameters
    # chunk_size: int = 32, -> chunk_size: int = 32):
    code = re.sub(r'chunk_size:\s*int\s*=\s*32\s*,?\s*$', r'chunk_size: int = 32):', code, flags=re.MULTILINE)
    
    # Fix 9: Missing closing parentheses in function definitions
    # Look for function definitions ending with a comma instead of closing paren
    code = re.sub(r'(\w+:\s*[\w\[\]]+(?:\s*=\s*[^,)]+)?)\s*,\s*$(\s*)"""', r'\1):\n\2"""', code, flags=re.MULTILINE)
    
    # Fix 10: Function docstrings that got mangled
    # chunk_size: int = 32,\n    """docstring"""
    code = re.sub(r'(chunk_size:\s*int\s*=\s*\d+)\s*,\s*\n\s*"""', r'\1):\n    """', code)
    
    # Fix 11: More specific cases
    # Fix def delta_rule_chunkwise(q, k, v, beta chunk_size: int = 32,
    code = re.sub(r'def\s+delta_rule_chunkwise\s*\(\s*([^)]+)\s+chunk_size:\s*int\s*=\s*32\s*,?', 
                  r'def delta_rule_chunkwise(\1,\n    chunk_size: int = 32):', code)
    
    # Fix 12: Fix function calls spanning lines incorrectly
    # Fixed patterns like: func(arg1\n        arg2=value without comma
    code = re.sub(r'(\w+)\s*\(\s*([^,\n()]+)\s*\n\s*(\w+\s*=)', r'\1(\2,\n        \3', code)
    
    # Fix 13: Malformed Sequential calls
    # nn.Sequential(nn.Linear(...), -> fix missing comma issues
    
    # Fix 14: Specific issues with unmatched parentheses
    # Look for patterns where there's an extra comma before a closing paren
    code = re.sub(r',\s*\)\s*:', r'):', code)
    
    # Fix 15: Assignment operator issues
    # Variables split across lines incorrectly
    
    return code

def process_file(filepath):
    """Process a single architecture file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        fixed_content = fix_targeted_syntax(original_content)
        
        if fixed_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Apply targeted fixes to all MLX architecture files"""
    architecture_dir = "/Users/daniel/dev/asi/mlx_architectures"
    pattern = os.path.join(architecture_dir, "*_mlx.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No MLX architecture files found in {architecture_dir}")
        return
    
    print(f"Processing {len(files)} MLX architecture files with targeted fixes...")
    
    fixed_count = 0
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        if process_file(filepath):
            print(f"✓ Fixed: {filename}")
            fixed_count += 1
        else:
            print(f"- No changes: {filename}")
    
    print(f"\nCompleted: {fixed_count}/{len(files)} files modified")

if __name__ == "__main__":
    main()