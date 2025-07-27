#!/usr/bin/env python3
"""
Comprehensive syntax fixer for MLX architectures
Targets all the specific syntax patterns found in the files
"""

import re
import os
import glob

def fix_comprehensive_syntax(code):
    """Apply comprehensive syntax fixes to architecture code"""
    
    # Fix 1: Type annotations with extra commas - tensor:, mx.array -> tensor: mx.array
    code = re.sub(r'(\w+):\s*,\s*(mx\.\w+)', r'\1: \2', code)
    
    # Fix 2: Function parameter missing commas - func(param1 param2) -> func(param1, param2)
    # This is more complex - look for patterns like kernel_size: int = 4\n    activation:
    code = re.sub(r'(\w+:\s*\w+(?:\s*=\s*[^,\n]+)?)\s*\n\s*(\w+:)', r'\1,\n        \2', code)
    
    # Fix 3: Missing commas in function calls - F.elu(x 1.0, False) -> F.elu(x, 1.0, False)
    code = re.sub(r'F\.elu\s*\(\s*([^,\s()]+)\s+([^,\s()]+)\s*,\s*([^)]+)\)', r'F.elu(\1, \2, \3)', code)
    
    # Fix 4: Missing commas in Conv1d calls - Conv1d(a, b, c\n        padding=d -> Conv1d(a, b, c,\n        padding=d
    code = re.sub(r'Conv1d\s*\(\s*([^,\n]+),\s*([^,\n]+),\s*([^,\n]+)\s*\n\s*(padding=)', r'Conv1d(\1, \2, \3,\n        \4', code)
    
    # Fix 5: Missing commas in Linear calls - Linear(a, b\n        bias=c -> Linear(a, b,\n        bias=c
    code = re.sub(r'Linear\s*\(\s*([^,\n]+),\s*([^,\n]+)\s*\n\s*(bias=)', r'Linear(\1, \2,\n        \3', code)
    
    # Fix 6: Function definitions missing commas - def __init__(self, a: int\n    b: int -> def __init__(self, a: int,\n    b: int
    code = re.sub(r'def\s+__init__\s*\(\s*([^)]+?):\s*int\s*\n\s*(\w+:)', r'def __init__(\1: int,\n        \2', code)
    
    # Fix 7: Missing commas in function calls with line breaks
    # Pattern: func(arg1\n        arg2 = value -> func(arg1,\n        arg2=value
    code = re.sub(r'(\w+)\s*\(\s*([^,\n()]+)\s*\n\s*(\w+\s*=)', r'\1(\2,\n        \3', code)
    
    # Fix 8: Unmatched parentheses - look for patterns like ...)))\n    attn = ...
    # This is complex, but we can fix some obvious patterns
    
    # Fix 9: Function call continuation issues - padding=kernel_size-1\n        bias=bias)
    code = re.sub(r'(padding=[^,\n]+)\s*\n\s*(bias=[^)]+)\)', r'\1,\n        \2)', code)
    
    # Fix 10: Assignment issues - fusion_weights = self.fusion_gate(, hidden_states,
    code = re.sub(r'self\.fusion_gate\(\s*,\s*', r'self.fusion_gate(', code)
    
    # Fix 11: Tuple assignment issues - q\n        conv_state_q = self.conv ->  q, conv_state_q = self.conv
    code = re.sub(r'\n\s*(\w+)\s*\n\s*(\w+)\s*=\s*self\.(\w+conv1d)', r'\n        \1, \2 = self.\3', code)
    
    # Fix 12: Missing commas in Sequential - nn.Sequential(, nn.Linear -> nn.Sequential(nn.Linear
    code = re.sub(r'nn\.Sequential\(\s*,\s*', r'nn.Sequential(', code)
    
    # Fix 13: Broken parameter lists - chunk_size: int = 32): -> chunk_size: int = 32,
    code = re.sub(r'chunk_size:\s*int\s*=\s*32\):', r'chunk_size: int = 32,', code)
    
    # Fix 14: Variable assignments split across lines - v_direct = v  # identity/value path --------------------------------
    # Look for orphaned assignment operators
    
    # Fix 15: Assert statement issues - assert(attention_mask.ndim, == 2
    code = re.sub(r'assert\s*\(\s*([^,)]+),\s*==\s*(\d+)', r'assert \1 == \2', code)
    
    # Fix 16: Missing operators in conditions - if t >= self.prune_end_step:\n            return self.prune_threshold
    # This looks OK, but check for missing colons
    
    # Fix 17: String formatting issues - already fixed in previous versions
    
    # Fix 18: Import statement cleanup
    code = re.sub(r'import mlx\.nn as F\nfrom mx\.nn import functional as F', 'import mlx.nn as nn\nfrom mlx.nn import functional as F', code)
    
    # Fix 19: Remove duplicate empty lines
    code = re.sub(r'\n\s*\n\s*\n', r'\n\n', code)
    
    # Fix 20: Function calls with missing arguments
    # Pattern: func(cache=conv_state_q\n        output_final_state=use_cache
    code = re.sub(r'(cache=[^,\n]+)\s*\n\s*(output_final_state=)', r'\1,\n        \2', code)
    
    # Fix 21: Unmatched parentheses in function definitions
    # Look for patterns ending with "**kwargs: Dict # Accept extra unused kwargs for, compatibility) -> None:"
    code = re.sub(r'\*\*kwargs:\s*Dict\s*#[^)]*compatibility\)\s*->\s*None:', r'**kwargs) -> None:', code)
    
    # Fix 22: More complex missing comma patterns
    # kernel_size: int = 4\n    activation: str = None\n    bias: bool = False
    code = re.sub(r'(kernel_size:\s*int\s*=\s*\d+)\s*\n\s*(activation:\s*str)', r'\1,\n        \2', code)
    code = re.sub(r'(activation:\s*str\s*=\s*None)\s*\n\s*(bias:\s*bool)', r'\1,\n        \2', code)
    
    # Fix 23: Specific patterns from the files
    # def __init__(self, num_heads:, int, -> def __init__(self, num_heads: int,
    code = re.sub(r'(\w+):,\s*(\w+),', r'\1: \2,', code)
    
    # Fix 24: Fix assignment operator spacing and placement
    code = re.sub(r'(\w+)\s*=\s*(\w+)\s*@\s*(\w+)\s*\n\s*(\w+)\s*=', r'\1 = \2 @ \3\n        \4 =', code)
    
    return code

def process_file(filepath):
    """Process a single architecture file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        fixed_content = fix_comprehensive_syntax(original_content)
        
        if fixed_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Apply comprehensive fixes to all MLX architecture files"""
    architecture_dir = "/Users/daniel/dev/asi/mlx_architectures"
    pattern = os.path.join(architecture_dir, "*_mlx.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No MLX architecture files found in {architecture_dir}")
        return
    
    print(f"Processing {len(files)} MLX architecture files...")
    
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