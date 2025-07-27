#!/usr/bin/env python3
"""
Targeted Parameter List Fixer
Specifically fixes the missing comma issues in function parameter lists
that are causing "line 68: invalid syntax. Perhaps you forgot a comma?" errors
"""

import os
import re
import glob

def fix_function_parameters(content: str) -> str:
    """Fix missing commas in function parameter lists."""
    
    # Pattern 1: Fix __init__ method parameter lists
    # Example: def __init__(self, hidden_size: int,\n    kernel_size: int = 4\n    activation: str = None
    pattern1 = r'def __init__\(self,\s*([^)]+?):\s*(\w+)\s*,?\s*\n\s+(\w+):\s*(\w+)(?:\s*=\s*[^,\n]+)?\s*\n\s+(\w+):\s*(\w+)(?:\s*=\s*[^,\n)]+)?\s*\):'
    def fix_init_params(match):
        first_param = match.group(1)
        param1_name = match.group(2)
        param2_name = match.group(3) 
        param2_type = match.group(4)
        param3_name = match.group(5)
        param3_type = match.group(6)
        
        # Get the rest of the parameters from the original match
        full_match = match.group(0)
        lines = full_match.split('\n')
        
        # Rebuild with proper commas
        result = f"def __init__(self, {first_param}: {param1_name},\n    {param2_name}: {param2_type}"
        if "=" in lines[2]:
            default_val = lines[2].split('=')[1].strip()
            result += f" = {default_val}"
        result += f",\n    {param3_name}: {param3_type}"
        if "=" in lines[3]:
            default_val = lines[3].split('=')[1].strip().rstrip('):')
            result += f" = {default_val}"
        result += "):"
        
        return result
    
    # Simpler approach: fix specific patterns
    # Pattern: parameter without comma followed by another parameter
    content = re.sub(
        r'(\w+:\s*\w+(?:\s*=\s*[^,\n)]+)?)\s*\n\s+(\w+:\s*\w+)', 
        r'\1,\n    \2', 
        content
    )
    
    # Pattern 2: Fix function call parameter lists
    # Example: self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size\n        padding=kernel_size-1
    content = re.sub(
        r'(nn\.Conv1d\([^)]+?)\s*\n\s+(padding=)',
        r'\1,\n        \2',
        content
    )
    
    # Pattern 3: Fix __call__ method parameters
    # Example: def __call__(self, x, cache=None\n        output_final_state=False
    content = re.sub(
        r'(def __call__\(self, [^)]+?)\s*\n\s+(\w+=)',
        r'\1,\n        \2',
        content
    )
    
    # Pattern 4: Fix general function call continuation
    # Example: func(arg1, arg2\n        arg3=value
    content = re.sub(
        r'(\w+\([^)]+?)\s*\n\s+(\w+=)',
        r'\1,\n        \2',
        content
    )
    
    # Pattern 5: Fix bias parameter specifically
    content = re.sub(
        r'(bias=bias)\s*\)',
        r'\1)',
        content
    )
    
    return content

def fix_other_syntax_issues(content: str) -> str:
    """Fix other common syntax issues found in the files."""
    
    # Fix: mx.zeros(b, h, d_k v.shape[-1]) - missing comma
    content = re.sub(r'mx\.zeros\(([^,)]+),\s*([^,)]+),\s*(\w+)\s+(\w+\.[^)]+)\)', r'mx.zeros(\1, \2, \3, \4)', content)
    
    # Fix: range(L_pad, // chunk_size) - extra comma
    content = re.sub(r'range\(([^,)]+),\s*//\s*([^)]+)\)', r'range(\1 // \2)', content)
    
    # Fix: assert(condition, == value) - wrong syntax
    content = re.sub(r'assert\s*\(\s*([^,)]+),\s*==\s*(\w+)', r'assert \1 == \2', content)
    
    # Fix: Sequential(, nn.Linear - extra comma
    content = re.sub(r'nn\.Sequential\(\s*,\s*', r'nn.Sequential(', content)
    
    # Fix: fusion_gate(, hidden_states - extra comma
    content = re.sub(r'\.fusion_gate\(\s*,\s*', r'.fusion_gate(', content)
    
    # Fix: mx.array(value), def - wrong comma placement
    content = re.sub(r'(mx\.array\([^)]+\)),\s*def\s+', r'\1\n\n    def ', content)
    
    # Fix: nn.nn.RMSNorm - duplicate nn
    content = re.sub(r'nn\.nn\.(\w+)', r'nn.\1', content)
    
    return content

def fix_unmatched_delimiters(content: str) -> str:
    """Fix obvious unmatched delimiter issues."""
    
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Count parentheses
        open_parens = line.count('(')
        close_parens = line.count(')')
        
        # If we have more opens than closes and line doesn't end with comma/colon
        if open_parens > close_parens and not line.rstrip().endswith((',', ':')):
            # Don't automatically add closing paren - might break multi-line expressions
            pass
            
        # Count brackets
        open_brackets = line.count('[')
        close_brackets = line.count(']')
        
        if open_brackets > close_brackets and '[' in line and ']' not in line:
            line = line + ']'
        elif close_brackets > open_brackets and ']' in line:
            # Remove extra closing bracket
            line = line.replace(']', '', 1)
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def process_single_file(filepath: str) -> bool:
    """Process a single MLX architecture file."""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply fixes in sequence
        content = original_content
        content = fix_function_parameters(content)
        content = fix_other_syntax_issues(content)
        content = fix_unmatched_delimiters(content)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Apply targeted fixes to all MLX architecture files."""
    
    mlx_dir = "/Users/daniel/dev/asi/mlx_architectures"
    if not os.path.exists(mlx_dir):
        print(f"Directory not found: {mlx_dir}")
        return
    
    # Find all MLX files
    pattern = os.path.join(mlx_dir, "delta_net_*_mlx.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No MLX files found in {mlx_dir}")
        return
    
    print(f"Processing {len(files)} MLX architecture files...")
    
    fixed_count = 0
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        if process_single_file(filepath):
            print(f"✓ Fixed: {filename}")
            fixed_count += 1
        else:
            print(f"- No changes: {filename}")
    
    print(f"\nFixed {fixed_count}/{len(files)} files")
    print("Running test to verify improvements...")
    
    # Run the test to see results
    os.system('cd /Users/daniel/dev/asi && python test_all_architectures.py')

if __name__ == "__main__":
    main()