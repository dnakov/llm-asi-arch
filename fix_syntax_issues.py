#!/usr/bin/env python3
"""
Fix specific syntax issues found in MLX architectures.
"""

import re
import os

def fix_spectral_fusion():
    """Fix syntax error in delta_net_spectral_fusion_mlx.py"""
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_spectral_fusion_mlx.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the syntax error on line 213
    content = content.replace(
        "self.mix_logit = mx.array(mx.zeros(num_heads)), # ------------------------------------------------------------------",
        "self.mix_logit = mx.array(mx.zeros(num_heads))  # ------------------------------------------------------------------"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed syntax error in delta_net_spectral_fusion_mlx.py")

def fix_hhmr():
    """Fix syntax error in delta_net_hhmr_mlx.py"""
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_hhmr_mlx.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the function parameter declaration
    content = content.replace(
        "def _rearrange(tensor:, mx.array, pattern: str, **kwargs) -> mx.array:",
        "def _rearrange(tensor: mx.array, pattern: str, **kwargs) -> mx.array:"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed syntax error in delta_net_hhmr_mlx.py")

def fix_unmatched_parentheses(file_path):
    """Fix unmatched parentheses in a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # Count parentheses
        open_parens = line.count('(')
        close_parens = line.count(')')
        
        if open_parens > close_parens:
            # Missing closing parentheses
            diff = open_parens - close_parens
            # Add closing parentheses at the end of the line (before newline)
            line = line.rstrip() + ')' * diff + '\n'
        elif close_parens > open_parens:
            # Extra closing parentheses - remove them
            diff = close_parens - open_parens
            for _ in range(diff):
                line = line.replace(')', '', 1)
        
        fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed parentheses in {os.path.basename(file_path)}")

def fix_entropy_floor():
    """Fix the ArrayAt.set issue in delta_net_entropy_floor_mlx.py"""
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_entropy_floor_mlx.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace .set() with direct assignment
    content = re.sub(
        r'(\w+)\.set\(([^)]+)\)',
        r'\1 = \2',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed ArrayAt.set issues in delta_net_entropy_floor_mlx.py")

def fix_tuple_unpacking_issues():
    """Fix tuple unpacking issues by checking return statements"""
    problem_files = [
        "delta_net_ahic_mlx.py",
        "delta_net_entropy_kl_floor_gate_mlx.py", 
        "delta_net_gae_ms3e_mlx.py"
    ]
    
    for filename in problem_files:
        file_path = f"/Users/daniel/dev/asi/mlx_architectures/{filename}"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find functions that return tuples when they should return single values
        # Look for return statements with multiple values
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check for return statements with tuples in __call__ methods
            if 'return' in line and ',' in line and 'def __call__' in content[content.rfind('def __call__', 0, content.find(line)):content.find(line) + len(line)]:
                # If it's returning a tuple but should return single value, take first element
                if line.strip().startswith('return') and line.count(',') == 1:
                    # Extract the first part of the tuple
                    return_part = line.split('return')[1].strip()
                    first_value = return_part.split(',')[0].strip()
                    line = line.split('return')[0] + f'return {first_value}'
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Fixed tuple unpacking issues in {filename}")

def main():
    print("Fixing syntax issues in MLX architectures...")
    
    # Fix specific syntax errors
    fix_spectral_fusion()
    fix_hhmr()
    fix_entropy_floor()
    
    # Fix unmatched parentheses in specific files
    paren_files = [
        "/Users/daniel/dev/asi/mlx_architectures/delta_net_sparsemax_temperature_mlx.py",
        "/Users/daniel/dev/asi/mlx_architectures/delta_net_ssg_sparsemax_temp_mlx.py",
        "/Users/daniel/dev/asi/mlx_architectures/delta_net_hybrid_floor_gt_mlx.py",
        "/Users/daniel/dev/asi/mlx_architectures/delta_net_triscale_mlx.py",
        "/Users/daniel/dev/asi/mlx_architectures/delta_net_ms_adaptive_gstat3_mlx.py"
    ]
    
    for file_path in paren_files:
        if os.path.exists(file_path):
            fix_unmatched_parentheses(file_path)
    
    # Fix tuple unpacking issues
    fix_tuple_unpacking_issues()
    
    print("\nAll syntax fixes completed!")

if __name__ == "__main__":
    main()