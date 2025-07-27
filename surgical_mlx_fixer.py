#!/usr/bin/env python3
"""
Surgical MLX Architecture Syntax Fixer
A very targeted approach to fix specific syntax errors without corrupting files.
"""

import os
import re
import glob
from typing import List, Tuple

def fix_docstring_issues(content: str) -> str:
    """Fix broken docstring quotes"""
    # Fix unterminated docstrings at the beginning
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix lines that start with just a quote (unterminated docstring)
        if line.strip() == '"':
            if i < len(lines) - 1 and lines[i+1].strip().startswith('MLX-converted'):
                # This is a broken docstring start
                line = '"""'
            elif line.strip() == '"' and any('"""' in lines[j] for j in range(max(0, i-3), min(len(lines), i+3))):
                # This is a broken docstring end
                line = '"""'
        
        # Fix docstrings with broken quotes
        line = re.sub(r'^"([^"]*)"$', r'"""\1"""', line.strip())
        if line != line.strip():
            line = '    ' + line  # Preserve some indentation
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_shortconv_class_issues(content: str) -> str:
    """Fix specific issues in _ShortConvolution class"""
    
    # Fix the duplicate bias parameter issue
    # Pattern: def __init__(self, ..., bias: bool = False):\n    bias: bool = False): super().__init__()
    pattern = r'(def __init__\(self,.*?bias: bool = False\):\s*)\n\s*bias: bool = False\):\s*super\(\)\.__init__\(\)'
    replacement = r'\1\n        super().__init__()'
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Fix Conv1d calls with broken parameters
    # Fix: self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1, bias=bias)
    pattern = r'nn\.Conv1d\(([^,]+),\s*([^,]+),\s*([^,]+),\s*\n\s*padding=([^,\n]+)\s*\n\s*bias=([^)]+)\)'
    replacement = r'nn.Conv1d(\1, \2, \3, padding=\4, bias=\5)'
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Fix broken return statements  
    # Fix: return out\n        None  # comment\n        return out
    pattern = r'return out\s*\n\s*None\s*#[^\n]*\n\s*return out'
    replacement = 'return out, None  # Simplified - no cache state\n        return out'
    content = re.sub(pattern, replacement, content)
    
    # Fix broken causal truncation line
    pattern = r'out = out\[:, :, :x\.shape\[1\]\]  # Causal truncation,\s*out = out\.transpose\(0, 2, 1\)  # \(B, L, D\)'
    replacement = 'out = out[:, :, :x.shape[1]]  # Causal truncation\n        out = out.transpose(0, 2, 1)  # (B, L, D)'
    content = re.sub(pattern, replacement, content)
    
    return content

def fix_l2norm_function(content: str) -> str:
    """Fix _l2norm function syntax issues"""
    
    # Fix: mx.linalg.norm(x, axis=-1, keepdims=True)mx.clip(min=1e-8)
    pattern = r'mx\.linalg\.norm\(x, axis=-1,\s*keepdims=True\)mx\.clip\(min=1e-8\)'
    replacement = 'mx.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)'
    content = re.sub(pattern, replacement, content)
    
    # Fix broken docstring
    pattern = r'"L2 normalization""\s*return'
    replacement = '"""L2 normalization"""\n    return'
    content = re.sub(pattern, replacement, content)
    
    return content

def fix_rearrange_function(content: str) -> str:
    """Fix _rearrange function syntax issues"""
    
    # Fix missing quotes in pattern matching
    patterns_to_fix = [
        (r'if "b l\(h, d\) -> b l h d in pattern:', r'if "b l (h d) -> b l h d" in pattern:'),
        (r'elif b l h d -> b l\(h, d\)" in pattern:', r'elif "b l h d -> b l (h d)" in pattern:'),
        (r'elif "b l h d -> b h l d in pattern:', r'elif "b l h d -> b h l d" in pattern:'),
        (r'elif b h l d -> b l h d" in pattern:', r'elif "b h l d -> b l h d" in pattern:'),
        (r'elif "b h\(n, c\) d -> b h n c d in pattern:', r'elif "b h (n c) d -> b h n c d" in pattern:'),
        (r'elif b h n c d -> b h\(n, c\) d" in pattern:', r'elif "b h n c d -> b h (n c) d" in pattern:'),
    ]
    
    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)
    
    # Fix broken variable assignments on same line
    # Fix: b, l, hd = tensor.shape, d = hd // h
    pattern = r'b, l, hd = tensor\.shape,\s*d = hd // h'
    replacement = 'b, l, hd = tensor.shape\n        d = hd // h'
    content = re.sub(pattern, replacement, content)
    
    # Similar fix for other patterns
    pattern = r'b, h, nc, d = tensor\.shape,\s*n = nc // c'
    replacement = 'b, h, nc, d = tensor.shape\n        n = nc // c'
    content = re.sub(pattern, replacement, content)
    
    return content

def fix_basic_syntax_errors(content: str) -> str:
    """Fix basic syntax errors throughout the file"""
    
    # Fix broken docstrings
    content = re.sub(r'""([^"]*)"', r'"""\1"""', content)
    content = re.sub(r'"([^"]*)""\s*$', r'"""\1"""', content, flags=re.MULTILINE)
    
    # Fix missing commas in function parameters
    # Only fix clear cases where there's a newline between parameters without comma
    content = re.sub(r'(\w+:\s*\w+)\s*\n\s*(\w+:\s*\w+\s*[=)])', r'\1,\n        \2', content)
    
    return content

def surgical_fix_file(file_path: str) -> Tuple[bool, str]:
    """Apply surgical fixes to a single file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes in order
        content = fix_docstring_issues(content)
        content = fix_shortconv_class_issues(content)
        content = fix_l2norm_function(content)
        content = fix_rearrange_function(content)
        content = fix_basic_syntax_errors(content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Test syntax
        try:
            compile(content, file_path, 'exec')
            return True, "Fixed successfully"
        except SyntaxError as e:
            return False, f"Syntax error after fix: line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Compile error after fix: {e}"
            
    except Exception as e:
        return False, f"File processing error: {e}"

def main():
    """Apply surgical fixes to all MLX architecture files"""
    
    mlx_dir = "mlx_architectures"
    if not os.path.exists(mlx_dir):
        print(f"❌ Directory {mlx_dir} not found!")
        return
    
    # Find all MLX architecture files
    pattern = os.path.join(mlx_dir, "*_mlx.py")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No MLX architecture files found in {mlx_dir}")
        return
    
    print(f"🔧 Found {len(files)} MLX architecture files to fix")
    print("=" * 80)
    
    success_count = 0
    failure_count = 0
    results = []
    
    for i, file_path in enumerate(sorted(files), 1):
        arch_name = os.path.basename(file_path).replace('_mlx.py', '')
        print(f"[{i:3d}/{len(files)}] Fixing {arch_name}...")
        
        success, message = surgical_fix_file(file_path)
        
        if success:
            success_count += 1
            print(f"  ✅ {message}")
        else:
            failure_count += 1
            print(f"  ❌ {message}")
        
        results.append({
            'file': arch_name,
            'success': success,
            'message': message
        })
    
    print("\n" + "=" * 80)
    print("SURGICAL FIX SUMMARY")
    print("=" * 80)
    print(f"📁 Total files processed: {len(files)}")
    print(f"✅ Successfully fixed: {success_count}")
    print(f"❌ Failed to fix: {failure_count}")
    print(f"📊 Success rate: {success_count/len(files)*100:.1f}%")
    
    if failure_count > 0:
        print(f"\n🔍 Failed fixes:")
        for result in results:
            if not result['success']:
                print(f"   {result['file']}: {result['message']}")
    
    print(f"\n🎯 Next step: Run 'python test_all_architectures.py' to validate fixes")

if __name__ == "__main__":
    main()