#!/usr/bin/env python3
"""
Comprehensive MLX Architecture Syntax Fixer
Systematically fixes all 106 MLX architecture files with common syntax errors.
"""

import os
import re
import glob
from typing import List, Tuple

def fix_short_convolution_class(content: str) -> str:
    """Fix the _ShortConvolution class syntax issues"""
    
    # Pattern 1: Fix duplicate bias parameter and unmatched parentheses
    # This handles lines like:
    # def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = None, bias: bool = False):
    #              bias: bool = False): super().__init__()
    
    pattern1 = r'(def __init__\([^)]+bias: bool = False\):\s*)\n\s*bias: bool = False\):\s*super\(\)\.__init__\(\)'
    replacement1 = r'\1\n        super().__init__()'
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    
    # Pattern 2: Fix the simpler case where there's just a duplicate bias line
    pattern2 = r'(def __init__\([^)]+\):\s*)\n\s*bias: bool = False\):\s*super\(\)\.__init__\(\)'
    replacement2 = r'\1\n        super().__init__()'
    content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
    
    # Pattern 3: Fix Conv1d calls with missing commas
    # self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size,
    # padding=kernel_size-1,
    # bias=bias)
    pattern3 = r'nn\.Conv1d\(([^,]+),\s*([^,]+),\s*([^,]+),\s*\n\s*padding=([^,\n]+)\s*\n\s*bias=([^)]+)\)'
    replacement3 = r'nn.Conv1d(\1, \2, \3, padding=\4, bias=\5)'
    content = re.sub(pattern3, replacement3, content, flags=re.MULTILINE)
    
    # Pattern 4: Fix indentation issues in __call__ method
    content = re.sub(r'\n    x_conv = ', '\n        x_conv = ', content)
    content = re.sub(r'\n    out = ', '\n        out = ', content)
    
    return content

def fix_rearrange_function(content: str) -> str:
    """Fix the _rearrange function syntax issues"""
    
    # Fix missing indentation and return statements
    lines = content.split('\n')
    fixed_lines = []
    in_rearrange = False
    
    for i, line in enumerate(lines):
        if 'def _rearrange(' in line:
            in_rearrange = True
            fixed_lines.append(line)
        elif in_rearrange and line.strip().startswith('def ') and 'def _rearrange(' not in line:
            in_rearrange = False
            fixed_lines.append(line)
        elif in_rearrange:
            # Fix indentation issues in _rearrange function
            if line.strip().startswith('d = ') or line.strip().startswith('n = '):
                if not line.startswith('        '):
                    line = '        ' + line.strip()
            elif line.strip().startswith('return ') and not line.startswith('        '):
                line = '        ' + line.strip()
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_l2norm_function(content: str) -> str:
    """Fix the _l2norm function syntax issues"""
    
    # Fix missing comma in function call
    pattern = r'mx\.linalg\.norm\(x, axis=-1\s*\n\s*keepdims=True\)'
    replacement = r'mx.linalg.norm(x, axis=-1, keepdims=True)'
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def fix_import_statements(content: str) -> str:
    """Fix common import statement issues"""
    
    # Fix wrong import: from mx.nn import functional as F
    content = re.sub(r'from mx\.nn import functional as F', 'import mlx.nn.functional as F', content)
    
    # Fix missing mlx.nn import
    if 'import mlx.nn as nn' not in content and 'nn.' in content:
        # Add missing import
        lines = content.split('\n')
        import_inserted = False
        for i, line in enumerate(lines):
            if line.startswith('import mlx.core as mx') and not import_inserted:
                lines.insert(i + 1, 'import mlx.nn as nn')
                import_inserted = True
                break
    
    return '\n'.join(lines) if 'lines' in locals() else content

def fix_class_definition_issues(content: str) -> str:
    """Fix various class definition syntax issues"""
    
    # Fix class constructor parameter issues
    # Pattern: def __init__(self, mode: str =, "value", ...)
    pattern = r'(def __init__\([^)]*mode: str =),\s*"([^"]*)"'
    replacement = r'\1 "\2"'
    content = re.sub(pattern, replacement, content)
    
    # Fix missing commas in function calls and parameter lists
    # Pattern: function(param1 param2)
    content = re.sub(r'(\w+)\s+(\w+\s*=)', r'\1, \2', content)
    
    return content

def fix_forward_method_issues(content: str) -> str:
    """Fix issues in forward method implementations"""
    
    # Fix broken variable assignments that span multiple lines
    # Pattern: variable = function(param
    #     param=value
    #     param=value)
    
    # Fix q, conv_state_q = assignment issues
    patterns_fixes = [
        # Fix broken multi-line function calls
        (r'(\w+)\s*\n\s*conv_state_\w+ = self\.\w+_conv1d\([^)]+\n\s*cache=[^)]+\n\s*output_final_state=[^)]+\n\s*cu_seqlens = [^)]+\)', 
         lambda m: m.group(0).replace('\n', ' ').replace('  ', ' ')),
        
        # Fix broken rearrange calls
        (r'_rearrange\([^,]+,\s*"[^"]+"\s*\n\s*[^)]+\)', 
         lambda m: m.group(0).replace('\n', ' ').replace('  ', ' ')),
         
        # Fix dict access syntax
        (r'attention_mask\[:, -seq_len:\]\s*\)', r'attention_mask[:, -seq_len:]'),
        
        # Fix missing commas in long parameter lists
        (r'(\w+=[^,\n]+)\s+(\w+=[^,\n]+)', r'\1, \2'),
    ]
    
    for pattern, replacement in patterns_fixes:
        if callable(replacement):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def fix_compilation_errors(content: str) -> str:
    """Fix various compilation/syntax errors"""
    
    fixes = [
        # Fix missing quotes around strings
        (r'assert (\w+) in \(([^)]+)\)', r'assert \1 in (\2)'),
        
        # Fix missing self parameter
        (r'def (\w+)\(([^)]*)\):', lambda m: f"def {m.group(1)}(self, {m.group(2)}):" if 'self' not in m.group(2) and m.group(1) != '__init__' else m.group(0)),
        
        # Fix missing commas in tuples and function calls
        (r'(\([^)]*\w)\s+(\w[^)]*\))', r'\1, \2'),
        
        # Fix broken string concatenation
        (r'"([^"]*)"([^"]*)"([^"]*)"', r'"\1\2\3"'),
        
        # Fix mx.array usage
        (r'mx\.array\(mx\.full\(([^)]+)\)\)', r'mx.full(\1)'),
        
        # Fix method calls
        (r'\.expand_dims\(([^)]+)\)\.expand\(([^)]+)\)', r'.expand_dims(\1).repeat(\2)'),
    ]
    
    for pattern, replacement in fixes:
        if callable(replacement):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def fix_mlx_specific_issues(content: str) -> str:
    """Fix MLX-specific conversion issues"""
    
    # Fix tensor operations
    fixes = [
        # Fix .masked_fill_() to mx.where()
        (r'\.masked_fill_\(([^,]+),\s*([^)]+)\)', r'_masked_fill(\1, \2)'),
        
        # Fix .clip() to mx.clip()
        (r'\.clip\(min=([^)]+)\)', r'mx.clip(min=\1)'),
        
        # Fix .detach() (not needed in MLX)
        (r'\.detach\(\)', ''),
        
        # Fix torch.compile to mx.compile
        (r'@torch\.compile', '@mx.compile'),
        
        # Fix device placement (not needed in MLX)
        (r', device=[^,)]+', ''),
        
        # Fix dtype specifications
        (r'dtype=torch\.bool_', 'dtype=mx.bool_'),
        (r'dtype=torch\.', 'dtype=mx.'),
        
        # Fix tensor creation
        (r'torch\.zeros', 'mx.zeros'),
        (r'torch\.ones', 'mx.ones'),
        (r'torch\.eye', 'mx.eye'),
        (r'torch\.randn_like', 'mx.random.normal'),
        
        # Fix functional calls
        (r'F\.elu\(([^,]+),\s*([^,]+),\s*False\)', r'F.elu(\1)'),
        (r'F\.softplus', 'mx.softplus'),
        (r'F\.sigmoid', 'mx.sigmoid'),
        (r'F\.conv1d', 'mx.conv1d'),
        (r'F\.pad', 'mx.pad'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def comprehensive_syntax_fix(content: str) -> str:
    """Apply all syntax fixes in the correct order"""
    
    # Apply fixes in order of dependency
    content = fix_import_statements(content)
    content = fix_rearrange_function(content)
    content = fix_l2norm_function(content)
    content = fix_short_convolution_class(content)
    content = fix_class_definition_issues(content)
    content = fix_forward_method_issues(content)
    content = fix_mlx_specific_issues(content)
    content = fix_compilation_errors(content)
    
    return content

def fix_single_architecture(file_path: str) -> Tuple[bool, str]:
    """Fix a single architecture file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply comprehensive fixes
        fixed_content = comprehensive_syntax_fix(original_content)
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Test syntax by compiling
        try:
            compile(fixed_content, file_path, 'exec')
            return True, "Fixed successfully"
        except SyntaxError as e:
            return False, f"Syntax error after fix: {e}"
        except Exception as e:
            return False, f"Compile error after fix: {e}"
            
    except Exception as e:
        return False, f"File processing error: {e}"

def main():
    """Fix all MLX architecture files"""
    
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
        
        success, message = fix_single_architecture(file_path)
        
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
    print("COMPREHENSIVE FIX SUMMARY")
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