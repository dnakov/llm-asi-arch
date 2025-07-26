#!/usr/bin/env python3
"""
Batch Architecture Syntax Fixer
===============================

Applies targeted fixes for the 5 most common syntax error patterns 
identified across all 106 MLX architecture files.
"""

import os
import re
from pathlib import Path
from typing import Dict, List

def fix_architecture_file(filepath: Path) -> Dict:
    """Apply all common syntax fixes to a single architecture file"""
    with open(filepath, 'r') as f:
        original_content = f.read()
    
    content = original_content
    fixes_applied = []
    
    # Fix 1: Type annotation syntax errors
    # Pattern: "tensor:, mx.array" -> "tensor: mx.array"
    pattern1 = r'(\w+):\s*,\s*(mx\.\w+)'
    if re.search(pattern1, content):
        content = re.sub(pattern1, r'\1: \2', content)
        fixes_applied.append("Fixed type annotation syntax")
    
    # Fix 2: kwargs.get missing commas
    # Pattern: "kwargs.get('h'\nkwargs.get('d', 1))" -> "kwargs.get('h', kwargs.get('d', 1))"
    pattern2 = r"kwargs\.get\s*\(\s*['\"]([^'\"]+)['\"]\s*\n\s*kwargs\.get\s*\(\s*['\"]([^'\"]+)['\"],\s*([^)]+)\)\)"
    if re.search(pattern2, content):
        content = re.sub(pattern2, r"kwargs.get('\1', kwargs.get('\2', \3))", content)
        fixes_applied.append("Fixed kwargs.get missing commas")
    
    # Fix 3: Function parameters missing commas
    # Fix __init__ and __call__ method parameters spread across lines
    # Pattern: parameters on separate lines without commas
    init_pattern = r'def __init__\(self,([^)]*?)(\n\s*\w+:\s*[^,\n)]+)(\n\s*\w+:\s*[^,\n)]+)*(\n\s*\w+:\s*[^,\n)]+)*\):'
    matches = list(re.finditer(init_pattern, content, re.MULTILINE | re.DOTALL))
    for match in reversed(matches):  # Process in reverse to maintain positions
        full_match = match.group(0)
        # Add commas to parameters that don't have them
        fixed_params = re.sub(r'(\w+:\s*[^,\n)]+)(\n\s*)(\w+:)', r'\1,\2\3', full_match)
        if fixed_params != full_match:
            content = content[:match.start()] + fixed_params + content[match.end():]
            fixes_applied.append("Fixed function parameter commas")
    
    # Fix 4: Conv1d and other function calls missing commas
    # Pattern: "nn.Conv1d(a, b, c\npadding=d\nbias=e)" -> "nn.Conv1d(a, b, c, padding=d, bias=e)"
    conv_pattern = r'nn\.Conv1d\s*\([^)]*?(\n\s*\w+\s*=\s*[^,\n)]+)(\n\s*\w+\s*=\s*[^,\n)]+)*\)'
    matches = list(re.finditer(conv_pattern, content, re.MULTILINE | re.DOTALL))
    for match in reversed(matches):
        full_match = match.group(0)
        # Add commas before parameters on new lines
        fixed_call = re.sub(r'([^,\s])(\n\s*)(\w+\s*=)', r'\1,\2\3', full_match)
        if fixed_call != full_match:
            content = content[:match.start()] + fixed_call + content[match.end():]
            fixes_applied.append("Fixed function call commas")
    
    # Fix 5: Unterminated string literals in assert statements
    # Pattern: 'assert condition "message\n more text"' -> 'assert condition, "message more text"'
    assert_pattern = r'assert\s+([^"\']+?)\s+"([^"]*?)"\s*\n\s*([^"]*?)"'
    matches = list(re.finditer(assert_pattern, content, re.MULTILINE | re.DOTALL))
    for match in reversed(matches):
        condition = match.group(1).strip()
        message_part1 = match.group(2)
        message_part2 = match.group(3)
        complete_message = f"{message_part1} {message_part2}".strip()
        fixed_assert = f'assert {condition}, "{complete_message}"'
        content = content[:match.start()] + fixed_assert + content[match.end():]
        fixes_applied.append("Fixed unterminated string in assert")
    
    # Fix 6: Simpler assert pattern
    # Pattern: 'assert condition "message' -> 'assert condition, "message"'
    simple_assert_pattern = r'assert\s+([^"\']+?)\s+"([^"]*?)$'
    matches = list(re.finditer(simple_assert_pattern, content, re.MULTILINE))
    for match in reversed(matches):
        condition = match.group(1).strip()
        message = match.group(2)
        fixed_assert = f'assert {condition}, "{message}"'
        content = content[:match.start()] + fixed_assert + content[match.end():]
        fixes_applied.append("Fixed assert statement syntax")
    
    # Fix 7: Standalone None statements
    # Pattern: "return something\nNone\nreturn other" -> "return something, None"
    none_pattern = r'(\s+return\s+[^,\n]+)\s*\n\s*None\s*#[^\n]*\n\s*return\s+'
    if re.search(none_pattern, content):
        content = re.sub(none_pattern, r'\1, None  # Simplified - no cache state\n        return ', content)
        fixes_applied.append("Fixed standalone None statements")
    
    # Fix 8: Missing commas in function calls with parameters spread across lines
    # Pattern: "function(param1\nparam2)" -> "function(param1, param2)"
    func_call_pattern = r'(\w+)\s*\(\s*([^,\n)]+)\s*\n\s*([^,\n)]+)\s*\)'
    matches = list(re.finditer(func_call_pattern, content, re.MULTILINE))
    for match in reversed(matches):
        func_name = match.group(1)
        param1 = match.group(2).strip()
        param2 = match.group(3).strip()
        fixed_call = f'{func_name}({param1}, {param2})'
        content = content[:match.start()] + fixed_call + content[match.end():]
        fixes_applied.append("Fixed function call missing commas")
    
    # Save the file if changes were made
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return {
            'success': True,
            'fixes_applied': fixes_applied,
            'fixes_count': len(fixes_applied)
        }
    else:
        return {
            'success': False,
            'fixes_applied': [],
            'fixes_count': 0
        }

def main():
    """Apply fixes to all architecture files"""
    mlx_dir = Path("mlx_architectures")
    
    if not mlx_dir.exists():
        print(f"❌ Directory {mlx_dir} not found!")
        return False
    
    arch_files = list(mlx_dir.glob("*_mlx.py"))
    if not arch_files:
        print(f"❌ No MLX architecture files found in {mlx_dir}")
        return False
    
    print(f"🔧 Applying batch fixes to {len(arch_files)} architecture files...")
    
    fixed_count = 0
    total_fixes = 0
    results = {}
    
    for i, filepath in enumerate(arch_files, 1):
        arch_name = filepath.stem.replace('_mlx', '')
        print(f"[{i:3d}/{len(arch_files)}] Processing {arch_name}...")
        
        try:
            result = fix_architecture_file(filepath)
            results[arch_name] = result
            
            if result['success']:
                fixed_count += 1
                total_fixes += result['fixes_count']
                print(f"  ✅ Applied {result['fixes_count']} fixes")
                for fix in result['fixes_applied']:
                    print(f"     - {fix}")
            else:
                print(f"  ℹ️  No fixes needed")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[arch_name] = {'success': False, 'error': str(e)}
    
    print(f"\n📊 Batch Fix Summary:")
    print(f"Files processed: {len(arch_files)}")
    print(f"Files modified: {fixed_count}")
    print(f"Total fixes applied: {total_fixes}")
    print(f"Success rate: {fixed_count/len(arch_files)*100:.1f}%")
    
    return fixed_count > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)