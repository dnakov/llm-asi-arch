#!/usr/bin/env python3
"""
Comprehensive Architecture Syntax Fixer
=======================================

Automatically fixes all common syntax errors across all 106 MLX architecture files.
Based on analysis of current error patterns, this script applies systematic fixes.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

class ArchitectureSyntaxFixer:
    """Fixes common syntax errors in MLX architecture files"""
    
    def __init__(self, mlx_dir: str = "mlx_architectures"):
        self.mlx_dir = Path(mlx_dir)
        self.fixes_applied = {}
        
    def fix_file(self, filepath: Path) -> Dict[str, any]:
        """Fix all syntax issues in a single architecture file"""
        with open(filepath, 'r') as f:
            original_code = f.read()
        
        code = original_code
        fixes_applied = []
        
        # Fix 1: Type annotation syntax errors
        # Pattern: tensor:, mx.array -> tensor: mx.array
        pattern1 = r'(\w+):\s*,\s*(mx\.\w+)'
        if re.search(pattern1, code):
            code = re.sub(pattern1, r'\1: \2', code)
            fixes_applied.append("Fixed type annotation syntax")
        
        # Fix 2: Missing commas in kwargs.get calls 
        # Pattern: kwargs.get('h' kwargs.get -> kwargs.get('h', kwargs.get
        pattern2 = r"kwargs\.get\s*\(\s*['\"]([^'\"]+)['\"][\s\n]+kwargs\.get"
        if re.search(pattern2, code):
            code = re.sub(pattern2, r"kwargs.get('\1', kwargs.get", code)
            fixes_applied.append("Fixed kwargs.get missing commas")
        
        # Fix 3: More complex kwargs.get patterns
        # Pattern: kwargs.get('h'\nkwargs.get('d', 1))
        pattern3 = r"kwargs\.get\s*\(\s*['\"]([^'\"]+)['\"][\s\n]+kwargs\.get\s*\(\s*['\"]([^'\"]+)['\"],\s*([^)]+)\)\)"
        if re.search(pattern3, code):
            code = re.sub(pattern3, r"kwargs.get('\1', kwargs.get('\2', \3))", code)
            fixes_applied.append("Fixed complex kwargs.get patterns")
        
        # Fix 4: Unterminated string literals in assert statements
        # Pattern: assert condition "message -> assert condition, "message"
        pattern4 = r'assert\s+([^"\']+)\s+"([^"]*)"([^"]*$)'
        matches = re.finditer(pattern4, code, re.MULTILINE)
        for match in matches:
            condition = match.group(1).strip()
            message_start = match.group(2)
            remainder = match.group(3)
            
            # Find the intended end of the string
            if '\n' in remainder and not remainder.strip().endswith('"'):
                # Look for the next line that might complete the string
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if match.group(0).split('\n')[0] in line:
                        # Check next few lines for string completion
                        for j in range(i+1, min(i+3, len(lines))):
                            if lines[j].strip().endswith('"') or '"' in lines[j]:
                                # Reconstruct the proper assert
                                message_parts = [message_start]
                                for k in range(i+1, j+1):
                                    message_parts.append(lines[k].strip().rstrip('"'))
                                complete_message = ' '.join(message_parts).strip()
                                
                                new_assert = f'assert {condition}, "{complete_message}"'
                                code = code.replace(match.group(0), new_assert)
                                fixes_applied.append("Fixed unterminated string in assert")
                                break
                        break
        
        # Fix 5: Missing commas in function parameters
        # Pattern: def func(self param1 param2) -> def func(self, param1, param2)
        pattern5 = r'def\s+(\w+)\s*\(\s*self\s+([^,)]+)'
        if re.search(pattern5, code):
            code = re.sub(pattern5, r'def \1(self, \2', code)
            fixes_applied.append("Fixed function parameter missing commas")
        
        # Fix 6: Missing commas in function calls
        # Pattern: func(arg1 arg2, arg3) -> func(arg1, arg2, arg3)
        # This is tricky - let's be conservative and fix specific known patterns
        
        # Fix nn.Linear calls
        pattern6a = r'nn\.Linear\s*\(\s*([^,\s]+)\s+([^,\s]+)\s*([,)])'
        if re.search(pattern6a, code):
            code = re.sub(pattern6a, r'nn.Linear(\1, \2\3', code)
            fixes_applied.append("Fixed nn.Linear calls")
        
        # Fix F.elu calls  
        pattern6b = r'F\.elu\s*\(\s*([^,\s]+)\s+([^,\s]+)\s*([,)])'
        if re.search(pattern6b, code):
            code = re.sub(pattern6b, r'F.elu(\1, \2\3', code)
            fixes_applied.append("Fixed F.elu calls")
        
        # Fix 7: Unmatched parentheses - detect and attempt to fix
        # Count parentheses to find imbalances
        paren_balance = 0
        bracket_balance = 0
        
        for char in code:
            if char == '(':
                paren_balance += 1
            elif char == ')':
                paren_balance -= 1
            elif char == '[':
                bracket_balance += 1
            elif char == ']':
                bracket_balance -= 1
        
        # If we have imbalances, try to fix common patterns
        if paren_balance != 0 or bracket_balance != 0:
            # Look for common patterns of missing closing parentheses
            # Pattern: function(args without closing )
            lines = code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Count parens in this line
                line_paren_balance = line.count('(') - line.count(')')
                line_bracket_balance = line.count('[') - line.count(']')
                
                # If line has unmatched opening parens/brackets, try to fix
                if line_paren_balance > 0 and not line.rstrip().endswith(')'):
                    if ('=' in line and '(' in line) or ('return' in line and '(' in line):
                        # Likely a function call or assignment that needs closing paren
                        line = line.rstrip() + ')'
                        fixes_applied.append("Fixed unmatched parentheses")
                
                if line_bracket_balance > 0 and not line.rstrip().endswith(']'):
                    if '[' in line and ('=' in line or 'return' in line):
                        # Likely an array access that needs closing bracket
                        line = line.rstrip() + ']'
                        fixes_applied.append("Fixed unmatched brackets")
                
                fixed_lines.append(line)
            
            code = '\n'.join(fixed_lines)
        
        # Fix 8: Invalid syntax patterns
        # Fix kwargs.get calls with missing quotes or commas
        pattern8 = r'kwargs\.get\s*\(\s*([^,\s\'"]+)\s+([^,)]+)\)'
        if re.search(pattern8, code):
            code = re.sub(pattern8, r"kwargs.get('\1', \2)", code)
            fixes_applied.append("Fixed kwargs.get syntax")
        
        # Fix 9: Missing commas in assert statements
        pattern9 = r'assert\s+([^,]+)\s+"([^"]+)"'
        if re.search(pattern9, code):
            code = re.sub(pattern9, r'assert \1, "\2"', code)
            fixes_applied.append("Fixed assert statement syntax")
        
        # Fix 10: Function parameter issues
        # Fix missing commas between parameters like "self param1"
        pattern10 = r'\(\s*self\s+(\w+[^,)]*)\)'
        if re.search(pattern10, code):
            code = re.sub(pattern10, r'(self, \1)', code)
            fixes_applied.append("Fixed function parameter syntax")
        
        # Save the fixed code if changes were made
        if code != original_code:
            with open(filepath, 'w') as f:
                f.write(code)
            
            return {
                'fixed': True,
                'fixes_applied': fixes_applied,
                'fixes_count': len(fixes_applied)
            }
        else:
            return {
                'fixed': False,
                'fixes_applied': [],
                'fixes_count': 0
            }
    
    def fix_all_architectures(self) -> Dict[str, any]:
        """Fix all architecture files in the MLX directory"""
        if not self.mlx_dir.exists():
            return {'error': f"Directory {self.mlx_dir} does not exist"}
        
        architecture_files = list(self.mlx_dir.glob("*_mlx.py"))
        
        if not architecture_files:
            return {'error': f"No MLX architecture files found in {self.mlx_dir}"}
        
        print(f"🔧 Found {len(architecture_files)} architecture files to fix")
        
        results = {
            'total_files': len(architecture_files),
            'files_fixed': 0,
            'files_unchanged': 0,
            'total_fixes': 0,
            'per_file_results': {}
        }
        
        for i, filepath in enumerate(architecture_files, 1):
            arch_name = filepath.stem.replace('_mlx', '')
            print(f"[{i:3d}/{len(architecture_files)}] Fixing {arch_name}...")
            
            try:
                file_result = self.fix_file(filepath)
                results['per_file_results'][arch_name] = file_result
                
                if file_result['fixed']:
                    results['files_fixed'] += 1
                    results['total_fixes'] += file_result['fixes_count']
                    print(f"  ✅ Applied {file_result['fixes_count']} fixes:")
                    for fix in file_result['fixes_applied']:
                        print(f"     - {fix}")
                else:
                    results['files_unchanged'] += 1
                    print(f"  ℹ️  No fixes needed")
                    
            except Exception as e:
                print(f"  ❌ Error fixing {arch_name}: {e}")
                results['per_file_results'][arch_name] = {
                    'fixed': False,
                    'error': str(e)
                }
        
        # Save results
        with open('architecture_fix_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Fix Summary:")
        print(f"Total files: {results['total_files']}")
        print(f"Files fixed: {results['files_fixed']}")
        print(f"Files unchanged: {results['files_unchanged']}")
        print(f"Total fixes applied: {results['total_fixes']}")
        print(f"Results saved to: architecture_fix_results.json")
        
        return results

def main():
    """Run the architecture syntax fixer"""
    print("🔧 Architecture Syntax Fixer")
    print("=" * 50)
    
    fixer = ArchitectureSyntaxFixer()
    results = fixer.fix_all_architectures()
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return False
    
    # Check if we made significant progress
    success_rate = results['files_fixed'] / results['total_files'] * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}% of files modified")
    
    return success_rate > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)