#!/usr/bin/env python3
"""
Comprehensive MLX Architecture Syntax Fixer
Fixes all 106 PyTorch to MLX converted architectures to achieve 100% valid syntax
"""

import os
import re
import ast
import sys
from typing import List, Tuple, Dict, Set
import traceback


class MLXSyntaxFixer:
    """Comprehensive syntax fixer for MLX architectures"""
    
    def __init__(self):
        self.fixes_applied = {}
        self.error_patterns = {
            'type_annotation': r'(\w+):\s*,\s*(mx\.array)',
            'unmatched_paren': [
                # Common patterns for mismatched delimiters
                r'\(\s*\]',  # ( followed by ]
                r'\[\s*\)',  # [ followed by )
                r'\(\s*$',   # unclosed ( at end of line
                r'\[\s*$',   # unclosed [ at end of line
            ],
            'incomplete_string': r'["\'][^"\']*$',  # unterminated strings
            'invalid_decimal': r'(\d+)\.(\w+)',     # invalid decimal like 1.e-3
            'missing_comma': r'(\w+)\s+(\w+)\s*=',  # missing comma in function args
        }
    
    def fix_type_annotations(self, content: str) -> str:
        """Fix invalid type annotation syntax like 'tensor:, mx.array'"""
        pattern = r'(\w+):\s*,\s*(mx\.array)'
        
        def replace_annotation(match):
            var_name = match.group(1)
            type_hint = match.group(2)
            return f'{var_name}: {type_hint}'
        
        fixed = re.sub(pattern, replace_annotation, content)
        if fixed != content:
            self.fixes_applied.setdefault('type_annotations', 0)
            self.fixes_applied['type_annotations'] += content.count(':,')
        
        return fixed
    
    def fix_unmatched_delimiters(self, content: str) -> str:
        """Fix unmatched parentheses and brackets"""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix common delimiter mismatches
            line = re.sub(r'\(\s*\]', '()', line)  # ( followed by ]
            line = re.sub(r'\[\s*\)', '[]', line)  # [ followed by )
            
            # Balance parentheses and brackets
            line = self._balance_delimiters(line)
            
            if line != original_line:
                self.fixes_applied.setdefault('delimiters', 0)
                self.fixes_applied['delimiters'] += 1
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _balance_delimiters(self, line: str) -> str:
        """Balance parentheses and brackets in a line"""
        # Count delimiters
        open_paren = line.count('(')
        close_paren = line.count(')')
        open_bracket = line.count('[')
        close_bracket = line.count(']')
        
        # Fix unbalanced parentheses
        if open_paren > close_paren:
            line += ')' * (open_paren - close_paren)
        elif close_paren > open_paren:
            # Remove extra closing parens
            for _ in range(close_paren - open_paren):
                line = line.rstrip().rstrip(')')
        
        # Fix unbalanced brackets
        if open_bracket > close_bracket:
            line += ']' * (open_bracket - close_bracket)
        elif close_bracket > open_bracket:
            # Remove extra closing brackets
            for _ in range(close_bracket - open_bracket):
                line = line.rstrip().rstrip(']')
        
        return line
    
    def fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated string literals"""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Check for unterminated strings
            if self._has_unterminated_string(line):
                # Try to fix by adding closing quote
                if line.count('"') % 2 == 1:
                    line += '"'
                elif line.count("'") % 2 == 1:
                    line += "'"
                
                self.fixes_applied.setdefault('strings', 0)
                self.fixes_applied['strings'] += 1
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _has_unterminated_string(self, line: str) -> bool:
        """Check if line has unterminated string"""
        # Simple check for odd number of quotes
        return (line.count('"') % 2 == 1) or (line.count("'") % 2 == 1)
    
    def fix_invalid_decimals(self, content: str) -> str:
        """Fix invalid decimal literals"""
        # Pattern for invalid decimals like 1.e-3
        pattern = r'(\d+)\.([a-zA-Z])'
        
        def fix_decimal(match):
            num = match.group(1)
            suffix = match.group(2)
            if suffix.lower() == 'e':
                return f'{num}.0e'
            return f'{num}.0{suffix}'
        
        fixed = re.sub(pattern, fix_decimal, content)
        if fixed != content:
            self.fixes_applied.setdefault('decimals', 0)
            self.fixes_applied['decimals'] += 1
        
        return fixed
    
    def fix_missing_commas(self, content: str) -> str:
        """Fix missing commas in function parameters"""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            original_line = line
            
            # Pattern: parameter_name parameter_name = value (missing comma)
            pattern = r'(\w+)\s+(\w+)\s*='
            if re.search(pattern, line) and 'def ' not in line:
                line = re.sub(pattern, r'\1, \2=', line)
                
                if line != original_line:
                    self.fixes_applied.setdefault('commas', 0)
                    self.fixes_applied['commas'] += 1
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_common_mlx_issues(self, content: str) -> str:
        """Fix common MLX-specific issues"""
        # Fix torch imports that weren't converted
        content = re.sub(r'import torch\b', 'import mlx.core as mx', content)
        content = re.sub(r'from torch', 'from mlx.core', content)
        content = re.sub(r'torch\.', 'mx.', content)
        
        # Fix tensor creation
        content = re.sub(r'\.to\(device\)', '', content)  # Remove .to(device)
        content = re.sub(r'\.cuda\(\)', '', content)      # Remove .cuda()
        content = re.sub(r'device=device[,\s]*', '', content)  # Remove device= args
        
        # Fix activation functions
        content = re.sub(r'torch\.nn\.functional\.', 'mx.', content)
        content = re.sub(r'F\.', 'mx.', content)
        
        return content
    
    def fix_line_continuations(self, content: str) -> str:
        """Fix broken line continuations and indentation"""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if line ends with comma and next line is indented
            if (line.rstrip().endswith(',') and 
                i + 1 < len(lines) and 
                lines[i + 1].strip() and
                not lines[i + 1].startswith(' ')):
                # Next line should be indented
                lines[i + 1] = '    ' + lines[i + 1].strip()
            
            # Fix broken function calls across lines
            if (line.rstrip().endswith('(') and 
                i + 1 < len(lines) and
                lines[i + 1].strip()):
                # Ensure proper indentation for continuation
                next_line = lines[i + 1]
                if not next_line.startswith('    '):
                    lines[i + 1] = '    ' + next_line.lstrip()
            
            fixed_lines.append(lines[i])
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def validate_syntax(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax"""
        try:
            compile(content, '<string>', 'exec')
            return True, "Syntax OK"
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def fix_file(self, file_path: str) -> Tuple[bool, str, Dict]:
        """Fix a single MLX architecture file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_valid, original_error = self.validate_syntax(content)
            if original_valid:
                return True, "Already valid", {}
            
            # Reset fixes counter for this file
            self.fixes_applied = {}
            
            # Apply all fixes in sequence
            content = self.fix_type_annotations(content)
            content = self.fix_common_mlx_issues(content)
            content = self.fix_unterminated_strings(content)
            content = self.fix_invalid_decimals(content)
            content = self.fix_missing_commas(content)
            content = self.fix_line_continuations(content)
            content = self.fix_unmatched_delimiters(content)
            
            # Validate fixed content
            is_valid, error_msg = self.validate_syntax(content)
            
            if is_valid:
                # Write fixed content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed successfully", self.fixes_applied.copy()
            else:
                return False, f"Still invalid after fixes: {error_msg}", self.fixes_applied.copy()
                
        except Exception as e:
            return False, f"Exception during fix: {str(e)}", {}
    
    def fix_all_architectures(self, mlx_dir: str = "mlx_architectures") -> Dict:
        """Fix all MLX architecture files"""
        if not os.path.exists(mlx_dir):
            return {"error": f"Directory {mlx_dir} not found"}
        
        results = {
            "total": 0,
            "fixed": 0,
            "already_valid": 0,
            "failed": 0,
            "details": []
        }
        
        files = [f for f in os.listdir(mlx_dir) if f.endswith('_mlx.py')]
        files.sort()
        
        for i, filename in enumerate(files, 1):
            file_path = os.path.join(mlx_dir, filename)
            arch_name = filename.replace('_mlx.py', '')
            
            print(f"[{i:3d}/106] Fixing {arch_name}...")
            
            success, message, fixes = self.fix_file(file_path)
            
            result = {
                "architecture": arch_name,
                "success": success,
                "message": message,
                "fixes_applied": fixes
            }
            
            results["details"].append(result)
            results["total"] += 1
            
            if success:
                if "Already valid" in message:
                    results["already_valid"] += 1
                    print(f"  ✅ {message}")
                else:
                    results["fixed"] += 1
                    fixes_str = ", ".join([f"{k}:{v}" for k, v in fixes.items()])
                    print(f"  🔧 {message} - Fixes: {fixes_str}")
            else:
                results["failed"] += 1
                print(f"  ❌ {message}")
        
        return results


def main():
    """Main execution function"""
    print("🔧 MLX Architecture Comprehensive Syntax Fixer")
    print("=" * 60)
    
    fixer = MLXSyntaxFixer()
    results = fixer.fix_all_architectures()
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE FIX SUMMARY")
    print("=" * 60)
    
    print(f"📁 Total architectures: {results['total']}")
    print(f"✅ Already valid: {results['already_valid']}")
    print(f"🔧 Successfully fixed: {results['fixed']}")
    print(f"❌ Failed to fix: {results['failed']}")
    
    success_rate = (results['already_valid'] + results['fixed']) / results['total'] * 100
    print(f"📊 Success rate: {success_rate:.1f}%")
    
    # Show failed fixes
    failed_fixes = [d for d in results['details'] if not d['success']]
    if failed_fixes:
        print(f"\n🔍 Failed fixes ({len(failed_fixes)}):")
        for detail in failed_fixes[:10]:  # Show first 10
            print(f"   {detail['architecture']}: {detail['message']}")
    
    # Save detailed results
    import json
    with open('comprehensive_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: comprehensive_fix_results.json")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())