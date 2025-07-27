#!/usr/bin/env python3
"""
Targeted MLX Architecture Syntax Fixer
Fixes specific syntax patterns found in the 106 MLX architectures
"""

import os
import re
import ast
import sys
from typing import List, Tuple, Dict
import traceback


class TargetedMLXFixer:
    """Targeted syntax fixer focusing on specific error patterns"""
    
    def __init__(self):
        self.fixes_applied = {}
    
    def fix_type_annotations(self, content: str) -> str:
        """Fix the specific type annotation error: tensor:, mx.array -> tensor: mx.array"""
        # Pattern: variable_name:, type_hint
        pattern = r'(\w+):\s*,\s+(mx\.array|mx\.Array)'
        
        def fix_annotation(match):
            var_name = match.group(1)
            type_hint = match.group(2)
            return f'{var_name}: {type_hint}'
        
        original_content = content
        content = re.sub(pattern, fix_annotation, content)
        
        if content != original_content:
            count = len(re.findall(pattern, original_content))
            self.fixes_applied['type_annotations'] = count
            print(f"    Fixed {count} type annotation errors")
        
        return content
    
    def fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses and brackets"""
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Count opening and closing delimiters
            open_paren = line.count('(')
            close_paren = line.count(')')
            open_bracket = line.count('[')
            close_bracket = line.count(']')
            
            # Fix simple mismatches
            if open_paren > close_paren:
                missing_close = open_paren - close_paren
                # Add missing closing parentheses at the end
                line = line.rstrip() + ')' * missing_close
                fixes_count += missing_close
            elif close_paren > open_paren:
                # Remove extra closing parentheses
                extra_close = close_paren - open_paren
                for _ in range(extra_close):
                    line = re.sub(r'\)(\s*)$', r'\1', line)
                fixes_count += extra_close
            
            if open_bracket > close_bracket:
                missing_close = open_bracket - close_bracket
                line = line.rstrip() + ']' * missing_close
                fixes_count += missing_close
            elif close_bracket > open_bracket:
                extra_close = close_bracket - open_bracket
                for _ in range(extra_close):
                    line = re.sub(r'\](\s*)$', r'\1', line)
                fixes_count += extra_close
            
            # Fix bracket/paren mismatches
            line = re.sub(r'\(\s*\]', ')', line)  # (] -> )
            line = re.sub(r'\[\s*\)', ']', line)  # [) -> ]
            
            if line != original_line:
                fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['delimiter_fixes'] = fixes_count
            print(f"    Fixed {fixes_count} delimiter issues")
        
        return '\n'.join(fixed_lines)
    
    def fix_string_literals(self, content: str) -> str:
        """Fix unterminated string literals"""
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        in_multiline_string = False
        quote_char = None
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Handle multiline strings
            if in_multiline_string:
                if quote_char * 3 in line:
                    in_multiline_string = False
                    quote_char = None
                fixed_lines.append(line)
                continue
            
            # Check for start of multiline string
            if '"""' in line or "'''" in line:
                if line.count('"""') % 2 == 1:
                    in_multiline_string = True
                    quote_char = '"'
                elif line.count("'''") % 2 == 1:
                    in_multiline_string = True
                    quote_char = "'"
                fixed_lines.append(line)
                continue
            
            # Fix single-line string issues
            stripped = line.strip()
            if stripped.startswith('"') and not stripped.endswith('"') and stripped.count('"') == 1:
                line = line + '"'
                fixes_count += 1
            elif stripped.startswith("'") and not stripped.endswith("'") and stripped.count("'") == 1:
                line = line + "'"
                fixes_count += 1
            
            # Fix strings that got broken across lines
            if (stripped.endswith('",') or stripped.endswith("',")) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith(('"""', "'''", '"', "'")):
                    # This might be a continuation - check if it looks like broken string
                    pass  # For now, just keep as is
            
            if line != original_line:
                fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['string_fixes'] = fixes_count
            print(f"    Fixed {fixes_count} string literal issues")
        
        return '\n'.join(fixed_lines)
    
    def fix_decimal_literals(self, content: str) -> str:
        """Fix invalid decimal literals like 1.e-5"""
        pattern = r'(\d+)\.([eE][+-]?\d+)'
        
        def fix_decimal(match):
            number = match.group(1)
            exponent = match.group(2)
            return f'{number}.0{exponent}'
        
        original_content = content
        content = re.sub(pattern, fix_decimal, content)
        
        if content != original_content:
            count = len(re.findall(pattern, original_content))
            self.fixes_applied['decimal_fixes'] = count
            print(f"    Fixed {count} decimal literal issues")
        
        return content
    
    def fix_import_issues(self, content: str) -> str:
        """Fix remaining PyTorch imports and related issues"""
        fixes_count = 0
        original_content = content
        
        # Fix torch imports
        if 'import torch' in content:
            content = re.sub(r'import torch\b', 'import mlx.core as mx', content)
            fixes_count += 1
        
        # Fix F. references (usually from torch.nn.functional)
        if 'F.' in content:
            content = re.sub(r'\bF\.', 'mx.', content)
            fixes_count += content.count('F.') - content.count('mx.')
        
        # Remove device-related code
        content = re.sub(r'\.to\(device\)', '', content)
        content = re.sub(r'\.cuda\(\)', '', content)
        content = re.sub(r',\s*device=device', '', content)
        content = re.sub(r'device=device,\s*', '', content)
        
        if content != original_content:
            self.fixes_applied['import_fixes'] = fixes_count
            if fixes_count > 0:
                print(f"    Fixed {fixes_count} import/device issues")
        
        return content
    
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
            
            # Check if already valid
            is_valid, error_msg = self.validate_syntax(content)
            if is_valid:
                return True, "Already valid", {}
            
            print(f"    Original error: {error_msg}")
            
            # Reset fixes counter
            self.fixes_applied = {}
            
            # Apply fixes in sequence
            content = self.fix_type_annotations(content)
            content = self.fix_import_issues(content)
            content = self.fix_string_literals(content)
            content = self.fix_decimal_literals(content)
            content = self.fix_unmatched_parentheses(content)
            
            # Validate after fixes
            is_valid, error_msg = self.validate_syntax(content)
            
            if is_valid:
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed successfully", self.fixes_applied.copy()
            else:
                return False, f"Still invalid: {error_msg}", self.fixes_applied.copy()
                
        except Exception as e:
            return False, f"Exception: {str(e)}", {}
    
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
        
        files = sorted([f for f in os.listdir(mlx_dir) if f.endswith('_mlx.py')])
        
        for i, filename in enumerate(files, 1):
            file_path = os.path.join(mlx_dir, filename)
            arch_name = filename.replace('_mlx.py', '')
            
            print(f"\n[{i:3d}/106] Fixing {arch_name}...")
            
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
                    print(f"  🔧 {message}")
            else:
                results["failed"] += 1
                print(f"  ❌ {message}")
        
        return results


def main():
    """Main execution function"""
    print("🎯 Targeted MLX Architecture Syntax Fixer")
    print("=" * 60)
    
    fixer = TargetedMLXFixer()
    results = fixer.fix_all_architectures()
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("TARGETED FIX SUMMARY")
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
        for detail in failed_fixes[:5]:
            print(f"   {detail['architecture']}: {detail['message']}")
    
    # Save results
    import json
    with open('targeted_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: targeted_fix_results.json")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())