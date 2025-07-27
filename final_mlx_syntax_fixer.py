#!/usr/bin/env python3
"""
Final MLX Architecture Syntax Fixer
Systematically fixes all remaining syntax issues in the 106 MLX architectures
"""

import os
import re
import ast
import sys
from typing import List, Tuple, Dict
import traceback


class FinalMLXSyntaxFixer:
    """Final comprehensive syntax fixer for all remaining MLX architecture issues"""
    
    def __init__(self):
        self.fixes_applied = {}
    
    def fix_line_continuation_comma_issues(self, content: str) -> str:
        """Fix the specific line continuation comma issue in functions like _l2norm"""
        fixes_count = 0
        
        # Pattern: function call that spans lines missing comma after axis=-1
        # Looking for: axis=-1\n        keepdims=True
        pattern = r'(axis=-1)\s*\n\s+(keepdims=True)'
        
        def fix_comma(match):
            axis_part = match.group(1)
            keepdims_part = match.group(2)
            return f'{axis_part},\n        {keepdims_part}'
        
        original_content = content
        content = re.sub(pattern, fix_comma, content)
        
        if content != original_content:
            fixes_count = len(re.findall(pattern, original_content))
            self.fixes_applied['line_continuation_commas'] = fixes_count
        
        return content
    
    def fix_common_syntax_errors(self, content: str) -> str:
        """Fix common syntax errors systematically"""
        fixes_count = 0
        
        # Fix missing commas in function parameters that span lines
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix cases where function parameters are split without proper comma
            if (i > 0 and 
                lines[i-1].strip().endswith('(') and
                line.strip() and
                not line.strip().startswith(('#', '"""', "'''")) and
                not line.strip().endswith(',') and
                i + 1 < len(lines) and
                lines[i+1].strip() and
                not lines[i+1].strip().startswith(')')):
                # Add comma if missing
                if not line.rstrip().endswith(','):
                    line = line.rstrip() + ','
                    fixes_count += 1
            
            # Fix missing commas before parameters that start on new lines
            if (line.strip() and 
                not line.strip().startswith(('#', '"""', "'''", ')', ']', '}')) and
                '=' in line and
                i > 0 and
                lines[i-1].strip() and
                not lines[i-1].strip().endswith(',') and
                not lines[i-1].strip().endswith('(') and
                not lines[i-1].strip().endswith('[') and
                not lines[i-1].strip().endswith('{')):
                # Check if previous line needs a comma
                prev_line = lines[i-1]
                if (not prev_line.strip().endswith(',') and 
                    not prev_line.strip().endswith(':') and
                    not prev_line.strip().endswith('=') and
                    not prev_line.strip().endswith('(')):
                    lines[i-1] = prev_line.rstrip() + ','
                    fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['syntax_commas'] = fixes_count
        
        return '\n'.join(fixed_lines)
    
    def fix_bracket_mismatches(self, content: str) -> str:
        """Fix bracket and parenthesis mismatches"""
        fixes_count = 0
        
        # Fix obvious mismatches
        original_content = content
        
        # Fix ) where ] should be
        content = re.sub(r'(\w+\s*\[.*?)\)(\s*(?:\.|$))', r'\1]\2', content)
        
        # Fix ] where ) should be
        content = re.sub(r'(\w+\s*\(.*?)\](\s*(?:\.|$))', r'\1)\2', content)
        
        # Count fixes
        if content != original_content:
            fixes_count += 1
            self.fixes_applied['bracket_fixes'] = fixes_count
        
        return content
    
    def fix_string_literals(self, content: str) -> str:
        """Fix unterminated string literals"""
        fixes_count = 0
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            original_line = line
            
            # Fix unterminated strings
            if (line.count('"') % 2 == 1 and 
                not line.strip().startswith('#') and
                len(line.strip()) < 200):  # Only fix shorter lines
                line = line + '"'
                fixes_count += 1
            elif (line.count("'") % 2 == 1 and 
                  not line.strip().startswith('#') and
                  len(line.strip()) < 200):
                line = line + "'"
                fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['string_fixes'] = fixes_count
        
        return '\n'.join(fixed_lines)
    
    def fix_decimal_literals(self, content: str) -> str:
        """Fix invalid decimal literals"""
        fixes_count = 0
        
        # Fix patterns like 1.e-5 -> 1.0e-5
        pattern = r'(\d+)\.([eE][+-]?\d+)'
        
        def fix_decimal(match):
            number = match.group(1)
            exponent = match.group(2)
            return f'{number}.0{exponent}'
        
        original_content = content
        content = re.sub(pattern, fix_decimal, content)
        
        if content != original_content:
            fixes_count = len(re.findall(pattern, original_content))
            self.fixes_applied['decimal_fixes'] = fixes_count
        
        return content
    
    def fix_indentation_issues(self, content: str) -> str:
        """Fix indentation issues"""
        fixes_count = 0
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Fix unexpected indentation after simple statements
            if (i > 0 and 
                lines[i-1].strip() and
                not lines[i-1].strip().endswith(':') and
                not lines[i-1].strip().endswith('\\') and
                line.startswith('        ') and  # 8 spaces
                not line.strip().startswith(('"""', "'''", '#'))):
                # Check if this line should be dedented
                if (not any(keyword in lines[i-1] for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with '])):
                    # Remove extra indentation
                    line = line[4:]  # Remove 4 spaces
                    fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['indentation_fixes'] = fixes_count
        
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
        """Fix a single file comprehensively"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already valid
            is_valid, error_msg = self.validate_syntax(content)
            if is_valid:
                return True, "Already valid", {}
            
            # Reset fixes counter
            self.fixes_applied = {}
            
            # Apply fixes in careful order
            content = self.fix_line_continuation_comma_issues(content)
            content = self.fix_common_syntax_errors(content)
            content = self.fix_bracket_mismatches(content)
            content = self.fix_string_literals(content)
            content = self.fix_decimal_literals(content)
            content = self.fix_indentation_issues(content)
            
            # Validate after fixes
            is_valid, new_error = self.validate_syntax(content)
            
            if is_valid:
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed successfully", self.fixes_applied.copy()
            else:
                return False, f"Still invalid: {new_error}", self.fixes_applied.copy()
                
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
            
            print(f"[{i:3d}/106] {arch_name}...")
            
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
                    print(f"  🔧 {message} ({fixes_str})")
            else:
                results["failed"] += 1
                print(f"  ❌ {message}")
        
        return results


def main():
    """Main execution function"""
    print("🔧 Final MLX Architecture Syntax Fixer")
    print("=" * 60)
    
    fixer = FinalMLXSyntaxFixer()
    results = fixer.fix_all_architectures()
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL FIX SUMMARY")
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
        for detail in failed_fixes[:10]:
            print(f"   {detail['architecture']}: {detail['message']}")
    
    # Save results
    import json
    with open('final_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: final_fix_results.json")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())