#!/usr/bin/env python3
"""
Surgical MLX Architecture Syntax Fixer
Precisely targets and fixes specific syntax errors without over-correcting
"""

import os
import re
import ast
import sys
from typing import List, Tuple, Dict
import traceback


class SurgicalMLXFixer:
    """Surgical syntax fixer that addresses specific patterns without breaking working code"""
    
    def __init__(self):
        self.fixes_applied = {}
    
    def fix_type_annotations_precisely(self, content: str) -> str:
        """Fix ONLY the specific type annotation error: variable:, type -> variable: type"""
        # Very specific pattern to avoid breaking other code
        pattern = r'(\b\w+):\s*,\s+(mx\.array|mx\.Array)\b'
        
        def fix_annotation(match):
            var_name = match.group(1)
            type_hint = match.group(2)
            return f'{var_name}: {type_hint}'
        
        original_content = content
        content = re.sub(pattern, fix_annotation, content)
        
        if content != original_content:
            count = len(re.findall(pattern, original_content))
            self.fixes_applied['type_annotations'] = count
            
        return content
    
    def fix_line_continuation_issues(self, content: str) -> str:
        """Fix line continuation issues that break syntax"""
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for line continuation issues
            if (line.strip().endswith('(') and 
                i + 1 < len(lines) and 
                lines[i + 1].strip() and
                not lines[i + 1].startswith(' ')):
                # Next line after ( should be indented
                next_line = lines[i + 1]
                if not next_line.startswith('    '):
                    lines[i + 1] = '    ' + next_line.lstrip()
                    fixes_count += 1
            
            # Fix cases where function call spans lines incorrectly
            if (line.rstrip().endswith(',') and 
                i + 1 < len(lines) and
                lines[i + 1].strip() and
                len(lines[i + 1]) - len(lines[i + 1].lstrip()) < 4):
                # Ensure proper indentation
                next_line = lines[i + 1]
                base_indent = len(line) - len(line.lstrip())
                lines[i + 1] = ' ' * (base_indent + 4) + next_line.strip()
                fixes_count += 1
            
            fixed_lines.append(lines[i])
            i += 1
        
        if fixes_count > 0:
            self.fixes_applied['line_continuation'] = fixes_count
        
        return '\n'.join(fixed_lines)
    
    def fix_specific_delimiter_errors(self, content: str) -> str:
        """Fix only specific delimiter errors that are clearly wrong"""
        fixes_count = 0
        
        # Fix obvious bracket/paren mismatches  
        content = re.sub(r'\(\s*\]', ')', content)
        if '()' in content:
            fixes_count += 1
        
        content = re.sub(r'\[\s*\)', ']', content)
        if '[]' in content:
            fixes_count += 1
        
        # Don't automatically balance - only fix obvious errors
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            original_line = line
            
            # Only fix lines that are clearly malformed
            # Example: function(...] should be function(...)
            if re.search(r'\([^)]*\]$', line.strip()):
                line = re.sub(r'\]$', ')', line)
                fixes_count += 1
            
            # Example: array[...] should be array[...]
            if re.search(r'\[[^]]*\)$', line.strip()):
                line = re.sub(r'\)$', ']', line)
                fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['delimiter_fixes'] = fixes_count
        
        return '\n'.join(fixed_lines)
    
    def fix_obvious_syntax_errors(self, content: str) -> str:
        """Fix only obvious syntax errors"""
        fixes_count = 0
        
        # Fix decimal literals like 1.e-5 -> 1.0e-5
        pattern = r'(\d+)\.([eE][+-]?\d+)'
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1.0\2', content)
            fixes_count += 1
        
        # Fix missing quotes in obvious cases
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Only fix obviously broken strings
            if (line.strip().startswith('"') and 
                not line.strip().endswith('"') and 
                line.count('"') == 1 and
                len(line.strip()) < 100):  # Short lines only
                line = line + '"'
                fixes_count += 1
            elif (line.strip().startswith("'") and 
                  not line.strip().endswith("'") and 
                  line.count("'") == 1 and
                  len(line.strip()) < 100):
                line = line + "'"
                fixes_count += 1
            
            fixed_lines.append(line)
        
        if fixes_count > 0:
            self.fixes_applied['syntax_fixes'] = fixes_count
        
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
        """Fix a single file with surgical precision"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already valid
            is_valid, error_msg = self.validate_syntax(content)
            if is_valid:
                return True, "Already valid", {}
            
            # Reset fixes counter
            self.fixes_applied = {}
            
            # Apply ONLY precise fixes
            content = self.fix_type_annotations_precisely(content)
            
            # Check if type annotation fix resolved the issue
            is_valid, new_error = self.validate_syntax(content)
            if is_valid:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed type annotations", self.fixes_applied.copy()
            
            # If still invalid, try other precise fixes
            content = self.fix_line_continuation_issues(content)
            is_valid, new_error = self.validate_syntax(content)
            if is_valid:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed line continuations", self.fixes_applied.copy()
            
            content = self.fix_specific_delimiter_errors(content)
            is_valid, new_error = self.validate_syntax(content)
            if is_valid:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed delimiters", self.fixes_applied.copy()
            
            content = self.fix_obvious_syntax_errors(content)
            is_valid, new_error = self.validate_syntax(content)
            if is_valid:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Fixed syntax errors", self.fixes_applied.copy()
            
            return False, f"Still invalid: {new_error}", self.fixes_applied.copy()
                
        except Exception as e:
            return False, f"Exception: {str(e)}", {}
    
    def fix_all_architectures(self, mlx_dir: str = "mlx_architectures") -> Dict:
        """Fix all MLX architecture files with surgical precision"""
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
    print("🔬 Surgical MLX Architecture Syntax Fixer")
    print("=" * 60)
    
    fixer = SurgicalMLXFixer()
    results = fixer.fix_all_architectures()
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SURGICAL FIX SUMMARY")
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
    with open('surgical_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: surgical_fix_results.json")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())