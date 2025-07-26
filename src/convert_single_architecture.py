#!/usr/bin/env python3
"""
Single Architecture Converter & Verifier
========================================

Tool for converting and verifying individual PyTorch architectures to MLX format.
Allows fixing issues one by one before processing the next architecture.
"""

import json
import ast
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SingleArchitectureConverter:
    """Convert and verify individual architectures"""
    
    def __init__(self, json_path: str = "106.json"):
        self.json_path = json_path
        self.architectures = self._load_architectures()
        self.mlx_dir = Path("mlx_architectures")
        self.mlx_dir.mkdir(exist_ok=True)
        
    def _load_architectures(self) -> List[Dict]:
        """Load architectures from JSON file"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def list_architectures(self) -> None:
        """List all available architectures with their status"""
        print(f"\n{'#':<3} {'Name':<40} {'Status':<12} {'Score':<8}")
        print("-" * 65)
        
        for i, arch in enumerate(self.architectures):
            name = arch['name']
            mlx_file = self.mlx_dir / f"{name}_mlx.py"
            
            # Check status
            if mlx_file.exists():
                status = self.verify_architecture(name, verbose=False)
                status_str = "✓ WORKING" if status['all_good'] else "✗ BROKEN"
            else:
                status_str = "⚪ NOT CONV"
            
            score = arch.get('score', arch.get('result', {}).get('test_accuracy', 0.0))
            print(f"{i:<3} {name:<40} {status_str:<12} {score:<8.4f}")
    
    def convert_architecture(self, index_or_name) -> bool:
        """Convert a single architecture by index or name"""
        # Find architecture
        if isinstance(index_or_name, int):
            if index_or_name >= len(self.architectures):
                print(f"Error: Index {index_or_name} out of range (0-{len(self.architectures)-1})")
                return False
            arch = self.architectures[index_or_name]
        else:
            arch = next((a for a in self.architectures if a['name'] == index_or_name), None)
            if not arch:
                print(f"Error: Architecture '{index_or_name}' not found")
                return False
        
        name = arch['name']
        pytorch_code = arch['program']
        
        print(f"\n🔄 Converting: {name}")
        print(f"Original score: {arch.get('score', 'N/A')}")
        
        # Convert using the existing converter
        from pytorch_to_mlx_converter import PyTorchToMLXConverter
        converter = PyTorchToMLXConverter()
        
        try:
            mlx_code = converter.convert_architecture(pytorch_code, name)
            
            # Save to file
            filepath = self.mlx_dir / f"{name}_mlx.py"
            with open(filepath, 'w') as f:
                f.write(mlx_code)
            
            print(f"✓ Converted and saved to: {filepath}")
            
            # Immediately verify
            verification = self.verify_architecture(name)
            if verification['all_good']:
                print(f"✅ Architecture verified successfully!")
                return True
            else:
                print(f"❌ Architecture has issues:")
                for issue in verification['issues']:
                    print(f"   - {issue}")
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            traceback.print_exc()
            return False
    
    def verify_architecture(self, name: str, verbose: bool = True) -> Dict:
        """Verify an MLX architecture for syntax, imports, and basic structure"""
        filepath = self.mlx_dir / f"{name}_mlx.py"
        
        if not filepath.exists():
            return {
                'all_good': False,
                'issues': [f"File {filepath} does not exist"]
            }
        
        issues = []
        
        try:
            # Read the file
            with open(filepath, 'r') as f:
                code = f.read()
            
            if verbose:
                print(f"\n🔍 Verifying: {name}")
            
            # 1. Check syntax
            try:
                ast.parse(code)
                if verbose:
                    print("✓ Syntax validation passed")
            except SyntaxError as e:
                issues.append(f"Syntax error on line {e.lineno}: {e.msg}")
                if verbose:
                    print(f"❌ Syntax error: {e}")
            
            # 2. Check imports
            try:
                # Create a temporary module
                spec = importlib.util.spec_from_file_location("temp_module", filepath)
                module = importlib.util.module_from_spec(spec)
                
                # Add to sys.modules temporarily
                sys.modules["temp_module"] = module
                
                # Try to execute just the imports
                import_lines = []
                for line in code.split('\n'):
                    stripped = line.strip()
                    if (stripped.startswith('import ') or 
                        stripped.startswith('from ') or
                        stripped.startswith('#') or
                        not stripped):
                        import_lines.append(line)
                    else:
                        break
                
                import_code = '\n'.join(import_lines)
                exec(import_code, module.__dict__)
                
                if verbose:
                    print("✓ Import validation passed")
                    
            except Exception as e:
                issues.append(f"Import error: {e}")
                if verbose:
                    print(f"❌ Import error: {e}")
            finally:
                # Clean up
                if "temp_module" in sys.modules:
                    del sys.modules["temp_module"]
            
            # 3. Check for required classes/functions
            required_items = ['DeltaNet']  # Main class that should exist
            found_items = []
            
            for item in required_items:
                if f"class {item}" in code:
                    found_items.append(item)
                    if verbose:
                        print(f"✓ Found class {item}")
                else:
                    issues.append(f"Missing required class: {item}")
                    if verbose:
                        print(f"❌ Missing class {item}")
            
            # 4. Check for common issues
            common_issues = [
                ("tensor:, mx.array", "Type annotation syntax error"),
                ("F.elu(x 1.0", "Missing comma in function call"),
                ("kwargs.get('h' kwargs.get", "Missing comma in kwargs.get"),
                ("mx.zeros(-1,)", "Invalid tensor shape"),
                ("import torch", "PyTorch import not converted"),
                (".device", "Device reference not removed"),
                ("register_buffer", "PyTorch register_buffer not handled"),
            ]
            
            for pattern, description in common_issues:
                if pattern in code:
                    issues.append(f"Common issue: {description}")
                    if verbose:
                        print(f"⚠️  Found issue: {description}")
            
            # 5. Check for proper MLX usage
            if "import mlx" in code or "mx." in code:
                if verbose:
                    print("✓ MLX imports found")
            else:
                issues.append("No MLX imports found")
                if verbose:
                    print("❌ No MLX imports found")
            
        except Exception as e:
            issues.append(f"Verification error: {e}")
            if verbose:
                print(f"❌ Verification error: {e}")
        
        all_good = len(issues) == 0
        
        if verbose:
            if all_good:
                print(f"✅ {name} verified successfully!")
            else:
                print(f"❌ {name} has {len(issues)} issues")
        
        return {
            'all_good': all_good,
            'issues': issues,
            'name': name
        }
    
    def fix_architecture(self, name: str) -> bool:
        """Attempt to automatically fix common issues in an architecture"""
        filepath = self.mlx_dir / f"{name}_mlx.py"
        
        if not filepath.exists():
            print(f"Error: {filepath} does not exist")
            return False
        
        print(f"\n🔧 Attempting to fix: {name}")
        
        # Read the file
        with open(filepath, 'r') as f:
            code = f.read()
        
        original_code = code
        
        # Apply comprehensive fixes
        fixes_applied = []
        
        # Fix 1: Type annotations with missing commas
        import re
        if re.search(r'(\w+):\s*,\s*(mx\.\w+)', code):
            code = re.sub(r'(\w+):\s*,\s*(mx\.\w+)', r'\1: \2', code)
            fixes_applied.append("Fixed type annotation syntax")
        
        # Fix 2: F.elu calls with missing commas
        if "F.elu(x 1.0" in code:
            code = re.sub(r'F\.elu\s*\(\s*([^,\s]+)\s+([^,\s]+)\s*,\s*([^)]+)\)', r'F.elu(\1, \2, \3)', code)
            code = re.sub(r'F\.elu\s*\(\s*([^,\s]+)\s+([^,)]+)\)', r'F.elu(\1, \2)', code)
            fixes_applied.append("Fixed F.elu function calls")
        
        # Fix 3: kwargs.get calls
        if "kwargs.get('h' kwargs.get" in code:
            code = re.sub(r"kwargs\.get\s*\(\s*'([^']+)'\s+kwargs\.get\s*\(\s*'([^']+)',\s*([^)]+)\)\)", 
                         r"kwargs.get('\1', kwargs.get('\2', \3))", code)
            fixes_applied.append("Fixed kwargs.get calls")
        
        # Fix 4: Function parameter lists
        if re.search(r'def\s+\w+\s*\(\s*([^,)]+)\s+([^,)]+)', code):
            code = re.sub(r'def\s+(__init__)\s*\(\s*self\s+([^,)]+)', r'def \1(self, \2', code)
            code = re.sub(r'def\s+(forward)\s*\(\s*self\s+([^,)]+)', r'def \1(self, \2', code)
            fixes_applied.append("Fixed function parameter lists")
        
        # Fix 5: Remove tensor.device references
        if ".device" in code:
            code = re.sub(r'\.device\b', '', code)
            code = re.sub(r',\s*[a-zA-Z_]\w*\.device', '', code)
            fixes_applied.append("Removed .device references")
        
        # Fix 6: Fix broken return statements
        if re.search(r'return\s+([^,\s]+)\s+([^,\s]+)\s*$', code, re.MULTILINE):
            code = re.sub(r'return\s+([^,\s]+)\s+([^,\s]+)\s*$', r'return \1, \2', code, flags=re.MULTILINE)
            fixes_applied.append("Fixed return statements")
        
        # Fix 7: Linear constructor calls
        if re.search(r'nn\.Linear\s*\(\s*([^,\s]+)\s+([^,\s]+)', code):
            code = re.sub(r'nn\.Linear\s*\(\s*([^,\s]+)\s+([^,\s]+)\s*,', r'nn.Linear(\1, \2,', code)
            code = re.sub(r'nn\.Linear\s*\(\s*([^,\s]+)\s+([^,)]+)\)', r'nn.Linear(\1, \2)', code)
            fixes_applied.append("Fixed nn.Linear calls")
        
        # Fix 8: mx.zeros calls with invalid shapes
        if "mx.zeros(-1," in code:
            code = re.sub(r'mx\.zeros\s*\(\s*([^,\s]+)\s*,\s*\)', r'mx.zeros(\1)', code)
            fixes_applied.append("Fixed mx.zeros calls")
        
        # Fix 9: Missing commas in other common functions
        patterns = [
            (r'\.sum\s*\(\s*([^,\s]+)\s+([^)]+)\)', r'.sum(\1, \2)'),
            (r'\.mean\s*\(\s*([^,\s]+)\s+([^)]+)\)', r'.mean(\1, \2)'),
            (r'\.clamp\s*\(\s*([^,\s]+)\s+([^)]+)\)', r'.clamp(\1, \2)'),
            (r'mx\.pad\s*\(\s*([^,\s]+)\s*,\s*\(\s*([^,\s]+)\s+([^,)]+)\)', r'mx.pad(\1, (\2, \3))'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, code):
                code = re.sub(pattern, replacement, code)
                fixes_applied.append(f"Fixed function call pattern")
        
        # Save the fixed code if any fixes were applied
        if code != original_code:
            with open(filepath, 'w') as f:
                f.write(code)
            
            print(f"✓ Applied {len(fixes_applied)} fixes:")
            for fix in fixes_applied:
                print(f"   - {fix}")
            
            # Verify the fixes
            verification = self.verify_architecture(name, verbose=False)
            if verification['all_good']:
                print(f"✅ Architecture fixed successfully!")
                return True
            else:
                print(f"⚠️  Some issues remain: {len(verification['issues'])} issues")
                for issue in verification['issues'][:3]:  # Show first 3 issues
                    print(f"   - {issue}")
                return False
        else:
            print("ℹ️  No automatic fixes could be applied")
            return False
    
    def show_architecture(self, name: str, lines: int = 50) -> None:
        """Show the content of an architecture file"""
        filepath = self.mlx_dir / f"{name}_mlx.py"
        
        if not filepath.exists():
            print(f"Error: {filepath} does not exist")
            return
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines_list = content.split('\n')
        if lines > 0:
            lines_list = lines_list[:lines]
        
        print(f"\n📄 Content of {name}_mlx.py (showing {len(lines_list)} lines):")
        print("-" * 60)
        for i, line in enumerate(lines_list, 1):
            print(f"{i:3}: {line}")
        
        if len(content.split('\n')) > lines:
            print(f"... ({len(content.split('\n')) - lines} more lines)")
    
    def get_status_summary(self) -> Dict:
        """Get overall status summary"""
        total = len(self.architectures)
        converted = 0
        working = 0
        broken = 0
        
        for arch in self.architectures:
            name = arch['name']
            mlx_file = self.mlx_dir / f"{name}_mlx.py"
            
            if mlx_file.exists():
                converted += 1
                status = self.verify_architecture(name, verbose=False)
                if status['all_good']:
                    working += 1
                else:
                    broken += 1
        
        return {
            'total': total,
            'converted': converted,
            'working': working,
            'broken': broken,
            'not_converted': total - converted
        }

def main():
    """Interactive CLI for converting architectures one by one"""
    converter = SingleArchitectureConverter()
    
    print("🏗️  Single Architecture Converter & Verifier")
    print("=" * 50)
    
    while True:
        print("\nCommands:")
        print("  list                 - List all architectures and their status")
        print("  convert <index|name> - Convert a specific architecture")
        print("  verify <name>        - Verify an MLX architecture")
        print("  fix <name>           - Attempt to fix an architecture")
        print("  show <name> [lines]  - Show architecture content")
        print("  status               - Show overall status summary")
        print("  quit                 - Exit")
        
        try:
            command = input("\n> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'list':
                converter.list_architectures()
            elif cmd == 'status':
                summary = converter.get_status_summary()
                print(f"\n📊 Status Summary:")
                print(f"Total architectures: {summary['total']}")
                print(f"Converted: {summary['converted']}")
                print(f"Working: {summary['working']} ✅")
                print(f"Broken: {summary['broken']} ❌")
                print(f"Not converted: {summary['not_converted']} ⚪")
                
                if summary['converted'] > 0:
                    success_rate = (summary['working'] / summary['converted']) * 100
                    print(f"Success rate: {success_rate:.1f}%")
            
            elif cmd == 'convert':
                if len(command) < 2:
                    print("Usage: convert <index|name>")
                    continue
                
                try:
                    # Try to parse as index first
                    index = int(command[1])
                    converter.convert_architecture(index)
                except ValueError:
                    # Treat as name
                    converter.convert_architecture(command[1])
            
            elif cmd == 'verify':
                if len(command) < 2:
                    print("Usage: verify <name>")
                    continue
                converter.verify_architecture(command[1])
            
            elif cmd == 'fix':
                if len(command) < 2:
                    print("Usage: fix <name>")
                    continue
                converter.fix_architecture(command[1])
            
            elif cmd == 'show':
                if len(command) < 2:
                    print("Usage: show <name> [lines]")
                    continue
                
                lines = 50  # default
                if len(command) > 2:
                    try:
                        lines = int(command[2])
                    except ValueError:
                        print("Invalid line count, using 50")
                
                converter.show_architecture(command[1], lines)
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()