#!/usr/bin/env python3
"""
Comprehensive test suite for all 106 converted MLX architectures
"""

import json
import os
import sys
import traceback
import importlib.util
from typing import Dict, List, Tuple
import time

def test_syntax_only(arch_name: str) -> Tuple[bool, str]:
    """Test if the converted file has valid Python syntax"""
    mlx_path = f"mlx_architectures/{arch_name}_mlx.py"
    
    if not os.path.exists(mlx_path):
        return False, f"File not found: {mlx_path}"
    
    try:
        # Try to compile the file
        with open(mlx_path, 'r') as f:
            source = f.read()
        compile(source, mlx_path, 'exec')
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Compile Error: {str(e)}"

def test_import_only(arch_name: str) -> Tuple[bool, str]:
    """Test if the converted file can be imported"""
    mlx_path = f"mlx_architectures/{arch_name}_mlx.py"
    
    try:
        spec = importlib.util.spec_from_file_location(arch_name, mlx_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "Import OK"
    except Exception as e:
        return False, f"Import Error: {str(e)}"

def find_architecture_class(module) -> str:
    """Find the main architecture class in the module"""
    classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and 
            hasattr(attr, '__bases__') and
            any('Module' in str(base) for base in attr.__bases__)):
            classes.append(attr_name)
    return classes

def load_original_data() -> Dict:
    """Load the original 106.json data"""
    with open('106.json', 'r') as f:
        architectures = json.load(f)
    return {arch['name']: arch for arch in architectures}

def test_all_architectures_comprehensive():
    """Run comprehensive tests on all 106 architectures"""
    
    # Get list of all MLX architectures
    if not os.path.exists('mlx_architectures'):
        print("❌ MLX architectures directory not found!")
        return
    
    architectures = []
    for filename in os.listdir('mlx_architectures'):
        if filename.endswith('_mlx.py'):
            arch_name = filename.replace('_mlx.py', '')
            architectures.append(arch_name)
    
    architectures = sorted(architectures)
    original_data = load_original_data()
    
    print(f"🔍 Testing {len(architectures)} converted MLX architectures")
    print("=" * 80)
    
    results = {
        'syntax_ok': [],
        'syntax_failed': [],
        'import_ok': [],
        'import_failed': [],
        'class_found': [],
        'class_not_found': []
    }
    
    for i, arch_name in enumerate(architectures, 1):
        print(f"\n[{i:3d}/106] {arch_name}")
        
        # Test 1: Syntax
        syntax_ok, syntax_msg = test_syntax_only(arch_name)
        if syntax_ok:
            results['syntax_ok'].append(arch_name)
            print(f"  ✅ Syntax: {syntax_msg}")
        else:
            results['syntax_failed'].append((arch_name, syntax_msg))
            print(f"  ❌ Syntax: {syntax_msg}")
            continue  # Skip further tests if syntax fails
        
        # Test 2: Import
        import_ok, import_msg = test_import_only(arch_name)
        if import_ok:
            results['import_ok'].append(arch_name)
            print(f"  ✅ Import: {import_msg}")
        else:
            results['import_failed'].append((arch_name, import_msg))
            print(f"  ❌ Import: {import_msg}")
            continue  # Skip further tests if import fails
        
        # Test 3: Find classes
        try:
            mlx_path = f"mlx_architectures/{arch_name}_mlx.py"
            spec = importlib.util.spec_from_file_location(arch_name, mlx_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            classes = find_architecture_class(module)
            if classes:
                results['class_found'].append((arch_name, classes))
                print(f"  ✅ Classes: {', '.join(classes)}")
            else:
                results['class_not_found'].append(arch_name)
                print(f"  ⚠️  No architecture classes found")
                
        except Exception as e:
            results['class_not_found'].append(arch_name)
            print(f"  ❌ Class detection failed: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    print(f"📁 Total architectures found: {len(architectures)}")
    print(f"✅ Syntax valid: {len(results['syntax_ok'])}")
    print(f"❌ Syntax failed: {len(results['syntax_failed'])}")
    print(f"✅ Import successful: {len(results['import_ok'])}")
    print(f"❌ Import failed: {len(results['import_failed'])}")
    print(f"✅ Classes found: {len(results['class_found'])}")
    print(f"⚠️  No classes found: {len(results['class_not_found'])}")
    
    # Success rates
    syntax_rate = len(results['syntax_ok']) / len(architectures) * 100
    import_rate = len(results['import_ok']) / len(architectures) * 100
    class_rate = len(results['class_found']) / len(architectures) * 100
    
    print(f"\n📊 Success Rates:")
    print(f"   Syntax: {syntax_rate:.1f}%")
    print(f"   Import: {import_rate:.1f}%")
    print(f"   Classes: {class_rate:.1f}%")
    
    # Show first few failures for debugging
    if results['syntax_failed']:
        print(f"\n🔍 First 5 syntax failures:")
        for arch_name, error in results['syntax_failed'][:5]:
            print(f"   {arch_name}: {error}")
    
    if results['import_failed']:
        print(f"\n🔍 First 5 import failures:")
        for arch_name, error in results['import_failed'][:5]:
            print(f"   {arch_name}: {error}")
    
    # Save detailed results
    with open('architecture_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total': len(architectures),
                'syntax_ok': len(results['syntax_ok']),
                'syntax_failed': len(results['syntax_failed']),
                'import_ok': len(results['import_ok']),
                'import_failed': len(results['import_failed']),
                'class_found': len(results['class_found']),
                'class_not_found': len(results['class_not_found']),
                'syntax_rate': syntax_rate,
                'import_rate': import_rate,
                'class_rate': class_rate
            },
            'details': {
                'syntax_ok': results['syntax_ok'],
                'syntax_failed': [{'arch': arch, 'error': err} for arch, err in results['syntax_failed']],
                'import_ok': results['import_ok'],
                'import_failed': [{'arch': arch, 'error': err} for arch, err in results['import_failed']],
                'class_found': [{'arch': arch, 'classes': classes} for arch, classes in results['class_found']],
                'class_not_found': results['class_not_found']
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: architecture_test_results.json")
    
    return syntax_rate > 50  # Return success if more than 50% syntax is OK

if __name__ == "__main__":
    success = test_all_architectures_comprehensive()
    sys.exit(0 if success else 1)