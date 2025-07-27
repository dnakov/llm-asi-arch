#!/usr/bin/env python3
"""
Summary script for Claude Code MLX Architecture Fixer

Provides current status and next steps for fixing PyTorch to MLX conversions.
"""

import subprocess
import json
from pathlib import Path

def check_claude_setup():
    """Check if Claude Code SDK is properly set up"""
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, "Claude CLI not working"
    except FileNotFoundError:
        return False, "Claude CLI not installed"

def get_architecture_status():
    """Get current status of all architectures"""
    cmd = ["python", "claude_code_mlx_fixer.py", "--test-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    working = []
    failing = []
    
    for line in result.stdout.split('\n'):
        if line.startswith('✅'):
            arch_name = line.split(': ')[0].replace('✅ ', '')
            working.append(arch_name)
        elif line.startswith('❌'):
            parts = line.split(': ', 2)
            arch_name = parts[0].replace('❌ ', '')
            error_msg = parts[1] if len(parts) > 1 else "Unknown error"
            failing.append((arch_name, error_msg))
    
    return working, failing

def analyze_errors(failing):
    """Analyze the types of errors in failing architectures"""
    error_types = {}
    
    for arch_name, error in failing:
        if "Syntax error" in error:
            if "unmatched ')'" in error:
                error_type = "Unmatched parenthesis"
            elif "invalid syntax" in error:
                error_type = "Invalid syntax"
            else:
                error_type = "Other syntax error"
        elif "Import error" in error:
            error_type = "Import error"
        elif "DeltaNet" in error:
            error_type = "DeltaNet class issue"
        else:
            error_type = "Other error"
        
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(arch_name)
    
    return error_types

def main():
    """Main summary function"""
    print("🔍 Claude Code MLX Architecture Fixer - Status Summary")
    print("=" * 60)
    
    # Check Claude setup
    claude_ok, claude_msg = check_claude_setup()
    if claude_ok:
        print(f"✅ Claude Code SDK: {claude_msg}")
    else:
        print(f"❌ Claude Code SDK: {claude_msg}")
        print("   Run: ./setup_claude_code.sh")
    
    # Check API key
    import os
    if os.getenv('ANTHROPIC_API_KEY'):
        print("✅ ANTHROPIC_API_KEY: Set")
    else:
        print("❌ ANTHROPIC_API_KEY: Not set")
        print("   Run: export ANTHROPIC_API_KEY='your-key-here'")
    
    print("\n📊 Architecture Status")
    print("-" * 30)
    
    # Get architecture status
    working, failing = get_architecture_status()
    total = len(working) + len(failing)
    
    print(f"Total architectures: {total}")
    print(f"✅ Working: {len(working)} ({len(working)/total*100:.1f}%)")
    print(f"❌ Failing: {len(failing)} ({len(failing)/total*100:.1f}%)")
    
    if failing:
        print(f"\n🔍 Error Analysis")
        print("-" * 20)
        
        error_types = analyze_errors(failing)
        for error_type, archs in error_types.items():
            print(f"{error_type}: {len(archs)} architectures")
            for arch in archs[:3]:  # Show first 3 examples
                print(f"  - {arch}")
            if len(archs) > 3:
                print(f"  ... and {len(archs) - 3} more")
    
    print(f"\n🚀 Next Steps")
    print("-" * 15)
    
    if not claude_ok:
        print("1. Set up Claude Code SDK:")
        print("   ./setup_claude_code.sh")
    elif not os.getenv('ANTHROPIC_API_KEY'):
        print("1. Set your API key:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
    elif not failing:
        print("🎉 All architectures are working! No fixes needed.")
    else:
        print("1. Test Claude setup:")
        print("   python test_claude_sdk.py")
        print()
        print("2. Fix failing architectures:")
        print(f"   python claude_code_mlx_fixer.py --max {len(failing)}")
        print()
        print("3. Or fix them one by one:")
        for i, (arch_name, _) in enumerate(failing[:5]):
            print(f"   python claude_code_mlx_fixer.py --start {i} --max 1  # {arch_name}")
        if len(failing) > 5:
            print(f"   ... and {len(failing) - 5} more")
    
    print(f"\n📁 Key Files")
    print("-" * 15)
    print("claude_code_mlx_fixer.py     - Main fixer script")
    print("test_claude_sdk.py           - Test Claude setup")
    print("setup_claude_code.sh         - Setup script")
    print("CLAUDE_CODE_MLX_FIXER_README.md - Full documentation")
    
    if Path("claude_fix_results.json").exists():
        print("claude_fix_results.json      - Previous results")
    if Path("claude_fix_progress.json").exists():
        print("claude_fix_progress.json     - Progress tracking")

if __name__ == "__main__":
    main()