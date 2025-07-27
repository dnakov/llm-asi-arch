#!/usr/bin/env python3
"""
Targeted script to fix only the failing architectures using Claude Code SDK
"""

import subprocess
import json
from pathlib import Path

def get_failing_architectures():
    """Get list of architectures that are currently failing tests"""
    cmd = ["python", "claude_code_mlx_fixer.py", "--test-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    failing = []
    for line in result.stdout.split('\n'):
        if line.startswith('❌'):
            # Extract architecture name and error
            parts = line.split(': ', 2)
            if len(parts) >= 2:
                arch_name = parts[0].replace('❌ ', '')
                error_msg = parts[1] if len(parts) > 1 else "Unknown error"
                failing.append((arch_name, error_msg))
    
    return failing

def main():
    """Fix only the failing architectures"""
    print("🎯 Targeting Failing Architectures Only")
    print("=" * 50)
    
    failing = get_failing_architectures()
    print(f"Found {len(failing)} failing architectures:")
    
    for arch_name, error in failing:
        print(f"  ❌ {arch_name}: {error}")
    
    if not failing:
        print("🎉 No failing architectures found! All are working.")
        return
    
    print(f"\n🚀 Running Claude Code fixer on {len(failing)} failing architectures...")
    
    # Create a list of architecture names to process
    arch_names = [arch for arch, _ in failing]
    
    # Run the fixer with a targeted approach
    for i, arch_name in enumerate(arch_names, 1):
        print(f"\n[{i}/{len(arch_names)}] Fixing {arch_name}...")
        
        cmd = [
            "python", "claude_code_mlx_fixer.py", 
            "--start", str(i-1), 
            "--max", "1"
        ]
        
        # Find the index of this architecture in the full list
        # This is a bit hacky but works for our targeted fixing
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Failed to process {arch_name}")
            print(result.stderr)

if __name__ == "__main__":
    main()