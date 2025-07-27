#!/usr/bin/env python3
"""
Quick fix for the specific delta_net_erfg pattern error that timed out.
"""

import subprocess
import sys

def quick_fix_erfg():
    """Quick fix for delta_net_erfg pattern error"""
    
    # Very focused prompt for the specific error
    prompt = """Fix the MLX file mlx_architectures/delta_net_erfg_mlx.py.

ERROR: "Pattern b l h -> b h l not implemented"

SOLUTION: Replace einops.rearrange with MLX transpose:
- Change: rearrange(x, 'b l h -> b h l') 
- To: mx.transpose(x, (0, 2, 1))

STEPS:
1. Find the line causing the error
2. Replace einops operation with mx.transpose
3. Remove einops import if no longer needed
4. Test that it works

Work quickly - this should be a simple 1-line fix."""

    cmd = [
        'claude', '-p', prompt,
        '--max-turns', '1',  # Just one turn
        '--output-format', 'text'  # Simpler output
    ]
    
    try:
        print("🚀 Quick fixing delta_net_erfg pattern error...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 min timeout
        
        if result.returncode == 0:
            print("✅ Quick fix completed!")
            print("Testing the fix...")
            
            # Test the fix
            test_cmd = ["python", "claude_code_mlx_fixer.py", "--test-only"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True)
            
            for line in test_result.stdout.split('\n'):
                if 'delta_net_erfg' in line:
                    print(f"Result: {line}")
                    break
        else:
            print(f"❌ Quick fix failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Quick fix also timed out")

if __name__ == "__main__":
    quick_fix_erfg()