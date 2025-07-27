#!/usr/bin/env python3
"""
Batch syntax fixer for MLX architectures
"""

import os
import re
import json
from typing import List, Tuple

def fix_common_syntax_errors(content: str) -> str:
    """Fix common syntax errors in MLX architecture files"""
    
    # Fix 1: Unterminated string literals
    content = re.sub(r'assert ([^,]+), ([^"]+)"$', r'assert \1, "\2"', content, flags=re.MULTILINE)
    content = re.sub(r'assert ([^,]+), ([^"]+)$', r'assert \1, "\2"', content, flags=re.MULTILINE)
    
    # Fix 2: Missing opening parentheses in function definitions
    content = re.sub(r'def ([^(]+)([^)]+), ([^)]+)\):', r'def \1(\2, \3):', content)
    
    # Fix 3: Unmatched parentheses and brackets
    # Fix missing commas before parameters
    content = re.sub(r'(\w+)\s+(\w+\s*=)', r'\1, \2', content)
    
    # Fix 4: Invalid syntax in comments that break lines
    content = re.sub(r'#\s*\[([^\]]+)\s*$', r'# [\1]', content, flags=re.MULTILINE)
    
    # Fix 5: Broken string formatting in comments
    content = re.sub(r'#\s*\[B\s+L,\s*H\]', r'# [B, L, H]', content)
    
    # Fix 6: Fix broken rearrange patterns
    content = re.sub(r'"b s d ->, \(b, s\) d"', r'"b s d -> (b s) d"', content)
    content = re.sub(r'"b s d ->, \(b s\) d"', r'"b s d -> (b s) d"', content)
    
    # Fix 7: Missing commas in function calls
    content = re.sub(r'(\w+)=([^,\)]+)\s+(\w+)=', r'\1=\2, \3=', content)
    
    # Fix 8: Fix invalid decimal literals
    content = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2', content)
    
    # Fix 9: Fix unmatched brackets in array indexing
    content = re.sub(r'\[([^\]]+)\s*$', r'[\1]', content, flags=re.MULTILINE)
    
    # Fix 10: Fix broken function parameter lists
    content = re.sub(r'(\w+)\s*\(\s*([^)]+)\s*\)\s*\(\s*([^)]+)\s*\)', r'\1(\2, \3)', content)
    
    return content

def fix_specific_architecture_errors(arch_name: str, content: str) -> str:
    """Fix architecture-specific errors"""
    
    if arch_name == "delta_net_adgr":
        # Fix the specific string literal issue
        content = re.sub(r'attention_mask must be \[batch, seq_len\]"', r'"attention_mask must be [batch, seq_len]"', content)
        # Fix the tuple unpacking issue
        content = re.sub(r'batch_size, seq_len, _ = hidden_states\.shape, last_state = None', 
                        r'batch_size, seq_len, _ = hidden_states.shape\n        last_state = None', content)
    
    elif arch_name == "delta_net_cagf_br":
        # Fix the rearrange pattern
        content = re.sub(r'"b s d ->, \(b, s\) d"', r'"b s d -> (b s) d"', content)
        # Fix indentation issues
        content = re.sub(r'^\s{12}hidden_states = ', r'            hidden_states = ', content, flags=re.MULTILINE)
    
    elif arch_name == "delta_net_csm":
        # Fix the broken comment
        content = re.sub(r'beta = self\.b_proj\(hidden_states\)\.sigmoid\(\)\s*#\s*\[B\s+L,\s*H\]', 
                        r'beta = self.b_proj(hidden_states).sigmoid()  # [B, L, H]', content)
    
    elif arch_name == "delta_net_ddfsanr":
        # Fix the function definition
        content = re.sub(r'def _delta_rule_chunkwiseq, k, v, beta', r'def _delta_rule_chunkwise(q, k, v, beta', content)
    
    elif arch_name == "delta_net_dlgm":
        # Fix the missing comma in past_key_values.update
        content = re.sub(r'recurrent_state=recurrent_state\s+conv_state=', 
                        r'recurrent_state=recurrent_state,\n                conv_state=', content)
    
    return content

def fix_architecture_file(arch_name: str) -> Tuple[bool, str]:
    """Fix a single architecture file"""
    mlx_path = f"mlx_architectures/{arch_name}_mlx.py"
    
    if not os.path.exists(mlx_path):
        return False, f"File not found: {mlx_path}"
    
    try:
        with open(mlx_path, 'r') as f:
            content = f.read()
        
        # Apply common fixes
        fixed_content = fix_common_syntax_errors(content)
        
        # Apply architecture-specific fixes
        fixed_content = fix_specific_architecture_errors(arch_name, fixed_content)
        
        # Write back if changed
        if fixed_content != content:
            with open(mlx_path, 'w') as f:
                f.write(fixed_content)
            return True, "Fixed and saved"
        else:
            return True, "No changes needed"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_syntax(arch_name: str) -> Tuple[bool, str]:
    """Test if the architecture has valid syntax"""
    mlx_path = f"mlx_architectures/{arch_name}_mlx.py"
    
    try:
        with open(mlx_path, 'r') as f:
            source = f.read()
        compile(source, mlx_path, 'exec')
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Compile Error: {str(e)}"

def main():
    """Main batch fixing function"""
    
    # Get list of failed architectures from the test results
    failed_architectures = [
        "delta_net_adgr", "delta_net_cagf_br", "delta_net_csm", "delta_net_ddfsanr", 
        "delta_net_dlgm", "delta_net_dmshf", "delta_net_dynfuse", "delta_net_entropy_kl_floor_gate",
        "delta_net_erfg", "delta_net_gtmlp", "delta_net_hafmg", "delta_net_hafs",
        "delta_net_hmgapf", "delta_net_hpaf", "delta_net_htcg", "delta_net_htfr",
        "delta_net_hwggm", "delta_net_hybrid_floor_gt", "delta_net_mafr", "delta_net_mfg",
        "delta_net_ms_gstat3_quota", "delta_net_ms_hsm_tempgate", "delta_net_ms_hsm_widefloor",
        "delta_net_ms_resgate", "delta_net_msdfdm", "delta_net_msfr_mn", "delta_net_ndg",
        "delta_net_oahmgr", "delta_net_pathgated", "delta_net_pfr", "delta_net_psafg",
        "delta_net_psfr", "delta_net_rggf", "delta_net_sigf_ptu", "delta_net_syngf",
        "delta_net_tapr", "delta_net_tareia"
    ]
    
    print(f"🔧 Batch fixing {len(failed_architectures)} architectures")
    print("=" * 60)
    
    fixed_count = 0
    still_broken = []
    
    for i, arch_name in enumerate(failed_architectures, 1):
        print(f"\n[{i:2d}/{len(failed_architectures)}] {arch_name}")
        
        # Test current syntax
        syntax_ok, syntax_msg = test_syntax(arch_name)
        if syntax_ok:
            print(f"  ✅ Already working: {syntax_msg}")
            fixed_count += 1
            continue
        
        print(f"  ❌ Before: {syntax_msg}")
        
        # Try to fix
        fix_ok, fix_msg = fix_architecture_file(arch_name)
        if not fix_ok:
            print(f"  ❌ Fix failed: {fix_msg}")
            still_broken.append((arch_name, fix_msg))
            continue
        
        # Test after fix
        syntax_ok, syntax_msg = test_syntax(arch_name)
        if syntax_ok:
            print(f"  ✅ After: {syntax_msg}")
            fixed_count += 1
        else:
            print(f"  ❌ Still broken: {syntax_msg}")
            still_broken.append((arch_name, syntax_msg))
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH FIX SUMMARY")
    print("=" * 60)
    print(f"✅ Fixed: {fixed_count}/{len(failed_architectures)}")
    print(f"❌ Still broken: {len(still_broken)}")
    
    if still_broken:
        print(f"\n🔍 Still broken architectures:")
        for arch_name, error in still_broken[:10]:  # Show first 10
            print(f"   {arch_name}: {error}")
        if len(still_broken) > 10:
            print(f"   ... and {len(still_broken) - 10} more")
    
    # Save results
    results = {
        'fixed_count': fixed_count,
        'total_attempted': len(failed_architectures),
        'still_broken': [{'arch': arch, 'error': err} for arch, err in still_broken]
    }
    
    with open('batch_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: batch_fix_results.json")
    
    return fixed_count > len(failed_architectures) // 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)