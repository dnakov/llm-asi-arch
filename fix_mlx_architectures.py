#!/usr/bin/env python3
"""
Script to fix all MLX architectures with .at[].set() issues
"""
import os
import re
import glob

def fix_architecture(file_path):
    """Fix a single architecture file"""
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if it actually has .at[].set() patterns
    if '.at[' not in content or '.set(' not in content:
        print(f"  ✅ {file_path} already fixed or no issues found")
        return True
    
    original_content = content
    
    # Fix 1: Filter initialization pattern
    pattern1 = r'(\s+)filters = mx\.zeros\([^)]+\)\s*\n\s+filters = filters\.at\[\.\.\.,-1\]\.set\(1\.0\)\s*\n\s+filters = filters \+ noise_std \* mx\.random\.normal\(filters\.shape\)'
    replacement1 = r'''\1filters = mx.zeros((num_heads, head_dim, self.kernel_size))
\1# MLX doesn't support .at[].set(), use direct assignment
\1filters_list = []
\1for i in range(num_heads):
\1    for j in range(head_dim):
\1        filter_row = mx.zeros(self.kernel_size)
\1        filter_row = mx.concatenate([filter_row[:-1], mx.array([1.0])])
\1        filters_list.append(filter_row)
\1filters = mx.stack(filters_list).reshape(num_heads, head_dim, self.kernel_size)
\1filters = filters + noise_std * mx.random.normal(filters.shape)'''
    
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    
    # Fix 2: Convolution loop pattern
    pattern2 = r'(\s+)y = mx\.zeros\([^)]+\)\s*\n\s+for i in range\(h \* d\):\s*\n\s+for j in range\(l\):\s*\n\s+start_idx = j\s*\n\s+end_idx = j \+ self\.kernel_size\s*\n\s+y = y\.at\[\.\.\..*?\]\.set\(\s*mx\.sum\([^)]+\)\s*\)'
    replacement2 = r'''\1y = mx.zeros((b, h * d, l))
\1# Replace .at[].set() with direct computation
\1y_list = []
\1for batch in range(b):
\1    batch_result = []
\1    for i in range(h * d):
\1        channel_result = []
\1        for j in range(l):
\1            start_idx = j
\1            end_idx = j + self.kernel_size
\1            conv_result = mx.sum(x_pad[batch, i, start_idx:end_idx] * weight[i, 0, :])
\1            channel_result.append(conv_result)
\1        batch_result.append(mx.stack(channel_result))
\1    y_list.append(mx.stack(batch_result))
\1y = mx.stack(y_list)'''
    
    content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE | re.DOTALL)
    
    # Fix 3: Attention update pattern
    pattern3 = r'(\s+)o = o\.at\[:, :, idx\]\.set\(q_i @ S \+ attn_local @ u_i\)'
    replacement3 = r'\1o_i = q_i @ S + attn_local @ u_i\n\1o_chunks.append(o_i)'
    
    content = re.sub(pattern3, replacement3, content)
    
    # Fix 4: Add chunk reconstruction if we have o_chunks
    if 'o_chunks.append(o_i)' in content and 'if o_chunks:' not in content:
        # Find the pattern where we need to add reconstruction
        pattern4 = r'(\s+)o = mx\.zeros_like\(v\)\s*\n(\s+)for idx in range\(L_pad // chunk_size\):'
        replacement4 = r'\1o = mx.zeros_like(v)\n\1\n\1# Build output list instead of using .at[].set()\n\1o_chunks = []\n\2for idx in range(L_pad // chunk_size):'
        content = re.sub(pattern4, replacement4, content)
        
        # Add reconstruction at the end
        pattern5 = r'(\s+)S = S \+ mx\.transpose\(k_i, \[0, 1, 3, 2\]\) @ u_i\s*\n(\s+)o = _rearrange'
        replacement5 = r'\1S = S + mx.transpose(k_i, [0, 1, 3, 2]) @ u_i\n\1\n\1# Reconstruct o from chunks\n\1if o_chunks:\n\1    o = mx.stack(o_chunks, axis=2)\n\2o = _rearrange'
        content = re.sub(pattern5, replacement5, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Fixed {file_path}")
        return True
    else:
        print(f"  ⚠️  No changes made to {file_path}")
        return False

def main():
    # Get all MLX architecture files that have .at[] patterns
    files_to_fix = []
    for file_path in glob.glob("mlx_architectures/*.py"):
        with open(file_path, 'r') as f:
            content = f.read()
            if '.at[' in content and '.set(' in content:
                files_to_fix.append(file_path)
    
    print(f"Found {len(files_to_fix)} files to fix:")
    for f in files_to_fix:
        print(f"  - {f}")
    
    print("\nStarting fixes...")
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_architecture(file_path):
            fixed_count += 1
    
    print(f"\n🎉 Fixed {fixed_count}/{len(files_to_fix)} files!")

if __name__ == "__main__":
    main()