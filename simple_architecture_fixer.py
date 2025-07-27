#!/usr/bin/env python3
"""
Simple, targeted fixes for specific MLX architecture issues.
"""

import os

def fix_files_individually():
    """Fix each file with specific targeted fixes"""
    
    # Fix delta_net_spectral_fusion_mlx.py
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_spectral_fusion_mlx.py"
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the major syntax issues in spectral fusion
    content = content.replace('    freq_f32 = freq.astype(mx.float32)\n        decay_f32 = decay.astype(mx.float32)\n        amp_f32 = amp.astype(mx.float32)', 
                             '    freq_f32 = freq.astype(mx.float32)\n    decay_f32 = decay.astype(mx.float32)\n    amp_f32 = amp.astype(mx.float32)')
    content = content.replace('def _spectral_conv_overlap_add(self, x: mx.array block: int) -> mx.array:', 
                             'def _spectral_conv_overlap_add(self, x: mx.array, block: int) -> mx.array:')
    content = content.replace('n_fft = _next_power_of_two(2, * block)', 
                             'n_fft = _next_power_of_two(2 * block)')
    content = content.replace('end = min(start, +, block, seq_len)', 
                             'end = min(start + block, seq_len)')
    content = content.replace('filt = _power_law_min_phase_filter(freq, amp, decay\n        dtype =fft.dtype)', 
                             'filt = _power_law_min_phase_filter(freq, amp, decay, fft.dtype)')
    content = content.replace('filt = _power_law_min_phase_filter(freq, amp, decay\n        dtype =mx.complex64)', 
                             'filt = _power_law_min_phase_filter(freq, amp, decay, mx.complex64)')
    content = content.replace('**kwargs -> None:', '**kwargs) -> None:')
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed delta_net_spectral_fusion_mlx.py")
    
    # Fix delta_net_hhmr_mlx.py
    file_path = "/Users/daniel/dev/asi/mlx_architectures/delta_net_hhmr_mlx.py"
    with open(file_path, 'r') as f:
        content = f.read()
    
    content = content.replace('def _elu_p1(x:, mx.array) -> mx.array:', 'def _elu_p1(x: mx.array) -> mx.array:')
    content = content.replace('def _sum_norm(x:, mx.array) -> mx.array:', 'def _sum_norm(x: mx.array) -> mx.array:')
    content = content.replace('def _mean_var(x:, mx.array) -> Tuple[mx.array, mx.array]:', 'def _mean_var(x: mx.array) -> Tuple[mx.array, mx.array]:')
    content = content.replace('int(hidden_size, * expand_k)', 'int(hidden_size * expand_k)')
    content = content.replace('int(hidden_size, * expand_v)', 'int(hidden_size * expand_v)')
    content = content.replace('int(gate1_in_dim, * fusion_hidden_mult)', 'int(gate1_in_dim * fusion_hidden_mult)')
    content = content.replace('int(gate_local_in_dim, * fusion_hidden_mult)', 'int(gate_local_in_dim * fusion_hidden_mult)')
    content = content.replace('int(gate_global_in_dim, * fusion_hidden_mult)', 'int(gate_global_in_dim * fusion_hidden_mult)')
    content = content.replace('x_ch = _rearrange(x, "b l h d -> b(h, d) l")', 'x_ch = _rearrange(x, "b l h d -> b (h d) l")')
    content = content.replace('return [_rearrange(F.conv1d(mx.pad(x_ch, (k - 1, 0)), filt, groups=h*d), "b(h, d) l -> b l h d", h=h)', 
                             'return [_rearrange(conv1d_result, "b (h d) l -> b l h d", h=h)')
    content = content.replace('q, k, v, k_beta = map(lambda t: _rearrange(t, "b h(n, c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))', 
                             'q, k, v, k_beta = map(lambda t: _rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))')
    content = content.replace('out = _rearrange(out, "b h n c d -> b h(n, c) d")', 
                             'out = _rearrange(out, "b h n c d -> b h (n c) d")')
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed delta_net_hhmr_mlx.py")

def fix_simple_syntax_issues():
    """Fix simple syntax issues across multiple files"""
    
    files_to_fix = [
        "delta_net_sparsemax_temperature_mlx.py",
        "delta_net_ssg_sparsemax_temp_mlx.py", 
        "delta_net_entropy_kl_floor_gate_mlx.py",
        "delta_net_ms_adaptive_gstat3_mlx.py",
        "delta_net_ms_hsm_tempgate_mlx.py"
    ]
    
    for filename in files_to_fix:
        file_path = f"/Users/daniel/dev/asi/mlx_architectures/{filename}"
        
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix basic type annotation issues  
        content = content.replace('tensor:, mx.array,', 'tensor: mx.array,')
        content = content.replace('x:, mx.array)', 'x: mx.array)')
        content = content.replace('def _l2norm(x:, mx.array)', 'def _l2norm(x: mx.array)')
        content = content.replace('def _masked_fill(tensor:, mx.array,', 'def _masked_fill(tensor: mx.array,')
        content = content.replace('def _index_first_axis(tensor:, mx.array,', 'def _index_first_axis(tensor: mx.array,')
        content = content.replace('def _pad_input(tensor:, mx.array,', 'def _pad_input(tensor: mx.array,')
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Fixed basic syntax in {filename}")

def main():
    print("Running simple architecture fixes...")
    
    fix_files_individually()
    fix_simple_syntax_issues()
    
    print("\nSimple fixes completed!")

if __name__ == "__main__":
    main()