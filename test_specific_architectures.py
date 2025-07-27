#!/usr/bin/env python3
"""
Test specific MLX architectures for import and basic functionality issues.
Focus on complex architectures that might need conversion improvements.
"""

import os
import sys
import traceback
import importlib.util
import mlx.core as mx
import mlx.nn as nn

def test_architecture_import(arch_path):
    """Test if an architecture can be imported without errors."""
    try:
        arch_name = os.path.basename(arch_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(arch_name, arch_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, None, module
    except Exception as e:
        return False, str(e), None

def test_architecture_forward(module, arch_name):
    """Test if the architecture can perform a basic forward pass."""
    try:
        # Standard test parameters
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        model_dim = 64
        
        # Try to find the main model class
        model_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, nn.Module) and 
                attr != nn.Module and
                'DeltaNet' in attr_name):
                model_class = attr
                break
        
        if model_class is None:
            return False, "No DeltaNet model class found"
        
        # Try to instantiate with common parameters
        try:
            model = model_class(
                vocab_size=vocab_size,
                model_dim=model_dim,
                n_layers=2
            )
        except TypeError:
            # Try with different parameter names
            try:
                model = model_class(
                    vocab_size=vocab_size,
                    d_model=model_dim,
                    num_layers=2
                )
            except TypeError:
                # Try minimal parameters
                try:
                    model = model_class(vocab_size=vocab_size)
                except TypeError as e:
                    return False, f"Could not instantiate model: {e}"
        
        # Create test input
        x = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        if len(output.shape) != 3:
            return False, f"Expected 3D output, got shape {output.shape}"
        
        if output.shape[0] != batch_size:
            return False, f"Expected batch size {batch_size}, got {output.shape[0]}"
        
        if output.shape[1] != seq_len:
            return False, f"Expected sequence length {seq_len}, got {output.shape[1]}"
        
        return True, f"Forward pass successful, output shape: {output.shape}"
        
    except Exception as e:
        return False, f"Forward pass failed: {str(e)}"

def main():
    # Target architectures to test
    test_architectures = [
        "delta_net_spectral_fusion_mlx.py",
        "delta_net_sparsemax_temperature_mlx.py", 
        "delta_net_ssg_sparsemax_temp_mlx.py",
        "delta_net_ahic_mlx.py",
        "delta_net_hhmr_mlx.py",
        # Add some additional complex ones
        "delta_net_entropy_floor_mlx.py",
        "delta_net_entropy_kl_floor_gate_mlx.py",
        "delta_net_hybrid_floor_gt_mlx.py",
        "delta_net_triscale_mlx.py",
        "delta_net_gae_ms3e_mlx.py",
        "delta_net_ms_adaptive_gstat3_mlx.py",
        "delta_net_ms_hsm_tempgate_mlx.py"
    ]
    
    base_path = "/Users/daniel/dev/asi/mlx_architectures"
    
    results = {}
    
    print("Testing MLX Architectures for Import and Forward Pass Issues")
    print("=" * 60)
    
    for arch_file in test_architectures:
        arch_path = os.path.join(base_path, arch_file)
        arch_name = arch_file.replace('_mlx.py', '')
        
        print(f"\nTesting: {arch_name}")
        print("-" * 40)
        
        if not os.path.exists(arch_path):
            print(f"❌ File not found: {arch_path}")
            results[arch_name] = "FILE_NOT_FOUND"
            continue
        
        # Test import
        import_success, import_error, module = test_architecture_import(arch_path)
        
        if not import_success:
            print(f"❌ Import failed: {import_error}")
            results[arch_name] = f"IMPORT_ERROR: {import_error}"
            continue
        
        print("✅ Import successful")
        
        # Test forward pass
        forward_success, forward_result = test_architecture_forward(module, arch_name)
        
        if not forward_success:
            print(f"❌ Forward pass failed: {forward_result}")
            results[arch_name] = f"FORWARD_ERROR: {forward_result}"
        else:
            print(f"✅ Forward pass successful: {forward_result}")
            results[arch_name] = "SUCCESS"
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(results)
    
    print(f"Successful: {success_count}/{total_count}")
    print(f"Failed: {total_count - success_count}/{total_count}")
    
    # Group by error type
    import_errors = [name for name, status in results.items() if status.startswith("IMPORT_ERROR")]
    forward_errors = [name for name, status in results.items() if status.startswith("FORWARD_ERROR")]
    not_found = [name for name, status in results.items() if status == "FILE_NOT_FOUND"]
    
    if import_errors:
        print(f"\nImport Errors ({len(import_errors)}):")
        for name in import_errors:
            print(f"  - {name}: {results[name]}")
    
    if forward_errors:
        print(f"\nForward Pass Errors ({len(forward_errors)}):")
        for name in forward_errors:
            print(f"  - {name}: {results[name]}")
    
    if not_found:
        print(f"\nNot Found ({len(not_found)}):")
        for name in not_found:
            print(f"  - {name}")
    
    # Save detailed results
    with open("/Users/daniel/dev/asi/specific_architecture_test_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: specific_architecture_test_results.json")

if __name__ == "__main__":
    main()