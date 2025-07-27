#!/usr/bin/env python3
"""
Test the architectures that seem to be in better shape.
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
                num_heads=2,
                hidden_size=model_dim
            )
        except TypeError as e:
            try:
                model = model_class(
                    d_model=model_dim,
                    num_heads=2,
                    hidden_size=model_dim
                )
            except TypeError as e:
                try:
                    model = model_class(hidden_size=model_dim, num_heads=2)
                except TypeError as e:
                    return False, f"Could not instantiate model: {e}"
        
        # Create test input
        x = mx.random.normal((batch_size, seq_len, model_dim))
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        if output is None:
            return False, "Model returned None"
        
        if not hasattr(output, 'shape'):
            return False, f"Output is not a tensor: {type(output)}"
        
        if len(output.shape) != 3:
            return False, f"Expected 3D output, got shape {output.shape}"
        
        if output.shape[0] != batch_size:
            return False, f"Expected batch size {batch_size}, got {output.shape[0]}"
        
        if output.shape[1] != seq_len:
            return False, f"Expected sequence length {seq_len}, got {output.shape[1]}"
        
        return True, f"Forward pass successful, output shape: {output.shape}"
        
    except Exception as e:
        return False, f"Forward pass failed: {str(e)}"

def test_known_working_architectures():
    """Test architectures that are known to import successfully."""
    base_path = "/Users/daniel/dev/asi/mlx_architectures"
    
    # Start with architectures that imported successfully before
    known_working = [
        "delta_net_ahic_mlx.py",
        "delta_net_entropy_floor_mlx.py", 
        "delta_net_gae_ms3e_mlx.py"
    ]
    
    # Add some others that might work
    potentially_working = [
        "delta_net_adgr_mlx.py",
        "delta_net_aegf_br_mlx.py",
        "delta_net_cagf_br_mlx.py",
        "delta_net_csm_mlx.py",
        "delta_net_ddfsanr_mlx.py",
        "delta_net_dfpcr_mlx.py",
        "delta_net_dlgm_mlx.py",
        "delta_net_dmshf_mlx.py"
    ]
    
    all_test_files = known_working + potentially_working
    
    results = {}
    
    print("Testing Working MLX Architectures")
    print("=" * 50)
    
    for arch_file in all_test_files:
        arch_path = os.path.join(base_path, arch_file)
        arch_name = arch_file.replace('_mlx.py', '')
        
        print(f"\nTesting: {arch_name}")
        print("-" * 30)
        
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
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(results)
    
    print(f"Successful: {success_count}/{total_count}")
    
    if success_count > 0:
        print(f"\nWorking architectures:")
        for name, status in results.items():
            if status == "SUCCESS":
                print(f"  ✅ {name}")
    
    if success_count < total_count:
        print(f"\nNeed fixes:")
        for name, status in results.items():
            if status != "SUCCESS":
                print(f"  ❌ {name}: {status}")
    
    return results

if __name__ == "__main__":
    results = test_known_working_architectures()
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    print(f"\nFinal result: {success_count} working architectures out of {len(results)} tested.")