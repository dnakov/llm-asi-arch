#!/usr/bin/env python3
"""Comprehensive test of the three fixed MLX architectures."""

import sys
import traceback
import mlx.core as mx
import mlx.nn as nn

def comprehensive_test(model_class, model_name):
    """Comprehensive test of a model architecture."""
    print(f"\n=== Comprehensive Testing {model_name} ===")
    
    tests_passed = 0
    total_tests = 6
    
    try:
        # Test 1: Basic initialization
        model = model_class(hidden_size=256, num_heads=4)
        print(f"✓ Test 1/6: Model initialization successful")
        tests_passed += 1
        
        # Test 2: Small batch forward pass
        batch_size, seq_len, hidden_size = 1, 8, 256
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        output = model(x)
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Test 2/6: Small batch forward pass successful")
        tests_passed += 1
        
        # Test 3: Regular batch forward pass
        batch_size, seq_len, hidden_size = 2, 16, 256
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        output = model(x)
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Test 3/6: Regular batch forward pass successful")
        tests_passed += 1
        
        # Test 4: Longer sequence
        batch_size, seq_len, hidden_size = 1, 64, 256
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        output = model(x)
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Test 4/6: Long sequence forward pass successful")
        tests_passed += 1
        
        # Test 5: Different hidden size model
        model_large = model_class(hidden_size=512, num_heads=8)
        batch_size, seq_len, hidden_size = 2, 16, 512
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        output = model_large(x)
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Test 5/6: Large model forward pass successful")
        tests_passed += 1
        
        # Test 6: Gradient flow (ensure no NaN/Inf)
        x = mx.random.normal((2, 16, 256))
        output = model(x)
        loss = mx.mean(output ** 2)
        
        # Check for NaN/Inf
        assert not mx.any(mx.isnan(output)), "Output contains NaN values"
        assert not mx.any(mx.isinf(output)), "Output contains Inf values"
        assert not mx.isnan(loss), "Loss is NaN"
        assert not mx.isinf(loss), "Loss is Inf"
        print(f"✓ Test 6/6: No NaN/Inf in output and loss")
        tests_passed += 1
        
        print(f"🎉 All {tests_passed}/{total_tests} tests passed for {model_name}!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed at {tests_passed}/{total_tests}: {type(e).__name__}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Test the three fixed models comprehensively."""
    
    sys.path.append('/Users/daniel/dev/asi/mlx_architectures')
    
    results = {}
    
    try:
        from delta_net_pathgated_mlx import DeltaNet as PathgatedDeltaNet
        results['pathgated'] = comprehensive_test(PathgatedDeltaNet, "delta_net_pathgated_mlx")
    except Exception as e:
        print(f"❌ Failed to import pathgated model: {e}")
        results['pathgated'] = False
    
    try:
        from delta_net_ms_adaptive_gstat3_mlx import DeltaNet as AdaptiveDeltaNet
        results['adaptive'] = comprehensive_test(AdaptiveDeltaNet, "delta_net_ms_adaptive_gstat3_mlx")
    except Exception as e:
        print(f"❌ Failed to import adaptive model: {e}")
        results['adaptive'] = False
    
    try:
        from delta_net_triscale_mlx import DeltaNet as TriscaleDeltaNet
        results['triscale'] = comprehensive_test(TriscaleDeltaNet, "delta_net_triscale_mlx")
    except Exception as e:
        print(f"❌ Failed to import triscale model: {e}")
        results['triscale'] = False
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"pathgated_mlx: {'✅ PASS' if results['pathgated'] else '❌ FAIL'}")
    print(f"adaptive_gstat3_mlx: {'✅ PASS' if results['adaptive'] else '❌ FAIL'}")
    print(f"triscale_mlx: {'✅ PASS' if results['triscale'] else '❌ FAIL'}")
    
    total_passed = sum(results.values())
    print(f"\nOverall: {total_passed}/3 models are now working correctly!")

if __name__ == "__main__":
    main()