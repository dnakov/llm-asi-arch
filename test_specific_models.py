#!/usr/bin/env python3
"""Test specific MLX architectures to identify runtime issues."""

import sys
import traceback
import mlx.core as mx
import mlx.nn as nn

def test_model(model_class, model_name):
    """Test a specific model architecture."""
    print(f"\n=== Testing {model_name} ===")
    
    try:
        # Initialize model
        model = model_class(hidden_size=256, num_heads=4)
        print(f"✓ Model initialization successful")
        
        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 16, 256
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        
        output = model(x)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test that output is properly formed
        if isinstance(output, (list, tuple)):
            print(f"✓ Output is tuple/list with {len(output)} elements")
            if len(output) > 0:
                print(f"  - First element shape: {output[0].shape}")
        else:
            print(f"✓ Output is array with shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Test the three problematic models."""
    
    # Test pathgated model
    sys.path.append('/Users/daniel/dev/asi/mlx_architectures')
    
    try:
        from delta_net_pathgated_mlx import DeltaNet as PathgatedDeltaNet
        test_model(PathgatedDeltaNet, "delta_net_pathgated_mlx")
    except Exception as e:
        print(f"❌ Failed to import pathgated model: {e}")
    
    try:
        from delta_net_ms_adaptive_gstat3_mlx import DeltaNet as AdaptiveDeltaNet
        test_model(AdaptiveDeltaNet, "delta_net_ms_adaptive_gstat3_mlx")
    except Exception as e:
        print(f"❌ Failed to import adaptive model: {e}")
    
    try:
        from delta_net_triscale_mlx import DeltaNet as TriscaleDeltaNet
        test_model(TriscaleDeltaNet, "delta_net_triscale_mlx")
    except Exception as e:
        print(f"❌ Failed to import triscale model: {e}")

if __name__ == "__main__":
    main()