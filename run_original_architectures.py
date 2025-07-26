#!/usr/bin/env python3
"""
Run and evaluate the 106 original MLX-converted architectures
"""

import json
import os
import importlib.util
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Dict, Optional
import time
import sys

class OriginalArchitectureRunner:
    """Load and run the 106 original architectures converted to MLX"""
    
    def __init__(self):
        self.architectures_dir = "mlx_architectures"
        self.original_data = self._load_original_data()
        
    def _load_original_data(self) -> Dict:
        """Load the original 106.json data"""
        with open('106.json', 'r') as f:
            architectures = json.load(f)
        
        # Create lookup by name
        return {arch['name']: arch for arch in architectures}
    
    def list_available_architectures(self) -> List[str]:
        """List all available converted architectures"""
        if not os.path.exists(self.architectures_dir):
            return []
        
        architectures = []
        for filename in os.listdir(self.architectures_dir):
            if filename.endswith('_mlx.py'):
                arch_name = filename.replace('_mlx.py', '')
                architectures.append(arch_name)
        
        return sorted(architectures)
    
    def load_architecture(self, arch_name: str):
        """Dynamically load a converted architecture"""
        mlx_filename = f"{arch_name}_mlx.py"
        mlx_path = os.path.join(self.architectures_dir, mlx_filename)
        
        if not os.path.exists(mlx_path):
            raise FileNotFoundError(f"Architecture {arch_name} not found at {mlx_path}")
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(arch_name, mlx_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the main architecture class (usually DeltaNet)
        architecture_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, nn.Module) and 
                attr.__name__ in ['DeltaNet', 'Model', 'Architecture']):
                architecture_class = attr
                break
        
        if architecture_class is None:
            # Fallback - look for any nn.Module subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, nn.Module) and 
                    attr.__name__ != 'Module'):
                    architecture_class = attr
                    break
        
        return architecture_class, module
    
    def create_test_data(self, batch_size: int = 2, seq_len: int = 64, vocab_size: int = 1000):
        """Create synthetic test data"""
        # Input tokens
        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # Attention mask
        attention_mask = mx.ones((batch_size, seq_len))
        
        # Labels for loss computation
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        return input_ids, attention_mask, labels
    
    def test_architecture(self, arch_name: str, verbose: bool = True) -> Dict:
        """Test a single architecture"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing Architecture: {arch_name}")
            print(f"{'='*60}")
        
        try:
            # Load architecture
            architecture_class, module = self.load_architecture(arch_name)
            
            if verbose:
                print(f"✅ Loaded architecture class: {architecture_class.__name__}")
            
            # Get original data
            original_data = self.original_data.get(arch_name, {})
            original_score = original_data.get('score', 'Unknown')
            original_params = original_data.get('parameters', 'Unknown')
            
            if verbose:
                print(f"📊 Original Score: {original_score}")
                print(f"📏 Original Parameters: {original_params}")
            
            # Create model instance
            try:
                model = architecture_class(
                    hidden_size=512,  # Smaller for testing
                    num_heads=8,
                    vocab_size=1000
                )
            except TypeError:
                # Try with minimal args
                model = architecture_class()
            
            if verbose:
                print(f"🏗️ Model created successfully")
            
            # Create test data
            input_ids, attention_mask, labels = self.create_test_data()
            
            if verbose:
                print(f"📝 Test data shape: {input_ids.shape}")
            
            # Test forward pass
            start_time = time.time()
            
            try:
                if hasattr(model, 'forward'):
                    output = model(input_ids, attention_mask=attention_mask)
                else:
                    output = model(input_ids)
                
                forward_time = time.time() - start_time
                
                if isinstance(output, tuple):
                    output = output[0]  # Take first element if tuple
                
                if verbose:
                    print(f"✅ Forward pass successful")
                    print(f"⚡ Forward time: {forward_time:.4f}s")
                    print(f"📤 Output shape: {output.shape}")
                
                return {
                    'status': 'success',
                    'architecture': arch_name,
                    'output_shape': output.shape,
                    'forward_time': forward_time,
                    'original_score': original_score,
                    'original_params': original_params
                }
                
            except Exception as e:
                if verbose:
                    print(f"❌ Forward pass failed: {e}")
                
                return {
                    'status': 'forward_failed',
                    'architecture': arch_name,
                    'error': str(e),
                    'original_score': original_score,
                    'original_params': original_params
                }
        
        except Exception as e:
            if verbose:
                print(f"❌ Architecture loading failed: {e}")
            
            return {
                'status': 'load_failed',
                'architecture': arch_name,
                'error': str(e)
            }
    
    def test_all_architectures(self, limit: Optional[int] = None) -> Dict:
        """Test all converted architectures"""
        architectures = self.list_available_architectures()
        
        if limit:
            architectures = architectures[:limit]
        
        print(f"🚀 Testing {len(architectures)} architectures...")
        
        results = {
            'successful': [],
            'failed': [],
            'summary': {}
        }
        
        for i, arch_name in enumerate(architectures, 1):
            print(f"\n[{i}/{len(architectures)}] Testing {arch_name}...")
            
            result = self.test_architecture(arch_name, verbose=False)
            
            if result['status'] == 'success':
                results['successful'].append(result)
                print(f"✅ {arch_name} - Forward time: {result['forward_time']:.4f}s")
            else:
                results['failed'].append(result)
                print(f"❌ {arch_name} - {result['status']}: {result.get('error', 'Unknown error')}")
        
        # Summary
        total = len(architectures)
        successful = len(results['successful'])
        failed = len(results['failed'])
        
        results['summary'] = {
            'total_tested': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0
        }
        
        print(f"\n{'='*60}")
        print(f"TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tested: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        
        return results
    
    def benchmark_top_architectures(self, top_n: int = 10):
        """Benchmark the top N architectures by original score"""
        # Get architectures sorted by score
        arch_scores = []
        for name, data in self.original_data.items():
            if isinstance(data.get('score'), (int, float)):
                arch_scores.append((name, data['score']))
        
        arch_scores.sort(key=lambda x: x[1], reverse=True)
        top_architectures = [name for name, score in arch_scores[:top_n]]
        
        print(f"🏆 Benchmarking Top {top_n} Architectures by Original Score")
        print(f"{'='*60}")
        
        for i, arch_name in enumerate(top_architectures, 1):
            score = self.original_data[arch_name]['score']
            print(f"\n[{i}/{top_n}] {arch_name} (Score: {score:.4f})")
            result = self.test_architecture(arch_name, verbose=True)

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the 106 original ASI-Arch architectures")
    parser.add_argument('--list', action='store_true', help='List available architectures')
    parser.add_argument('--test', type=str, help='Test specific architecture by name')
    parser.add_argument('--test-all', action='store_true', help='Test all architectures')
    parser.add_argument('--limit', type=int, help='Limit number of architectures to test')
    parser.add_argument('--benchmark', type=int, default=10, help='Benchmark top N architectures')
    
    args = parser.parse_args()
    
    runner = OriginalArchitectureRunner()
    
    if args.list:
        architectures = runner.list_available_architectures()
        print(f"📋 Available Architectures ({len(architectures)}):")
        for i, arch in enumerate(architectures, 1):
            original_data = runner.original_data.get(arch, {})
            score = original_data.get('score', 'N/A')
            print(f"  {i:3d}. {arch} (Score: {score})")
    
    elif args.test:
        result = runner.test_architecture(args.test)
        return result['status'] == 'success'
    
    elif args.test_all:
        results = runner.test_all_architectures(limit=args.limit)
        return results['summary']['success_rate'] > 0.5
    
    else:
        # Default: benchmark top architectures
        runner.benchmark_top_architectures(args.benchmark)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)