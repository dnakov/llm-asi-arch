#!/usr/bin/env python3
"""
Claude Code SDK automation script for fixing PyTorch to MLX architecture conversions.

This script:
1. Iterates through all PyTorch architectures in pytorch_arch/
2. Finds corresponding MLX files in mlx_architectures/
3. Uses Claude Code SDK to fix MLX implementations to match PyTorch
4. Tests each fix using the existing test framework
5. Tracks progress and results
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil

class ClaudeCodeMLXFixer:
    def __init__(self):
        self.pytorch_dir = Path("pytorch_arch")
        self.mlx_dir = Path("mlx_architectures")
        self.results_file = Path("claude_fix_results.json")
        self.progress_file = Path("claude_fix_progress.json")
        
        # Load existing progress if available
        self.progress = self.load_progress()
        
    def load_progress(self) -> Dict:
        """Load existing progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "completed": [],
            "failed": [],
            "skipped": [],
            "current_index": 0
        }
    
    def save_progress(self):
        """Save current progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_architecture_pairs(self) -> List[Tuple[str, Path, Path]]:
        """Get list of (name, pytorch_path, mlx_path) tuples"""
        pairs = []
        
        for pytorch_file in sorted(self.pytorch_dir.glob("*.py")):
            arch_name = pytorch_file.stem
            mlx_file = self.mlx_dir / f"{arch_name}_mlx.py"
            
            if mlx_file.exists():
                pairs.append((arch_name, pytorch_file, mlx_file))
            else:
                print(f"⚠️  No MLX file found for {arch_name}")
        
        return pairs
    
    def test_architecture(self, arch_name: str) -> Tuple[bool, str]:
        """Test if an architecture works using comprehensive testing including DeltaNet class"""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import importlib.util
            
            # Test 1: Syntax
            mlx_path = f"mlx_architectures/{arch_name}_mlx.py"
            if not os.path.exists(mlx_path):
                return False, f"File not found: {mlx_path}"
            
            try:
                with open(mlx_path, 'r') as f:
                    source = f.read()
                compile(source, mlx_path, 'exec')
            except SyntaxError as e:
                return False, f"Syntax error line {e.lineno}: {e.msg}"
            except Exception as e:
                return False, f"Compile error: {str(e)}"
            
            # Test 2: Import
            try:
                spec = importlib.util.spec_from_file_location(arch_name, mlx_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                return False, f"Import error: {str(e)}"
            
            # Test 3: DeltaNet class exists and can be instantiated
            if not hasattr(module, 'DeltaNet'):
                return False, "No DeltaNet class found"
            
            DeltaNetClass = module.DeltaNet
            
            # Test 4: Can instantiate DeltaNet
            try:
                model = DeltaNetClass(
                    hidden_size=256,
                    num_heads=4,
                    use_gate=False,
                    use_short_conv=True,
                    layer_idx=0
                )
            except Exception as e:
                return False, f"DeltaNet instantiation error: {str(e)}"
            
            # Test 5: Forward pass works
            try:
                batch_size, seq_len = 2, 32
                x = mx.random.normal((batch_size, seq_len, 256))
                output = model(x)
                
                # Handle different return formats
                if isinstance(output, tuple):
                    output = output[0]
                    
                if output.shape != (batch_size, seq_len, 256):
                    return False, f"Wrong output shape: {output.shape}, expected: {(batch_size, seq_len, 256)}"
                    
            except Exception as e:
                return False, f"Forward pass error: {str(e)}"
            
            return True, "All tests passed (syntax, import, DeltaNet class, instantiation, forward pass)"
            
        except Exception as e:
            return False, f"Test framework error: {str(e)}"
    
    def create_claude_prompt(self, arch_name: str, pytorch_path: Path, mlx_path: Path) -> str:
        """Create a comprehensive prompt for Claude to fix the MLX architecture"""
        
        prompt = f"""I need you to fix the MLX architecture implementation to properly match the PyTorch implementation.

ARCHITECTURE: {arch_name}

TASK: Fix the MLX implementation in mlx_architectures/{arch_name}_mlx.py to:
1. Match the exact functionality of the PyTorch version
2. Use proper MLX framework patterns and best practices
3. Ensure all imports are correct for MLX
4. Fix any syntax errors or type issues
5. Maintain the same architecture logic and parameters

PYTORCH REFERENCE (pytorch_arch/{arch_name}.py):
This is the reference implementation that the MLX version should match.

MLX CURRENT IMPLEMENTATION (mlx_architectures/{arch_name}_mlx.py):
This is the current MLX implementation that needs to be fixed.

REQUIREMENTS:
- Use mlx.nn instead of torch.nn
- Use mlx.core instead of torch
- Convert torch.Tensor to mlx.core.array
- Fix any MLX-specific syntax issues
- Ensure proper MLX initialization patterns
- Keep the same class names and method signatures
- Maintain the same forward pass logic
- Fix any import errors

SOME PREVIOUS ERRORS AND FIXES:
  1. Array Updates

  Problem: Used PyTorch-style tensor.at[indices].set(values)
  Fix: Use concatenation: mx.concatenate([prefix, new_values, suffix], axis=dim)

  2. Transpose Operations

  Problem: Used .transpose(0, 1, 2, 4, 3) with too many dimensions
  Fix: Use mx.swapaxes(tensor, -2, -1) for last two dimensions

  3. Missing Rearrange Patterns

  Problem: einops replacement missing common patterns
  Fix: Add missing patterns like 'b l h d -> b l (h d)' to rearrange function

  4. Conv1d Format

  Problem: Assumed PyTorch format (batch, channels, length)
  Fix: MLX uses (batch, length, channels) - remove unnecessary transposes

  5. Module Registration

  Problem: Convolutions in lists not properly registered as parameters
  Fix: Use setattr(self, f'conv_{{i}}', conv) to register each conv layer

  6. MLX-Specific Syntax

  Problem: Used PyTorch function signatures
  Fix:
  - nn.elu(x, alpha) → nn.elu(x)
  - F.pad() → mx.pad()
  - torch.Tensor → mx.array

  Quick Fix Template

  # 1. Import fixes
  import mlx.core as mx
  import mlx.nn as nn

  # 2. Array updates
  # OLD: tensor.at[indices].set(values)
  # NEW: mx.concatenate([prefix, values, suffix], axis=dim)

  # 3. Transposes
  # OLD: tensor.transpose(0,1,2,4,3)
  # NEW: mx.swapaxes(tensor, -2, -1)

  # 4. Conv1d usage
  # Input format: (batch, length, channels) - no transpose needed

  # 5. Module registration
  for i, layer in enumerate(layers):
      setattr(self, f'layer_{{i}}', layer)

TESTING:
After making changes, the architecture will be tested with:
1. Syntax validation (compile check)
2. Import validation (can be imported without errors)
3. Class detection (proper MLX Module inheritance)

Please read both files and fix the MLX implementation to work correctly while maintaining the same functionality as the PyTorch version."""

        return prompt
    
    def run_claude_code(self, prompt: str, max_turns: int = 3) -> Tuple[bool, str]:
        """Run Claude Code SDK with the given prompt"""
        try:
            # Run claude-code with the prompt
            cmd = [
                'claude', '--dangerously-skip-permissions', '--verbose', '--output-format', 'stream-json', '-p', prompt
            ]
            
            # Use Popen for streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Stream stdout in real-time
            stdout_lines = []
            stderr_lines = []
            
            # Read stdout line by line
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Print the line for real-time feedback
                    print(f"      📝 {output.rstrip()}")
                    stdout_lines.append(output)
            
            # Get any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                stderr_lines.append(stderr_output)
                print(f"      ⚠️  {stderr_output.rstrip()}")
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            
            # Combine all output
            full_stdout = ''.join(stdout_lines)
            full_stderr = ''.join(stderr_lines)
            
            if return_code == 0:
                try:
                    response_data = json.loads(full_stdout)
                    return True, response_data.get('result', 'Success')
                except json.JSONDecodeError:
                    return True, full_stdout
            else:
                return False, f"Claude failed: {full_stderr}"
                
        except Exception as e:
            return False, f"Error running Claude: {str(e)}"
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file"""
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def fix_architecture(self, arch_name: str, pytorch_path: Path, mlx_path: Path) -> Dict:
        """Fix a single architecture using Claude Code"""
        
        print(f"\n🔧 Fixing {arch_name}...")
        
        # Check if already completed
        if arch_name in self.progress["completed"]:
            print(f"  ✅ Already completed, skipping")
            return {"status": "already_completed", "message": "Previously fixed"}
        
        # Test current state
        print(f"  🧪 Testing current state...")
        test_ok, test_msg = self.test_architecture(arch_name)
        if test_ok:
            print(f"  ✅ Already working: {test_msg}")
            self.progress["completed"].append(arch_name)
            self.save_progress()
            return {"status": "already_working", "message": test_msg}
        
        print(f"  ❌ Current issues: {test_msg}")
        
        # Create backup
        backup_path = self.backup_file(mlx_path)
        print(f"  💾 Created backup: {backup_path}")
        
        # Create Claude prompt
        prompt = self.create_claude_prompt(arch_name, pytorch_path, mlx_path)
        
        # Run Claude Code with adaptive max_turns based on error type
        max_turns = 3  # Default
        
        if "Pattern" in test_msg and "not implemented" in test_msg:
            max_turns = 2  # Fewer turns for focused pattern fixes
        elif "Syntax error" in test_msg:
            max_turns = 2  # Fewer turns for simple syntax fixes
        
        print(f"  🤖 Running Claude Code...")
        claude_ok, claude_msg = self.run_claude_code(prompt, max_turns=max_turns)
        
        if not claude_ok:
            print(f"  ❌ Claude failed: {claude_msg}")
            return {"status": "claude_failed", "message": claude_msg}
        
        print(f"  ✅ Claude completed")
        
        # Test the fix
        print(f"  🧪 Testing fix...")
        test_ok, test_msg = self.test_architecture(arch_name)
        
        if test_ok:
            print(f"  ✅ Fix successful: {test_msg}")
            self.progress["completed"].append(arch_name)
            self.save_progress()
            return {
                "status": "success", 
                "message": test_msg,
                "claude_response": claude_msg
            }
        else:
            print(f"  ❌ Fix failed: {test_msg}")
            # Restore backup
            shutil.copy2(backup_path, mlx_path)
            print(f"  🔄 Restored from backup")
            
            self.progress["failed"].append(arch_name)
            self.save_progress()
            return {
                "status": "fix_failed", 
                "message": test_msg,
                "claude_response": claude_msg
            }
    
    def run_batch_fix(self, start_index: int = 37, max_architectures: Optional[int] = None):
        """Run batch fixing process"""
        
        pairs = self.get_architecture_pairs()
        total = len(pairs)
        
        print(f"🚀 Starting Claude Code MLX Fixer")
        print(f"📁 Found {total} architecture pairs")
        print(f"📊 Progress: {len(self.progress['completed'])} completed, {len(self.progress['failed'])} failed")
        
        if max_architectures:
            pairs = pairs[start_index:start_index + max_architectures]
            print(f"🎯 Processing {len(pairs)} architectures (starting from index {start_index})")
        else:
            pairs = pairs[start_index:]
            print(f"🎯 Processing {len(pairs)} architectures (starting from index {start_index})")
        
        results = []
        
        for i, (arch_name, pytorch_path, mlx_path) in enumerate(pairs, start_index + 1):
            print(f"\n{'='*60}")
            print(f"[{i:3d}/{total}] Processing {arch_name}")
            print(f"{'='*60}")
            
            result = self.fix_architecture(arch_name, pytorch_path, mlx_path)
            result["architecture"] = arch_name
            result["index"] = i
            results.append(result)
            
            # Save intermediate results
            self.save_results(results)
            
            # Small delay to be respectful
            time.sleep(1)
        
        # Final summary
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save results to file"""
        summary = {
            "total_processed": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "already_working": len([r for r in results if r["status"] == "already_working"]),
            "already_completed": len([r for r in results if r["status"] == "already_completed"]),
            "claude_failed": len([r for r in results if r["status"] == "claude_failed"]),
            "fix_failed": len([r for r in results if r["status"] == "fix_failed"]),
            "timestamp": time.time()
        }
        
        data = {
            "summary": summary,
            "results": results
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self, results: List[Dict]):
        """Print final summary"""
        print(f"\n{'='*60}")
        print("🎉 CLAUDE CODE MLX FIXER SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        successful = len([r for r in results if r["status"] == "success"])
        already_working = len([r for r in results if r["status"] == "already_working"])
        already_completed = len([r for r in results if r["status"] == "already_completed"])
        claude_failed = len([r for r in results if r["status"] == "claude_failed"])
        fix_failed = len([r for r in results if r["status"] == "fix_failed"])
        
        print(f"📊 Total processed: {total}")
        print(f"✅ Newly fixed: {successful}")
        print(f"✅ Already working: {already_working}")
        print(f"✅ Previously completed: {already_completed}")
        print(f"❌ Claude failed: {claude_failed}")
        print(f"❌ Fix failed: {fix_failed}")
        
        working_total = successful + already_working + already_completed
        success_rate = (working_total / total * 100) if total > 0 else 0
        
        print(f"\n🎯 Overall success rate: {success_rate:.1f}%")
        print(f"💾 Results saved to: {self.results_file}")
        print(f"📈 Progress saved to: {self.progress_file}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix MLX architectures using Claude Code SDK")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--max", type=int, help="Maximum number of architectures to process")
    parser.add_argument("--test-only", action="store_true", help="Only test current architectures")
    parser.add_argument("--resume", action="store_true", help="Resume from last position")
    
    args = parser.parse_args()
    
    fixer = ClaudeCodeMLXFixer()
    
    if args.test_only:
        # Just run tests on all architectures
        pairs = fixer.get_architecture_pairs()
        working = 0
        for arch_name, _, _ in pairs:
            test_ok, test_msg = fixer.test_architecture(arch_name)
            status = "✅" if test_ok else "❌"
            print(f"{status} {arch_name}: {test_msg}")
            if test_ok:
                working += 1
        
        print(f"\n📊 {working}/{len(pairs)} architectures working ({working/len(pairs)*100:.1f}%)")
        return
    
    start_index = args.start
    if args.resume:
        start_index = fixer.progress.get("current_index", 0)
        print(f"🔄 Resuming from index {start_index}")
    
    # Run the batch fix
    fixer.run_batch_fix(start_index=start_index, max_architectures=args.max)

if __name__ == "__main__":
    main()