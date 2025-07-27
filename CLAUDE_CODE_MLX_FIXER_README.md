# Claude Code MLX Architecture Fixer

This script uses the Claude Code SDK to automatically fix PyTorch to MLX architecture conversions by having Claude analyze and correct the MLX implementations to match their PyTorch counterparts.

## Setup

1. **Install Claude Code CLI:**
   ```bash
   ./setup_claude_code.sh
   ```

2. **Set your Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

3. **Test the setup:**
   ```bash
   python test_claude_sdk.py
   ```

## Usage

### Test Current Architecture Status
```bash
# Check which architectures are currently working
python claude_code_mlx_fixer.py --test-only
```

### Fix Architectures

```bash
# Fix first 5 architectures (recommended for testing)
python claude_code_mlx_fixer.py --max 5

# Fix all architectures starting from index 10
python claude_code_mlx_fixer.py --start 10

# Resume from where you left off
python claude_code_mlx_fixer.py --resume

# Fix all architectures (full run)
python claude_code_mlx_fixer.py
```

## How It Works

1. **Architecture Pairing**: Finds matching PyTorch and MLX architecture files
2. **Current State Testing**: Tests if MLX architecture already works
3. **Claude Analysis**: Uses Claude Code SDK to analyze both implementations
4. **Automated Fixing**: Claude makes necessary changes to fix MLX compatibility
5. **Verification**: Tests the fixed architecture using existing test framework
6. **Progress Tracking**: Saves progress and results for resumability

## Features

- **Intelligent Prompting**: Creates comprehensive prompts for Claude with context
- **Backup System**: Automatically backs up files before making changes
- **Progress Tracking**: Saves progress to resume interrupted sessions
- **Comprehensive Testing**: Uses existing test framework to verify fixes
- **Detailed Logging**: Tracks all results and errors for analysis
- **Batch Processing**: Can process all 106 architectures systematically

## Output Files

- `claude_fix_results.json` - Detailed results of all fixing attempts
- `claude_fix_progress.json` - Progress tracking for resumability
- `*.backup` files - Automatic backups of original MLX files

## Architecture Fix Process

For each architecture, Claude:

1. **Analyzes** the PyTorch reference implementation
2. **Identifies** issues in the current MLX implementation
3. **Applies** MLX-specific fixes:
   - Convert `torch.nn` → `mlx.nn`
   - Convert `torch.Tensor` → `mlx.core.array`
   - Fix MLX initialization patterns
   - Correct import statements
   - Maintain same functionality and API

4. **Verifies** the fix works through automated testing

## Example Session

```bash
$ python claude_code_mlx_fixer.py --max 3

🚀 Starting Claude Code MLX Fixer
📁 Found 106 architecture pairs
🎯 Processing 3 architectures (starting from index 0)

============================================================
[  1/106] Processing delta_net_abrgf
============================================================

🔧 Fixing delta_net_abrgf...
  🧪 Testing current state...
  ❌ Current issues: Import error: No module named 'torch'
  💾 Created backup: mlx_architectures/delta_net_abrgf_mlx.py.backup
  🤖 Running Claude Code...
  ✅ Claude completed
  🧪 Testing fix...
  ✅ Fix successful: All tests passed

============================================================
[  2/106] Processing delta_net_acfg
============================================================
...
```

## Troubleshooting

### Claude Code Not Found
```bash
npm install -g @anthropic-ai/claude-code
```

### API Key Issues
```bash
export ANTHROPIC_API_KEY='your-key-here'
# Test with: python test_claude_sdk.py
```

### Permission Issues
```bash
chmod +x claude_code_mlx_fixer.py
chmod +x setup_claude_code.sh
```

### Resume After Interruption
```bash
python claude_code_mlx_fixer.py --resume
```

## Success Metrics

The script tracks:
- **Syntax Success Rate**: Architectures with valid Python syntax
- **Import Success Rate**: Architectures that can be imported without errors
- **Overall Success Rate**: Architectures that pass all tests
- **Fix Success Rate**: Architectures successfully fixed by Claude

Expected improvement: 50%+ → 90%+ working architectures after Claude fixes.