#!/bin/bash
"""
Setup script for Claude Code SDK MLX fixer
"""

echo "🚀 Setting up Claude Code SDK for MLX Architecture Fixing"
echo "========================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed"
    exit 1
fi

echo "✅ npm found: $(npm --version)"

# Install Claude Code CLI globally
echo "📦 Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Check if installation was successful
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code installation failed"
    exit 1
fi

echo "✅ Claude Code CLI installed: $(claude --version)"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY environment variable not set"
    echo "Please set your Anthropic API key:"
    echo "export ANTHROPIC_API_KEY='your-api-key-here'"
    echo ""
    echo "You can get an API key from: https://console.anthropic.com/"
else
    echo "✅ ANTHROPIC_API_KEY is set"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Make the fixer script executable
chmod +x claude_code_mlx_fixer.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage examples:"
echo "  # Test current architectures:"
echo "  python claude_code_mlx_fixer.py --test-only"
echo ""
echo "  # Fix first 5 architectures:"
echo "  python claude_code_mlx_fixer.py --max 5"
echo ""
echo "  # Resume from where you left off:"
echo "  python claude_code_mlx_fixer.py --resume"
echo ""
echo "  # Fix all architectures:"
echo "  python claude_code_mlx_fixer.py"