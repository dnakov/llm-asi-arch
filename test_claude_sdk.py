#!/usr/bin/env python3
"""
Test script to verify Claude Code SDK is working properly
"""

import subprocess
import json
import sys

def test_claude_installation():
    """Test if Claude Code CLI is installed and working"""
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Claude Code CLI installed: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Claude Code CLI not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Claude Code CLI not found. Please run: npm install -g @anthropic-ai/claude-code")
        return False

def test_claude_api():
    """Test if Claude Code can make API calls"""
    try:
        # Simple test prompt
        cmd = [
            'claude', '-p', 'Say "Hello from Claude Code SDK test"',
            '--output-format', 'json',
            '--max-turns', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout)
                response_text = response_data.get('result', '')
                print(f"✅ Claude API test successful")
                print(f"   Response: {response_text[:100]}...")
                return True
            except json.JSONDecodeError:
                print(f"✅ Claude API working (non-JSON response)")
                print(f"   Response: {result.stdout[:100]}...")
                return True
        else:
            print(f"❌ Claude API test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Claude API test timed out")
        return False
    except Exception as e:
        print(f"❌ Claude API test error: {str(e)}")
        return False

def test_file_access():
    """Test if Claude can access files in the current directory"""
    try:
        cmd = [
            'claude', '-p', 'List the files in the current directory and tell me how many .py files there are',
            '--output-format', 'json',
            '--max-turns', '2'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        if result.returncode == 0:
            print("✅ Claude file access test successful")
            return True
        else:
            print(f"❌ Claude file access test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Claude file access test timed out")
        return False
    except Exception as e:
        print(f"❌ Claude file access test error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Claude Code SDK Integration")
    print("=" * 50)
    
    tests = [
        ("Claude Installation", test_claude_installation),
        ("Claude API", test_claude_api),
        ("File Access", test_file_access)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   💡 This may indicate missing API key or network issues")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Claude Code SDK is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check your setup:")
        print("   1. Ensure ANTHROPIC_API_KEY is set")
        print("   2. Check internet connection")
        print("   3. Verify Claude Code CLI installation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)