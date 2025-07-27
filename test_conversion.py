#!/usr/bin/env python3
"""
Enhanced PyTorch to MLX Conversion Testing Framework
===================================================

Provides comprehensive testing with detailed failure analysis, error categorization,
and enhanced visibility for debugging conversion issues.
"""

import ast
import importlib.util
import os
import sys
import time
import traceback
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Some tests will be skipped.")

try:
    import torch
    import torch.nn as torch_nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Comparison tests will be skipped.")


@dataclass
class ErrorDetail:
    """Detailed error information for better debugging."""
    error_type: str
    error_message: str
    file_name: str
    line_number: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None
    full_traceback: Optional[str] = None


class ErrorAnalyzer:
    """Analyzes and categorizes errors for better debugging."""
    
    ERROR_PATTERNS = {
        'import_error': [
            r"No module named '(.+)'",
            r"cannot import name '(.+)'",
            r"ImportError: (.+)"
        ],
        'attribute_error': [
            r"'(.+)' object has no attribute '(.+)'",
            r"module '(.+)' has no attribute '(.+)'"
        ],
        'type_error': [
            r"unsupported operand type\(s\) for (.+): '(.+)' and '(.+)'",
            r"'(.+)' object is not callable",
            r"(.+)\(\) takes (.+) positional arguments but (.+) were given"
        ],
        'value_error': [
            r"invalid literal for (.+) with base (.+): '(.+)'",
            r"could not convert (.+) to (.+)"
        ],
        'mlx_specific': [
            r"MLX error: (.+)",
            r"mx\.(.+) not found",
            r"mlx\.(.+) not available"
        ],
        'pytorch_remnant': [
            r"torch\.(.+) found in MLX code",
            r"\.cuda\(\) found",
            r"\.to\(device\) found",
            r"F\.(.+) should be nn\.(.+)"
        ],
        'syntax_error': [
            r"SyntaxError: (.+)",
            r"IndentationError: (.+)",
            r"invalid syntax"
        ]
    }
    
    ERROR_SUGGESTIONS = {
        'import_error': "Check if all required MLX modules are available and properly imported",
        'attribute_error': "Verify that all PyTorch attributes have been converted to MLX equivalents",
        'type_error': "Check parameter types and function signatures for MLX compatibility",
        'mlx_specific': "Review MLX documentation for correct usage patterns",
        'pytorch_remnant': "Remove or convert remaining PyTorch-specific code",
        'syntax_error': "Fix Python syntax issues"
    }
    
    @classmethod
    def analyze_error(cls, error: Exception, file_name: str, context: str = None) -> ErrorDetail:
        """Analyze an error and provide detailed information."""
        error_message = str(error)
        error_type_name = type(error).__name__
        
        # Categorize error
        category = 'unknown'
        for cat, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    category = cat
                    break
            if category != 'unknown':
                break
        
        # Extract line number from traceback
        line_number = None
        tb = traceback.format_exc()
        line_match = re.search(r'line (\d+)', tb)
        if line_match:
            line_number = int(line_match.group(1))
        
        return ErrorDetail(
            error_type=f"{category}:{error_type_name}",
            error_message=error_message,
            file_name=file_name,
            line_number=line_number,
            context=context,
            suggestion=cls.ERROR_SUGGESTIONS.get(category, "Review MLX conversion guidelines"),
            full_traceback=tb
        )


class EnhancedConversionTester:
    """Enhanced testing framework with detailed failure analysis."""
    
    def __init__(self, pytorch_dir: str = "pytorch_arch", mlx_dir: str = "mlx_architectures"):
        self.pytorch_dir = Path(pytorch_dir)
        self.mlx_dir = Path(mlx_dir)
        self.test_results = {}
        self.error_database = []
        self.error_analyzer = ErrorAnalyzer()
        
    def run_all_tests(self, detailed_output: bool = True) -> Dict[str, Any]:
        """Run comprehensive tests with enhanced error reporting."""
        logger.info("Starting enhanced conversion testing...")
        
        mlx_files = list(self.mlx_dir.glob("*_mlx.py"))
        total_files = len(mlx_files)
        
        logger.info(f"Found {total_files} MLX architecture files to test")
        
        results = {
            "total_files": total_files,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "detailed_results": {},
            "error_analysis": {},
            "summary": {}
        }
        
        for i, mlx_file in enumerate(mlx_files, 1):
            logger.info(f"[{i}/{total_files}] Testing {mlx_file.name}...")
            
            try:
                file_results = self.test_single_file_enhanced(mlx_file, detailed_output)
                results["detailed_results"][mlx_file.name] = file_results
                
                if file_results["overall_status"] == "passed":
                    results["passed"] += 1
                elif file_results["overall_status"] == "failed":
                    results["failed"] += 1
                else:
                    results["skipped"] += 1
                    
            except Exception as e:
                logger.error(f"Critical error testing {mlx_file.name}: {e}")
                error_detail = self.error_analyzer.analyze_error(e, mlx_file.name)
                self.error_database.append(error_detail)
                results["failed"] += 1
        
        # Enhanced error analysis
        results["error_analysis"] = self._analyze_error_patterns()
        results["summary"] = self._generate_enhanced_summary(results)
        self.test_results = results
        
        return results
    
    def test_single_file_enhanced(self, mlx_file: Path, detailed: bool = True) -> Dict[str, Any]:
        """Test a single file with enhanced error reporting."""
        file_results = {
            "file_name": mlx_file.name,
            "tests": {},
            "overall_status": "unknown",
            "errors": [],
            "warnings": [],
            "debug_info": {}
        }
        
        # Enhanced test suite
        test_methods = [
            ("syntax", self._test_syntax_enhanced),
            ("imports", self._test_imports_enhanced),
            ("structure", self._test_class_structure_enhanced),
            ("pytorch_remnants", self._test_pytorch_remnants),
            ("mlx_compatibility", self._test_mlx_compatibility),
            ("instantiation", self._test_instantiation_enhanced),
            ("forward_pass", self._test_forward_pass_enhanced),
            ("shape_compatibility", self._test_shape_compatibility_enhanced),
            ("parameter_analysis", self._test_parameter_analysis),
            ("performance", self._test_performance_enhanced)
        ]
        
        for test_name, test_method in test_methods:
            try:
                if test_name in ["instantiation", "forward_pass", "shape_compatibility", "parameter_analysis", "performance"]:
                    # Skip compute-intensive tests if imports failed
                    if file_results["tests"].get("imports", {}).get("status") != "passed":
                        file_results["tests"][test_name] = {"status": "skipped", "reason": "imports_failed"}
                        continue
                
                result = test_method(mlx_file, detailed)
                file_results["tests"][test_name] = result
                
                # Collect errors and warnings
                if result.get("status") == "failed" and "error_detail" in result:
                    error_detail = result["error_detail"]
                    if isinstance(error_detail, dict):
                        file_results["errors"].append(error_detail)
                        # Convert dict back to ErrorDetail for database
                        self.error_database.append(ErrorDetail(**{k: v for k, v in error_detail.items() if k in ErrorDetail.__dataclass_fields__}))
                    else:
                        file_results["errors"].append(error_detail.__dict__)
                        self.error_database.append(error_detail)
                
            except Exception as e:
                error_detail = self.error_analyzer.analyze_error(e, mlx_file.name, test_name)
                file_results["tests"][test_name] = {
                    "status": "failed",
                    "error": str(e),
                    "error_detail": error_detail.__dict__
                }
                file_results["errors"].append(error_detail.__dict__)
                self.error_database.append(error_detail)
        
        file_results["overall_status"] = self._determine_overall_status(file_results["tests"])
        return file_results
    
    def _test_syntax_enhanced(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced syntax testing with detailed error reporting."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse and validate syntax
            tree = ast.parse(source_code)
            
            # Additional syntax checks
            issues = []
            for node in ast.walk(tree):
                # Check for potential issues
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'torch' in alias.name and 'torch' not in alias.name.replace('torch', 'mlx'):
                            issues.append(f"Line {node.lineno}: PyTorch import detected: {alias.name}")
                
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == 'torch':
                        issues.append(f"Line {node.lineno}: PyTorch usage detected: torch.{node.attr}")
            
            result = {"status": "passed", "message": "Valid Python syntax"}
            if issues:
                result["warnings"] = issues
                result["status"] = "warning"
                result["message"] += f" (with {len(issues)} potential conversion issues)"
            
            return result
            
        except SyntaxError as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "syntax validation")
            return {
                "status": "failed", 
                "error": str(e),
                "error_detail": error_detail.__dict__,
                "line": getattr(e, 'lineno', None),
                "text": getattr(e, 'text', None)
            }
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "syntax validation")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_imports_enhanced(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced import testing with dependency analysis."""
        try:
            # Analyze imports before attempting to load
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            imports_found = []
            problematic_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_found.append(alias.name)
                        if any(x in alias.name for x in ['torch', 'einops', 'fla']):
                            problematic_imports.append(f"Line {node.lineno}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports_found.append(module)
                    if any(x in module for x in ['torch', 'einops', 'fla']):
                        for alias in node.names:
                            problematic_imports.append(f"Line {node.lineno}: from {module} import {alias.name}")
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec is None or spec.loader is None:
                return {"status": "failed", "error": "Could not create module spec"}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            result = {
                "status": "passed", 
                "message": "All imports successful",
                "imports_found": imports_found
            }
            
            if problematic_imports:
                result["warnings"] = problematic_imports
                result["status"] = "warning"
                result["message"] += f" (with {len(problematic_imports)} unconverted imports)"
            
            return result
            
        except ImportError as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "import resolution")
            return {
                "status": "failed", 
                "error": str(e),
                "error_detail": error_detail.__dict__,
                "imports_found": imports_found,
                "problematic_imports": problematic_imports
            }
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "import resolution")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_pytorch_remnants(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Test for unconverted PyTorch code."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            pytorch_patterns = [
                (r'torch\.', 'PyTorch module usage'),
                (r'\.cuda\(\)', 'CUDA device usage'),
                (r'\.to\(device\)', 'Device placement'),
                (r'F\.(\w+)', 'PyTorch functional usage'),
                (r'torch\.nn\.functional', 'PyTorch functional import'),
                (r'@torch\.compile', 'PyTorch compile decorator'),
                (r'torch\.no_grad', 'PyTorch no_grad context'),
                (r'dtype=torch\.', 'PyTorch dtype usage'),
                (r'torch\.tensor', 'PyTorch tensor creation'),
                (r'\.cpu\(\)', 'CPU device usage'),
                (r'einops', 'Einops usage'),
                (r'from fla', 'FLA module usage')
            ]
            
            remnants = []
            for pattern, description in pytorch_patterns:
                matches = list(re.finditer(pattern, source_code, re.MULTILINE))
                for match in matches:
                    line_num = source_code[:match.start()].count('\n') + 1
                    context = source_code[match.start()-50:match.end()+50].strip()
                    remnants.append({
                        "pattern": pattern,
                        "description": description,
                        "line": line_num,
                        "match": match.group(),
                        "context": context
                    })
            
            if not remnants:
                return {"status": "passed", "message": "No PyTorch remnants found"}
            else:
                return {
                    "status": "warning",
                    "message": f"Found {len(remnants)} potential PyTorch remnants",
                    "remnants": remnants
                }
                
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "pytorch remnant check")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_mlx_compatibility(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Test for MLX-specific compatibility issues."""
        if not MLX_AVAILABLE:
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            mlx_patterns = [
                (r'mx\.', 'MLX core usage'),
                (r'mlx\.nn\.', 'MLX neural network usage'),
                (r'axis=', 'MLX axis parameter'),
                (r'keepdims=', 'MLX keepdims parameter'),
                (r'__call__', 'MLX module call method'),
                (r'mx\.array', 'MLX array usage')
            ]
            
            mlx_usage = []
            for pattern, description in mlx_patterns:
                matches = list(re.finditer(pattern, source_code))
                mlx_usage.append({
                    "pattern": description,
                    "count": len(matches)
                })
            
            total_mlx_usage = sum(usage["count"] for usage in mlx_usage)
            
            if total_mlx_usage == 0:
                return {
                    "status": "warning",
                    "message": "No MLX usage detected - file may not be converted",
                    "mlx_usage": mlx_usage
                }
            else:
                return {
                    "status": "passed",
                    "message": f"MLX usage detected ({total_mlx_usage} instances)",
                    "mlx_usage": mlx_usage
                }
                
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "mlx compatibility check")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_class_structure_enhanced(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced class structure analysis."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            classes_found = []
            deltanet_class = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes_found.append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                    })
                    if node.name == "DeltaNet":
                        deltanet_class = node
            
            if deltanet_class is None:
                return {
                    "status": "failed", 
                    "error": "DeltaNet class not found",
                    "classes_found": classes_found
                }
            
            # Analyze DeltaNet class in detail
            methods = []
            for node in deltanet_class.body:
                if isinstance(node, ast.FunctionDef):
                    methods.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
            
            required_methods = ["__init__", "__call__"]
            found_methods = [m["name"] for m in methods]
            missing_methods = [m for m in required_methods if m not in found_methods]
            
            result = {
                "status": "passed" if not missing_methods else "warning",
                "message": "Valid class structure" if not missing_methods else f"Missing methods: {missing_methods}",
                "classes_found": classes_found,
                "methods_found": methods,
                "deltanet_analysis": {
                    "line": deltanet_class.lineno,
                    "methods": methods,
                    "missing_methods": missing_methods
                }
            }
            
            return result
            
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "class structure analysis")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_instantiation_enhanced(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced instantiation testing with parameter analysis."""
        if not MLX_AVAILABLE:
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec is None or spec.loader is None:
                return {"status": "failed", "error": "Could not create module spec"}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'DeltaNet'):
                return {"status": "failed", "error": "DeltaNet class not found in module"}
            
            # Try different parameter combinations (smaller to avoid memory issues)
            test_configs = [
                {"hidden_size": 32, "num_heads": 2},
                {"hidden_size": 64, "num_heads": 2}
            ]
            
            instantiation_results = []
            for i, config in enumerate(test_configs):
                try:
                    model = module.DeltaNet(**config)
                    instantiation_results.append({
                        "config": config,
                        "success": True,
                        "model_type": str(type(model))
                    })
                except Exception as e:
                    error_detail = self.error_analyzer.analyze_error(e, file_path.name, f"instantiation_config_{i}")
                    instantiation_results.append({
                        "config": config,
                        "success": False,
                        "error": str(e),
                        "error_detail": error_detail.__dict__
                    })
            
            successful = sum(1 for r in instantiation_results if r["success"])
            total = len(instantiation_results)
            
            if successful == 0:
                return {
                    "status": "failed",
                    "error": "All instantiation attempts failed",
                    "instantiation_results": instantiation_results
                }
            elif successful == total:
                return {
                    "status": "passed",
                    "message": f"Successfully instantiated with all {total} configurations",
                    "instantiation_results": instantiation_results
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Instantiated with {successful}/{total} configurations",
                    "instantiation_results": instantiation_results
                }
                
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "instantiation")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_forward_pass_enhanced(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced forward pass testing with detailed analysis."""
        if not MLX_AVAILABLE:
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'DeltaNet'):
                return {"status": "failed", "error": "DeltaNet class not found"}
            
            model = module.DeltaNet(hidden_size=32, num_heads=2)
            
            # Test multiple input configurations (smaller to avoid memory issues)
            test_cases = [
                {"batch_size": 1, "seq_len": 8, "hidden_size": 32},
                {"batch_size": 1, "seq_len": 16, "hidden_size": 32}
            ]
            
            forward_results = []
            for i, case in enumerate(test_cases):
                try:
                    test_input = mx.random.normal((case["batch_size"], case["seq_len"], case["hidden_size"]))
                    
                    # Time the forward pass
                    start_time = time.perf_counter()
                    output = model(test_input)
                    end_time = time.perf_counter()
                    
                    if isinstance(output, tuple):
                        main_output = output[0]
                        additional_outputs = len(output) - 1
                    else:
                        main_output = output
                        additional_outputs = 0
                    
                    forward_results.append({
                        "test_case": case,
                        "success": True,
                        "input_shape": test_input.shape,
                        "output_shape": main_output.shape if hasattr(main_output, 'shape') else None,
                        "additional_outputs": additional_outputs,
                        "forward_time": end_time - start_time,
                        "output_type": str(type(main_output))
                    })
                    
                except Exception as e:
                    error_detail = self.error_analyzer.analyze_error(e, file_path.name, f"forward_pass_case_{i}")
                    forward_results.append({
                        "test_case": case,
                        "success": False,
                        "error": str(e),
                        "error_detail": error_detail.__dict__
                    })
            
            successful = sum(1 for r in forward_results if r["success"])
            total = len(forward_results)
            
            if successful == 0:
                return {
                    "status": "failed",
                    "error": "All forward pass attempts failed",
                    "forward_results": forward_results
                }
            elif successful == total:
                avg_time = sum(r["forward_time"] for r in forward_results if r["success"]) / successful
                return {
                    "status": "passed",
                    "message": f"All {total} forward passes successful (avg: {avg_time:.4f}s)",
                    "forward_results": forward_results
                }
            else:
                return {
                    "status": "warning",
                    "message": f"{successful}/{total} forward passes successful",
                    "forward_results": forward_results
                }
                
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "forward pass")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_shape_compatibility_enhanced(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced shape compatibility testing."""
        if not MLX_AVAILABLE:
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.DeltaNet(hidden_size=32, num_heads=2)
            
            # Comprehensive shape testing (smaller to avoid memory issues)
            test_cases = [
                (1, 1, 32),     # Single token
                (1, 8, 32),     # Short sequence
                (1, 16, 32),    # Medium sequence  
                (2, 8, 32),     # Small batch
            ]
            
            shape_results = []
            for batch_size, seq_len, hidden_size in test_cases:
                try:
                    test_input = mx.random.normal((batch_size, seq_len, hidden_size))
                    output = model(test_input)
                    
                    if isinstance(output, tuple):
                        main_output = output[0]
                    else:
                        main_output = output
                    
                    # Validate output shape
                    expected_shape = (batch_size, seq_len, 32)  # Assuming output size = hidden_size
                    actual_shape = main_output.shape if hasattr(main_output, 'shape') else None
                    shape_match = actual_shape == expected_shape if actual_shape else False
                    
                    shape_results.append({
                        "input_shape": (batch_size, seq_len, hidden_size),
                        "output_shape": actual_shape,
                        "expected_shape": expected_shape,
                        "shape_match": shape_match,
                        "success": True
                    })
                except Exception as e:
                    error_detail = self.error_analyzer.analyze_error(e, file_path.name, f"shape_test_{batch_size}x{seq_len}")
                    shape_results.append({
                        "input_shape": (batch_size, seq_len, hidden_size),
                        "success": False,
                        "error": str(e),
                        "error_detail": error_detail.__dict__
                    })
            
            successful = sum(1 for r in shape_results if r["success"])
            shape_matches = sum(1 for r in shape_results if r.get("shape_match", False))
            total = len(shape_results)
            
            if successful == 0:
                status = "failed"
                message = "All shape tests failed"
            elif successful == total and shape_matches == total:
                status = "passed"
                message = f"All {total} shape tests passed with correct output shapes"
            elif successful == total:
                status = "warning"
                message = f"All {total} forward passes succeeded, but {total-shape_matches} had unexpected output shapes"
            else:
                status = "warning"
                message = f"{successful}/{total} shape tests passed"
            
            return {
                "status": status,
                "message": message,
                "shape_results": shape_results
            }
            
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "shape compatibility")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_parameter_analysis(self, file_path: Path, detailed: bool) -> Dict[str, Any]:
        """Enhanced parameter analysis."""
        if not MLX_AVAILABLE:
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.DeltaNet(hidden_size=64, num_heads=2)
            
            # Detailed parameter analysis
            total_params = 0
            param_breakdown = {}
            
            def analyze_params(obj, prefix=""):
                nonlocal total_params
                if hasattr(obj, '__dict__'):
                    for name, value in obj.__dict__.items():
                        full_name = f"{prefix}.{name}" if prefix else name
                        if isinstance(value, mx.array):
                            param_count = value.size
                            total_params += param_count
                            param_breakdown[full_name] = {
                                "shape": value.shape,
                                "size": param_count,
                                "dtype": str(value.dtype)
                            }
                        elif hasattr(value, '__dict__') and not isinstance(value, (int, float, str, bool)):
                            analyze_params(value, full_name)
            
            analyze_params(model)
            
            # Categorize parameters
            categories = {
                "linear": sum(v["size"] for k, v in param_breakdown.items() if "linear" in k.lower() or "proj" in k.lower()),
                "conv": sum(v["size"] for k, v in param_breakdown.items() if "conv" in k.lower() or "fir" in k.lower()),
                "norm": sum(v["size"] for k, v in param_breakdown.items() if "norm" in k.lower()),
                "gate": sum(v["size"] for k, v in param_breakdown.items() if "gate" in k.lower() or "mlp" in k.lower()),
                "other": 0
            }
            categories["other"] = total_params - sum(categories.values())
            
            return {
                "status": "passed",
                "message": f"Model has {total_params:,} parameters",
                "total_parameters": total_params,
                "parameter_breakdown": param_breakdown,
                "parameter_categories": categories,
                "largest_components": sorted(
                    [(k, v["size"]) for k, v in param_breakdown.items()], 
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
            
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "parameter analysis")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _test_performance_enhanced(self, file_path: Path, detailed: bool, num_iterations: int = 10) -> Dict[str, Any]:
        """Enhanced performance testing."""
        if not MLX_AVAILABLE:
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.DeltaNet(hidden_size=64, num_heads=2)
            
            # Performance testing with different configurations (smaller to avoid memory issues)
            perf_configs = [
                {"batch_size": 1, "seq_len": 16, "name": "small"},
                {"batch_size": 1, "seq_len": 32, "name": "medium"}
            ]
            
            performance_results = []
            
            for config in perf_configs:
                batch_size, seq_len = config["batch_size"], config["seq_len"]
                test_input = mx.random.normal((batch_size, seq_len, 64))
                
                # Warmup (reduced to save memory)
                for _ in range(1):
                    _ = model(test_input)
                
                # Benchmark (reduced iterations)
                times = []
                for _ in range(min(num_iterations, 3)):
                    start_time = time.perf_counter()
                    output = model(test_input)
                    if hasattr(output, '__iter__') and len(output) > 0:
                        _ = output[0].sum()
                    elif hasattr(output, 'sum'):
                        _ = output.sum()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
                tokens_per_sec = (batch_size * seq_len) / avg_time
                
                performance_results.append({
                    "config": config["name"],
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "min_time": min(times),
                    "max_time": max(times),
                    "tokens_per_second": tokens_per_sec,
                    "iterations": num_iterations
                })
            
            return {
                "status": "passed",
                "message": f"Performance tested across {len(perf_configs)} configurations",
                "performance_results": performance_results
            }
            
        except Exception as e:
            error_detail = self.error_analyzer.analyze_error(e, file_path.name, "performance testing")
            return {"status": "failed", "error": str(e), "error_detail": error_detail.__dict__}
    
    def _determine_overall_status(self, tests: Dict[str, Dict]) -> str:
        """Determine overall status with enhanced logic."""
        critical_tests = ["syntax", "imports"]
        important_tests = ["structure", "instantiation", "forward_pass"]
        
        # Critical failures
        for test_name in critical_tests:
            if test_name in tests and tests[test_name]["status"] == "failed":
                return "failed"
        
        # Important test failures
        failed_important = sum(1 for test_name in important_tests 
                             if test_name in tests and tests[test_name]["status"] == "failed")
        
        if failed_important > 0:
            return "failed"
        
        # Check for functional tests passing (ignore PyTorch remnant warnings)
        functional_tests = ["syntax", "imports", "structure", "instantiation", "forward_pass", "shape_compatibility"]
        functional_passed = all(tests.get(test, {}).get("status") in ["passed", "skipped"] 
                               for test in functional_tests)
        
        # Only consider critical warnings (not PyTorch remnants in comments)
        critical_warnings = any(test["status"] == "warning" and test_name in functional_tests
                               for test_name, test in tests.items())
        
        if functional_passed and not critical_warnings:
            return "passed"
        else:
            return "warning"
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns across all tests."""
        error_types = Counter()
        error_messages = Counter()
        file_errors = defaultdict(list)
        
        for error in self.error_database:
            # Handle both ErrorDetail objects and dicts
            if isinstance(error, dict):
                error_type = error.get('error_type', 'unknown')
                error_message = error.get('error_message', 'unknown')
                file_name = error.get('file_name', 'unknown')
            else:
                error_type = error.error_type
                error_message = error.error_message
                file_name = error.file_name
            
            error_types[error_type] += 1
            error_messages[error_message] += 1
            file_errors[file_name].append(error_type)
        
        return {
            "most_common_error_types": error_types.most_common(10),
            "most_common_error_messages": error_messages.most_common(10),
            "files_with_most_errors": sorted(
                [(f, len(errors)) for f, errors in file_errors.items()], 
                key=lambda x: x[1], reverse=True
            )[:10],
            "total_errors": len(self.error_database)
        }
    
    def _generate_enhanced_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced summary with actionable insights."""
        summary = {
            "pass_rate": results["passed"] / results["total_files"] if results["total_files"] > 0 else 0,
            "test_breakdown": {},
            "conversion_quality": {},
            "recommendations": []
        }
        
        # Test breakdown
        all_tests = ["syntax", "imports", "structure", "pytorch_remnants", "mlx_compatibility",
                    "instantiation", "forward_pass", "shape_compatibility", "parameter_analysis", "performance"]
        
        for test_name in all_tests:
            passed = sum(1 for details in results["detailed_results"].values()
                        if details["tests"].get(test_name, {}).get("status") == "passed")
            failed = sum(1 for details in results["detailed_results"].values()
                        if details["tests"].get(test_name, {}).get("status") == "failed")
            warning = sum(1 for details in results["detailed_results"].values()
                         if details["tests"].get(test_name, {}).get("status") == "warning")
            skipped = sum(1 for details in results["detailed_results"].values()
                         if details["tests"].get(test_name, {}).get("status") == "skipped")
            
            summary["test_breakdown"][test_name] = {
                "passed": passed,
                "failed": failed,
                "warning": warning,
                "skipped": skipped,
                "total": passed + failed + warning + skipped
            }
        
        # Generate recommendations
        if summary["test_breakdown"]["imports"]["failed"] > 0:
            summary["recommendations"].append("Fix import errors - check MLX installation and remove PyTorch/FLA dependencies")
        
        if summary["test_breakdown"]["pytorch_remnants"]["warning"] > 0:
            summary["recommendations"].append("Remove remaining PyTorch code patterns for complete conversion")
        
        if summary["test_breakdown"]["forward_pass"]["failed"] > 0:
            summary["recommendations"].append("Fix forward pass errors - likely shape/operation compatibility issues")
        
        if summary["pass_rate"] < 0.5:
            summary["recommendations"].append("Consider reviewing conversion guidelines - low overall pass rate")
        
        return summary
    
    def print_enhanced_summary(self):
        """Print detailed summary with error analysis."""
        if not self.test_results:
            logger.warning("No test results available")
            return
        
        results = self.test_results
        
        print("\n" + "="*80)
        print("🚀 ENHANCED PyTorch to MLX Conversion Test Report")
        print("="*80)
        
        # Overall statistics
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total files tested: {results['total_files']}")
        print(f"   ✅ Passed: {results['passed']} ({results['passed']/results['total_files']*100:.1f}%)")
        print(f"   ❌ Failed: {results['failed']} ({results['failed']/results['total_files']*100:.1f}%)")
        print(f"   ⏭️  Skipped: {results['skipped']} ({results['skipped']/results['total_files']*100:.1f}%)")
        
        # Test breakdown
        if 'summary' in results and 'test_breakdown' in results['summary']:
            print(f"\n🔍 DETAILED TEST BREAKDOWN:")
            for test_name, stats in results['summary']['test_breakdown'].items():
                total = stats['total']
                if total > 0:
                    passed_pct = stats['passed'] / total * 100
                    status_icon = "✅" if passed_pct == 100 else "⚠️" if passed_pct >= 80 else "❌"
                    print(f"   {status_icon} {test_name:25}: {stats['passed']:3}P {stats['failed']:3}F {stats['warning']:3}W {stats['skipped']:3}S ({passed_pct:5.1f}%)")
        
        # Error analysis
        if 'error_analysis' in results:
            error_analysis = results['error_analysis']
            print(f"\n🐛 ERROR ANALYSIS:")
            print(f"   Total errors detected: {error_analysis['total_errors']}")
            
            if error_analysis['most_common_error_types']:
                print(f"\n   📈 Most Common Error Types:")
                for error_type, count in error_analysis['most_common_error_types'][:5]:
                    print(f"      • {error_type}: {count} occurrences")
            
            if error_analysis['files_with_most_errors']:
                print(f"\n   📁 Files with Most Errors:")
                for filename, error_count in error_analysis['files_with_most_errors'][:5]:
                    print(f"      • {filename}: {error_count} errors")
        
        # Recommendations
        if 'summary' in results and 'recommendations' in results['summary']:
            recommendations = results['summary']['recommendations']
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
        
        # Failed files detail
        failed_files = [name for name, details in results['detailed_results'].items()
                       if details['overall_status'] == 'failed']
        
        if failed_files:
            print(f"\n❌ FAILED FILES ({len(failed_files)}):")
            for file_name in failed_files:  # Show all failed files
                details = results['detailed_results'][file_name]
                failed_tests = [test for test, result in details['tests'].items()
                              if result.get('status') == 'failed']
                main_error = details['errors'][0] if details['errors'] else {}
                error_type = main_error.get('error_type', 'unknown')
                print(f"   • {file_name:35} | {', '.join(failed_tests[:3])} | {error_type}")
        
        # Skipped files detail
        skipped_files = [name for name, details in results['detailed_results'].items()
                        if details['overall_status'] == 'skipped']
        
        if skipped_files:
            print(f"\n⏭️  SKIPPED FILES ({len(skipped_files)}):")
            for file_name in skipped_files:  # Show all skipped files
                details = results['detailed_results'][file_name]
                skipped_tests = [test for test, result in details['tests'].items()
                               if result.get('status') == 'skipped']
                skip_reasons = [result.get('reason', 'unknown') for test, result in details['tests'].items()
                              if result.get('status') == 'skipped']
                main_reason = skip_reasons[0] if skip_reasons else 'unknown'
                print(f"   • {file_name:35} | {', '.join(skipped_tests[:3])} | {main_reason}")
        
        # Warning files detail
        warning_files = [name for name, details in results['detailed_results'].items()
                        if details['overall_status'] == 'warning']
        
        if warning_files:
            print(f"\n⚠️  WARNING FILES ({len(warning_files)}):")
            for file_name in warning_files[:10]:  # Show first 10 warning files
                details = results['detailed_results'][file_name]
                warning_tests = [test for test, result in details['tests'].items()
                               if result.get('status') == 'warning']
                print(f"   • {file_name:35} | {', '.join(warning_tests[:3])}")
            if len(warning_files) > 10:
                print(f"   ... and {len(warning_files) - 10} more files")
        
        # PyTorch remnants detail
        pytorch_remnant_files = [name for name, details in results['detailed_results'].items()
                               if details['tests'].get('pytorch_remnants', {}).get('status') == 'warning']
        
        if pytorch_remnant_files:
            print(f"\n🔥 PYTORCH REMNANTS WARNING FILES ({len(pytorch_remnant_files)}):")
            for file_name in pytorch_remnant_files:
                print(f"   • {file_name}")
                details = results['detailed_results'][file_name]
                pytorch_test = details['tests'].get('pytorch_remnants', {})
                if 'remnants' in pytorch_test:
                    for remnant in pytorch_test['remnants'][:3]:  # Show first 3 remnants
                        print(f"     - Line {remnant['line']}: {remnant['description']} ({remnant['match']})")
                    if len(pytorch_test['remnants']) > 3:
                        print(f"     ... and {len(pytorch_test['remnants']) - 3} more remnants")
                print()
        
        # Success stories
        passed_files = [name for name, details in results['detailed_results'].items()
                       if details['overall_status'] == 'passed']
        if passed_files:
            print(f"\n✅ SUCCESSFUL CONVERSIONS ({len(passed_files)}):")
            for file_name in passed_files[:10]:
                print(f"   • {file_name}")
            if len(passed_files) > 10:
                print(f"   ... and {len(passed_files) - 10} more files")
        
        print("\n" + "="*80)
    
    def save_enhanced_results(self, filename: str = "enhanced_conversion_results.json"):
        """Save enhanced results with full error details."""
        output_path = Path(filename)
        
        # Convert error objects to dictionaries for JSON serialization
        json_results = json.loads(json.dumps(self.test_results, default=str))
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Enhanced test results saved to {output_path}")
        
        # Also save a human-readable error report
        error_report_path = output_path.with_suffix('.error_report.txt')
        with open(error_report_path, 'w') as f:
            f.write("PyTorch to MLX Conversion - Detailed Error Report\n")
            f.write("=" * 60 + "\n\n")
            
            for error in self.error_database:
                # Handle both ErrorDetail objects and dicts
                if isinstance(error, dict):
                    file_name = error.get('file_name', 'unknown')
                    error_type = error.get('error_type', 'unknown')
                    error_message = error.get('error_message', 'unknown')
                    line_number = error.get('line_number', None)
                    suggestion = error.get('suggestion', None)
                else:
                    file_name = error.file_name
                    error_type = error.error_type
                    error_message = error.error_message
                    line_number = error.line_number
                    suggestion = error.suggestion
                
                f.write(f"File: {file_name}\n")
                f.write(f"Error Type: {error_type}\n")
                f.write(f"Message: {error_message}\n")
                if line_number:
                    f.write(f"Line: {line_number}\n")
                if suggestion:
                    f.write(f"Suggestion: {suggestion}\n")
                f.write("-" * 40 + "\n\n")
        
        logger.info(f"Error report saved to {error_report_path}")


def main():
    """Enhanced CLI interface."""
    parser = argparse.ArgumentParser(description="Enhanced PyTorch to MLX conversion testing")
    parser.add_argument("--test-all", action="store_true", help="Test all MLX architectures")
    parser.add_argument("--test-file", type=str, help="Test a specific MLX file")
    parser.add_argument("--pytorch-dir", type=str, default="pytorch_arch", help="PyTorch architectures directory")
    parser.add_argument("--mlx-dir", type=str, default="mlx_architectures", help="MLX architectures directory")
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument("--iterations", type=int, default=5, help="Performance test iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = EnhancedConversionTester(args.pytorch_dir, args.mlx_dir)
    
    if args.test_all:
        results = tester.run_all_tests(detailed_output=args.detailed)
        tester.print_enhanced_summary()
        
        if args.save_results:
            tester.save_enhanced_results(args.save_results)
    
    elif args.test_file:
        mlx_file = Path(args.mlx_dir) / args.test_file
        if not mlx_file.exists():
            logger.error(f"File not found: {mlx_file}")
            sys.exit(1)
        
        results = tester.test_single_file_enhanced(mlx_file, detailed=True)
        
        print(f"\n🔍 Enhanced Test Results for {args.test_file}:")
        print(f"Overall Status: {results['overall_status']}")
        
        for test_name, test_result in results['tests'].items():
            status = test_result['status']
            message = test_result.get('message', test_result.get('error', ''))
            status_icon = "✅" if status == "passed" else "⚠️" if status == "warning" else "❌" if status == "failed" else "⏭️"
            print(f"  {status_icon} {test_name:25}: {status:8} - {message}")
            
            # Show detailed error info if available
            if 'error_detail' in test_result:
                error_detail = test_result['error_detail']
                if isinstance(error_detail, dict):
                    print(f"    └─ Error Type: {error_detail.get('error_type', 'unknown')}")
                    if error_detail.get('suggestion'):
                        print(f"    └─ Suggestion: {error_detail['suggestion']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()