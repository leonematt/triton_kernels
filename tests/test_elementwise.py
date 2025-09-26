#!/usr/bin/env python3
"""
Comprehensive test suite for Triton elementwise kernels
Tests all variants, operations, and compares with PyTorch implementations
"""

import torch
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple
import argparse
import json

# Import your kernels (adjust import path as needed)
from elementwise import (
    ElementwiseOp, universal_elementwise, 
    triton_gelu, triton_silu, triton_swiglu, triton_geglu,
    triton_gelu_optimized, triton_silu_optimized, triton_swiglu_optimized,
    VARIANTS
)

class KernelTester:
    def __init__(self, device='cuda', dtype=torch.float32, warmup_iters=20, test_iters=100):
        self.device = device
        self.dtype = dtype
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.results = []
        
    def benchmark_function(self, func: Callable, tensor: torch.Tensor, *args, name: str = "function") -> Dict:
        """Benchmark a function and return timing + throughput metrics"""
        # Warmup
        for _ in range(self.warmup_iters):
            _ = func(tensor, *args)
        torch.cuda.synchronize()
        
        # Benchmark using CUDA events for accuracy
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.test_iters):
            result = func(tensor, *args)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        total_time = start_event.elapsed_time(end_event)  # ms
        avg_time = total_time / self.test_iters
        
        # Memory bandwidth (GB/s) - assumes read input + write output
        bytes_per_element = tensor.element_size()
        io_elements = tensor.numel() * 2  # read + write
        total_bytes = io_elements * bytes_per_element
        throughput = total_bytes / (avg_time / 1000) / 1e9  # GB/s
        
        return {
            'name': name,
            'avg_time_ms': avg_time,
            'throughput_gbps': throughput,
            'tensor_shape': tuple(tensor.shape),
            'num_elements': tensor.numel(),
            'result_tensor': result
        }
    
    def test_accuracy(self, pytorch_result: torch.Tensor, triton_result: torch.Tensor, 
                     atol: float = 1e-4, rtol: float = 1e-3) -> Dict:
        """Test accuracy between PyTorch and Triton implementations"""
        are_close = torch.allclose(pytorch_result, triton_result, atol=atol, rtol=rtol)
        
        if are_close:
            max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
            mean_diff = torch.mean(torch.abs(pytorch_result - triton_result)).item()
        else:
            max_diff = float('inf')
            mean_diff = float('inf')
        
        return {
            'accurate': are_close,
            'max_absolute_diff': max_diff,
            'mean_absolute_diff': mean_diff,
            'atol_used': atol,
            'rtol_used': rtol
        }
    
    def test_unary_operation(self, op_name: str, tensor_sizes: List[Tuple], test_configs: Dict):
        """Test unary operations (GELU, SiLU, ReLU, etc.)"""
        print(f"\n{'='*60}")
        print(f"TESTING UNARY OPERATION: {op_name.upper()}")
        print(f"{'='*60}")
        
        op_results = []
        
        for shape in tensor_sizes:
            print(f"\nTensor shape: {shape} ({np.prod(shape):,} elements)")
            x = torch.randn(shape, device=self.device, dtype=self.dtype)
            
            # Get reference implementation and test functions
            pytorch_func = test_configs[op_name]['pytorch_func']
            triton_funcs = test_configs[op_name]['triton_funcs']
            accuracy_tolerance = test_configs[op_name].get('tolerance', {'atol': 1e-4, 'rtol': 1e-3})
            
            # Benchmark PyTorch reference
            pytorch_metrics = self.benchmark_function(pytorch_func, x, name=f"PyTorch {op_name}")
            pytorch_result = pytorch_metrics['result_tensor']
            
            shape_results = {
                'operation': op_name,
                'shape': shape,
                'pytorch': pytorch_metrics
            }
            
            # Test all Triton variants
            triton_results = {}
            for variant_name, triton_func in triton_funcs.items():
                try:
                    # Benchmark Triton implementation
                    triton_metrics = self.benchmark_function(triton_func, x, name=f"Triton {op_name} ({variant_name})")
                    triton_result = triton_metrics['result_tensor']
                    
                    # Test accuracy
                    accuracy = self.test_accuracy(pytorch_result, triton_result, **accuracy_tolerance)
                    
                    # Calculate speedup
                    speedup = pytorch_metrics['avg_time_ms'] / triton_metrics['avg_time_ms']
                    
                    triton_results[variant_name] = {
                        **triton_metrics,
                        'accuracy': accuracy,
                        'speedup': speedup
                    }
                    
                    # Print results
                    status = "✓" if accuracy['accurate'] else "✗"
                    print(f"  {status} {variant_name:20s}: {triton_metrics['avg_time_ms']:.3f}ms, "
                          f"{triton_metrics['throughput_gbps']:.1f} GB/s, "
                          f"{speedup:.2f}x speedup, "
                          f"max_diff: {accuracy['max_absolute_diff']:.2e}")
                    
                except Exception as e:
                    print(f"  ✗ {variant_name:20s}: ERROR - {str(e)}")
                    triton_results[variant_name] = {'error': str(e)}
            
            shape_results['triton'] = triton_results
            op_results.append(shape_results)
        
        return op_results
    
    def test_binary_operation(self, op_name: str, tensor_sizes: List[Tuple], test_configs: Dict):
        """Test binary operations (ADD, MUL, SwiGLU, etc.)"""
        print(f"\n{'='*60}")
        print(f"TESTING BINARY OPERATION: {op_name.upper()}")
        print(f"{'='*60}")
        
        op_results = []
        
        for shape in tensor_sizes:
            print(f"\nTensor shape: {shape} ({np.prod(shape):,} elements)")
            x = torch.randn(shape, device=self.device, dtype=self.dtype)
            y = torch.randn(shape, device=self.device, dtype=self.dtype)
            
            # Get reference implementation and test functions
            pytorch_func = test_configs[op_name]['pytorch_func']
            triton_funcs = test_configs[op_name]['triton_funcs']
            accuracy_tolerance = test_configs[op_name].get('tolerance', {'atol': 1e-4, 'rtol': 1e-3})
            
            # Benchmark PyTorch reference
            pytorch_metrics = self.benchmark_function(pytorch_func, x, y, name=f"PyTorch {op_name}")
            pytorch_result = pytorch_metrics['result_tensor']
            
            shape_results = {
                'operation': op_name,
                'shape': shape,
                'pytorch': pytorch_metrics
            }
            
            # Test all Triton variants
            triton_results = {}
            for variant_name, triton_func in triton_funcs.items():
                try:
                    # Benchmark Triton implementation
                    triton_metrics = self.benchmark_function(triton_func, x, y, name=f"Triton {op_name} ({variant_name})")
                    triton_result = triton_metrics['result_tensor']
                    
                    # Test accuracy
                    accuracy = self.test_accuracy(pytorch_result, triton_result, **accuracy_tolerance)
                    
                    # Calculate speedup
                    speedup = pytorch_metrics['avg_time_ms'] / triton_metrics['avg_time_ms']
                    
                    triton_results[variant_name] = {
                        **triton_metrics,
                        'accuracy': accuracy,
                        'speedup': speedup
                    }
                    
                    # Print results
                    status = "✓" if accuracy['accurate'] else "✗"
                    print(f"  {status} {variant_name:20s}: {triton_metrics['avg_time_ms']:.3f}ms, "
                          f"{triton_metrics['throughput_gbps']:.1f} GB/s, "
                          f"{speedup:.2f}x speedup, "
                          f"max_diff: {accuracy['max_absolute_diff']:.2e}")
                    
                except Exception as e:
                    print(f"  ✗ {variant_name:20s}: ERROR - {str(e)}")
                    triton_results[variant_name] = {'error': str(e)}
            
            shape_results['triton'] = triton_results
            op_results.append(shape_results)
        
        return op_results
    
    def test_all_variants(self, tensor_sizes: List[Tuple] = None):
        """Test all operation variants with comprehensive comparisons"""
        if tensor_sizes is None:
            # vLLM-typical tensor sizes: (batch_size * seq_len, hidden_size)
            tensor_sizes = [
                (1024, 4096),     # Small
                (2048, 4096),     # Medium
                (2048, 8192),     # Large  
                (4096, 8192),     # Very Large
                (2048, 16384),    # Extra Large
                (8192, 16384),    # Huge
            ]
        
        print(f"Testing with tensor sizes: {tensor_sizes}")
        print(f"Device: {self.device}, Dtype: {self.dtype}")
        print(f"Warmup iterations: {self.warmup_iters}, Test iterations: {self.test_iters}")
        
        # Define test configurations for each operation
        test_configs = {
            'gelu': {
                'pytorch_func': torch.nn.functional.gelu,
                'triton_funcs': {
                    'universal_standard': lambda x: triton_gelu(x, fast=False),
                    'universal_fast': lambda x: triton_gelu(x, fast=True),
                    'optimized_standard': lambda x: triton_gelu_optimized(x, fast=False),
                    'optimized_fast': lambda x: triton_gelu_optimized(x, fast=True),
                },
                'tolerance': {'atol': 1e-2, 'rtol': 1e-2}  # GELU approximations are less precise
            },
            'silu': {
                'pytorch_func': torch.nn.functional.silu,
                'triton_funcs': {
                    'universal': triton_silu,
                    'optimized': triton_silu_optimized,
                },
                'tolerance': {'atol': 1e-4, 'rtol': 1e-4}
            },
            'relu': {
                'pytorch_func': torch.nn.functional.relu,
                'triton_funcs': {
                    'universal': lambda x: universal_elementwise(x, op_code=ElementwiseOp.RELU),
                },
                'tolerance': {'atol': 1e-6, 'rtol': 1e-6}  # ReLU should be exact
            },
            'sigmoid': {
                'pytorch_func': torch.sigmoid,
                'triton_funcs': {
                    'universal': lambda x: universal_elementwise(x, op_code=ElementwiseOp.SIGMOID),
                },
                'tolerance': {'atol': 1e-4, 'rtol': 1e-4}
            }
        }
        
        binary_configs = {
            'add': {
                'pytorch_func': torch.add,
                'triton_funcs': {
                    'universal': lambda x, y: universal_elementwise(x, y, ElementwiseOp.ADD),
                },
                'tolerance': {'atol': 1e-6, 'rtol': 1e-6}
            },
            'mul': {
                'pytorch_func': torch.mul,
                'triton_funcs': {
                    'universal': lambda x, y: universal_elementwise(x, y, ElementwiseOp.MUL),
                },
                'tolerance': {'atol': 1e-6, 'rtol': 1e-6}
            },
            'swiglu': {
                'pytorch_func': lambda x, y: torch.nn.functional.silu(x) * y,
                'triton_funcs': {
                    'universal': triton_swiglu,
                    'optimized': triton_swiglu_optimized,
                },
                'tolerance': {'atol': 1e-4, 'rtol': 1e-4}
            }
        }
        
        # Run all tests
        all_results = {}
        
        # Test unary operations
        for op_name, config in test_configs.items():
            all_results[op_name] = self.test_unary_operation(op_name, tensor_sizes, test_configs)
        
        # Test binary operations  
        for op_name, config in binary_configs.items():
            all_results[op_name] = self.test_binary_operation(op_name, tensor_sizes, binary_configs)
        
        return all_results
    
    def generate_summary_report(self, results: Dict) -> pd.DataFrame:
        """Generate a summary report of all test results"""
        summary_data = []
        
        for op_name, op_results in results.items():
            for shape_result in op_results:
                shape = shape_result['shape']
                pytorch_time = shape_result['pytorch']['avg_time_ms']
                pytorch_throughput = shape_result['pytorch']['throughput_gbps']
                
                base_row = {
                    'operation': op_name,
                    'shape': f"{shape[0]}x{shape[1]}",
                    'elements': np.prod(shape),
                    'pytorch_time_ms': pytorch_time,
                    'pytorch_throughput_gbps': pytorch_throughput,
                }
                
                # Add results for each Triton variant
                for variant_name, variant_result in shape_result['triton'].items():
                    if 'error' in variant_result:
                        continue
                        
                    row = base_row.copy()
                    row.update({
                        'variant': variant_name,
                        'triton_time_ms': variant_result['avg_time_ms'],
                        'triton_throughput_gbps': variant_result['throughput_gbps'],
                        'speedup': variant_result['speedup'],
                        'accurate': variant_result['accuracy']['accurate'],
                        'max_diff': variant_result['accuracy']['max_absolute_diff'],
                    })
                    
                    summary_data.append(row)
        
        return pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Triton Kernel Test Suite')
    parser.add_argument('--device', default='cuda', help='Device to run tests on')
    parser.add_argument('--dtype', default='float32', choices=['float16', 'float32'], help='Data type for tensors')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Number of test iterations')
    parser.add_argument('--output', help='Output file for detailed results (JSON)')
    parser.add_argument('--summary', help='Output file for summary report (CSV)')
    parser.add_argument('--sizes', nargs='*', help='Custom tensor sizes as "HxW" (e.g., "2048x4096")')
    
    args = parser.parse_args()
    
    # Parse dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    
    # Parse custom sizes if provided
    tensor_sizes = None
    if args.sizes:
        tensor_sizes = []
        for size_str in args.sizes:
            h, w = map(int, size_str.split('x'))
            tensor_sizes.append((h, w))
    
    print("="*80)
    print("COMPREHENSIVE TRITON KERNEL TEST SUITE")
    print("="*80)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    if args.device == 'cpu':
        print("WARNING: Running on CPU. Performance comparisons may not be meaningful.")
    
    # Initialize tester
    tester = KernelTester(
        device=args.device,
        dtype=dtype,
        warmup_iters=args.warmup,
        test_iters=args.iterations
    )
    
    # Run comprehensive tests
    results = tester.test_all_variants(tensor_sizes)
    
    # Generate summary
    summary_df = tester.generate_summary_report(results)
    
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Convert tensors to lists for JSON serialization
            def serialize_results(obj):
                if isinstance(obj, torch.Tensor):
                    return "TENSOR_REMOVED_FOR_SERIALIZATION"
                return obj
            
            json.dump(results, f, indent=2, default=serialize_results)
        print(f"\nDetailed results saved to: {args.output}")
    
    if args.summary:
        summary_df.to_csv(args.summary, index=False)
        print(f"Summary report saved to: {args.summary}")
    
    # Print best performers
    print(f"\n{'='*80}")
    print("BEST PERFORMING VARIANTS")
    print(f"{'='*80}")
    
    for op_name in summary_df['operation'].unique():
        op_data = summary_df[summary_df['operation'] == op_name]
        if len(op_data) > 0:
            best_speedup = op_data.loc[op_data['speedup'].idxmax()]
            best_accurate = op_data[op_data['accurate'] == True]
            if len(best_accurate) > 0:
                best_accurate = best_accurate.loc[best_accurate['speedup'].idxmax()]
                print(f"{op_name.upper():10s}: {best_accurate['variant']:20s} "
                      f"({best_accurate['speedup']:.2f}x speedup, accurate)")
            else:
                print(f"{op_name.upper():10s}: No accurate variants found!")

if __name__ == "__main__":
    main()