"""Benchmarking utilities for comparing CPU vs GPU convolution performance."""

import time
import numpy as np
import cupy as cp
from typing import Dict, List, Tuple
from .api import convolve, convolve_cpu, convolve_cupy_builtin


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, synchronize_cuda=False):
        """Initialize timer.
        
        Args:
            synchronize_cuda: If True, synchronize CUDA stream before measuring time
        """
        self.synchronize_cuda = synchronize_cuda
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.synchronize_cuda:
            cp.cuda.Stream.null.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.synchronize_cuda:
            cp.cuda.Stream.null.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def benchmark_single(image, kernel, method='gpu_optimized', warmup_runs=2, timed_runs=10):
    """Benchmark a single convolution method.
    
    Args:
        image: Input image as NumPy array
        kernel: Convolution kernel as NumPy array
        method: One of 'cpu', 'gpu_naive', 'gpu_optimized', 'gpu_builtin'
        warmup_runs: Number of warmup iterations to discard
        timed_runs: Number of timed iterations to average
        
    Returns:
        Dictionary with timing results:
        - 'method': method name
        - 'mean_time': mean execution time in seconds
        - 'std_time': standard deviation of execution time
        - 'min_time': minimum execution time
        - 'max_time': maximum execution time
        - 'throughput': pixels per second
    """
    times = []
    
    # Warmup runs
    for _ in range(warmup_runs):
        if method == 'cpu':
            _ = convolve_cpu(image, kernel)
        elif method == 'gpu_naive':
            _ = convolve(image, kernel, use_shared_mem=False, return_numpy=True)
        elif method == 'gpu_optimized':
            _ = convolve(image, kernel, use_shared_mem=True, return_numpy=True)
        elif method == 'gpu_builtin':
            _ = convolve_cupy_builtin(image, kernel)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Timed runs
    for _ in range(timed_runs):
        synchronize = method != 'cpu'
        
        with Timer(synchronize_cuda=synchronize) as t:
            if method == 'cpu':
                _ = convolve_cpu(image, kernel)
            elif method == 'gpu_naive':
                _ = convolve(image, kernel, use_shared_mem=False, return_numpy=True)
            elif method == 'gpu_optimized':
                _ = convolve(image, kernel, use_shared_mem=True, return_numpy=True)
            elif method == 'gpu_builtin':
                _ = convolve_cupy_builtin(image, kernel)
        
        times.append(t.elapsed)
    
    times = np.array(times)
    num_pixels = image.shape[0] * image.shape[1]
    if image.ndim == 3:
        num_pixels *= image.shape[2]
    
    return {
        'method': method,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput': num_pixels / np.mean(times)
    }


def benchmark_all(image, kernel, warmup_runs=2, timed_runs=10, include_builtin=True):
    """Benchmark all convolution methods.
    
    Args:
        image: Input image as NumPy array
        kernel: Convolution kernel as NumPy array
        warmup_runs: Number of warmup iterations
        timed_runs: Number of timed iterations
        include_builtin: If True, also benchmark CuPy's built-in convolution
        
    Returns:
        List of dictionaries with timing results for each method
    """
    methods = ['cpu', 'gpu_naive', 'gpu_optimized']
    if include_builtin:
        methods.append('gpu_builtin')
    
    results = []
    for method in methods:
        print(f"Benchmarking {method}...")
        result = benchmark_single(image, kernel, method, warmup_runs, timed_runs)
        results.append(result)
    
    return results


def benchmark_kernel_only(image, kernel, method='gpu_optimized', warmup_runs=2, timed_runs=10):
    """Benchmark kernel execution time only (excluding memory transfers).
    
    Args:
        image: Input image as NumPy array
        kernel: Convolution kernel as NumPy array
        method: 'gpu_naive' or 'gpu_optimized'
        warmup_runs: Number of warmup iterations
        timed_runs: Number of timed iterations
        
    Returns:
        Dictionary with timing results (kernel time only)
    """
    if method not in ['gpu_naive', 'gpu_optimized']:
        raise ValueError("Kernel-only timing only supported for GPU methods")
    
    # Transfer data to GPU once
    image_gpu = cp.asarray(image, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel, dtype=cp.float32)
    
    # Handle multi-channel
    if image_gpu.ndim == 3:
        image_gpu = image_gpu[:, :, 0]  # Use first channel for kernel timing
    
    from .kernels.conv2d import convolve_naive, convolve_optimized
    conv_func = convolve_optimized if method == 'gpu_optimized' else convolve_naive
    
    # Warmup
    for _ in range(warmup_runs):
        _ = conv_func(image_gpu, kernel_gpu)
    cp.cuda.Stream.null.synchronize()
    
    # Timed runs
    times = []
    for _ in range(timed_runs):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        _ = conv_func(image_gpu, kernel_gpu)
        end_event.record()
        end_event.synchronize()
        
        elapsed = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0  # Convert to seconds
        times.append(elapsed)
    
    times = np.array(times)
    num_pixels = image_gpu.shape[0] * image_gpu.shape[1]
    
    return {
        'method': f"{method}_kernel_only",
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput': num_pixels / np.mean(times)
    }


def calculate_speedup(results, baseline='cpu'):
    """Calculate speedup relative to baseline method.
    
    Args:
        results: List of result dictionaries from benchmark_all
        baseline: Name of baseline method (default: 'cpu')
        
    Returns:
        Dictionary mapping method names to speedup factors
    """
    baseline_time = None
    for result in results:
        if result['method'] == baseline:
            baseline_time = result['mean_time']
            break
    
    if baseline_time is None:
        raise ValueError(f"Baseline method '{baseline}' not found in results")
    
    speedups = {}
    for result in results:
        speedups[result['method']] = baseline_time / result['mean_time']
    
    return speedups


def format_results_table(results, speedups=None):
    """Format benchmark results as a readable table string.
    
    Args:
        results: List of result dictionaries
        speedups: Optional dictionary of speedup factors
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Method':<20} {'Time (ms)':<15} {'Throughput (MP/s)':<20} {'Speedup':<10}")
    lines.append("=" * 80)
    
    for result in results:
        method = result['method']
        mean_time_ms = result['mean_time'] * 1000
        throughput_mp = result['throughput'] / 1e6
        
        speedup_str = ""
        if speedups and method in speedups:
            speedup_str = f"{speedups[method]:.2f}x"
        
        lines.append(
            f"{method:<20} {mean_time_ms:>10.3f} ms   "
            f"{throughput_mp:>10.2f} MP/s      {speedup_str:<10}"
        )
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def print_results(results):
    """Print benchmark results in a formatted table.
    
    Args:
        results: List of result dictionaries from benchmark_all
    """
    speedups = calculate_speedup(results, baseline='cpu')
    table = format_results_table(results, speedups)
    print(table)

