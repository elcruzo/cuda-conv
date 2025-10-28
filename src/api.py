"""High-level Python API for CUDA convolution."""

import numpy as np
import cupy as cp
from .kernels.conv2d import convolve_naive, convolve_optimized


def convolve(image, kernel, use_shared_mem=True, return_numpy=True, warmup=False):
    """Convolve an image with a kernel using CUDA acceleration.
    
    This is the main entry point for the convolution API. It handles:
    - Automatic conversion between NumPy and CuPy arrays
    - Support for grayscale and RGB images
    - Memory transfer management
    - Optional warmup runs for accurate benchmarking
    
    Args:
        image: Input image as NumPy or CuPy array
               - Grayscale: shape (H, W)
               - RGB: shape (H, W, 3)
        kernel: Convolution kernel as NumPy or CuPy array
                - Shape (K, K) where K is odd
        use_shared_mem: If True, use optimized tiled kernel with shared memory.
                       If False, use naive kernel. Default: True
        return_numpy: If True, return result as NumPy array.
                     If False, keep result on GPU as CuPy array. Default: True
        warmup: If True, run kernel once before actual computation to eliminate
               cold-start effects. Useful for benchmarking. Default: False
    
    Returns:
        Convolved image with same shape and type as input
        
    Raises:
        ValueError: If kernel size is even or dimensions don't match
    """
    # Validate inputs
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Kernel must be square, got shape {kernel.shape}")
    if kernel.shape[0] % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {kernel.shape[0]}")
    if kernel.shape[0] > 9:
        raise ValueError(
            f"Kernel size too large ({kernel.shape[0]}). "
            "Maximum supported size is 9x9 for optimized kernel."
        )
    
    # Check if input is already on GPU
    is_cupy_input = isinstance(image, cp.ndarray)
    is_cupy_kernel = isinstance(kernel, cp.ndarray)
    
    # Convert to CuPy arrays and ensure float32
    if not is_cupy_input:
        image_gpu = cp.asarray(image, dtype=cp.float32)
    else:
        image_gpu = image.astype(cp.float32, copy=False)
    
    if not is_cupy_kernel:
        kernel_gpu = cp.asarray(kernel, dtype=cp.float32)
    else:
        kernel_gpu = kernel.astype(cp.float32, copy=False)
    
    # Handle RGB images (process each channel separately)
    if image_gpu.ndim == 3:
        channels = []
        for c in range(image_gpu.shape[2]):
            # Make sure channel is contiguous for CUDA kernel
            channel_img = cp.ascontiguousarray(image_gpu[:, :, c])
            channel_result = _convolve_single_channel(
                channel_img, kernel_gpu, use_shared_mem, warmup
            )
            channels.append(channel_result)
        result_gpu = cp.stack(channels, axis=2)
    elif image_gpu.ndim == 2:
        result_gpu = _convolve_single_channel(
            image_gpu, kernel_gpu, use_shared_mem, warmup
        )
    else:
        raise ValueError(
            f"Image must be 2D (grayscale) or 3D (RGB), got shape {image.shape}"
        )
    
    # Convert back to NumPy if requested
    if return_numpy:
        return cp.asnumpy(result_gpu)
    else:
        return result_gpu


def _convolve_single_channel(image_gpu, kernel_gpu, use_shared_mem, warmup):
    """Internal function to convolve a single channel.
    
    Args:
        image_gpu: CuPy array of shape (H, W)
        kernel_gpu: CuPy array of shape (K, K)
        use_shared_mem: Whether to use optimized kernel
        warmup: Whether to run warmup iteration
        
    Returns:
        CuPy array of shape (H, W) with convolution result
    """
    # Select kernel implementation
    if use_shared_mem:
        conv_func = convolve_optimized
    else:
        conv_func = convolve_naive
    
    # Warmup run (discard result)
    if warmup:
        _ = conv_func(image_gpu, kernel_gpu)
        cp.cuda.Stream.null.synchronize()
    
    # Actual computation
    result = conv_func(image_gpu, kernel_gpu)
    
    return result


def convolve_cpu(image, kernel):
    """CPU baseline convolution using SciPy for comparison.
    
    Args:
        image: NumPy array (H, W) or (H, W, 3)
        kernel: NumPy array (K, K)
        
    Returns:
        NumPy array with same shape as input
    """
    from scipy.ndimage import convolve as scipy_convolve
    
    # Ensure float32 for fair comparison
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)
    
    if image.ndim == 3:
        # Process each channel separately
        channels = []
        for c in range(image.shape[2]):
            channel_result = scipy_convolve(
                image[:, :, c], kernel, mode='nearest'
            )
            channels.append(channel_result)
        return np.stack(channels, axis=2)
    else:
        return scipy_convolve(image, kernel, mode='nearest')


def convolve_cupy_builtin(image, kernel):
    """GPU convolution using CuPy's built-in implementation for comparison.
    
    Args:
        image: NumPy array (H, W) or (H, W, 3)
        kernel: NumPy array (K, K)
        
    Returns:
        NumPy array with same shape as input
    """
    from cupyx.scipy.ndimage import convolve as cupy_convolve
    
    image_gpu = cp.asarray(image, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel, dtype=cp.float32)
    
    if image_gpu.ndim == 3:
        channels = []
        for c in range(image_gpu.shape[2]):
            channel_result = cupy_convolve(
                image_gpu[:, :, c], kernel_gpu, mode='nearest'
            )
            channels.append(channel_result)
        result = cp.stack(channels, axis=2)
    else:
        result = cupy_convolve(image_gpu, kernel_gpu, mode='nearest')
    
    return cp.asnumpy(result)

