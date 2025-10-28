"""Predefined convolution kernels for common image processing tasks."""

import numpy as np


def get_kernel(name):
    """Get a predefined convolution kernel by name.
    
    Args:
        name: Kernel name. One of: 'sobel_x', 'sobel_y', 'gaussian', 'box_blur',
              'sharpen', 'edge_detect', 'emboss'
    
    Returns:
        2D numpy array representing the kernel
        
    Raises:
        ValueError: If kernel name is not recognized
    """
    kernels = {
        'sobel_x': np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),
        
        'sobel_y': np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32),
        
        'gaussian': np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0,
        
        'box_blur': np.ones((3, 3), dtype=np.float32) / 9.0,
        
        'sharpen': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32),
        
        'edge_detect': np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32),
        
        'emboss': np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ], dtype=np.float32),
        
        'gaussian_5x5': np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=np.float32) / 256.0,
        
        'box_blur_5x5': np.ones((5, 5), dtype=np.float32) / 25.0,
    }
    
    if name not in kernels:
        raise ValueError(
            f"Unknown kernel '{name}'. Available kernels: {', '.join(kernels.keys())}"
        )
    
    return kernels[name]


def list_kernels():
    """List all available predefined kernels.
    
    Returns:
        List of kernel names
    """
    return [
        'sobel_x', 'sobel_y', 'gaussian', 'box_blur',
        'sharpen', 'edge_detect', 'emboss',
        'gaussian_5x5', 'box_blur_5x5'
    ]


def normalize_kernel(kernel):
    """Normalize a kernel so its elements sum to 1.
    
    Useful for blur kernels to maintain image brightness.
    
    Args:
        kernel: 2D numpy array
        
    Returns:
        Normalized kernel
    """
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        return kernel / kernel_sum
    return kernel


def create_gaussian_kernel(size, sigma=None):
    """Create a Gaussian blur kernel.
    
    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation. If None, defaults to size/6
        
    Returns:
        2D numpy array with Gaussian kernel
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    if sigma is None:
        sigma = size / 6.0
    
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    return (kernel / np.sum(kernel)).astype(np.float32)

