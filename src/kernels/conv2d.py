"""CUDA kernels for 2D convolution using CuPy RawKernel."""

import cupy as cp
import numpy as np

# Naive kernel - direct global memory access
NAIVE_KERNEL = r'''
extern "C" __global__
void convolve2d_naive(
    const float* input,
    const float* kernel,
    float* output,
    int height,
    int width,
    int ksize
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int khalf = ksize / 2;
    float sum = 0.0f;
    
    // Convolve with clamp-to-edge boundary handling
    for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
            int iy = row + ky - khalf;
            int ix = col + kx - khalf;
            
            // Clamp to edge
            iy = max(0, min(iy, height - 1));
            ix = max(0, min(ix, width - 1));
            
            // Flip kernel for true convolution (not cross-correlation)
            int k_idx = (ksize - 1 - ky) * ksize + (ksize - 1 - kx);
            sum += input[iy * width + ix] * kernel[k_idx];
        }
    }
    
    output[row * width + col] = sum;
}
'''

# Optimized kernel - tiled with shared memory
OPTIMIZED_KERNEL = r'''
#define TILE_WIDTH 16
#define MAX_KERNEL_SIZE 9

extern "C" __global__
void convolve2d_optimized(
    const float* input,
    const float* kernel,
    float* output,
    int height,
    int width,
    int ksize
) {
    // Shared memory for tile + halo
    __shared__ float tile[TILE_WIDTH + MAX_KERNEL_SIZE - 1][TILE_WIDTH + MAX_KERNEL_SIZE - 1];
    __shared__ float s_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int khalf = ksize / 2;
    
    int tile_size = TILE_WIDTH + ksize - 1;
    
    // Cooperatively load kernel into shared memory
    int kid = ty * blockDim.x + tx;
    if (kid < ksize * ksize) {
        s_kernel[kid] = kernel[kid];
    }
    
    // Load tile with halo into shared memory
    // Each thread may need to load multiple elements
    for (int i = ty; i < tile_size; i += blockDim.y) {
        for (int j = tx; j < tile_size; j += blockDim.x) {
            int iy = blockIdx.y * TILE_WIDTH + i - khalf;
            int ix = blockIdx.x * TILE_WIDTH + j - khalf;
            
            // Clamp to edge
            iy = max(0, min(iy, height - 1));
            ix = max(0, min(ix, width - 1));
            
            tile[i][j] = input[iy * width + ix];
        }
    }
    
    __syncthreads();
    
    // Compute convolution from shared memory
    if (row < height && col < width) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Flip kernel for true convolution (not cross-correlation)
                int k_idx = (ksize - 1 - ky) * ksize + (ksize - 1 - kx);
                sum += tile[ty + ky][tx + kx] * s_kernel[k_idx];
            }
        }
        
        output[row * width + col] = sum;
    }
}
'''


class Conv2DKernel:
    """Wrapper class for CUDA convolution kernels."""
    
    def __init__(self, use_optimized=True):
        """Initialize the kernel wrapper.
        
        Args:
            use_optimized: If True, use optimized tiled kernel. Otherwise use naive kernel.
        """
        self.use_optimized = use_optimized
        
        if use_optimized:
            self.kernel = cp.RawKernel(OPTIMIZED_KERNEL, 'convolve2d_optimized')
        else:
            self.kernel = cp.RawKernel(NAIVE_KERNEL, 'convolve2d_naive')
    
    def __call__(self, input_array, kernel_array):
        """Execute convolution on GPU.
        
        Args:
            input_array: CuPy array of shape (H, W) with dtype float32
            kernel_array: CuPy array of shape (K, K) with dtype float32
            
        Returns:
            CuPy array of shape (H, W) with convolution result
        """
        height, width = input_array.shape
        ksize = kernel_array.shape[0]
        
        # Allocate output
        output_array = cp.empty_like(input_array)
        
        # Configure grid and block dimensions
        if self.use_optimized:
            # Use 16x16 tile size for optimized kernel
            block_size = (16, 16)
            grid_size = (
                (width + block_size[0] - 1) // block_size[0],
                (height + block_size[1] - 1) // block_size[1]
            )
        else:
            # Use 16x16 blocks for naive kernel
            block_size = (16, 16)
            grid_size = (
                (width + block_size[0] - 1) // block_size[0],
                (height + block_size[1] - 1) // block_size[1]
            )
        
        # Launch kernel
        self.kernel(
            grid_size,
            block_size,
            (input_array, kernel_array, output_array, height, width, ksize)
        )
        
        return output_array


def convolve_naive(input_array, kernel_array):
    """Convolve image with kernel using naive CUDA implementation.
    
    Args:
        input_array: CuPy array of shape (H, W)
        kernel_array: CuPy array of shape (K, K)
        
    Returns:
        CuPy array of shape (H, W) with convolution result
    """
    kernel = Conv2DKernel(use_optimized=False)
    return kernel(input_array, kernel_array)


def convolve_optimized(input_array, kernel_array):
    """Convolve image with kernel using optimized tiled CUDA implementation.
    
    Args:
        input_array: CuPy array of shape (H, W)
        kernel_array: CuPy array of shape (K, K)
        
    Returns:
        CuPy array of shape (H, W) with convolution result
    """
    kernel = Conv2DKernel(use_optimized=True)
    return kernel(input_array, kernel_array)

