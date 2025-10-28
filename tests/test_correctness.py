"""Tests for convolution correctness against NumPy/SciPy baseline."""

import pytest
import numpy as np
from scipy.ndimage import convolve as scipy_convolve

# These tests will only run in Colab with CUDA available
try:
    import cupy as cp
    from src.api import convolve
    from src.presets import get_kernel
    CUDA_AVAILABLE = True
except (ImportError, RuntimeError):
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestConvolutionCorrectness:
    """Test convolution correctness against SciPy reference implementation."""
    
    def test_simple_3x3_identity(self):
        """Test identity kernel on small image."""
        image = np.random.rand(8, 8).astype(np.float32)
        kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        np.testing.assert_allclose(result_naive, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(result_optimized, expected, rtol=1e-5, atol=1e-5)
    
    def test_simple_3x3_box_blur(self):
        """Test box blur on small image."""
        image = np.random.rand(10, 10).astype(np.float32) * 100
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        np.testing.assert_allclose(result_naive, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result_optimized, expected, rtol=1e-4, atol=1e-4)
    
    def test_sobel_edge_detection(self):
        """Test Sobel edge detection kernel."""
        image = np.random.rand(16, 16).astype(np.float32)
        kernel = get_kernel('sobel_x')
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        np.testing.assert_allclose(result_naive, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result_optimized, expected, rtol=1e-4, atol=1e-4)
    
    def test_gaussian_blur(self):
        """Test Gaussian blur kernel."""
        image = np.random.rand(20, 20).astype(np.float32) * 255
        kernel = get_kernel('gaussian')
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        np.testing.assert_allclose(result_naive, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result_optimized, expected, rtol=1e-4, atol=1e-4)
    
    def test_5x5_kernel(self):
        """Test larger 5x5 kernel."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = get_kernel('gaussian_5x5')
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        np.testing.assert_allclose(result_naive, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result_optimized, expected, rtol=1e-4, atol=1e-4)
    
    def test_integer_kernel(self):
        """Test with integer kernel (exact match expected)."""
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32)
        
        kernel = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ], dtype=np.float32)
        
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        np.testing.assert_allclose(result_optimized, expected, rtol=1e-5, atol=1e-5)
    
    def test_rgb_image(self):
        """Test RGB image convolution."""
        image = np.random.rand(16, 16, 3).astype(np.float32) * 255
        kernel = get_kernel('box_blur')
        
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        
        # Compare each channel separately
        for c in range(3):
            expected = scipy_convolve(image[:, :, c], kernel, mode='nearest')
            np.testing.assert_allclose(
                result_optimized[:, :, c], expected, rtol=1e-4, atol=1e-4
            )
    
    def test_naive_vs_optimized_consistency(self):
        """Ensure naive and optimized kernels produce same results."""
        image = np.random.rand(64, 64).astype(np.float32)
        kernel = get_kernel('sharpen')
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        
        np.testing.assert_allclose(result_naive, result_optimized, rtol=1e-5, atol=1e-5)
    
    def test_medium_image(self):
        """Test on medium-sized image (512x512)."""
        image = np.random.rand(512, 512).astype(np.float32)
        kernel = get_kernel('edge_detect')
        
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        
        # Spot check a few pixels against expected result
        expected = scipy_convolve(image, kernel, mode='nearest')
        
        # Check corners and center
        test_points = [(0, 0), (0, 511), (511, 0), (511, 511), (256, 256)]
        for y, x in test_points:
            np.testing.assert_allclose(
                result_optimized[y, x], expected[y, x], rtol=1e-4, atol=1e-4
            )

