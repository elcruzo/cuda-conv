"""Tests for edge cases and error handling."""

import pytest
import numpy as np

try:
    import cupy as cp
    from src.api import convolve
    from src.presets import get_kernel
    CUDA_AVAILABLE = True
except (ImportError, RuntimeError):
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_tiny_1x1_image(self):
        """Test convolution on 1x1 image."""
        image = np.array([[5.0]], dtype=np.float32)
        kernel = get_kernel('box_blur')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert result.shape == (1, 1)
        # With clamp-to-edge, single pixel should be convolved with itself
        assert result[0, 0] == pytest.approx(5.0, rel=1e-4)
    
    def test_small_7x7_image(self):
        """Test on very small image."""
        image = np.random.rand(7, 7).astype(np.float32)
        kernel = get_kernel('gaussian')
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        
        assert result_naive.shape == image.shape
        assert result_optimized.shape == image.shape
        np.testing.assert_allclose(result_naive, result_optimized, rtol=1e-5, atol=1e-5)
    
    def test_non_square_image(self):
        """Test on non-square image."""
        image = np.random.rand(32, 64).astype(np.float32)
        kernel = get_kernel('sobel_x')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert result.shape == image.shape
    
    def test_wide_image(self):
        """Test on very wide image."""
        image = np.random.rand(16, 256).astype(np.float32)
        kernel = get_kernel('box_blur')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert result.shape == image.shape
    
    def test_tall_image(self):
        """Test on very tall image."""
        image = np.random.rand(256, 16).astype(np.float32)
        kernel = get_kernel('box_blur')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert result.shape == image.shape
    
    def test_extreme_values(self):
        """Test with extreme pixel values."""
        # Very large values
        image = np.random.rand(32, 32).astype(np.float32) * 10000
        kernel = get_kernel('gaussian')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Very small values
        image = np.random.rand(32, 32).astype(np.float32) * 0.0001
        result = convolve(image, kernel, use_shared_mem=True)
        assert not np.any(np.isnan(result))
    
    def test_zero_image(self):
        """Test on all-zero image."""
        image = np.zeros((32, 32), dtype=np.float32)
        kernel = get_kernel('sobel_x')
        
        result = convolve(image, kernel, use_shared_mem=True)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)
    
    def test_constant_image(self):
        """Test on constant-valued image."""
        image = np.full((32, 32), 42.0, dtype=np.float32)
        kernel = get_kernel('sobel_y')
        
        # Sobel on constant image should be near zero
        result = convolve(image, kernel, use_shared_mem=True)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)
    
    def test_asymmetric_kernel(self):
        """Test with asymmetric kernel."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=np.float32)
        
        result_naive = convolve(image, kernel, use_shared_mem=False)
        result_optimized = convolve(image, kernel, use_shared_mem=True)
        
        np.testing.assert_allclose(result_naive, result_optimized, rtol=1e-5, atol=1e-5)
    
    def test_negative_kernel_values(self):
        """Test with negative kernel values."""
        image = np.random.rand(32, 32).astype(np.float32) * 100
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert not np.any(np.isnan(result))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_even_kernel_size(self):
        """Test that even kernel size raises error."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((4, 4), dtype=np.float32) / 16.0  # Even size
        
        with pytest.raises(ValueError, match="must be odd"):
            convolve(image, kernel)
    
    def test_non_square_kernel(self):
        """Test that non-square kernel raises error."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((3, 5), dtype=np.float32)  # Not square
        
        with pytest.raises(ValueError, match="must be square"):
            convolve(image, kernel)
    
    def test_kernel_too_large(self):
        """Test that kernel larger than 9x9 raises error (for optimized)."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((11, 11), dtype=np.float32) / 121.0  # Too large
        
        with pytest.raises(ValueError, match="too large"):
            convolve(image, kernel, use_shared_mem=True)
    
    def test_wrong_image_dimensions(self):
        """Test that 1D or 4D images raise error."""
        kernel = get_kernel('box_blur')
        
        # 1D array
        image_1d = np.random.rand(32).astype(np.float32)
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            convolve(image_1d, kernel)
        
        # 4D array
        image_4d = np.random.rand(2, 32, 32, 3).astype(np.float32)
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            convolve(image_4d, kernel)
    
    def test_unknown_preset_kernel(self):
        """Test that unknown kernel name raises error."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            get_kernel('nonexistent_kernel')


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestDataTypes:
    """Test handling of different data types."""
    
    def test_float64_conversion(self):
        """Test automatic conversion from float64 to float32."""
        image = np.random.rand(32, 32).astype(np.float64)
        kernel = get_kernel('box_blur')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert result.dtype == np.float32
    
    def test_int_conversion(self):
        """Test automatic conversion from int to float32."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        kernel = get_kernel('gaussian')
        
        result = convolve(image, kernel, use_shared_mem=True)
        assert result.dtype == np.float32
    
    def test_cupy_input(self):
        """Test that CuPy arrays work as input."""
        image_np = np.random.rand(32, 32).astype(np.float32)
        image_cp = cp.asarray(image_np)
        kernel = get_kernel('sobel_x')
        
        result = convolve(image_cp, kernel, use_shared_mem=True)
        assert isinstance(result, np.ndarray)
    
    def test_return_cupy_array(self):
        """Test returning CuPy array instead of NumPy."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = get_kernel('box_blur')
        
        result = convolve(image, kernel, use_shared_mem=True, return_numpy=False)
        assert isinstance(result, cp.ndarray)

