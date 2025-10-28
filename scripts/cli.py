#!/usr/bin/env python3
"""Command-line interface for CUDA convolution accelerator."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import convolve
from src.presets import get_kernel, list_kernels


def load_image(image_path):
    """Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        NumPy array with image data (normalized to 0-1 range)
    """
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB' and img.mode != 'L':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to 0-1 range
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    return img_array


def save_image(image_array, output_path):
    """Save image array to file.
    
    Args:
        image_array: NumPy array with image data (0-1 range)
        output_path: Path to save image
    """
    # Denormalize and clip
    img_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    if img_array.ndim == 2:
        img = Image.fromarray(img_array, mode='L')
    else:
        img = Image.fromarray(img_array, mode='RGB')
    
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Apply CUDA-accelerated convolution to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available preset kernels:
  {', '.join(list_kernels())}

Examples:
  # Apply Sobel edge detection
  python scripts/cli.py --image data/lena.png --kernel sobel_x --output result.png
  
  # Apply Gaussian blur
  python scripts/cli.py --image photo.jpg --kernel gaussian --output blurred.jpg
  
  # Use naive kernel (for comparison)
  python scripts/cli.py --image data/lena.png --kernel box_blur --naive
"""
    )
    
    parser.add_argument(
        '--image', '-i',
        required=True,
        help='Input image path'
    )
    
    parser.add_argument(
        '--kernel', '-k',
        required=True,
        help='Kernel name (see available presets below) or path to .npy file'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output image path'
    )
    
    parser.add_argument(
        '--naive',
        action='store_true',
        help='Use naive kernel instead of optimized (for benchmarking)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Show detailed timing information'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=1,
        help='Number of warmup runs (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    try:
        import cupy as cp
        print(f"✓ CUDA available - GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except (ImportError, RuntimeError) as e:
        print(f"✗ CUDA not available: {e}")
        print("This tool requires CUDA. Please run in Google Colab or on a machine with NVIDIA GPU.")
        return 1
    
    # Load image
    print(f"Loading image: {args.image}")
    try:
        image = load_image(args.image)
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return 1
    
    # Load kernel
    print(f"Loading kernel: {args.kernel}")
    try:
        if args.kernel.endswith('.npy'):
            # Load custom kernel from file
            kernel = np.load(args.kernel).astype(np.float32)
            print(f"  Custom kernel loaded: {kernel.shape}")
        else:
            # Use preset kernel
            kernel = get_kernel(args.kernel)
            print(f"  Preset kernel: {kernel.shape}")
    except Exception as e:
        print(f"✗ Error loading kernel: {e}")
        return 1
    
    # Warmup runs
    if args.warmup > 0:
        print(f"Running {args.warmup} warmup iteration(s)...")
        for _ in range(args.warmup):
            _ = convolve(
                image, kernel,
                use_shared_mem=not args.naive,
                return_numpy=True
            )
    
    # Run convolution
    kernel_type = "naive" if args.naive else "optimized"
    print(f"Running convolution ({kernel_type} kernel)...")
    
    start_time = time.perf_counter()
    result = convolve(
        image, kernel,
        use_shared_mem=not args.naive,
        return_numpy=True
    )
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000
    
    print(f"✓ Convolution completed in {elapsed_ms:.2f} ms")
    print(f"  Result shape: {result.shape}")
    print(f"  Result range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Show benchmark info if requested
    if args.benchmark:
        num_pixels = image.shape[0] * image.shape[1]
        if image.ndim == 3:
            num_pixels *= image.shape[2]
        throughput = num_pixels / (elapsed_ms / 1000) / 1e6  # MP/s
        
        print("\nBenchmark results:")
        print(f"  Execution time: {elapsed_ms:.3f} ms")
        print(f"  Throughput: {throughput:.2f} MP/s")
        print(f"  Pixels processed: {num_pixels:,}")
    
    # Save result
    print(f"Saving result: {args.output}")
    try:
        save_image(result, args.output)
        print(f"✓ Image saved successfully")
    except Exception as e:
        print(f"✗ Error saving image: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

