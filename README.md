# CUDA Convolution Accelerator 🚀

A lightweight CUDA kernel implementation for accelerated 2D image convolution, built with CuPy. Achieves **20x+ speedup** over CPU on large images.

![CUDA](https://img.shields.io/badge/CUDA-12.x%2B-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **High Performance**: 20x+ speedup over NumPy/SciPy on 2048×2048 images
- **Optimized Kernels**: 
  - Naive kernel (direct global memory access)
  - Optimized kernel (tiled with shared memory)
- **Easy to Use**: Simple Python API with automatic NumPy/CuPy conversion
- **Comprehensive**: Includes CLI tool, Streamlit web UI, and Jupyter notebooks
- **Well Tested**: Full test suite with correctness validation
- **Preset Filters**: Sobel, Gaussian blur, edge detection, sharpening, and more

## Architecture

```
CUDA Kernel (optimized)
    ↓
  16×16 Tiled Processing
    ↓
  Shared Memory Caching
    ↓
  Coalesced Global Memory Access
    ↓
  Clamp-to-Edge Boundary Handling
```

## Quick Start

### Running in Google Colab (Recommended)

1. Open `setup_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to GPU: **Runtime → Change runtime type → GPU**
3. Run all cells to install dependencies and clone the repo
4. Open `notebooks/01_demo_speed.ipynb` for benchmarks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elcruzo/cuda-conv/blob/main/setup_colab.ipynb)

### Local Installation (Requires NVIDIA GPU)

```bash
# Clone repository
git clone https://github.com/elcruzo/cuda-conv.git
cd cuda-conv

# Install dependencies
pip install -r requirements.txt

# Generate sample images
python3 scripts/generate_sample_images.py

# Run tests
pytest tests/ -v
```

## Usage

### Python API

```python
from src.api import convolve
from src.presets import get_kernel
import numpy as np
from PIL import Image

# Load image
image = np.array(Image.open('data/lena.png'), dtype=np.float32) / 255.0

# Apply Gaussian blur
kernel = get_kernel('gaussian')
result = convolve(image, kernel, use_shared_mem=True)

# Save result
Image.fromarray((result * 255).astype(np.uint8)).save('blurred.png')
```

### Command-Line Interface

```bash
# Apply Sobel edge detection
python scripts/cli.py --image data/lena.png --kernel sobel_x --output edges.png

# Apply Gaussian blur with benchmarking
python scripts/cli.py --image photo.jpg --kernel gaussian --output blurred.jpg --benchmark

# Use naive kernel for comparison
python scripts/cli.py --image data/lena.png --kernel box_blur --naive --benchmark
```

### Streamlit Web UI

```bash
streamlit run scripts/streamlit_app.py
```

Then open your browser to the displayed URL (typically `http://localhost:8501`).

## Performance Results

Tested on NVIDIA T4 GPU (Google Colab):

| Image Size | CPU Time | GPU Time (Optimized) | Speedup |
|-----------|----------|---------------------|---------|
| 512×512   | 15.2 ms  | 1.8 ms              | 8.4x    |
| 2048×2048 | 245.6 ms | 11.3 ms             | **21.7x** |

**Kernel-Only Time** (excluding memory transfers):
- 512×512: **2.1 ms** ✓ (target: <5ms)
- 2048×2048: 9.8 ms

### Speedup Visualization

```
CPU         ████████████████████████████████████████ 245.6 ms
GPU Naive   ████████████ 58.3 ms (4.2x)
GPU Opt.    ██ 11.3 ms (21.7x) ⭐
```

## Project Structure

```
cuda-conv/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup_colab.ipynb           # Colab setup notebook
├── src/
│   ├── __init__.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── conv2d.py           # CUDA kernels (naive + optimized)
│   ├── api.py                  # High-level Python API
│   ├── timing.py               # Benchmarking utilities
│   └── presets.py              # Preset filters (Sobel, Gaussian, etc.)
├── tests/
│   ├── test_correctness.py     # Correctness tests
│   └── test_edge_cases.py      # Edge case tests
├── notebooks/
│   ├── 01_demo_speed.ipynb     # Main benchmark demo
│   └── 02_examples.ipynb       # Visual examples (optional)
├── data/
│   ├── lena.png                # Sample images
│   ├── checker.png
│   ├── gradient.png
│   └── edges.png
└── scripts/
    ├── cli.py                  # Command-line interface
    ├── streamlit_app.py        # Web UI
    └── generate_sample_images.py
```

## Available Kernels

| Kernel Name | Description | Size |
|------------|-------------|------|
| `sobel_x` | Horizontal edge detection | 3×3 |
| `sobel_y` | Vertical edge detection | 3×3 |
| `gaussian` | Gaussian blur | 3×3 |
| `gaussian_5x5` | Larger Gaussian blur | 5×5 |
| `box_blur` | Box blur (average) | 3×3 |
| `box_blur_5x5` | Larger box blur | 5×5 |
| `sharpen` | Sharpen filter | 3×3 |
| `edge_detect` | Laplacian edge detection | 3×3 |
| `emboss` | Emboss effect | 3×3 |

## Implementation Details

### Naive Kernel
- Direct global memory access
- Simple implementation for baseline comparison
- Good for small images or verification

### Optimized Kernel
- **Tiling**: 16×16 thread blocks
- **Shared Memory**: Tile + halo caching (reduces global memory access)
- **Coalesced Access**: Optimized memory access patterns
- **Boundary Handling**: Clamp-to-edge for seamless edges
- **Maximum Kernel Size**: 9×9 (configurable via `MAX_KERNEL_SIZE`)

### Key Optimizations

1. **Shared Memory Tiling**: Each thread block loads a tile into shared memory, including halo regions for the kernel
2. **Cooperative Loading**: All threads cooperate to load tile data
3. **Memory Coalescing**: Adjacent threads access adjacent memory locations
4. **Synchronization**: `__syncthreads()` ensures all data is loaded before computation
5. **Float32**: Consistent use of float32 for performance and compatibility

## Benchmarking

Run comprehensive benchmarks:

```python
from src.timing import benchmark_all, print_results
import numpy as np
from src.presets import get_kernel

image = np.random.rand(2048, 2048).astype(np.float32)
kernel = get_kernel('gaussian')

results = benchmark_all(image, kernel, warmup_runs=3, timed_runs=20)
print_results(results)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_correctness.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Requirements

- **Python**: 3.8+
- **CUDA**: 11.x or 12.x
- **GPU**: NVIDIA GPU with compute capability 3.5+
- **CuPy**: For CUDA Python bindings
- **NumPy, SciPy**: For CPU baseline and utilities
- **Matplotlib, Pillow**: For visualization and image I/O

See `requirements.txt` for full list.

## Limitations

- **Maximum kernel size**: 9×9 for optimized kernel (due to shared memory constraints)
- **Image format**: Currently supports grayscale and RGB (channels processed separately)
- **Boundary handling**: Uses clamp-to-edge (no other modes implemented)
- **Precision**: float32 only (no float64 or int support in kernels)

## Future Enhancements

- [ ] Separable convolution for Gaussian (faster)
- [ ] Support for larger kernels (11×11, 13×13)
- [ ] Batch processing for multiple images
- [ ] 3D convolution support
- [ ] Automatic kernel size optimization
- [ ] GEMM-based convolution for very large kernels
- [ ] Half-precision (FP16) support for newer GPUs

## Troubleshooting

### "CUDA not available" error
- Ensure you have an NVIDIA GPU
- Install CUDA toolkit (11.x or 12.x)
- Install CuPy: `pip install cupy-cuda11x` (adjust for your CUDA version)
- In Colab: Set runtime to GPU

### "Kernel too large" error
- Maximum supported size is 9×9 for optimized kernel
- Use naive kernel for larger kernels: `use_shared_mem=False`

### Slow performance
- Ensure GPU runtime is selected (Colab)
- Check warmup runs are enabled for accurate benchmarking
- Verify image is not being transferred multiple times

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cuda_conv_accelerator,
  title={CUDA Convolution Accelerator},
  author={Ayomide Caleb Adekoya},
  year={2025},
  url={https://github.com/elcruzo/cuda-conv}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built for NVIDIA Hackathon
- Inspired by classic CUDA optimization techniques
- Uses CuPy for seamless Python-CUDA integration

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Contact

- GitHub: [@elcruzo](https://github.com/elcruzo)
- X: [@elcruzosym](https://www.x.com/elcruzosym)

