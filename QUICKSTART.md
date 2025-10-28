# Quick Start Guide

Get up and running with CUDA Convolution Accelerator in 5 minutes!

## Option 1: Google Colab (Easiest - No Local GPU Required)

### Step 1: Open Colab Notebook

Click this button:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/cuda-conv/blob/main/setup_colab.ipynb)

Or manually:
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Upload `setup_colab.ipynb` from this repo

### Step 2: Enable GPU

**CRITICAL**: Set runtime to GPU!
- Click: **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU**
- Click **Save**

### Step 3: Run Setup

Run all cells in `setup_colab.ipynb`:
- Click **Runtime** → **Run all**
- Wait ~2 minutes for installation

### Step 4: Run Benchmarks

Open and run `notebooks/01_demo_speed.ipynb` to see the 20x speedup!

## Option 2: Local Installation (Requires NVIDIA GPU)

### Prerequisites

- NVIDIA GPU (compute capability 3.5+)
- CUDA 11.x or 12.x installed
- Python 3.8+

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/cuda-conv.git
cd cuda-conv

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate sample images
make data
# Or: python3 scripts/generate_sample_images.py

# 5. Verify installation
python3 -c "import cupy as cp; print(f'GPU: {cp.cuda.Device().name.decode()}')"
```

### Test It

```bash
# Run tests
make test

# Try CLI
make cli

# Launch Streamlit UI
make streamlit

# Or manually
python3 scripts/cli.py --image data/lena.png --kernel gaussian --output result.png --benchmark
```

## Quick Examples

### Python API

```python
from src.api import convolve
from src.presets import get_kernel
import numpy as np
from PIL import Image

# Load image
img = np.array(Image.open('data/lena.png'), dtype=np.float32) / 255.0

# Apply Gaussian blur
kernel = get_kernel('gaussian')
result = convolve(img, kernel, use_shared_mem=True)

# Save
Image.fromarray((result * 255).astype(np.uint8)).save('blurred.png')
print("✓ Done!")
```

### Command Line

```bash
# Edge detection
python3 scripts/cli.py \
    --image data/lena.png \
    --kernel sobel_x \
    --output edges.png \
    --benchmark

# Gaussian blur
python3 scripts/cli.py \
    --image data/lena.png \
    --kernel gaussian \
    --output blurred.png

# Use naive kernel (for comparison)
python3 scripts/cli.py \
    --image data/lena.png \
    --kernel box_blur \
    --naive \
    --output result.png
```

### Streamlit Web UI

```bash
streamlit run scripts/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Benchmarking

### Quick Benchmark

```python
from src.timing import benchmark_all, print_results
from src.presets import get_kernel
import numpy as np

# Create test image
image = np.random.rand(2048, 2048).astype(np.float32)
kernel = get_kernel('gaussian')

# Run benchmark
results = benchmark_all(image, kernel, warmup_runs=3, timed_runs=20)
print_results(results)
```

### Expected Output (NVIDIA T4)

```
================================================================================
Method               Time (ms)       Throughput (MP/s)    Speedup   
================================================================================
cpu                     245.600 ms        17.08 MP/s      1.00x     
gpu_naive                58.300 ms        71.94 MP/s      4.21x     
gpu_optimized            11.300 ms       371.24 MP/s      21.74x ⭐  
gpu_builtin              12.100 ms       346.69 MP/s      20.30x    
================================================================================
```

## Available Filters

Try these preset filters:

| Filter | Effect | Command |
|--------|--------|---------|
| `sobel_x` | Horizontal edges | `--kernel sobel_x` |
| `sobel_y` | Vertical edges | `--kernel sobel_y` |
| `gaussian` | Blur | `--kernel gaussian` |
| `sharpen` | Sharpen | `--kernel sharpen` |
| `edge_detect` | All edges | `--kernel edge_detect` |
| `emboss` | 3D emboss | `--kernel emboss` |
| `box_blur` | Average blur | `--kernel box_blur` |

## Troubleshooting

### "CUDA not available"

**In Colab:**
- Ensure GPU runtime is selected (Runtime → Change runtime type → GPU)
- Restart runtime if needed

**Locally:**
- Check GPU: `nvidia-smi`
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Install correct CuPy: `pip install cupy-cuda11x` (or cuda12x)

### "Module not found"

```bash
# Ensure you're in the project directory
cd cuda-conv

# Install dependencies
pip install -r requirements.txt

# Verify
python3 -c "from src.api import convolve; print('✓ OK')"
```

### "No such file: data/lena.png"

```bash
# Generate sample images
python3 scripts/generate_sample_images.py
```

### Poor Performance

- Check GPU is actually being used: `nvidia-smi` (should show Python process)
- Use warmup runs: `--warmup 3`
- Ensure image is on GPU (automatic in our API)
- Try larger images (GPU shines on 1024×1024+)

## Next Steps

1. **Run Benchmarks**: Open `notebooks/01_demo_speed.ipynb`
2. **See Examples**: Open `notebooks/02_examples.ipynb`
3. **Read Architecture**: Check `ARCHITECTURE.md` for technical details
4. **Customize**: Modify kernels in `src/presets.py`

## Getting Help

- Check `README.md` for full documentation
- Check `ARCHITECTURE.md` for technical details
- Open an issue on GitHub
- Read the source code (it's well-commented!)

## Performance Tips

1. **Use optimized kernel** (default): `use_shared_mem=True`
2. **Batch processing**: Process multiple images in a loop
3. **Keep data on GPU**: Use `return_numpy=False` if chaining operations
4. **Warmup**: Always do 1-3 warmup runs for accurate timing
5. **Larger images**: GPU speedup increases with image size

## Quick Reference

```bash
# Generate data
make data

# Run tests
make test

# CLI demo
make cli

# Web UI
make streamlit

# Jupyter
make notebook

# Clean up
make clean

# Help
make help
```

---

**Ready to see 20x speedup?** Open `notebooks/01_demo_speed.ipynb` and run all cells! 🚀

