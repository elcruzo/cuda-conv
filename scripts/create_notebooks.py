#!/usr/bin/env python3
"""Script to create Jupyter notebooks programmatically."""

import json
from pathlib import Path


def create_demo_notebook():
    """Create the main demo notebook."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CUDA Convolution Accelerator - Speed Demo\\n",
                    "\\n",
                    "This notebook demonstrates the speedup achieved by using CUDA-accelerated convolution compared to CPU baseline.\\n",
                    "\\n",
                    "## Setup\\n",
                    "\\n",
                    "First, let's check if we have GPU access and install dependencies."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check GPU availability\\n",
                    "!nvidia-smi"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies (if running in Colab)\\n",
                    "import sys\\n",
                    "IN_COLAB = 'google.colab' in sys.modules\\n",
                    "\\n",
                    "if IN_COLAB:\\n",
                    "    !pip install cupy-cuda11x scipy matplotlib pillow tqdm -q\\n",
                    "    print(\"✓ Dependencies installed\")\\n",
                    "else:\\n",
                    "    print(\"Running locally - ensure dependencies are installed\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Imports"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "from PIL import Image\\n",
                    "import cupy as cp\\n",
                    "from tqdm import tqdm\\n",
                    "import time\\n",
                    "\\n",
                    "# Import our modules\\n",
                    "import sys\\n",
                    "sys.path.insert(0, '..')\\n",
                    "\\n",
                    "from src.api import convolve, convolve_cpu\\n",
                    "from src.presets import get_kernel, list_kernels\\n",
                    "from src.timing import benchmark_all, benchmark_kernel_only, print_results\\n",
                    "\\n",
                    "print(f\"✓ CuPy version: {cp.__version__}\")\\n",
                    "print(f\"✓ GPU: {cp.cuda.Device().name.decode()}\")\\n",
                    "print(f\"✓ Available kernels: {', '.join(list_kernels())}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Load Sample Images and Run Benchmark\\n\\nSee notebook for full analysis!"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def create_colab_setup_notebook():
    """Create the Colab setup notebook."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CUDA Convolution Accelerator - Colab Setup\\n",
                    "\\n",
                    "This notebook sets up the CUDA convolution project in Google Colab.\\n",
                    "\\n",
                    "## Prerequisites\\n",
                    "\\n",
                    "**IMPORTANT:** Set runtime to GPU!\\n",
                    "- Runtime → Change runtime type → Hardware accelerator → GPU\\n",
                    "- Recommended: T4 or better"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 1: Verify GPU Access"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check NVIDIA GPU\\n",
                    "!nvidia-smi\\n",
                    "\\n",
                    "import torch\\n",
                    "if torch.cuda.is_available():\\n",
                    "    print(f\"\\n✓ GPU Available: {torch.cuda.get_device_name(0)}\")\\n",
                    "    print(f\"✓ CUDA Version: {torch.version.cuda}\")\\n",
                    "else:\\n",
                    "    print(\"\\n✗ No GPU detected. Please change runtime type to GPU.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 2: Install Dependencies"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install CuPy and other dependencies\\n",
                    "!pip install cupy-cuda11x scipy matplotlib pillow tqdm pytest -q\\n",
                    "\\n",
                    "print(\"✓ Dependencies installed\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Clone Repository\\n",
                    "\\n",
                    "Choose one of the following options:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Option A: Clone from GitHub (replace with your repo URL)\\n",
                    "!git clone https://github.com/YOUR_USERNAME/cuda-conv.git\\n",
                    "%cd cuda-conv"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Option B: Upload files manually\\n",
                    "from google.colab import files\\n",
                    "import zipfile\\n",
                    "\\n",
                    "# Upload your project zip file\\n",
                    "uploaded = files.upload()\\n",
                    "\\n",
                    "# Extract\\n",
                    "for filename in uploaded.keys():\\n",
                    "    if filename.endswith('.zip'):\\n",
                    "        with zipfile.ZipFile(filename, 'r') as zip_ref:\\n",
                    "            zip_ref.extractall('.')\\n",
                    "        print(f\"✓ Extracted {filename}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 4: Verify Installation"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Test imports\\n",
                    "import cupy as cp\\n",
                    "from src.api import convolve\\n",
                    "from src.presets import get_kernel, list_kernels\\n",
                    "\\n",
                    "print(f\"✓ CuPy version: {cp.__version__}\")\\n",
                    "print(f\"✓ GPU: {cp.cuda.Device().name.decode()}\")\\n",
                    "print(f\"✓ Available kernels: {', '.join(list_kernels())}\")\\n",
                    "\\n",
                    "# Quick test\\n",
                    "import numpy as np\\n",
                    "test_img = np.random.rand(32, 32).astype(np.float32)\\n",
                    "test_kernel = get_kernel('box_blur')\\n",
                    "result = convolve(test_img, test_kernel)\\n",
                    "\\n",
                    "print(f\"\\n✓ Test convolution successful! Result shape: {result.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Generate Sample Images (if needed)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate sample images\\n",
                    "!python3 scripts/generate_sample_images.py\\n",
                    "\\n",
                    "print(\"✓ Sample images generated\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Run Tests"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run test suite\\n",
                    "!pytest tests/ -v\\n",
                    "\\n",
                    "print(\"\\n✓ All tests completed\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Next Steps\\n",
                    "\\n",
                    "Now you're ready to use the CUDA convolution accelerator!\\n",
                    "\\n",
                    "Try:\\n",
                    "- Open `notebooks/01_demo_speed.ipynb` for benchmarks\\n",
                    "- Open `notebooks/02_examples.ipynb` for visual examples\\n",
                    "- Use the CLI: `!python scripts/cli.py --help`"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """Create all notebooks."""
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    notebooks_dir.mkdir(exist_ok=True)
    
    # Create demo notebook
    demo_nb = create_demo_notebook()
    with open(notebooks_dir / '01_demo_speed.ipynb', 'w') as f:
        json.dump(demo_nb, f, indent=1)
    print(f"✓ Created {notebooks_dir / '01_demo_speed.ipynb'}")
    
    # Create Colab setup notebook
    colab_nb = create_colab_setup_notebook()
    setup_path = Path(__file__).parent.parent / 'setup_colab.ipynb'
    with open(setup_path, 'w') as f:
        json.dump(colab_nb, f, indent=1)
    print(f"✓ Created {setup_path}")
    
    print("\\nDone! Notebooks created successfully.")


if __name__ == '__main__':
    main()

