"""Streamlit web UI for CUDA convolution accelerator."""

import sys
from pathlib import Path
import time

import streamlit as st
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import convolve
from src.presets import get_kernel, list_kernels


# Page config
st.set_page_config(
    page_title="CUDA Convolution Accelerator",
    page_icon="🚀",
    layout="wide"
)


@st.cache_resource
def check_cuda():
    """Check if CUDA is available."""
    try:
        import cupy as cp
        device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return True, device_name
    except (ImportError, RuntimeError) as e:
        return False, str(e)


def normalize_image(img_array):
    """Normalize image to 0-1 range."""
    if img_array.max() > 1.0:
        return img_array / 255.0
    return img_array


def denormalize_image(img_array):
    """Denormalize image from 0-1 to 0-255 range."""
    return np.clip(img_array * 255, 0, 255).astype(np.uint8)


def main():
    st.title("🚀 CUDA Convolution Accelerator")
    st.markdown("Real-time GPU-accelerated image convolution demo")
    
    # Check CUDA availability
    cuda_available, cuda_info = check_cuda()
    
    if cuda_available:
        st.success(f"✓ CUDA Available - GPU: {cuda_info}")
    else:
        st.error(f"✗ CUDA Not Available: {cuda_info}")
        st.info("This app requires CUDA. Please run in Google Colab with GPU runtime.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Kernel selection
    kernel_name = st.sidebar.selectbox(
        "Select Filter",
        list_kernels(),
        index=list_kernels().index('box_blur')
    )
    
    # Implementation selection
    use_optimized = st.sidebar.radio(
        "Kernel Implementation",
        ["Optimized (Shared Memory)", "Naive (Global Memory)"],
        index=0
    ) == "Optimized (Shared Memory)"
    
    # Show benchmark info
    show_benchmark = st.sidebar.checkbox("Show Timing Info", value=True)
    
    # Warmup runs
    warmup_runs = st.sidebar.slider("Warmup Runs", 0, 5, 1)
    
    # File uploader
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to apply convolution"
    )
    
    # Use sample images if no upload
    use_sample = st.sidebar.checkbox("Use Sample Image", value=True)
    
    if uploaded_file is None and not use_sample:
        st.info("👈 Please upload an image or check 'Use Sample Image' to get started")
        return
    
    # Load image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        # Generate sample image
        st.sidebar.info("Using generated sample image")
        # Create a simple pattern
        x = np.linspace(-5, 5, 512)
        y = np.linspace(-5, 5, 512)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        Z = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)
        image = Image.fromarray(Z, mode='L')
    
    # Convert to RGB if needed
    if image.mode != 'RGB' and image.mode != 'L':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    img_array = normalize_image(img_array)
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"Shape: {img_array.shape} | "
                  f"Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    # Load kernel
    kernel = get_kernel(kernel_name)
    
    # Apply convolution
    with st.spinner("Processing..."):
        # Warmup
        for _ in range(warmup_runs):
            _ = convolve(img_array, kernel, use_shared_mem=use_optimized)
        
        # Timed run
        start_time = time.perf_counter()
        result = convolve(img_array, kernel, use_shared_mem=use_optimized)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
    
    # Display result
    with col2:
        st.subheader("Convolved Image")
        result_display = denormalize_image(result)
        st.image(result_display, use_container_width=True)
        st.caption(f"Shape: {result.shape} | "
                  f"Range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Show timing info
    if show_benchmark:
        st.divider()
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Execution Time", f"{elapsed_ms:.2f} ms")
        
        with col_b:
            num_pixels = img_array.shape[0] * img_array.shape[1]
            if img_array.ndim == 3:
                num_pixels *= img_array.shape[2]
            throughput = num_pixels / (elapsed_ms / 1000) / 1e6
            st.metric("Throughput", f"{throughput:.2f} MP/s")
        
        with col_c:
            st.metric("Pixels Processed", f"{num_pixels:,}")
    
    # Show kernel info
    with st.expander("ℹ️ Kernel Information"):
        st.write(f"**Kernel Name:** {kernel_name}")
        st.write(f"**Kernel Size:** {kernel.shape}")
        st.write(f"**Implementation:** {'Optimized (Tiled + Shared Memory)' if use_optimized else 'Naive (Global Memory)'}")
        
        # Display kernel values
        st.write("**Kernel Values:**")
        st.dataframe(kernel, use_container_width=False)
    
    # Download button
    st.divider()
    
    # Convert result to bytes for download
    result_image = Image.fromarray(result_display, mode='L' if result.ndim == 2 else 'RGB')
    
    from io import BytesIO
    buf = BytesIO()
    result_image.save(buf, format='PNG')
    byte_data = buf.getvalue()
    
    st.download_button(
        label="📥 Download Result",
        data=byte_data,
        file_name=f"convolved_{kernel_name}.png",
        mime="image/png"
    )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
        Built with CUDA + CuPy | 
        <a href='https://github.com' target='_blank'>GitHub</a>
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()

