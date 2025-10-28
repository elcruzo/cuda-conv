"""Generate sample images for testing and demo."""

import numpy as np
from PIL import Image
from pathlib import Path


def generate_lena():
    """Generate a Lena-like test pattern (512x512 grayscale)."""
    # Create a complex test pattern with various features
    size = 512
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Combine multiple patterns
    Z1 = np.sin(np.sqrt(X**2 + Y**2))
    Z2 = np.cos(X) * np.sin(Y)
    Z3 = np.exp(-0.1 * (X**2 + Y**2))
    
    Z = 0.3 * Z1 + 0.3 * Z2 + 0.4 * Z3
    
    # Normalize to 0-255
    Z = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)
    
    # Add some structure
    # Add circles
    center_y, center_x = size // 2, size // 2
    for r in [50, 100, 150]:
        rr, cc = np.ogrid[:size, :size]
        circle = (rr - center_y)**2 + (cc - center_x)**2
        mask = (circle > (r-2)**2) & (circle < (r+2)**2)
        Z[mask] = 255
    
    # Add rectangles
    Z[size//4:size//4+10, size//4:3*size//4] = 200
    Z[3*size//4:3*size//4+10, size//4:3*size//4] = 200
    Z[size//4:3*size//4, size//4:size//4+10] = 200
    Z[size//4:3*size//4, 3*size//4:3*size//4+10] = 200
    
    return Image.fromarray(Z, mode='L')


def generate_checker():
    """Generate a checkerboard pattern (512x512 grayscale)."""
    size = 512
    square_size = 32
    
    # Create checkerboard
    img = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255
    
    return Image.fromarray(img, mode='L')


def generate_gradient():
    """Generate a smooth gradient (512x512 grayscale)."""
    size = 512
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Diagonal gradient
    Z = (X + Y) / 2
    Z = (Z * 255).astype(np.uint8)
    
    return Image.fromarray(Z, mode='L')


def generate_edges():
    """Generate an image with sharp edges (512x512 grayscale)."""
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add various shapes
    # Rectangle
    img[100:200, 100:300] = 255
    
    # Circle
    center_y, center_x = 350, 350
    rr, cc = np.ogrid[:size, :size]
    circle = (rr - center_y)**2 + (cc - center_x)**2 < 80**2
    img[circle] = 255
    
    # Triangle
    for i in range(150):
        img[300+i, 100:100+i] = 128
    
    return Image.fromarray(img, mode='L')


def main():
    # Create data directory
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    print("Generating sample images...")
    
    # Generate and save images
    lena = generate_lena()
    lena.save(data_dir / 'lena.png')
    print(f"✓ Saved {data_dir / 'lena.png'}")
    
    checker = generate_checker()
    checker.save(data_dir / 'checker.png')
    print(f"✓ Saved {data_dir / 'checker.png'}")
    
    gradient = generate_gradient()
    gradient.save(data_dir / 'gradient.png')
    print(f"✓ Saved {data_dir / 'gradient.png'}")
    
    edges = generate_edges()
    edges.save(data_dir / 'edges.png')
    print(f"✓ Saved {data_dir / 'edges.png'}")
    
    print("\nDone! Sample images generated in data/ directory.")


if __name__ == '__main__':
    main()

