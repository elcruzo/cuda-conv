.PHONY: help install test clean data cli streamlit notebook

help:
	@echo "CUDA Convolution Accelerator - Available Commands"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make data       - Generate sample images"
	@echo "  make test       - Run test suite"
	@echo "  make cli        - Run CLI demo"
	@echo "  make streamlit  - Launch Streamlit web UI"
	@echo "  make notebook   - Launch Jupyter notebook server"
	@echo "  make clean      - Remove generated files"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

data:
	python3 scripts/generate_sample_images.py
	@echo "✓ Sample images generated"

test:
	pytest tests/ -v
	@echo "✓ Tests completed"

cli:
	python3 scripts/cli.py --image data/lena.png --kernel gaussian --output result.png --benchmark
	@echo "✓ CLI demo completed - check result.png"

streamlit:
	streamlit run scripts/streamlit_app.py

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f result.png
	@echo "✓ Cleaned up generated files"

