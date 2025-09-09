# Makefile for Algorithmic Trading Engine

.PHONY: help install test clean run lint format docs build-cpp

# Default target
help:
	@echo "Algorithmic Trading Engine - Available Commands:"
	@echo ""
	@echo "  install    - Install the package and dependencies"
	@echo "  test       - Run the test suite"
	@echo "  run        - Run the trading engine"
	@echo "  run-paper  - Run in paper trading mode"
	@echo "  run-live   - Run in live trading mode"
	@echo "  run-opt    - Run in optimization mode"
	@echo "  lint       - Run code linting and style checks"
	@echo "  format     - Format code using black and isort"
	@echo "  clean      - Clean build artifacts and cache"
	@echo "  docs       - Build documentation"
	@echo "  package    - Create distribution package"
	@echo "  build-cpp  - Build C++ extensions"
	@echo ""

# Install the package
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installing package in development mode..."
	pip install -e .
	@echo "Installation complete!"

# Build C++ extensions
build-cpp:
	@echo "Building C++ extensions..."
	cd src/cpp && python -c "import pybind11; print('pybind11 found')" || pip install pybind11
	cd src/cpp && python setup.py build_ext --inplace
	@echo "C++ extensions built!"

# Run tests
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v --cov=src --cov-report=html
	@echo "Test coverage report generated in htmlcov/"

# Run the trading engine (default: paper trading)
run:
	@echo "Running trading engine in paper trading mode..."
	python main.py --paper

# Run in paper trading mode
run-paper:
	@echo "Running trading engine in paper trading mode..."
	python main.py --paper

# Run in live trading mode
run-live:
	@echo "Running trading engine in live trading mode..."
	@echo "WARNING: This will use real money! Make sure you have proper risk management in place."
	python main.py --live

# Run in optimization mode
run-opt:
	@echo "Running trading engine in optimization mode..."
	python main.py --optimize

# Run basic example
example:
	@echo "Running basic trading example..."
	python examples/basic_pricing_example.py

# Lint code
lint:
	@echo "Running code linting..."
	flake8 src/ tests/ examples/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/ examples/ --line-length=100
	isort src/ tests/ examples/
	@echo "Code formatting complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf logs/
	rm -rf data/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.pyd" -delete
	@echo "Cleanup complete!"

# Build documentation
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

# Create distribution package
package:
	@echo "Creating distribution package..."
	python setup.py sdist bdist_wheel
	@echo "Package created in dist/"

# Install development dependencies
install-dev: install
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "Development dependencies installed!"

# Install trading dependencies
install-trading: install
	@echo "Installing trading dependencies..."
	pip install -e ".[trading]"
	@echo "Trading dependencies installed!"

# Run all checks
check: lint test
	@echo "All checks passed!"

# Quick start (install and run in paper mode)
quickstart: install run-paper
	@echo "Quick start completed!"

# Development setup
dev-setup: install-dev format lint test
	@echo "Development environment setup complete!"

# Full setup with C++ extensions
full-setup: install build-cpp
	@echo "Full setup with C++ extensions complete!"

# Create necessary directories
setup-dirs:
	@echo "Creating necessary directories..."
	mkdir -p logs data tests backtest_results strategy_results
	@echo "Directories created!"
