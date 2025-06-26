.PHONY: help env install test check clean setup

ENV_NAME = nonlinear-tsw
CONDA_ENV_FILE = environment.yaml

help:
	@echo "🔧 NonLinear TSW - Available Commands:"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  setup         Complete setup: create env + install + test"
	@echo "  env           Create conda environment from environment.yaml"
	@echo "  install       Install package with all dependencies"
	@echo ""
	@echo "🧪 Testing & Development:"
	@echo "  test          Run import tests + test suite"
	@echo "  check         Quick environment and import check"
	@echo ""
	@echo "🛠️  Maintenance:"
	@echo "  clean         Clean build artifacts"
	@echo "  env-update    Update existing conda environment"
	@echo "  env-export    Export current environment to file"
	@echo "  env-remove    Remove conda environment"
	@echo ""

env:
	@echo "🐍 Setting up conda environment: $(ENV_NAME)"
	@echo "Using environment file: $(CONDA_ENV_FILE)"
	@if conda env list | grep -q "^$(ENV_NAME) "; then \
		echo "Environment $(ENV_NAME) already exists. Updating..."; \
		conda env update -n $(ENV_NAME) -f $(CONDA_ENV_FILE) --prune; \
		echo "✅ Environment updated! Activate with: conda activate $(ENV_NAME)"; \
	else \
		echo "Creating new environment..."; \
		conda env create -f $(CONDA_ENV_FILE); \
		echo "✅ Environment created! Activate with: conda activate $(ENV_NAME)"; \
	fi

env-update:
	@echo "🔄 Updating conda environment: $(ENV_NAME)"
	conda env update -n $(ENV_NAME) -f $(CONDA_ENV_FILE) --prune
	@echo "✅ Environment updated successfully!"

env-remove:
	@echo "Removing conda environment: $(ENV_NAME)"
	conda env remove -n $(ENV_NAME)

env-export:
	@echo "💾 Exporting conda environment: $(ENV_NAME)"
	@echo "Saving to: $(CONDA_ENV_FILE)"
	conda env export -n $(ENV_NAME) > $(CONDA_ENV_FILE)
	@echo "✅ Environment exported successfully!"

install:
	@echo "📦 Installing package with all dependencies..."
	pip install -e .
	@echo "✅ Package installed successfully!"

check:
	@echo "🔍 Quick environment check..."
	@echo "Environment: $(ENV_NAME)"
	@conda info --envs | grep $(ENV_NAME) || echo "❌ Environment not found"
	@echo "Testing imports..."
	@python -c "from tsw import TSW; from power_spherical import PowerSpherical; print('✅ All imports successful!')" || echo "❌ Import failed"

test:
	@echo "🧪 Running comprehensive tests..."
	@echo "Running import tests..."
	python -c "from tsw import TSW; from power_spherical import PowerSpherical; print('✅ Import test passed!')"
	@echo "Running test suite..."
	@if [ -d "tests" ]; then \
		python -m pytest tests/ -v; \
	else \
		echo "⚠️  No tests directory found, skipping test suite"; \
	fi
	@echo "✅ All tests completed!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

setup: env install test
	@echo ""
	@echo "🎉 Setup completed successfully!"
	@echo ""
	@echo "To get started:"
	@echo "  conda activate $(ENV_NAME)"
	@echo "  python -c 'from tsw import TSW; print(\"Ready to use TSW!\")'"
	@echo ""
	@echo "Available modules:"
	@echo "  - from tsw import TSW, generate_trees_frames"
	@echo "  - from power_spherical import PowerSpherical, HypersphericalUniform"
	@echo "" 