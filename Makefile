# Variables
PYTHON = python
VENV = venv
VENV_BIN = $(VENV)/bin
VENV_ACTIVATE = $(VENV_BIN)/activate
SRC_DIR = src
RESULTS_DIR = optimization_results

# Default target
.PHONY: all
all: setup run

# Setup virtual environment and install dependencies
.PHONY: setup
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	. $(VENV_ACTIVATE) && pip install -r requirements.txt
	@echo "Setup complete!"

# Run the optimizer
.PHONY: run
run:
	@echo "Running the optimizer..."
	. $(VENV_ACTIVATE) && $(PYTHON) $(SRC_DIR)/main.py

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(RESULTS_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

# Clean everything including virtual environment
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "All cleanup complete!"

# Help command
.PHONY: help
help:
	@echo "Resource Allocation Optimizer Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup     - Create virtual environment and install dependencies"
	@echo "  make run       - Run the optimizer"
	@echo "  make clean     - Remove generated files and caches"
	@echo "  make clean-all - Remove everything including virtual environment"
	@echo "  make help      - Show this help message"
	@echo "  make all       - Setup and run the optimizer" 