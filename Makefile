.PHONY: setup test lint explore train clean

PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

ifeq ($(strip $(PYTHON)),)
	$(error No python interpreter found. Please install Python 3: https://www.python.org/downloads/)
endif

setup:
	@$(PYTHON) -c "import sys; sys.exit('Python 3.10 or higher is required. Please upgrade your Python version.') if sys.version_info < (3,10) else None"
	@echo "Using $(PYTHON) version $(shell $(PYTHON) --version )"
	
	$(PYTHON) -m venv .venv && \
	.venv/bin/pip install --upgrade pip && \
	.venv/bin/pip install -r requirements.txt && \
	.venv/bin/pip install -e .

test:
	.venv/bin/python -m pytest tests/ -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

explore:
	.venv/bin/python scripts/00_explore_data.py

train:
	.venv/bin/python scripts/01_train.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + && \
	rm -rf .pytest_cache .ruff_cache && \
	rm -rf outputs/models/* outputs/figures/*
