## Makefile for managing workspace

# Get directory of this Makefile
MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Set UV_PATH to ~/.local/bin/uv
UV_INSTALL_DIR := $(HOME)/.local/bin
UV_PATH := $(UV_INSTALL_DIR)/uv


## Makefile help
help:
	@echo "Available commands"
	@echo "============================================================================================================="
	@echo " - check                    : Runs all checks:"
	@echo "                              - formatting"
	@echo "                              - docstring checks"
	@echo "                              - tests and coverage reports"
	@echo "                              - type checking"
	@echo "                              - dependency vulnerability checks"
	@echo " - docstring-check          : Run docstring checks"
	@echo " - format                   : Lint and format code with ruff"
	@echo " - init                     : Initialize workspace for development:"
	@echo "                              - install & update uv"
	@echo "                              - sync workspace environment"
	@echo "                              - install pre-commit hooks"
	@echo " - install-uv               : Download and install uv"
	@echo " - test                     : Run all tests using workspace Python version"
	@echo " - test-cov                 : Run all tests and generate coverage report using workspace Python version "
	@echo " - test-all-python-versions : Run all tests over supported Python versions"
	@echo " - type-check               : Run type checking with mypy"
	@echo " - update                   : Update uv, all dependencies, and pre-commit hooks in workspace"
	@echo " - vulnerability-check      : Run dependency vulnerability checks"

## Download and install uv
# 1) Check if uv is already installed
# 2) If uv is not installed, download and install it
install-uv:
	@if [ -e "$(UV_PATH)" ]; then \
		echo "Found existing installation of uv at: $(UV_PATH)"; \
	else \
		echo "Downloading and installing uv."; \
		curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(UV_INSTALL_DIR)" sh; \
	fi

## Update all dependencies in workspace
# 1) Update UV tool
# 2) Prune UV Cache
# 3) Upgrade Python version
# 4) Upgrade all packages listed in uv.lock
# 5) Sync workspace environment
# 6) Install (new) pre-commit hooks
# 7) Update pre-commit hooks
update:
	@uv self update
	@uv cache prune
	@uv uv python upgrade
	@uv lock --upgrade
	@uv sync
	@uv run pre-commit install-hooks
	@uv run pre-commit autoupdate

## Initialize workspace
# 1) Set directory environment variables
# 2) Download UV and install it (if not already installed)
# 3) Update UV
# 4) Sync workspace environment (create virtual environment and install dependencies)
# 5) Install pre-commit hooks
init: install-uv
	@echo "Updating uv"
	@$(UV_PATH) self update
	@echo "Syncing workspace environment"
	@$(UV_PATH) sync
	@echo "Installing pre-commit hooks"
	@$(UV_PATH) run pre-commit install
	@echo "success: Workspace initialized"

## Lint and format code with ruff
# 1) Run ruff linter to check code adhearence to rules and fix when possible
# 2) Run ruff formatting
format:
	@echo "Checking code formatting"
	@uv run ruff check --fix .
	@uv run ruff format .

## Run all tests using workspace Python version
PYTHON_VERSION := $(shell cat .python-version)
test:
	@echo "Running tests using Python $(PYTHON_VERSION)"
	@uv run --python $(PYTHON_VERSION) pytest

## Run all tests and generate coverage report in workspace using workspace Python version 
test-cov:
	@echo "Running tests and generating coverage report using Python $(PYTHON_VERSION)"
	@uv run --python $(PYTHON_VERSION) pytest --cov

## Run all tests in workspace over supported Python versions
test-all-python-versions:
	@echo "Running tests for Python 3.12"
	@uv run --python 3.12 pytest
	@echo "Running tests for Python 3.13"
	@uv run --python 3.13 pytest

## Run type checking with mypy
type-check:
	@echo "Type checking with mypy"
	@uv run mypy src/ tests/
	@echo "Type checking with ty"
	@uv run ty check src/ tests/

## Run docstring checks
docstring-check:
	@echo "Checking docstrings"
	uv run pydoclint src/ tests/

# Run dependency vulnerability checks
vulnerability-check:
	@echo "Checking for dependencies with vulnerabilities"
	@uv run uv-secure uv.lock
	
## Run all checks
# 1) Run ruff linter and formatter
# 2) Run docstring checks
# 3) Run all tests and generate coverage report
# 4) Run type checking
# 5) Run dependency vulnerability checks
check: format docstring-check test-cov type-check vulnerability-check 

