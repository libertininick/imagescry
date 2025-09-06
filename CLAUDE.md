# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Essential Commands
- `make init` - Initialize the development workspace (installs uv, syncs dependencies, sets up pre-commit hooks)
- `make check` - Run all quality checks (formatting, docstrings, tests with coverage, type checking, vulnerability scanning)
- `make test` - Run tests with current Python version
- `make test-cov` - Run tests with coverage report
- `make format` - Lint and format code with ruff
- `make type-check` - Run type checking with mypy and ty

### Testing Commands
- `uv run pytest` - Run all tests
- `uv run pytest --cov` - Run tests with coverage
- `uv run pytest tests/test_image/` - Run tests for specific module
- `uv run pytest tests/test_image/test_info.py` - Run tests for specific file
- `uv run pytest -k "test_name"` - Run specific test by name

### Environment Management
- `make sync` - Sync dependencies and prune uv cache
- `make update` - Update all dependencies and pre-commit hooks

## Project Architecture

**ImageScry** is a Python tool for using embeddings to intelligently analyze and organize unlabeled imagery. The project uses a modular architecture with clear separation of concerns:

### Core Modules

- **`src/imagescry/`** - Main package root
- **`src/imagescry/image/`** - Image processing and I/O operations
  - `io.py` - Image reading/writing with encoding utilities
  - `info.py` - Image metadata and shape information
  - `transforms.py` - Image transformation operations
- **`src/imagescry/models/`** - Machine learning models and pipelines
  - `embedding.py` - Image embedding generation
  - `decomposition.py` - Dimensionality reduction techniques
  - `pipelines.py` - ML pipeline orchestration
- **`src/imagescry/storage/`** - Data persistence layer
  - `database.py` - Database operations
  - `models.py` - Data models for storage
  - `utils.py` - Storage utilities
- **`src/imagescry/app/`** - Web application interface
  - `app.py` - Dash-based web interface for image annotation
  - `app_utils.py` - Utility functions for the web app

### Key Technologies

- **uv** for dependency management and virtual environments
- **Dash/Plotly** for the interactive web interface
- **PyTorch/Lightning** for machine learning models
- **Rasterio/Shapely** for geospatial image processing
- **SQLModel/Pydantic** for data modeling
- **Ruff** for linting and formatting
- **MyPy + ty** for type checking


## Development Conventions
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Use consistent naming conventions, file structure, and architecture patterns** 
- Prioritize clean, simplistic, and readable code
- Use composition over inheritance
- Fail fast with clear error messages
- **Don't** over engineer code
- Follow the DRY principle, but do not create strong couplings to avoid repetition
- All functions and classes must have descriptive and consistent names
- Use docstrings to explain the why not the how

### Script Development
- Use `uv add --script <script> <deps>` for single-file scripts
- Use typer for CLI interfaces
- Shebang: `#!/usr/bin/env -S uv run --script`
- Make executable with `chmod u+x script.py`


## Code Quality Requirements
- **Follow PEP8**, use type hints, and format with `ruff`.
- All public functions/classes/modules must have Google-style docstrings
- **Use `pydantic` for data validation**.
- Type annotations required on all functions and classes
- For tensor functions use `@jaxtyped(typechecker=typechecker)` decorator to check tensor shapes and datatypes

  For example :

  ```python
  from jaxtyping import Float, jaxtyped
  from torch import Tensor
  from imagescry.typechecking import typechecker

  @jaxtyped(typechecker=typechecker)
  def sum_columns(x: Float[Tensor, "N 3"]) -> Float["N"]:
    """Sum the columns of input tensor.

    Args:
        x (Float[Tensor, 'N 3']): Input tensor to sum columns.

    Returns:
        Float[Tensor, 'N']: Summed columns.
    """
    return x.sum(dim=1)
  ```

- Async functions should have `async_` prefix

## 🧪 Testing Guidelines
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main project structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case
- Pytest with doctest integration enabled
- Supports Python 3.12+ (test with `make test-all-python-versions`)


## 📚 Documentation Standards
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.
- Provide example usage in docstrings whenever possible. This examples should be concise, clear and tested by `doctest`


## 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
