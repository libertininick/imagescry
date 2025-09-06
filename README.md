# imagescry
Use embeddings to intelligently sift through a maze of unlabeled imagery.

## Motivation

## Quick Start

WIP Coming soon ⏳

## Installation

Use the following steps to install `imagescry` from its github repository. 

### Install `uv` and create  a virtual environment

1. Install `uv` if you haven't already:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Create a virtual environment:

    ```bash
    uv venv <my_env_name>
    ```

3. Activate your virtual environment:

    ```bash
    source <my_env_name>/bin/activate
    ```

### Install `imagescry` using `uv pip install`:

```bash
uv pip install git+https://github.com/libertininick/imagescry.git
```

Verify the installation by running:

```bash
uv pip show imagescry
```

## Developer Info

### Makefile commands
This repository uses a [Makefile](Makefile) for managing useful commands used for development.

Available commands:

```plain
Available commands
=============================================================================================================
 - check                    : Runs all checks:
                              - formatting
                              - docstring checks
                              - tests and coverage reports
                              - type checking
                              - dependency vulnerability checks
 - docstring-check          : Run docstring checks
 - format                   : Lint and format code with ruff
 - init                     : Initialize workspace for development:
                              - install & update uv
                              - sync workspace environment
                              - install pre-commit hooks
 - install-uv               : Download and install uv
 - sync                     : Sync workspace environment and prune uv cache
 - test                     : Run all tests using workspace Python version
 - test-cov                 : Run all tests and generate coverage report using workspace Python version 
 - test-all-python-versions : Run all tests over supported Python versions
 - type-check               : Run type checking with mypy
 - update                   : Update uv, all dependencies, and pre-commit hooks in workspace
 - vulnerability-check      : Run dependency vulnerability checks
 ```

### Developer installation

1. Clone the repository:

    ```bash
    git clone git@github.com:libertininick/imagescry.git
    cd imagescry
    ```

2. Install `make` if you haven't already

    ```bash
    sudo apt update && sudo apt install make
    ```

3. Initialize the workspace environment

    ```bash
    make init
    ```

4. Restart your terminal and run all workspace checks:

    ```bash
    make check
    ```