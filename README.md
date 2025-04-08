# imagescry
Use embeddings to intelligently sift through a maze of unlabeled imagery.

## Motivation

## Quick Start

WIP Coming soon ⏳

## Installation

First, clone the repository:

```bash
git clone git@github.com:libertininick/imagescry.git
cd imagescry
```

### User Installation
1. Install `uv` if you haven't already:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Create virtual environment and ensure all project dependencies are installed and up-to-date with the lockfile:

    ```bash
    uv sync --no-dev
    ```

The package is now installed and ready to use. You can verify the installation by running:

  ```bash
  uv pip show imagescry
  ```

### Developer installation

This workspace uses a [Makefile](Makefile) to define a recipe of convience commands. To view a list of available commands run `make help`.

1. Install `make` if you haven't already

    ```bash
    sudo apt update && sudo apt install make
    ```

2. Initialize the workspace environment

    ```bash
    make init
    ```

3. Restart your terminal and run all worspace checks:

    ```bash
    make check
    ```

