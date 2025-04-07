"""Tests for the embedding module."""

import math

import pytest
import torch

from imagescry.embedding import EfficientNetEmbedder


@pytest.fixture(scope="session")
def efficientnet_embedder() -> EfficientNetEmbedder:
    """Fixture for an EfficientNet embedder."""
    return EfficientNetEmbedder()


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
@pytest.mark.parametrize("height", [35, 64, 128])
@pytest.mark.parametrize("width", [42, 73, 96])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_embedding_predict_step(
    efficientnet_embedder: EfficientNetEmbedder,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> None:
    """Test the embedding predict step across different input shapes and dtypes produces the correct output shape."""
    # Create batch
    if dtype == torch.float32:
        batch = torch.rand(batch_size, 3, height, width)
    else:
        batch = torch.randint(0, 256, (batch_size, 3, height, width))

    # Extract embedding
    embedding = efficientnet_embedder.predict_step(batch)

    # Check embedding shape
    expected_height = math.ceil(height / efficientnet_embedder.downsample_factor)
    expected_width = math.ceil(width / efficientnet_embedder.downsample_factor)
    assert embedding.shape == (batch_size, efficientnet_embedder.embedding_dim, expected_height, expected_width)
