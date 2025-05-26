"""Tests for the embedding module."""

import math

import pytest
import torch

from imagescry.embedding import EfficientNetEmbedder
from imagescry.image.dataset import ImageBatch


@pytest.fixture(scope="session")
def efficientnet_embedder() -> EfficientNetEmbedder:
    """Fixture for an EfficientNet embedder."""
    return EfficientNetEmbedder()


@pytest.mark.parametrize("height", [35, 64, 128])
@pytest.mark.parametrize("width", [42, 73, 96])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_embedding_predict_step(
    efficientnet_embedder: EfficientNetEmbedder,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """Test the embedding predict step across different input shapes and dtypes produces the correct output shape."""
    # Create batch
    image_batch = ImageBatch(
        indices=torch.arange(batch_size),
        images=torch.randint(0, 256, (batch_size, 3, height, width)).to(torch.uint8),
    )

    # Extract embedding
    embedding_batch = efficientnet_embedder.predict_step(image_batch)

    # Check embedding shape
    expected_height = math.ceil(height / efficientnet_embedder.downsample_factor)
    expected_width = math.ceil(width / efficientnet_embedder.downsample_factor)
    assert embedding_batch.embeddings.shape == (
        batch_size,
        efficientnet_embedder.embedding_dim,
        expected_height,
        expected_width,
    )
