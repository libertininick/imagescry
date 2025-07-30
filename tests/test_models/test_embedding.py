"""Tests for the embedding module."""

import math

import pytest
import torch
from pytest_check import check_functions

from imagescry.image.dataset import ImageBatch
from imagescry.models.embedding import EfficientNetEmbedder, EmbeddingBatch

# Seed
SEED = 1234
torch.manual_seed(SEED)


# Fixtures
@pytest.fixture(scope="session")
def device() -> str:
    """Compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture(scope="session")
def embedding_batch() -> EmbeddingBatch:
    """Fixture for an embedding batch."""
    batch_size = 3
    embedding_dim = 128
    spatial_dims = (7, 10)
    batch = EmbeddingBatch(
        indices=torch.arange(batch_size),
        embeddings=torch.randn(batch_size, embedding_dim, *spatial_dims),
    )

    # Check properties
    check_functions.equal(len(batch), batch_size)
    check_functions.equal(batch.embedding_dim, embedding_dim)
    check_functions.equal(batch.spatial_dims, spatial_dims)

    return batch


@pytest.fixture(scope="session")
def efficientnet_embedder(device: str) -> EfficientNetEmbedder:
    """Fixture for an EfficientNet embedder."""
    model = EfficientNetEmbedder().to(device)
    return model


# Tests
def test_embedding_batch_get_flat_vectors(embedding_batch: EmbeddingBatch) -> None:
    """Test the get_flat_vectors method of an embedding batch."""
    # Get flat vectors
    flat_vectors = embedding_batch.get_flat_vectors()

    # Check shape
    check_functions.equal(
        (
            len(embedding_batch) * embedding_batch.spatial_dims[0] * embedding_batch.spatial_dims[1],
            embedding_batch.embedding_dim,
        ),
        flat_vectors.shape,
    )

    # Check values
    check_functions.is_true(
        torch.allclose(
            flat_vectors, embedding_batch.embeddings.permute(0, 2, 3, 1).reshape(-1, embedding_batch.embedding_dim)
        )
    )


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
    ).to(efficientnet_embedder.device)

    # Extract embedding
    embedding_batch = efficientnet_embedder.predict_step(image_batch)

    # Check embedding shape
    downsample_factor = 32
    expected_height = math.ceil(height / downsample_factor)
    expected_width = math.ceil(width / downsample_factor)
    assert embedding_batch.embeddings.shape == (
        batch_size,
        efficientnet_embedder.embedding_dim,
        expected_height,
        expected_width,
    )
