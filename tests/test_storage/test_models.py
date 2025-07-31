"""Tests for database object models."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from pytest_check import check_functions
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from imagescry.image.info import ImageInfo, ImageShape
from imagescry.models.decomposition import PCA
from imagescry.storage.models import Embedding, LightningCheckpoint


# Fixtures
@pytest.fixture(scope="session")
def engine() -> Generator[Engine, None, None]:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def pca() -> PCA:
    """Fixture to create a PCA instance for testing."""
    pca = PCA(min_num_components=10, max_num_components=20, min_explained_variance=0.50)
    pca.fit(torch.randn(100, 20))
    return pca


# Tests
def test_pca_checkpoint_creation_and_insertion(engine: Engine, pca: PCA) -> None:
    """Test the LightningCheckpoint model creation and insertion from a PCA model."""
    # Create a sample PCA checkpoint
    pca_checkpoint = LightningCheckpoint.create(pca, model_name="test_pca_checkpoint")

    # Add PCA checkpoint to the database
    with Session(engine) as session:
        session.add(pca_checkpoint)
        session.commit()
        session.refresh(pca_checkpoint)

        # Verify the PCA checkpoint was added and has an ID
        check_functions.is_not_none(pca_checkpoint.id)

    # Get the PCA checkpoint from the database and verify attributes match the original
    with Session(engine) as session:
        statement = select(LightningCheckpoint).where(LightningCheckpoint.id == pca_checkpoint.id)
        db_pca_checkpoint = session.exec(statement).one()

    # Test model class retrieval
    check_functions.equal(db_pca_checkpoint.get_model_class(), PCA)

    # Load the PCA model from the checkpoint and verify component vectors match expected
    loaded_pca = db_pca_checkpoint.load_from_checkpoint(PCA)
    check_functions.is_true(torch.allclose(loaded_pca.component_vectors, pca.component_vectors))


def test_embedding_model_creation_and_insertion(engine: Engine) -> None:
    """Test the Embedding model creation and insertion."""
    # Create a sample ImageInfo
    image_info = ImageInfo(
        source=Path("/path/to/image.jpg"),
        shape=ImageShape(width=800, height=600),
        md5_hash="test-hash",
    )

    # Create a sample embedding tensor
    embedding_tensor = torch.randn(128, 20, 20)  # Example tensor with shape (C, H, W)

    # Create an Embedding instance from the ImageInfo and tensor
    embedding = Embedding.create(image_info=image_info, embedding_tensor=embedding_tensor)

    # Add embedding to the database
    with Session(engine) as session:
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

        # Verify the embedding was added and has an ID
        check_functions.is_not_none(embedding.id)

    # Get the embedding from the database and verify attributes match the original
    with Session(engine) as session:
        statement = select(Embedding).where(Embedding.md5_hash == "test-hash")
        db_embedding = session.exec(statement).one()

    assert_embeddings_equal(embedding, db_embedding)


# Helpers
def assert_embeddings_equal(embedding1: Embedding, embedding2: Embedding) -> None:
    """Assert that two Embedding instances are equal."""
    check_functions.equal(embedding1.id, embedding2.id)
    check_functions.equal(embedding1.checkpoint_id, embedding2.checkpoint_id)
    check_functions.equal(embedding1.md5_hash, embedding2.md5_hash)
    check_functions.equal(embedding1.filepath, embedding2.filepath)
    check_functions.equal(embedding1.image_height, embedding2.image_height)
    check_functions.equal(embedding1.image_width, embedding2.image_width)
    check_functions.equal(embedding1.embedding_dim, embedding2.embedding_dim)
    check_functions.equal(embedding1.embedding_height, embedding2.embedding_height)
    check_functions.equal(embedding1.embedding_width, embedding2.embedding_width)
    check_functions.is_true(torch.allclose(embedding1.embedding_tensor, embedding2.embedding_tensor))
