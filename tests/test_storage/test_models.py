"""Tests for database object models."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from pytest_check import check_functions
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from imagescry.decomposition import PCA
from imagescry.image.info import ImageInfo, ImageShape
from imagescry.storage.models import Embedding, PCACheckpoint


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


def test_pca_checkpoint_creation_and_insertion(engine: Engine, pca: PCA) -> None:
    """Test the PCACheckpoint model creation and insertion."""
    # Create a sample PCA checkpoint
    pca_checkpoint = PCACheckpoint.create(pca)

    # Add PCA checkpoint to the database
    with Session(engine) as session:
        session.add(pca_checkpoint)
        session.commit()
        session.refresh(pca_checkpoint)

        # Verify the PCA checkpoint was added and has an ID
        check_functions.is_not_none(pca_checkpoint.id)

    # Get the PCA checkpoint from the database and verify attributes match the original
    with Session(engine) as session:
        statement = select(PCACheckpoint).where(PCACheckpoint.id == pca_checkpoint.id)
        db_pca_checkpoint = session.exec(statement).one()

    check_functions.equal(db_pca_checkpoint.num_features, pca_checkpoint.num_features)
    check_functions.equal(db_pca_checkpoint.num_components, pca_checkpoint.num_components)
    check_functions.almost_equal(db_pca_checkpoint.explained_variance, pca_checkpoint.explained_variance)

    # Load the PCA model from the checkpoint and verify component vectors match expected
    loaded_pca = db_pca_checkpoint.load_from_checkpoint()
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

    check_functions.equal(db_embedding.md5_hash, "test-hash")
    check_functions.equal(db_embedding.filepath, Path("/path/to/image.jpg"))
    check_functions.equal(db_embedding.image_height, 600)
    check_functions.equal(db_embedding.image_width, 800)
    check_functions.equal(db_embedding.embedding_dim, 128)
    check_functions.equal(db_embedding.embedding_height, 20)
    check_functions.equal(db_embedding.embedding_width, 20)
    check_functions.is_true(torch.allclose(db_embedding.embedding_tensor, embedding_tensor))
