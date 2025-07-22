"""Tests for database object models."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from pytest_check import check_functions
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from imagescry.image.info import ImageInfo, ImageShape
from imagescry.storage.models import Embedding


@pytest.fixture(scope="session")
def engine() -> Generator[Engine, None, None]:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


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
