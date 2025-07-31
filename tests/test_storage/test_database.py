"""Tests for database manager."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from pytest_check import check_functions

from imagescry.image.info import ImageInfo, ImageShape
from imagescry.storage.database import DatabaseManager
from imagescry.storage.models import Embedding
from tests.test_storage.test_models import assert_embeddings_equal


# Fixtures
@pytest.fixture(scope="function")
def embedding1() -> Embedding:
    """Fixture to create a sample Embedding instance."""
    # Create a sample ImageInfo
    image_info = ImageInfo(
        source=Path("/path/to/image-1.jpg"),
        shape=ImageShape(width=800, height=800),
        md5_hash="1",
    )

    # Create a sample embedding tensor
    embedding_tensor = torch.randn(128, 20, 20)

    # Create an Embedding instance from the ImageInfo and tensor
    return Embedding.create(image_info=image_info, embedding_tensor=embedding_tensor, checkpoint_id=1)


@pytest.fixture(scope="function")
def embedding2() -> Embedding:
    """Fixture to create a sample Embedding instance."""
    # Create a sample ImageInfo
    image_info = ImageInfo(
        source=Path("/path/to/image-2.jpg"),
        shape=ImageShape(width=800, height=600),
        md5_hash="2",
    )

    # Create a sample embedding tensor
    embedding_tensor = torch.randn(128, 20, 15)

    # Create an Embedding instance from the ImageInfo and tensor
    return Embedding.create(image_info=image_info, embedding_tensor=embedding_tensor, checkpoint_id=1)


@pytest.fixture(scope="function")
def embedding3() -> Embedding:
    """Fixture to create a sample Embedding instance."""
    # Create a sample ImageInfo
    image_info = ImageInfo(
        source=Path("/path/to/image-3.jpg"),
        shape=ImageShape(width=600, height=800),
        md5_hash="3",
    )

    # Create a sample embedding tensor
    embedding_tensor = torch.randn(128, 15, 20)

    # Create an Embedding instance from the ImageInfo and tensor
    return Embedding.create(image_info=image_info, embedding_tensor=embedding_tensor, checkpoint_id=1)


@pytest.fixture(scope="function")
def embeddings(embedding1: Embedding, embedding2: Embedding, embedding3: Embedding) -> list[Embedding]:
    """Fixture to create a list of sample Embedding instances."""
    return [embedding1, embedding2, embedding3]


@pytest.fixture(scope="function")
def db_manager(tmp_path_factory: pytest.TempPathFactory) -> Generator[DatabaseManager, None, None]:
    """Fixture to create a DatabaseManager instance."""
    with DatabaseManager(db_dir=tmp_path_factory.mktemp("db_dir")) as db_manager:
        yield db_manager


# Tests
def test_create_delete_database(tmp_path: Path) -> None:
    """Test creating and deleting a database."""
    # Create the database by initializing DatabaseManager
    db_manager = DatabaseManager(db_dir=tmp_path)
    check_functions.is_true(db_manager.database_exists, "Database file should be created at the specified path.")
    check_functions.is_true(db_manager.is_connected, "Database manager should be connected after creation.")

    # Delete the database
    check_functions.is_true(db_manager.delete_database(), "Database should be deleted successfully.")
    check_functions.is_false(db_manager.is_connected, "Database manager should be disconnected after deletion.")
    check_functions.is_false(db_manager.database_exists, "Database file should not exist after deletion.")
    with pytest.raises(RuntimeError):
        db_manager.get_session()


def test_add_and_get_embedding(db_manager: DatabaseManager, embedding1: Embedding) -> None:
    """Test adding and retrieving an embedding."""
    # Clone the embedding
    embedding1_clone = embedding1.model_copy(deep=True)

    # Add the embedding to the database
    item_id = db_manager.add_item(embedding1)

    # Retrieve the item by ID
    retrieved_item = db_manager.get_item(Embedding, item_id)
    if retrieved_item is None:
        pytest.fail("Retrieved item is None, expected an Embedding instance.")

    # Check that the retrieved item matches the original
    assert_embeddings_equal(retrieved_item, embedding1)

    # Test trying to add the same item again raises an error because of unique hash/filepath constraints
    with pytest.raises(RuntimeError):
        db_manager.add_item(embedding1_clone)


def test_add_and_get_multiple_embeddings(db_manager: DatabaseManager, embeddings: list[Embedding]) -> None:
    """Test adding and retrieving multiple embeddings."""
    # Add multiple embeddings to the database
    item_ids = db_manager.add_items(embeddings)
    check_functions.equal([1, 2, 3], item_ids, "Item IDs do not match expected values")

    # Retrieve the items by IDs
    retrieved_items = db_manager.get_items(Embedding, item_ids)
    assert len(retrieved_items) == len(embeddings), "Number of retrieved items does not match number of added items"

    # Check that the retrieved items match the originals
    for original, retrieved in zip(embeddings, retrieved_items, strict=True):
        assert_embeddings_equal(retrieved, original)


def test_get_all_items(db_manager: DatabaseManager, embeddings: list[Embedding]) -> None:
    """Test retrieving all items from the database."""
    # Add multiple embeddings to the database
    db_manager.add_items(embeddings)

    # Retrieve all items
    retrieved_items = db_manager.get_items(Embedding)
    check_functions.equal(
        len(retrieved_items), len(embeddings), "Number of retrieved items does not match number of added items"
    )

    # Check that the retrieved items match the originals
    for original, retrieved in zip(embeddings, retrieved_items, strict=True):
        assert_embeddings_equal(retrieved, original)


def test_get_item_ids(db_manager: DatabaseManager, embeddings: list[Embedding]) -> None:
    """Test retrieving item IDs from the database."""
    # Add multiple embeddings to the database
    db_manager.add_items(embeddings)

    # Retrieve item IDs
    item_ids = db_manager.get_item_ids(Embedding)

    # Check retrieved IDs match the expected range
    expected_ids = list(range(1, len(embeddings) + 1))
    check_functions.equal(item_ids, expected_ids, "Retrieved IDs does not match expected range")


def test_add_and_get_empty_items(tmp_path: Path) -> None:
    """Test retrieving items when no items exist."""
    db_manager = DatabaseManager(db_dir=tmp_path)

    # Try to retrieve all items when the database is empty
    items = db_manager.get_items(Embedding)
    check_functions.equal(len(items), 0, "Should retrieve an empty list when no items exist")

    # Try to get an item by ID when no items exist
    item = db_manager.get_item(Embedding, 1)
    check_functions.is_none(item, "Should return None when no items exist")

    # Try adding an empty list of items
    item_ids = db_manager.add_items([])
    check_functions.equal(item_ids, [], "Should return an empty list when adding no items")

    # Try to retrieve items with empty list of IDs
    items = db_manager.get_items(Embedding, [])
    check_functions.equal(len(items), 0, "Should retrieve an empty list when no IDs are provided")
