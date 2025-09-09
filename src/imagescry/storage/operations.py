"""Database operations for ImageScry storage models.

This module provides concrete database operations using DatabaseManager instances and storage model objects.
"""

import torch
from jaxtyping import Float32, jaxtyped
from sqlmodel import select
from torch import Tensor

from imagescry.image.info import ImageInfo
from imagescry.storage.database import DatabaseManager
from imagescry.storage.models import Embedding, Image
from imagescry.typechecking import typechecker


def create_image_records(database_manager: DatabaseManager, image_infos: list[ImageInfo]) -> list[int]:
    """Create image records in the database from ImageInfo objects.

    Args:
        database_manager (DatabaseManager): Database manager instance to use for operations.
        image_infos (list[ImageInfo]): List of ImageInfo objects to create records for.

    Returns:
        list[int]: List of database IDs for the created image records.

    Raises:
        ValueError: If image_infos is empty or contains invalid data.
    """
    if not image_infos:
        raise ValueError("image_infos cannot be empty")

    # Create Image instances from ImageInfo objects
    image_records = [Image.create(image_info, database_manager.db_dir) for image_info in image_infos]

    # Add all image records to the database
    return database_manager.add_items(image_records)


@jaxtyped(typechecker=typechecker)
def create_embedding_records(
    database_manager: DatabaseManager,
    image_ids: list[int],
    embeddings: Float32[Tensor, "N C H W"],
    *,
    checkpoint_id: int | None = None,
) -> list[int]:
    """Create embedding records in the database from image IDs and embedding tensors.

    Args:
        database_manager (DatabaseManager): Database manager instance to use for operations.
        image_ids (list[int]): List of image database IDs that correspond to the embeddings.
        embeddings (Float32[Tensor, 'N C H W']): Tensor containing embeddings for each image.
        checkpoint_id (int | None): Optional checkpoint ID to associate with embeddings.

    Returns:
        list[int]: List of database IDs for the created embedding records.

    Raises:
        ValueError: If image_ids and embeddings lengths don't match or if inputs are empty.
    """
    if not image_ids:
        raise ValueError("image_ids cannot be empty")

    if len(image_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Number of image_ids ({len(image_ids)}) must match first dimension "
            f"of embeddings tensor ({embeddings.shape[0]})"
        )

    # Create Embedding instances
    embedding_records = [
        Embedding.create(image_id, embedding, checkpoint_id=checkpoint_id)
        for image_id, embedding in zip(image_ids, embeddings, strict=True)
    ]

    # Add all embedding records to the database
    return database_manager.add_items(embedding_records)


def get_image_infos_by_id(database_manager: DatabaseManager, image_ids: list[int]) -> list[ImageInfo]:
    """Retrieve ImageInfo objects from the database by image IDs.

    Args:
        database_manager (DatabaseManager): Database manager instance to use for operations.
        image_ids (list[int]): List of image database IDs to retrieve.

    Returns:
        list[ImageInfo]: List of ImageInfo objects corresponding to the requested IDs.
            Order matches the order of input image_ids. Missing IDs are skipped.

    Raises:
        ValueError: If image_ids is empty.
    """
    if not image_ids:
        raise ValueError("image_ids cannot be empty")

    # Get Image records from database
    image_records = database_manager.get_items(Image, image_ids)

    # Create mapping of id to record for efficient lookup to preserve order
    id_to_record = {record.id: record for record in image_records if record.id is not None}

    # Convert to ImageInfo objects preserving query order
    return [id_to_record[image_id].info(database_manager.db_dir) for image_id in image_ids if image_id in id_to_record]


@jaxtyped(typechecker=typechecker)
def get_embeddings_by_image_id(database_manager: DatabaseManager, image_ids: list[int]) -> Float32[Tensor, "N C H W"]:
    """Retrieve embedding tensors from the database by image IDs.

    Args:
        database_manager (DatabaseManager): Database manager instance to use for operations.
        image_ids (list[int]): List of image database IDs to retrieve embeddings for.

    Returns:
        Float32[Tensor, 'N C H W']: Tensor containing embeddings for the requested images.
            Order matches the order of input image_ids. Images without embeddings are skipped.

    Raises:
        ValueError: If image_ids is empty.
        RuntimeError: If no embeddings are found for the provided image IDs.
    """
    if not image_ids:
        raise ValueError("image_ids cannot be empty")

    # Query embeddings by image_id using SQLModel session
    with database_manager.get_session() as session:
        statement = select(Embedding).where(Embedding.image_id.in_(image_ids))  # type: ignore[union-attr]
        embedding_records = session.exec(statement).all()

    if not embedding_records:
        raise RuntimeError(f"No embeddings found for image IDs: {image_ids}")

    # Create mapping from image_id to embedding for ordered retrieval
    id_to_embedding = {record.image_id: record for record in embedding_records}

    # Collect embeddings in order using list comprehension
    embedding_tensors = [
        id_to_embedding[image_id].embedding_tensor for image_id in image_ids if image_id in id_to_embedding
    ]

    # Stack all embedding tensors into a single tensor
    return torch.stack(embedding_tensors, dim=0)
