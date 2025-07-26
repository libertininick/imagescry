"""Database object models."""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float32, jaxtyped
from sqlmodel import Column, Field, LargeBinary, SQLModel, String, TypeDecorator
from torch import Tensor

from imagescry.decomposition import PCA
from imagescry.image.info import ImageInfo
from imagescry.storage.utils import create_lightning_checkpoint
from imagescry.typechecking import typechecker


class PathType(TypeDecorator):
    """Custom SQLAlchemy type for python Path objects."""

    impl = String
    cache_ok = True

    @staticmethod
    def process_bind_param(value: Path | None, _: Any) -> str | None:
        """Convert Path to string when storing."""
        if value is not None:
            return str(value)
        return value

    @staticmethod
    def process_result_value(value: str | None, _: Any) -> Path | None:
        """Convert string back to Path when loading."""
        if value is not None:
            return Path(value)
        return value


class Embedding(SQLModel, table=True):
    """SQLModel for storing an image embedding record in the embeddings table.

    Attributes:
        id (int | None): Primary key, auto-incremented.
        md5_hash (str): Unique MD5 hash of the image file. Indexed for fast lookup.
        filepath (Path): Unique file path of the image. Indexed for fast lookup.
        image_height (int): Height of the original image in pixels. Must be greater than 0.
        image_width (int): Width of the original image in pixels. Must be greater than 0.
        embedding_dim (int): Number of channels in the embedding. Must be greater than 0.
        embedding_height (int): Height of the embedding in pixels. Must be greater than 0.
        embedding_width (int): Width of the embedding in pixels. Must be greater than 0.
        embedding_data (bytes): Binary data of the embedding stored as a byte array.
    """

    __tablename__: str = "embeddings"  # Manually set the table name

    id: int | None = Field(default=None, primary_key=True)
    md5_hash: str = Field(unique=True, index=True)
    filepath: Path = Field(sa_column=Column(PathType, unique=True, index=True))
    image_height: int = Field(gt=0)
    image_width: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    embedding_height: int = Field(gt=0)
    embedding_width: int = Field(gt=0)
    embedding_data: bytes = Field(sa_column=Column(LargeBinary), repr=False)

    @property
    @jaxtyped(typechecker=typechecker)
    def embedding_tensor(self) -> Float32[Tensor, "C H W"]:
        """Float32[Tensor, 'C H W']: Get the embedding as a PyTorch tensor."""
        return torch.from_numpy(
            np.frombuffer(self.embedding_data, dtype=np.float32)
            .reshape(self.embedding_dim, self.embedding_height, self.embedding_width)
            .copy()
        )

    @classmethod
    @jaxtyped(typechecker=typechecker)
    def create(
        cls,
        image_info: ImageInfo,
        embedding_tensor: Float32[Tensor, "C H W"],
    ) -> "Embedding":
        """Create an Embedding instance from image information and embedding tensor.

        Args:
            image_info (ImageInfo): Information about the image.
            embedding_tensor (Float32[Tensor, 'C H W']): PyTorch tensor representing the embedding.

        Returns:
            Embedding: An instance of the Embedding class.
        """
        # Unpack embedding tensor shape
        embedding_dim, embedding_height, embedding_width = embedding_tensor.shape

        return cls(
            md5_hash=image_info.md5_hash,
            filepath=image_info.source,
            image_height=image_info.shape.height,
            image_width=image_info.shape.width,
            embedding_dim=embedding_dim,
            embedding_height=embedding_height,
            embedding_width=embedding_width,
            embedding_data=embedding_tensor.numpy().tobytes(),
        )


class PCACheckpoint(SQLModel, table=True):
    """SQLModel for storing a PCA model checkpoint in the pca_checkpoints table.

    Attributes:
        id (int | None): Primary key, auto-incremented.
        timestamp (str): Timestamp of when the checkpoint was created, indexed for fast lookup.
        num_features (int): Number of input features, must be greater than 0.
        num_components (int): Number of PCA components, must be greater than 0.
        explained_variance (float): Explained variance ratio of the PCA components, must be between 0.0 and 1.0.
        checkpoint (bytes): Binary data of the PCA model checkpoint stored as a byte array.
    """

    __tablename__: str = "pca_checkpoints"  # Manually set the table name

    id: int | None = Field(default=None, primary_key=True)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), index=True)
    num_features: int = Field(gt=0, description="Number of input features")
    num_components: int = Field(gt=0, description="Number of PCA components")
    explained_variance: float = Field(ge=0.0, le=1.0, description="Explained variance ratio of the PCA components")
    checkpoint: bytes = Field(sa_column=Column(LargeBinary), repr=False)

    def load_from_checkpoint(self) -> PCA:
        """Load the PCA model from the checkpoint data.

        Returns:
            PCA: The PCA model loaded from the checkpoint.
        """
        # Load the checkpoint data into a BytesIO stream
        checkpoint_stream = BytesIO(self.checkpoint)

        # Load the PCA model from the checkpoint stream
        pca_model = PCA.load_from_checkpoint(checkpoint_path=checkpoint_stream)

        return pca_model

    @classmethod
    def create(
        cls,
        pca_model: PCA,
    ) -> "PCACheckpoint":
        """Create a PCACheckpoint instance from a PCA model.

        Args:
            pca_model (PCA): The PCA model to create a checkpoint for.

        Returns:
            PCACheckpoint: An instance of the PCACheckpoint class.
        """
        # Create a Lightning checkpoint from the PCA model
        checkpoint_data = create_lightning_checkpoint(pca_model)

        return cls(
            num_features=pca_model.num_features,
            num_components=pca_model.num_components,
            explained_variance=pca_model.explained_variance.item(),
            checkpoint=checkpoint_data,
        )
