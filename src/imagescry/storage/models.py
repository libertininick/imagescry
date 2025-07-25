"""Database object models."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float32, jaxtyped
from sqlmodel import Column, Field, LargeBinary, SQLModel, String, TypeDecorator
from torch import Tensor

from imagescry.image.info import ImageInfo
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
