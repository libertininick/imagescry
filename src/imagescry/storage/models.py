"""Database object models."""

from datetime import datetime
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from jaxtyping import Float32, jaxtyped
from lightning import LightningModule
from sqlmodel import Column, Field, LargeBinary
from torch import Tensor

from imagescry.image.info import ImageInfo
from imagescry.storage.base import BaseStorageModel, PathType
from imagescry.storage.utils import create_lightning_checkpoint
from imagescry.typechecking import typechecker


class Embedding(BaseStorageModel, table=True):
    """SQLModel for storing an image embedding record in the embeddings table.

    Attributes:
        checkpoint_id (int | None): Foreign key referencing the PCA model checkpoint used for this embedding.
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

    checkpoint_id: int | None = Field(default=None, foreign_key="checkpoints.id")
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
        cls, image_info: ImageInfo, embedding_tensor: Float32[Tensor, "C H W"], *, checkpoint_id: int | None = None
    ) -> "Embedding":
        """Create an Embedding instance from image information and embedding tensor.

        Args:
            image_info (ImageInfo): Information about the image.
            embedding_tensor (Float32[Tensor, 'C H W']): PyTorch tensor representing the embedding.
            checkpoint_id (int | None): Foreign key referencing the PCA model checkpoint used for this embedding.

        Returns:
            Embedding: An instance of the Embedding class.
        """
        # Unpack embedding tensor shape
        embedding_dim, embedding_height, embedding_width = embedding_tensor.shape

        return cls(
            checkpoint_id=checkpoint_id,
            md5_hash=image_info.md5_hash,
            filepath=image_info.source,
            image_height=image_info.shape.height,
            image_width=image_info.shape.width,
            embedding_dim=embedding_dim,
            embedding_height=embedding_height,
            embedding_width=embedding_width,
            embedding_data=embedding_tensor.numpy().tobytes(),
        )


ModelType = TypeVar("ModelType", bound="LightningModule")


class LightningCheckpoint(BaseStorageModel, table=True):
    """SQLModel for storing a lightning model checkpoint in the checkpoints table.

    Attributes:
        timestamp (str): Timestamp of when the checkpoint was created, indexed for fast lookup.
        model_name (str | None): Name of the model checkpoint.
        model_class (str): Class name of the model checkpoint.
        model_module (str): Module name of the model checkpoint.
        package_version (str): Version of the package containing the model.
        size (int): Size of the checkpoint in bytes, must be greater than 0.
        checkpoint (bytes): Binary data of the lightning model checkpoint stored as a byte array.
    """

    __tablename__: str = "checkpoints"  # Manually set the table name

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), index=True)
    model_name: str | None
    model_class: str
    model_module: str
    package_version: str
    size: int = Field(gt=0, description="Size of the checkpoint in bytes")
    checkpoint: bytes = Field(sa_column=Column(LargeBinary), repr=False)

    def get_model_class(self) -> type[LightningModule]:
        """Get the class of the model checkpoint via dynamic import.

        Returns:
            type[ModelType]: The class of the model checkpoint.

        Raises:
            TypeError: If the model class is not a subclass of LightningModule.
        """
        module = import_module(self.model_module)
        model_class = getattr(module, self.model_class)
        if not issubclass(model_class, LightningModule):
            raise TypeError(f"Model class {self.model_class} in module {self.model_module} is not a LightningModule.")
        return model_class

    def load_from_checkpoint(self, model_class: type[ModelType]) -> ModelType:
        """Load a model instance from the checkpoint data.

        Args:
            model_class (type[ModelType]): The class of the model to load.

        Returns:
            ModelType: The model loaded from the checkpoint.
        """
        # Load the checkpoint data into a BytesIO stream
        checkpoint_stream = BytesIO(self.checkpoint)

        # Load the model from the checkpoint stream
        model = model_class.load_from_checkpoint(checkpoint_path=checkpoint_stream)

        return model

    @classmethod
    def create(cls, model: LightningModule, model_name: str | None = None) -> "LightningCheckpoint":
        """Create a LightningCheckpoint from a model instance.

        Args:
            model (LightningModule): The model to create a checkpoint for.
            model_name (str | None, optional): Name for the model. Defaults to None.

        Returns:
            LightningCheckpoint: An instance of the LightningCheckpoint class.
        """
        # Create a Lightning checkpoint from the model
        checkpoint_data = create_lightning_checkpoint(model)

        return cls(
            model_name=model_name,
            model_class=model.__class__.__name__,
            model_module=model.__module__,
            package_version=version(model.__module__.split(".")[0]),
            size=len(checkpoint_data),
            checkpoint=checkpoint_data,
        )
