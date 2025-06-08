"""Embedding model for featurizing images."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Literal, cast

import torch
from jaxtyping import Float, Int64, UInt8, jaxtyped
from lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)

from imagescry.decomposition import PCA
from imagescry.image.dataset import ImageBatch
from imagescry.image.transforms import normalize_per_channel, resize
from imagescry.typechecking import typechecker


@jaxtyped(typechecker=typechecker)
@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Batch of image embeddings and their dataset indices.

    Attributes:
        indices (Int64[Tensor, "B"]): Dataset indices of the embeddings.
        embeddings (Float[Tensor, "B E H W"]): Batch of image embeddings, with embedding dimension `E`.
    """

    indices: Int64[Tensor, "B"]
    embeddings: Float[Tensor, "B E H W"]

    def __len__(self) -> int:
        """Return the number of embeddings in the batch."""
        return len(self.indices)

    def __post_init__(self) -> None:
        """Check that both tensors are on the same device."""
        if self.indices.device != self.embeddings.device:
            raise ValueError(
                "Tensors must be on the same device. "
                f"Got indices on {self.indices.device} and embeddings on {self.embeddings.device}"
            )

    def cpu(self) -> "EmbeddingBatch":
        """Move tensors to the CPU.

        Returns:
            EmbeddingBatch: A new EmbeddingBatch with tensors on the CPU.
        """
        return self.to("cpu")

    def get_flat_vectors(self) -> Float[Tensor, "N E"]:
        """Flatten batch and spatial dimensions of embeddings.

        Returns:
            Float[Tensor, "N E"]: Flattened spatial embeddings.
        """
        return self.embeddings.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

    def to(self, device: str | torch.device) -> "EmbeddingBatch":
        """Move tensors to the specified device.

        Args:
            device (str | torch.device): The device to move the tensors to.

        Returns:
            EmbeddingBatch: A new EmbeddingBatch with tensors on the specified device.
        """
        return EmbeddingBatch(indices=self.indices.to(device), embeddings=self.embeddings.to(device))

    @property
    def device(self) -> torch.device:
        """Get the device that the tensors are on."""
        return self.indices.device

    @property
    def embedding_dim(self) -> int:
        """int: Embedding dimension."""
        return self.embeddings.size(1)

    @property
    def spatial_dims(self) -> tuple[int, int]:
        """tuple[int, int]: Spatial dimensions of the embeddings."""
        return self.embeddings.size(2), self.embeddings.size(3)


# Interfaces
class EmbeddingModule(ABC, LightningModule):
    """Embedding module interface.

    This class defines the interface for all embedding models. It inherits from `LightningModule` and implements the
    `predict_step` and `embed_images` methods to standardize the prediction interface for all embedding models.

    Subclasses must implement the `__init__`, `preprocess` and `forward` methods and define an `embedding_dim` property.
    - The `__init__` method should save required initialization parameters using `self.save_hyperparameters()`.
    - The `preprocess` method should preprocess the images to the same format as the model expects.
    - The `forward` method should extract the embedding feature maps from the images.
    - The `predict_step` will call the `preprocess` and `forward` methods in sequence and return an `EmbeddingBatch`.
    """

    @abstractmethod
    def preprocess(self, images: UInt8[Tensor, "B C H1 W1"]) -> Float[Tensor, "B C H2 W2"]:
        """Preprocess images to the same format as the model expects.

        Args:
            images (UInt8[Tensor, "B C H1 W1"]): Batch of images.

        Returns:
            Float[Tensor, "B C H2 W2"]: Preprocessed images.
        """
        ...  # pragma: no cover

    @abstractmethod
    def forward(self, x: Float[Tensor, "B C H1 W1"]) -> Float[Tensor, "B E H2 W2"]:
        """Forward pass."""
        ...  # pragma: no cover

    @jaxtyped(typechecker=typechecker)
    def predict_step(self, batch: ImageBatch) -> EmbeddingBatch:
        """Preprocess images, extract embedding feature maps and L2 normalizes each embedding vector for an ImageBatch.

        Args:
            batch (ImageBatch): Batch of images and their dataset indices.

        Returns:
            EmbeddingBatch: Embedding feature map for the batch of images.
        """
        # Preprocess images
        x = self.preprocess(batch.images)

        # Extract embedding feature map
        x = self.forward(x)

        # L2 normalize embedding
        x = nn.functional.normalize(x, p=2, dim=1)

        return EmbeddingBatch(indices=batch.indices, embeddings=x)

    def embed_images(
        self, dataloader: DataLoader, *, accelerator: str = "auto", devices: list[int] | str | int = "auto"
    ) -> list[EmbeddingBatch]:
        """Run the embedding module on a dataloader in inference mode and return a list of `EmbeddingBatch` objects.

        Args:
            dataloader (DataLoader): Dataloader that yields `ImageBatch` objects.
            accelerator (str): The accelerator to use. Defaults to "auto".
            devices (list[int] | str | int): The devices to use. Defaults to "auto".

        Returns:
            list[EmbeddingBatch]: The embeddings of the images.

        Raises:
            ValueError: If no results were returned from the inference run.
        """
        with TemporaryDirectory() as temp_dir:
            trainer = Trainer(accelerator=accelerator, devices=devices, default_root_dir=temp_dir)
            results = trainer.predict(self, dataloader)
            if results is None:
                raise ValueError("No results were returned from the inference run.")
            return cast(list[EmbeddingBatch], results)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """int: Embedding dimension."""
        ...  # pragma: no cover


class EmbeddingPCAPipeline(LightningModule):
    """Pipeline that embeds images and then transforms the embeddings to lower-dimensional space using PCA."""

    def __init__(self, embedding_model: EmbeddingModule, pca: PCA) -> None:
        """Initialize the pipeline.

        Args:
            embedding_model (EmbeddingModule): Pretrained embedding model to use.
            pca (PCA): Pretrained PCA model to use.

        Raises:
            ValueError: If the PCA model is not fitted.
        """
        super().__init__()
        self.embedding_model = embedding_model
        if not pca.fitted:
            raise ValueError("PCA model must be fitted before it can be used in the pipeline.")
        self.pca = pca

    @jaxtyped(typechecker=typechecker)
    def predict_step(self, batch: ImageBatch) -> EmbeddingBatch:
        """Embed images and then transform the embeddings to lower-dimensional space using PCA.

        Args:
            batch (ImageBatch): Batch of images and their dataset indices.

        Returns:
            EmbeddingBatch: Compressed embedding feature map for the batch of images.
        """
        # Embed images
        batch_size = len(batch)
        full_embeddings = self.embedding_model.predict_step(batch)

        # Transform embedding vectors to lower-dimensional space
        compressed_flat_embeddings = self.pca.transform(full_embeddings.get_flat_vectors())

        # Reshape embeddings to original spatial dimensions
        compressed_embeddings = compressed_flat_embeddings.reshape(
            batch_size, *full_embeddings.spatial_dims, self.pca.num_components
        ).permute(0, 3, 1, 2)

        return EmbeddingBatch(indices=batch.indices, embeddings=compressed_embeddings)


# Concrete implementations
class EfficientNetEmbedder(EmbeddingModule):
    """Embedding model using EfficientNetV2 as the backbone feature extractor."""

    def __init__(
        self, *, backbone_size: Literal["s", "m", "l"] = "s", max_side_length: int = 640, pretrained: bool = False
    ) -> None:
        """Initialize the model.

        Args:
            backbone_size (Literal["s", "m", "l"]): The size of the backbone model to load. Defaults to "s".
            max_side_length (int): The maximum side length of the images to resize to. Defaults to 640.
            pretrained (bool): Whether to load pretrained weights. Defaults to False.

        Raises:
            ValueError: If the model size is invalid.
        """
        super().__init__()

        self._embedding_dim = 1_280

        # Save initialization parameters
        self.save_hyperparameters({"backbone_size": backbone_size, "max_side_length": max_side_length})
        self.backbone_size = backbone_size
        self.max_side_length = max_side_length

        # Set architecture & weights
        if self.backbone_size == "s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            efficientnet_v2 = efficientnet_v2_s
        elif self.backbone_size == "m":
            weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            efficientnet_v2 = efficientnet_v2_m
        elif self.backbone_size == "l":
            weights = EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            efficientnet_v2 = efficientnet_v2_l
        else:
            raise ValueError(f"Invalid model size: {self.backbone_size}")  # pragma: no cover

        # Load EfficientNetV2 model & extract feature layers
        self.feature_layers = efficientnet_v2(weights=weights).features

    @jaxtyped(typechecker=typechecker)
    def preprocess(self, images: UInt8[Tensor, "B C H1 W1"]) -> Float[Tensor, "B C H2 W2"]:
        """Resize and normalize images prior to extracting embedding feature maps.

        Args:
            images (UInt8[Tensor, "B C H1 W1"]): Batch of images.

        Returns:
            Float[Tensor, "B C H2 W2"]: Preprocessed images.
        """
        # Resize images so that the long side is at most `max_side_length`
        h, w = images.shape[-2:]
        if max(h, w) > self.max_side_length:
            images = resize(images, output_size=self.max_side_length, side_ref="long")

        # Normalize pixel values to [-3, 3]
        return normalize_per_channel(images, min_value=-3, max_value=3)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "B C H1 W1"]) -> Float[Tensor, "B E H2 W2"]:
        """Extract embedding feature map.

        Args:
            x (Float[Tensor, 'B C H1 W1']): Image tensor.

        Returns:
            Float[Tensor, 'B E H2 W2']: Image embedding feature map.
        """
        return self.feature_layers.forward(x)

    @property
    def embedding_dim(self) -> int:
        """int: Embedding dimension."""
        return self._embedding_dim
