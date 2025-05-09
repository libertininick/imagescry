"""Embedding model for featurizing images."""

from abc import ABC, abstractmethod
from typing import Literal

from jaxtyping import Float, Num, jaxtyped
from lightning import LightningModule
from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)

from imagescry.image import normalize_per_channel
from imagescry.typechecking import typechecker


class AbstractEmbeddingModel(ABC, LightningModule):
    """Abstract embedding model.

    This class defines the interface for all embedding models. It inherits from `LightningModule` and implements the
    `predict_step` method to standardize the prediction interface for all embedding models.

    Subclasses must implement the `__init__` and `forward` methods and define the `embedding_dim` and
    `downsample_factor` properties. The `__init__` method should save required initialization parameters using
    `self.save_hyperparameters()`. The `forward` method should handle any preprocessing steps required for the model.
    """

    @abstractmethod
    def forward(self, x: Num[Tensor, "B C H1 W1"]) -> Float[Tensor, "B E H2 W2"]:
        """Forward pass."""
        ...  # pragma: no cover

    @jaxtyped(typechecker=typechecker)
    def predict_step(self, batch: Num[Tensor, "B C H1 W1"]) -> Float[Tensor, "B E H2 W2"]:
        """Extract embedding feature maps and L2 normalizes each embedding vector for a batch of images.

        Args:
            batch (Num[Tensor, 'B C H1 W1']): Batch of images.

        Returns:
            Float[Tensor, 'B E H2 W2']: L2 normalized embedding feature map for the batch of images.
        """
        # Extract embedding feature map
        x = self.forward(batch)

        # L2 normalize embedding
        x = nn.functional.normalize(x, p=2, dim=1)

        return x

    @property
    @abstractmethod
    def downsample_factor(self) -> int:
        """int: Spatial downsample factor."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """int: Embedding dimension."""
        ...  # pragma: no cover


class EfficientNetEmbedder(AbstractEmbeddingModel):
    """Embedding model using EfficientNetV2 as the backbone feature extractor."""

    def __init__(self, *, backbone_size: Literal["s", "m", "l"] = "s", pretrained: bool = False) -> None:
        """Initialize the model.

        Args:
            backbone_size (Literal["s", "m", "l"]): The size of the backbone model to load. Defaults to "s".
            pretrained (bool): Whether to load pretrained weights. Defaults to False.

        Raises:
            ValueError: If the model size is invalid.
        """
        super().__init__()

        self._embedding_dim = 1_280
        self._downsample_factor = 32

        # Save initialization parameters
        self.save_hyperparameters({"backbone_size": backbone_size})
        self.backbone_size = backbone_size

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
    def forward(self, x: Num[Tensor, "B C H1 W1"]) -> Float[Tensor, "B E H2 W2"]:
        """Extract embedding feature map.

        Args:
            x (Num[Tensor, 'B C H1 W1']): Image tensor.

        Returns:
            Float[Tensor, 'B E H2 W2']: Image embedding feature map.
        """
        # Normalize pixel values to [-3, 3]
        x = normalize_per_channel(x, min_value=-3, max_value=3)

        # Extract features
        x = self.feature_layers.forward(x)

        return x

    @property
    def downsample_factor(self) -> int:
        """int: Spatial downsample factor."""
        return self._downsample_factor

    @property
    def embedding_dim(self) -> int:
        """int: Embedding dimension."""
        return self._embedding_dim
