"""Matrix decomposition algorithms."""

import torch
from jaxtyping import Float, jaxtyped
from lightning import LightningModule
from torch import Tensor, nn

from imagescry.typechecking import typechecker


class PCA(LightningModule):
    """Principal component analysis (PCA).

    Linear projection of the data to a lower dimensional space using Singular Value Decomposition.
    """

    def __init__(
        self, *, min_num_components: int = 1, max_num_components: int | None = None, min_explained_variance: float = 0.0
    ) -> None:
        """Initialize the PCA model.

        Args:
            min_num_components (int, optional): The minimum number of principal components to keep. Defaults to 1.
            max_num_components (int | None, optional): The maximum number of principal components to keep. If not
                provided, the number of principal components will be determined by the minimum number of components that
                explain at least `min_explained_variance` of the total variance. Defaults to None.
            min_explained_variance (float, optional): The minimum percentage of variance that the PCA should explain.
                If `max_num_components` is not provided, this will be used to determine the maximum number of
                components. Defaults to 0.0.

        Raises:
            ValueError: If the number of components is less than 1, or if the minimum number of components is greater
                than the maximum number of components, or if the minimum explained variance is not between 0.0 and 1.0.
        """
        super().__init__()

        # Validate arguments
        if min_num_components < 1:
            raise ValueError(f"min_num_components must be at least 1, got {min_num_components}")
        if max_num_components is not None and max_num_components < min_num_components:
            raise ValueError(f"max_num_components must be at least {min_num_components}, got {max_num_components}")
        if min_explained_variance < 0.0 or min_explained_variance > 1.0:
            raise ValueError(f"min_explained_variance must be between 0.0 and 1.0, got {min_explained_variance}")

        # PCA parameters
        self.min_num_components = min_num_components
        self.max_num_components = max_num_components
        self.min_explained_variance = min_explained_variance
        self.save_hyperparameters({
            "min_num_components": min_num_components,
            "max_num_components": max_num_components,
            "min_explained_variance": min_explained_variance,
        })

        # Define parameters to fit
        self._fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self._num_features = nn.Parameter(torch.tensor(0), requires_grad=False)
        self._num_components = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.feature_means: nn.Parameter
        self.explained_variance: nn.Parameter
        self.component_vectors: nn.Parameter

    def __repr__(self) -> str:
        """str: Representation of the PCA model."""
        num_features = self.num_features if self.fitted else "not fitted"
        num_components = self.num_components if self.fitted else "not fitted"
        return f"{self.__class__.__name__}(num_features={num_features}, num_components={num_components})"

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Tensor, "num_samples {self.num_features}"]
    ) -> Float[Tensor, "num_samples {self.num_components}"]:
        """Project the input data to a lower dimensional space.

        Args:
            x (Float[Tensor, 'num_samples {self.num_features}']): Input data.

        Returns:
            Float[Tensor, 'num_samples {self.num_components}']: Projected data.
        """
        return torch.matmul(x - self.feature_means, self.component_vectors)

    @jaxtyped(typechecker=typechecker)
    def fit(self, x: Float[Tensor, "num_samples num_features"]) -> "PCA":
        """Fit the PCA model to the input data.

        The input data is centered but not scaled for each feature before applying the SVD.
        See [torch.pca_lowrank](https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html) for more details.

        Args:
            x (Float[Tensor, 'num_samples num_features']): Input data with shape (num_samples, num_features).

        Returns:
            PCA: The fitted PCA model.

        Raises:
            ValueError: If the number of samples is less than 2.
        """
        # Validate input data
        num_samples, num_features = x.shape
        if num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {num_samples}")  # pragma: no cover

        # Calculate the mean of the input data
        self._num_features = nn.Parameter(torch.tensor(num_features), requires_grad=False)
        self.feature_means = nn.Parameter(x.mean(dim=0, keepdim=True), requires_grad=False)

        # Center the input data
        x_centered = x - self.feature_means

        # Perform singular value decomposition of centered matrix
        _, s, vt = torch.linalg.svd(x_centered)

        # Calculate explained variance ratio
        eigenvalues = s**2 / (num_samples - 1)
        total_variance = torch.sum(eigenvalues)
        self.explained_variance = nn.Parameter(eigenvalues / total_variance, requires_grad=False)
        cumulative_explained_variance = torch.cumsum(self.explained_variance, dim=0)

        # Determine the number of components to keep
        num_components_to_meet_min_explained_variance = int(
            torch.sum(cumulative_explained_variance < self.min_explained_variance).item() + 1
        )
        num_components = max(self.min_num_components, num_components_to_meet_min_explained_variance)
        if self.max_num_components is not None:
            num_components = min(self.max_num_components, num_components)
        self._num_components = nn.Parameter(torch.tensor(num_components), requires_grad=False)

        # Select components
        self.component_vectors = nn.Parameter(vt[:num_components, :].T, requires_grad=False)

        # Set fitted flag
        self._fitted = nn.Parameter(torch.tensor(True), requires_grad=False)

        return self

    @jaxtyped(typechecker=typechecker)
    def transform(self, x: Float[Tensor, "num_samples num_features"]) -> Float[Tensor, "num_samples num_components"]:
        """Project the input data to a lower dimensional space.

        Args:
            x (Float[Tensor, 'num_samples num_features']): Input data.

        Returns:
            Float[Tensor, 'num_samples num_components']: Projected data.

        Raises:
            RuntimeError: If the PCA model is not fitted.
        """
        if not self.fitted:
            raise RuntimeError("PCA model not fitted")  # pragma: no cover
        return self(x)

    @property
    def fitted(self) -> bool:
        """bool: Whether the PCA model has been fitted."""
        return bool(self._fitted.item())

    @property
    def num_features(self) -> int:
        """int: Number of features in the input data."""
        return int(self._num_features.item())

    @property
    def num_components(self) -> int:
        """int: Number of principal components."""
        return int(self._num_components.item())
