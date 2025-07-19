"""Tests for the decomposition module."""

import pytest
import torch
from jaxtyping import Float
from pytest_check import check
from torch import Tensor
from torch.distributions import MultivariateNormal

from imagescry.decomposition import PCA

# Fixtures
num_samples = 1_000
feature_locs = torch.tensor([0.0, 1.0, -1.0, 0.0])
num_features = feature_locs.shape[0]


@pytest.fixture(scope="module")
def uncorrelated_features() -> Float[Tensor, f"{num_samples} {num_features}"]:
    """Fixture generating uncorrelated features."""
    torch.manual_seed(1234)
    dist = MultivariateNormal(loc=feature_locs, covariance_matrix=torch.eye(num_features))
    return dist.sample((num_samples,))


@pytest.fixture(scope="module")
def correlated_features() -> Float[Tensor, f"{num_samples} {num_features}"]:
    """Fixture generating correlated features."""
    torch.manual_seed(1234)
    dist = MultivariateNormal(
        loc=feature_locs,
        covariance_matrix=torch.tensor([
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.5],
            [0.0, 0.0, -0.5, 1.0],
        ]),
    )
    return dist.sample((num_samples,))


@pytest.mark.parametrize("min_explained_variance, expected_num_components", [(0.2, 1), (0.4, 2), (0.6, 3), (1.0, 4)])
def test_pca_uncorrelated_features(
    uncorrelated_features: Float[Tensor, f"{num_samples} {num_features}"],
    min_explained_variance: float,
    expected_num_components: int,
) -> None:
    """Test principal component decomposition on uncorrelated features."""
    # Initialize PCA model
    pca = PCA(min_explained_variance=min_explained_variance)

    # Check that the PCA object is not fitted before fitting
    check.is_false(pca.fitted)

    # Fit PCA model
    pca.fit(uncorrelated_features)

    # Check that the PCA object is fitted after fitting
    check.is_true(pca.fitted)

    # Check length of feature means and explained variances
    check.equal(pca.feature_means.data.size(1), num_features)
    check.equal(len(pca.explained_variance.data), num_features)

    # Project features
    projected_features = pca.transform(uncorrelated_features)

    # Check that the number of projected components is as expected
    check.equal((num_samples, expected_num_components), projected_features.shape)

    # Check explained variance is as expected
    check.greater_equal(
        pca.explained_variance.data[:expected_num_components].sum().item(),
        min_explained_variance - 1e-6,  # account for numerical precision
    )

    # Check correlation of projected features are close to zero
    if expected_num_components > 1:
        atol = 1e-4
        corr_matrix_abs = torch.abs(torch.corrcoef(projected_features.T))
        check.is_true(torch.all(torch.tril(corr_matrix_abs, diagonal=-1) <= atol))


@pytest.mark.parametrize(
    "min_explained_variance, expected_num_components", [(0.2, 1), (0.4, 2), (0.6, 2), (0.8, 3), (1.0, 4)]
)
def test_pca_correlated_features(
    correlated_features: Float[Tensor, f"{num_samples} {num_features}"],
    min_explained_variance: float,
    expected_num_components: int,
) -> None:
    """Test principal component decomposition on correlated features."""
    # Initialize PCA model
    pca = PCA(min_explained_variance=min_explained_variance)

    # Check that the PCA object is not fitted before fitting
    check.is_false(pca.fitted)

    # Fit PCA model
    pca.fit(correlated_features)

    # Check that the PCA object is fitted after fitting
    check.is_true(pca.fitted)

    # Check length of feature means and explained variances
    check.equal(pca.feature_means.data.size(1), num_features)
    check.equal(len(pca.explained_variance.data), num_features)

    # Project features
    projected_features = pca.transform(correlated_features)

    # Check that the number of projected components is as expected
    check.equal((num_samples, expected_num_components), projected_features.shape)

    # Check explained variance is as expected
    check.greater_equal(pca.explained_variance.data[:expected_num_components].sum().item(), min_explained_variance)

    # Check correlation of projected features are close to zero
    if expected_num_components > 1:
        atol = 1e-4
        corr_matrix_abs = torch.abs(torch.corrcoef(projected_features.T))
        check.is_true(torch.all(torch.tril(corr_matrix_abs, diagonal=-1) <= atol))
