"""Tests for typechecking utilities."""

from typing import Any

import pytest
import torch
from jaxtyping import Float, TypeCheckError, jaxtyped
from torch import Tensor

from imagescry.typechecking import typechecker


@jaxtyped(typechecker=typechecker)
def mock_function(x: Float[Tensor, "1 3"], a: float = 1.0) -> Float[Tensor, "1"]:
    """Test function for checking that typechecker works."""
    return x.sum(dim=1) + a


@pytest.mark.parametrize(
    "x",
    [
        # Case 0: not a tensor
        1.0,
        # Case 1: tensor with incorrect shape
        torch.randn((1,)),
        # Case 2: tensor with incorrect shape
        torch.randn(1, 4),
        # Case 3: tensor with incorrect dtype
        torch.randint(0, 10, (1, 3), dtype=torch.int64),
    ],
)
def test_typchecker_raises(x: Any) -> None:
    """Test that typechecker raises an error when it should."""
    with pytest.raises(TypeCheckError):
        mock_function(x)


def test_typchecker_respects_numerical_tower() -> None:
    """Test that typechecker respects Python's numerical tower and allows float and int arguments."""
    x = torch.rand(1, 3)
    y = mock_function(x, a=2)
    assert y.shape == (1,)
