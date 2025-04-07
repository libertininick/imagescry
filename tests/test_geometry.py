"""Tests for geometry tools."""

import torch
from pytest_check import check
from shapely.geometry import Polygon

from imagescry.geometry import create_roi_mask


def test_create_roi_mask_single_polygon() -> None:
    """Test create_roi_mask function."""
    # Define a region of interest
    roi = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])

    # Define the shape of the original image and the feature map
    original_image_shape = (6, 8)
    feature_map_shape = (3, 4)

    # Create the mask
    mask = create_roi_mask(roi, original_image_shape, feature_map_shape)

    # Check mask is the correct shape
    check.equal(mask.shape, feature_map_shape)

    # Check the mask is correct
    expected_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    check.is_true(torch.allclose(mask, expected_mask))


def test_create_roi_mask_multiple_polygons() -> None:
    """Test create_roi_mask function with multiple polygons."""
    # Define multiple regions of interest
    roi = [
        # Upper left
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        # Lower right
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
    ]

    # Define the shape of the original image and the feature map
    original_image_shape = (4, 4)
    feature_map_shape = (2, 2)

    # Create the mask
    mask = create_roi_mask(roi, original_image_shape, feature_map_shape)

    # Check mask is the correct shape
    check.equal(mask.shape, feature_map_shape)

    # Check the mask is correct
    expected_mask = torch.tensor([[1, 0], [0, 1]])
    check.is_true(torch.allclose(mask, expected_mask))
