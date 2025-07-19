"""Tests for image info module."""

from io import BytesIO
from pathlib import Path

import pytest
from pytest_check import check_functions

from imagescry.image.info import ImageInfo, ImageShape


def test_image_info_from_source(image_source_file: Path, image_shape: ImageShape, image_hash: str) -> None:
    """Test creating an ImageInfo instance from a valid image source."""
    # Create ImageInfo instance
    image_info = ImageInfo.from_source(image_source_file)

    # Check attributes
    check_functions.equal(image_source_file.absolute(), image_info.source, msg="Image source path is not correct")
    check_functions.equal(image_shape, image_info.shape, msg="Image shape is not correct")
    check_functions.equal(image_hash, image_info.hash, msg="Image hash is not correct")


def test_image_shape_from_source(image_source: Path | bytes | BytesIO, image_shape: ImageShape) -> None:
    """Test reading the shape of an image."""
    check_functions.equal(image_shape, ImageShape.from_source(image_source))


def test_image_info_invalid_source_raises(tmp_path: Path) -> None:
    """Test creating an ImageInfo instance from an invalid image source."""
    # Try to create ImageInfo from non-existent file
    non_existent_file = tmp_path / "non_existent.png"
    with pytest.raises(FileNotFoundError, match=f"{non_existent_file} does not exist or is not a file"):
        ImageInfo.from_source(non_existent_file)

    # Try to create ImageInfo from a directory
    with pytest.raises(FileNotFoundError, match=f"{tmp_path} does not exist or is not a file"):
        ImageInfo.from_source(tmp_path)
