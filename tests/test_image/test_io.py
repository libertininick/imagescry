"""Tests for image I/O module."""

from io import BytesIO
from pathlib import Path

import torch
from jaxtyping import UInt8
from pytest_check import check_functions
from torch import Tensor

from imagescry.image.io import get_image_hash, read_image_as_rgb_tensor


def test_get_image_hash(image_source: Path | bytes | BytesIO, image_hash: str) -> None:
    """Test getting the hash of an image is consistent."""
    if isinstance(image_source, (Path, bytes)):
        check_functions.equal(image_hash, get_image_hash(image_source))
    elif isinstance(image_source, BytesIO):
        cloned_buffer = BytesIO(image_source.getvalue())
        check_functions.equal(image_hash, get_image_hash(cloned_buffer))


def test_read_image_as_rgb_tensor(image_tensor: UInt8[Tensor, "C H W"], image_source: Path | bytes | BytesIO) -> None:
    """Test image read from source matches the original image."""
    check_functions.is_true(torch.allclose(image_tensor, read_image_as_rgb_tensor(image_source), atol=1))
