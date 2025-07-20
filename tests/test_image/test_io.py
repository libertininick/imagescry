"""Tests for image I/O module."""

import base64
from io import BytesIO
from pathlib import Path

import torch
from jaxtyping import UInt8
from pytest_check import check_functions
from torch import Tensor

from imagescry.image.io import get_image_hash, open_image_source, read_image_and_encode, read_image_as_rgb_tensor


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


def test_read_image_and_encode(image_tensor: UInt8[Tensor, "C H W"], image_source: Path | bytes | BytesIO) -> None:
    """Test reading and encoding an image preserves the image data."""
    # Read and encode the image
    encoded = read_image_and_encode(image_source)

    # Check the encoding prefix
    check_functions.is_true(
        encoded.startswith("data:image/jpeg;base64,"), "Encoded image should start with data URI scheme"
    )

    # Decode and verify the image is a JPEG with the same shape as the original
    # NOTE: the exact bytes may differ due to JPEG compression
    _, b64 = encoded.split(",", 1)
    decoded = base64.b64decode(b64)
    with open_image_source(decoded) as img:
        check_functions.equal(img.format, "JPEG", "Decoded image should be a JPEG")
        width, height = img.size
        check_functions.equal(width, image_tensor.shape[2], "Decoded image width should match original")
        check_functions.equal(height, image_tensor.shape[1], "Decoded image height should match original")
