"""Tests for image tools."""

from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
import torch
from jaxtyping import UInt8, jaxtyped
from pytest_check import check
from torch import Tensor
from torchvision.io import write_png

from imagescry.image import normalize_per_channel, read_as_tensor, resize
from imagescry.typechecking import typechecker


# Fixtures
@jaxtyped(typechecker=typechecker)
@pytest.fixture(scope="module")
def image_tensor() -> UInt8[Tensor, "3 30 45"]:
    """Create a uint8 image tensor test fixture."""
    torch.manual_seed(1234)
    return torch.randint(low=0, high=256, size=(3, 30, 45), dtype=torch.uint8)


# Tests
def test_normalize_per_channel(image_tensor: UInt8[Tensor, "C H W"]) -> None:
    """Test normalizing an image."""
    # Normalize the image
    normalized_image = normalize_per_channel(image_tensor.float().unsqueeze(0))

    channel_means = normalized_image.mean((-2, -1))
    channel_stds = normalized_image.std((-2, -1))

    # Check the image is normalized
    check.is_true(torch.allclose(torch.zeros_like(channel_means), channel_means, atol=1e-4))
    check.is_true(torch.allclose(torch.ones_like(channel_means), channel_stds, atol=1e-4))


def test_read_as_tensor_from_file(image_tensor: UInt8[Tensor, "C H W"], tmp_path: Path) -> None:
    """Test reading an image as a tensor from a file."""
    # Write the image to a temporary file
    tempfile = tmp_path / "test.png"
    write_png(image_tensor, tempfile)

    # Read the image as a tensor
    tensor = read_as_tensor(tempfile)

    # Check the image is read correctly
    check.equal(tensor.shape, image_tensor.shape)
    check.is_true(torch.allclose(tensor, image_tensor, atol=1))


def test_read_as_tensor_from_buffer(image_tensor: UInt8[Tensor, "C H W"], tmp_path: Path) -> None:
    """Test reading an image as a tensor from a buffer."""
    # Write the image to a temporary file
    tempfile = tmp_path / "test.png"
    write_png(image_tensor, tempfile)

    # Read the image as a tensor
    tensor = read_as_tensor(BytesIO(tempfile.read_bytes()))

    # Check the image is read correctly
    check.equal(tensor.shape, image_tensor.shape)
    check.is_true(torch.allclose(tensor, image_tensor, atol=1))


def test_read_as_tensor_from_bytes(image_tensor: UInt8[Tensor, "C H W"], tmp_path: Path) -> None:
    """Test reading an image as a tensor from bytes."""
    # Write the image to a temporary file
    tempfile = tmp_path / "test.png"
    write_png(image_tensor, tempfile)

    # Read the image as a tensor
    tensor = read_as_tensor(tempfile.read_bytes(), device=torch.device("cpu"))

    # Check the image is read correctly
    check.equal(tensor.shape, image_tensor.shape)
    check.is_true(torch.allclose(tensor, image_tensor, atol=1))
    check.equal(tensor.device, torch.device("cpu"))


@pytest.mark.parametrize("add_batch", [False, True])
@pytest.mark.parametrize("output_size", [(4, 4), (5, 5), (5, 7), (7, 5), (33, 38)])
def test_resize_exact_output_size(
    *,
    image_tensor: UInt8[Tensor, "C H W"],
    output_size: tuple[int, int],
    add_batch: bool,
) -> None:
    """Test resizing an image to an exact output size."""
    # Add a batch dimension if needed
    if add_batch:
        image_tensor = image_tensor.unsqueeze(0)

    # Resize the image
    resized_image = resize(image_tensor, output_size=output_size, side_ref="height")
    check.equal(resized_image.shape[-2:], output_size)


@pytest.mark.parametrize("transpose_input", [False, True])
@pytest.mark.parametrize("side_ref", ["height", "width", "long", "short"])
@pytest.mark.parametrize("output_size", [16, 31, 46])
def test_resize_side_ref(
    *,
    image_tensor: UInt8[Tensor, "C H W"],
    output_size: int,
    side_ref: Literal["height", "width", "long", "short"],
    transpose_input: bool,
) -> None:
    """Test resizing an image using a single dimension as reference for a specified size."""
    # Transpose the image if needed
    if transpose_input:
        image_tensor = image_tensor.transpose(-2, -1)

    # Get the original image shape
    original_height, original_width = image_tensor.shape[-2:]

    # Resize the image
    resized_image = resize(image_tensor, output_size, side_ref=side_ref)

    # Check the image is resized correctly
    resized_height, resized_width = resized_image.shape[-2:]
    if (
        side_ref == "height"
        or (side_ref == "long" and original_height >= original_width)
        or (side_ref == "short" and original_height < original_width)
    ):
        # Check the height is exactly the output size
        check.equal(
            resized_height,
            output_size,
            f"Resized height is not equal to output size: {resized_height} != {output_size}",
        )

        # Check the width changed proportionally to the height
        check.equal(
            resized_width,
            pytest.approx(original_width * output_size / original_height, abs=1),
        )
    else:
        # Check the width is exactly the output size
        check.equal(
            resized_width,
            output_size,
            f"Resized width is not equal to output size: {resized_width} != {output_size}",
        )

        # Check the height changed proportionally to the width
        check.equal(
            resized_height,
            pytest.approx(original_height * output_size / original_width, abs=1),
        )
