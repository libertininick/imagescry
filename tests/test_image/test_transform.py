"""Tests for image transforms module."""

from typing import Literal

import pytest
import torch
from jaxtyping import UInt8
from pytest_check import check_functions
from torch import Tensor

from imagescry.image.transforms import normalize_per_channel, resize


def test_normalize_per_channel(image_tensor: UInt8[Tensor, "C H W"]) -> None:
    """Test normalizing an image."""
    # Normalize the image
    normalized_image = normalize_per_channel(image_tensor.float().unsqueeze(0))

    channel_means = normalized_image.mean((-2, -1))
    channel_stds = normalized_image.std((-2, -1))

    # Check the image is normalized
    check_functions.is_true(torch.allclose(torch.zeros_like(channel_means), channel_means, atol=1e-4))
    check_functions.is_true(torch.allclose(torch.ones_like(channel_means), channel_stds, atol=1e-4))

    # # Read the image shape
    # image_shape = read_image_shape(tempfile)
    # check_functions.equal(image_shape, ImageShape(*image_tensor.shape[-2:]))


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
    check_functions.equal(resized_image.shape[-2:], output_size)

    # Check that image tensor dtype is float
    check_functions.is_true(resized_image.dtype.is_floating_point)


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
        check_functions.equal(
            resized_height,
            output_size,
            f"Resized height is not equal to output size: {resized_height} != {output_size}",
        )

        # Check the width changed proportionally to the height
        check_functions.equal(
            resized_width,
            pytest.approx(original_width * output_size / original_height, abs=1),
        )
    else:
        # Check the width is exactly the output size
        check_functions.equal(
            resized_width,
            output_size,
            f"Resized width is not equal to output size: {resized_width} != {output_size}",
        )

        # Check the height changed proportionally to the width
        check_functions.equal(
            resized_height,
            pytest.approx(original_height * output_size / original_width, abs=1),
        )
