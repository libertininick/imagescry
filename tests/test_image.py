"""Tests for image tools."""

from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
import torch
from jaxtyping import UInt8, jaxtyped
from pytest import FixtureRequest, TempPathFactory
from pytest_check import check
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.io import write_png

from imagescry.image import (
    ImageFilesDataset,
    ImageShape,
    SimilarShapeBatcher,
    normalize_per_channel,
    read_image_as_rgb_tensor,
    read_image_shape,
    resize,
)
from imagescry.typechecking import typechecker

# Seed
SEED = 1234
torch.manual_seed(SEED)


# Fixtures
@pytest.fixture(scope="module")
def variable_size_image_dataset(tmp_path_factory: TempPathFactory) -> ImageFilesDataset:
    """Create a variable size image dataset test fixture."""
    # Generate a set of image with random shapes and save them to disk
    image_dir = tmp_path_factory.mktemp("images")
    image_files: list[Path] = []
    shapes = [
        (7, 7),
        (7, 8),
        (8, 8),
        (8, 8),
        (7, 7),
        (5, 7),
        (8, 8),
        (8, 7),
        (8, 7),
        (7, 7),
        (7, 7),
        (2, 2),
        (3, 2),
        (2, 2),
    ]
    for i, shape in enumerate(shapes):
        image_tensor = torch.randint(0, 255, (3, *shape), dtype=torch.uint8)
        image_file = image_dir / f"{i}.png"
        image_files.append(image_file)
        write_png(image_tensor, image_file)

    return ImageFilesDataset(sources=image_files)


@pytest.fixture(scope="module")
def image_shape() -> ImageShape:
    """Create a image shape test fixture."""
    return ImageShape(30, 45)


@jaxtyped(typechecker=typechecker)
@pytest.fixture(scope="module")
def image_tensor(image_shape: ImageShape) -> UInt8[Tensor, "3 {image_shape.height} {image_shape.width}"]:
    """Create a uint8 image tensor test fixture."""
    torch.manual_seed(1234)
    return torch.randint(low=0, high=256, size=(3, *image_shape), dtype=torch.uint8)


@pytest.fixture(scope="module")
def image_source_file(image_tensor: UInt8[Tensor, "3 30 45"], tmp_path_factory: TempPathFactory) -> Path:
    """Create a test image source file."""
    temp_file = tmp_path_factory.mktemp("images") / "test.png"
    write_png(image_tensor, temp_file)
    return temp_file


@pytest.fixture(scope="module")
def image_source_bytes(image_source_file: Path) -> bytes:
    """Create a test image source bytes."""
    with image_source_file.open("rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def image_source_buffer(image_source_bytes: bytes) -> BytesIO:
    """Create a test image source buffer."""
    return BytesIO(image_source_bytes)


@pytest.fixture(params=["image_source_file", "image_source_bytes", "image_source_buffer"])
def image_source(request: FixtureRequest) -> Path | bytes | BytesIO:
    """Create a test image source."""
    # Get the fixture dynamically by name
    return request.getfixturevalue(request.param)


# Tests
@pytest.mark.parametrize("max_batch_size", [1, 2, 3, 4])
def test_similar_shape_batcher(variable_size_image_dataset: ImageFilesDataset, max_batch_size: int) -> None:
    """Test the similar shape batcher groups images into batches by imagesize."""
    # Create a dataloader with the similar shape batcher
    dataloader = DataLoader(
        variable_size_image_dataset,
        batch_sampler=SimilarShapeBatcher(
            image_shapes=variable_size_image_dataset.image_shapes, max_batch_size=max_batch_size
        ),
        shuffle=False,
        drop_last=False,
    )

    observed_image_indexes = set()
    for indexes, images in dataloader:
        # Check the batch size is less than or equal to max_batch_size
        check.less_equal(images.size(0), max_batch_size)

        # Check all images in the batch have the same shape
        shapes = [ImageShape(*img.shape[-2:]) for img in images]
        expected_shapes = variable_size_image_dataset.image_shapes[indexes.tolist()].tolist()
        num_unique_shapes = len(set(expected_shapes))
        check.equal(expected_shapes, shapes)
        check.equal(num_unique_shapes, 1)

        # Add image indexes to the observed set
        observed_image_indexes.update(indexes.tolist())

    # Check that all image indexes were observed while iterating through the dataloader using the batch sampler
    expected_image_indexes = set(range(len(variable_size_image_dataset)))
    check.equal(expected_image_indexes, observed_image_indexes)


def test_normalize_per_channel(image_tensor: UInt8[Tensor, "C H W"]) -> None:
    """Test normalizing an image."""
    # Normalize the image
    normalized_image = normalize_per_channel(image_tensor.float().unsqueeze(0))

    channel_means = normalized_image.mean((-2, -1))
    channel_stds = normalized_image.std((-2, -1))

    # Check the image is normalized
    check.is_true(torch.allclose(torch.zeros_like(channel_means), channel_means, atol=1e-4))
    check.is_true(torch.allclose(torch.ones_like(channel_means), channel_stds, atol=1e-4))

    # # Read the image shape
    # image_shape = read_image_shape(tempfile)
    # check.equal(image_shape, ImageShape(*image_tensor.shape[-2:]))


def test_read_image_shape(image_shape: ImageShape, image_source: Path | bytes | BytesIO) -> None:
    """Test reading the shape of an image."""
    check.equal(image_shape, read_image_shape(image_source))


def test_read_image_as_rgb_tensor(image_tensor: UInt8[Tensor, "C H W"], image_source: Path | bytes | BytesIO) -> None:
    """Test image read from source matches the original image."""
    check.is_true(torch.allclose(image_tensor, read_image_as_rgb_tensor(image_source), atol=1))


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
