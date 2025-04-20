"""Tests for image tools."""

from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
import torch
from jaxtyping import Int64, UInt8, jaxtyped
from pytest_check import check
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision.io import write_png

from imagescry.image import ImageShape, SimilarShapeBatcher, normalize_per_channel, read_as_tensor, resize
from imagescry.typechecking import typechecker

# Seed
SEED = 1234
torch.manual_seed(SEED)


# Fixtures
class VariableSizeImageDataset(TorchDataset):
    """Dataset of images of different sizes."""

    def __init__(self, num_channels: Literal[1, 3], sizes: list[tuple[int, int]]) -> None:
        """Initialize the dataset."""
        self.num_channels = num_channels
        self.image_shapes = [ImageShape(num_channels, h, w) for h, w in sizes]
        self.data = [
            torch.randint(0, 255, image_shape.to_tuple(), dtype=torch.uint8) for image_shape in self.image_shapes
        ]

    def __len__(self) -> int:
        """Get the number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Int64[Tensor, ""], UInt8[Tensor, "C H W"]]:
        """Get an image and its index from the dataset.

        Args:
            idx (int): The index of the image to get.

        Returns:
            tuple[Int64[Tensor, ""], UInt8[Tensor, "C H W"]]: A tuple containing the index and the image.
        """
        return torch.tensor(idx), self.data[idx]


@pytest.fixture(scope="module", params=[1, 3])
def variable_size_image_dataset(request: pytest.FixtureRequest) -> VariableSizeImageDataset:
    """Create a variable size image dataset test fixture."""
    return VariableSizeImageDataset(
        num_channels=request.param,
        # Define a random set of image sizes
        sizes=[
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
        ],
    )


@jaxtyped(typechecker=typechecker)
@pytest.fixture(scope="module")
def image_tensor() -> UInt8[Tensor, "3 30 45"]:
    """Create a uint8 image tensor test fixture."""
    torch.manual_seed(1234)
    return torch.randint(low=0, high=256, size=(3, 30, 45), dtype=torch.uint8)


# Tests
@pytest.mark.parametrize("max_batch_size", [1, 2, 3, 4])
def test_similar_shape_batcher(variable_size_image_dataset: VariableSizeImageDataset, max_batch_size: int) -> None:
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
        check.equal(len({ImageShape(*img.shape[-3:]) for img in images}), 1)

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
