"""Tests for image dataset module."""

from pathlib import Path

import pytest
import torch
from pytest import TempPathFactory
from pytest_check import check
from torch.utils.data import Subset
from torchvision.io import write_png

from imagescry.image.dataset import ImageBatch, ImageFilesDataset
from imagescry.image.info import ImageShape


def test_image_batch_device() -> None:
    """Test getting the device of an `ImageBatch` and moving tensors to a device."""
    # Create a batch of images
    image_batch = ImageBatch(
        indices=torch.tensor([0, 1, 2]), images=torch.randint(0, 255, (3, 3, 3, 3)).to(torch.uint8)
    )

    # Check the device
    check.equal(image_batch.device, torch.device("cpu"))

    # Move to GPU
    if torch.cuda.is_available():
        image_batch = image_batch.to("cuda")
        check.is_true(str(image_batch.device).startswith("cuda"))
        check.is_true(image_batch.images.is_cuda)
        check.is_true(image_batch.indices.is_cuda)

    # Move to CPU
    image_batch = image_batch.cpu()
    check.equal(image_batch.device, torch.device("cpu"))


def test_sample_integer_size(variable_size_image_dataset: ImageFilesDataset) -> None:
    """Test sampling from an `ImageFilesDataset` with integer size."""
    # Test sampling 5 items
    subset = variable_size_image_dataset.sample(5)

    # Check it's a Subset instance
    check.is_instance(subset, Subset)

    # Check the size is correct
    check.equal(len(subset), 5)

    # Check all indices are valid
    check.is_true(all(0 <= idx < len(variable_size_image_dataset) for idx in subset.indices))


def test_sample_float_size(variable_size_image_dataset: ImageFilesDataset) -> None:
    """Test sampling from an `ImageFilesDataset` with with float size (percentage)."""
    # Test sampling 50% of items
    subset = variable_size_image_dataset.sample(0.5)

    # Check it's a Subset instance
    check.is_instance(subset, Subset)

    # Check the size is correct (should be half rounded down)
    expected_size = len(variable_size_image_dataset) // 2
    check.equal(len(subset), expected_size)

    # Test edge cases
    empty_subset = variable_size_image_dataset.sample(0.0)
    check.equal(len(empty_subset), 0)

    full_subset = variable_size_image_dataset.sample(1.0)
    check.equal(len(full_subset), len(variable_size_image_dataset))


def test_invalid_sample_size_raises(variable_size_image_dataset: ImageFilesDataset) -> None:
    """Test error conditions for sampling from an `ImageFilesDataset` with invalid sizes."""
    # Test negative integer size
    with pytest.raises(ValueError, match="size must be between 0 and the number of images in the dataset"):
        variable_size_image_dataset.sample(-1)

    # Test size larger than dataset
    with pytest.raises(ValueError, match="size must be between 0 and the number of images in the dataset"):
        variable_size_image_dataset.sample(len(variable_size_image_dataset) + 1)

    # Test negative float size
    with pytest.raises(ValueError, match="size must be between 0 and 1"):
        variable_size_image_dataset.sample(-0.5)

    # Test float size > 1.0
    with pytest.raises(ValueError, match="size must be between 0 and 1"):
        variable_size_image_dataset.sample(1.5)


def test_sample_seed_reproducibility(variable_size_image_dataset: ImageFilesDataset) -> None:
    """Test that sampling is reproducible with the same seed."""
    # Sample with same seed
    subset1 = variable_size_image_dataset.sample(5, seed=1234)
    subset2 = variable_size_image_dataset.sample(5, seed=1234)

    # Check indices are identical
    check.equal(subset1.indices, subset2.indices)

    # Sample with different seed
    subset3 = variable_size_image_dataset.sample(5, seed=1235)

    # Check indices are different
    check.not_equal(subset1.indices, subset3.indices)


@pytest.mark.parametrize("max_batch_size", [1, 2, 3, 4])
def test_dataloader_similar_shape_batcher(variable_size_image_dataset: ImageFilesDataset, max_batch_size: int) -> None:
    """Test dataset's dataloader uses the similar shape batcher to group images into batches by imagesize."""
    # Create a dataloader that uses the similar shape batcher to group images into batches by imagesize
    dataloader = variable_size_image_dataset.get_loader(max_batch_size=max_batch_size)

    # Iterate through the batches and check that the images are grouped into batches by imagesize
    observed_image_indexes = set()
    for batch in dataloader:
        # Check the batch size is less than or equal to max_batch_size
        check.less_equal(len(batch), max_batch_size)

        # Check all images in the batch have the same shape
        shapes = [ImageShape(*img.shape[-2:]) for img in batch.images]
        num_unique_shapes = len(set(shapes))
        check.equal(num_unique_shapes, 1)

        # Get expected shape for each image in the batch
        index_list: list[int] = batch.indices.tolist()
        expected_shapes = [info.shape for info in variable_size_image_dataset.image_infos[index_list]]

        # Check the expected shapes match the observed shapes
        check.equal(expected_shapes, shapes)

        # Add image indexes to the observed set
        observed_image_indexes.update(index_list)

    # Check that all image indexes were observed while iterating through the dataloader using the batch sampler
    expected_image_indexes = set(range(len(variable_size_image_dataset)))
    check.equal(expected_image_indexes, observed_image_indexes)


def test_from_directory(tmp_path_factory: TempPathFactory) -> None:
    """Test creating a dataset from a directory."""
    # Create test directory with some images
    image_dir = tmp_path_factory.mktemp("images")
    shapes = [(7, 7), (8, 8), (9, 9)]
    for i, shape in enumerate(shapes):
        image_tensor = torch.randint(0, 255, (3, *shape), dtype=torch.uint8)
        write_png(image_tensor, image_dir / f"{i}.png")

    # Create dataset from directory
    dataset = ImageFilesDataset.from_directory(image_dir)

    # Check dataset size
    check.equal(len(dataset), len(shapes))

    # Check all images can be loaded
    for i in range(len(dataset)):
        idx, img = dataset[i]
        check.equal(idx, i)
        check.equal(img.shape[0], 3)  # RGB channels
        check.equal(img.shape[1:], shapes[i])


def test_from_directory_no_images(tmp_path: Path) -> None:
    """Test creating a dataset from a directory with no images."""
    with pytest.raises(FileNotFoundError, match="No images found in directory"):
        ImageFilesDataset.from_directory(tmp_path)


def test_from_directory_not_exists(tmp_path: Path) -> None:
    """Test creating a dataset from a non-existent directory."""
    non_existent_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError, match=r"Directory .* does not exist"):
        ImageFilesDataset.from_directory(non_existent_dir)
