"""Image dataset module.

This module contains classes for creating image datasets and dataloaders.
"""

from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Self

import torch
from jaxtyping import Int64, UInt8, jaxtyped
from more_itertools import chunked, split_when
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm.contrib.concurrent import thread_map

from imagescry.image.info import ImageInfo, ImageInfos, ImageShape
from imagescry.image.io import read_image_as_rgb_tensor
from imagescry.typechecking import typechecker


@jaxtyped(typechecker=typechecker)
@dataclass(frozen=True, slots=True)
class ImageBatch:
    """Batch of RGB uint8 images and their dataset indices.

    Attributes:
        indices (Int64[Tensor, "B"]): Dataset indices of the images.
        images (UInt8[Tensor, "B 3 H W"]): Batch of images.
    """

    indices: Int64[Tensor, "B"]
    images: UInt8[Tensor, "B 3 H W"]

    def __len__(self) -> int:
        """Return the number of images in the batch."""
        return len(self.indices)

    def __post_init__(self) -> None:
        """Check that both tensors are on the same device."""
        if self.indices.device != self.images.device:
            raise ValueError(
                "Tensors must be on the same device. "
                f"Got indices on {self.indices.device} and images on {self.images.device}"
            )

    def cpu(self) -> "ImageBatch":
        """Move tensors to the CPU.

        Returns:
            ImageBatch: A new ImageBatch with tensors on the CPU.
        """
        return self.to("cpu")

    def to(self, device: str | torch.device) -> "ImageBatch":
        """Move tensors to the specified device.

        Args:
            device (str | torch.device): The device to move the tensors to.

        Returns:
            ImageBatch: A new ImageBatch with tensors on the specified device.
        """
        return ImageBatch(indices=self.indices.to(device), images=self.images.to(device))

    @property
    def device(self) -> torch.device:
        """Get the device that the tensors are on."""
        return self.indices.device


class ImageFilesDataset(Dataset):
    """Dataset of UInt8 RGB images stored on disk.

    - When an image is accessed, it is read as a uint8 tensor with shape `(3, H, W)`
    - The image's index in the dataset is returned as a tuple with the image tensor: `(index, image_tensor)`
    - Not all images need to have the same spatial dimensions.

    Examples:
        Create a dataset from a directory of images:
        >>> dataset = ImageFilesDataset.from_directory("path/to/images") # doctest: +SKIP

        Get the first image in the dataset:
        >>> dataset[0] # doctest: +SKIP
        (tensor(0),
         tensor([[[ 36,  44,  58,  ...,  35,  42,  50],
                  [ 46,  38,  38,  ...,  40,  46,  49],
                  [ 66,  52,  46,  ...,  51,  52,  49],
                  ...,
                  [ 42,  49,  56,  ..., 171, 172, 173],
                  [ 35,  33,  42,  ..., 169, 169, 171],
                  [ 26,  26,  47,  ..., 167, 166, 167]],
                 [[ 40,  48,  62,  ...,  40,  47,  55],
                  [ 50,  42,  42,  ...,  45,  51,  54],
                  [ 70,  56,  47,  ...,  56,  57,  54],
                  ...,
                  [ 42,  49,  56,  ..., 189, 190, 191],
                  [ 35,  33,  44,  ..., 187, 187, 189],
                  [ 26,  26,  49,  ..., 185, 184, 185]],
                 [[ 43,  51,  65,  ...,  36,  43,  51],
                  [ 53,  45,  45,  ...,  41,  47,  50],
                  [ 73,  59,  51,  ...,  52,  53,  50],
                  ...,
                  [ 42,  49,  56,  ..., 199, 200, 201],
                  [ 35,  33,  43,  ..., 197, 197, 199],
                  [ 26,  26,  48,  ..., 195, 194, 195]]], dtype=torch.uint8))
    """

    def __init__(self, sources: Sequence[str | PathLike]) -> None:
        """Initialize the dataset by indexing image sources.

        Args:
            sources (Sequence[str | PathLike]): Sequence of image sources to create the dataset from.
        """
        self.image_infos = ImageInfos(
            thread_map(
                ImageInfo.from_source,
                sources,
                desc="Indexing images",
                unit="img",
                total=len(sources),
            )
        )

    @jaxtyped(typechecker=typechecker)
    def __getitem__(self, idx: int) -> tuple[Int64[Tensor, ""], UInt8[Tensor, "3 H W"]]:
        """Get an image and its index from the dataset.

        Args:
            idx (int): The index of the image to get.

        Returns:
            tuple[Int64[Tensor, ''], UInt8[Tensor, '3 H W']]: The image index and tensor.
        """
        # Read image and extract tensor
        image_tensor = read_image_as_rgb_tensor(self.image_infos[idx].source)

        # Return image index and tensor
        return torch.tensor(idx), image_tensor

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_infos)

    def get_loader(
        self,
        max_batch_size: int,
        *,
        sample_size: int | float | None = None,
        num_workers: int = 0,
        seed: int | None = None,
    ) -> DataLoader:
        """Get a `DataLoader` for the dataset using a `SimilarShapeBatcher` to batch images by shape.

        Loader will return `ImageBatch` objects.

        Args:
            max_batch_size (int): The maximum batch size for the `DataLoader`.
            sample_size (int | float | None, optional): The size of the subset to sample. If a float, it is interpreted
                as a fraction of the dataset. Defaults to None, which means no sampling.
            num_workers (int, optional): The number of worker processes to use for loading images. Defaults to 0.
            seed (int | None, optional): The seed to use for the random number generator. Defaults to None.

        Returns:
            DataLoader: A `DataLoader` for the dataset.
        """
        # Sample subset if requested
        if sample_size is not None:
            subset = self.sample(sample_size, seed=seed)
            shapes = [info.shape for info in self.image_infos[subset.indices]]
        else:
            subset = None
            shapes = [info.shape for info in self.image_infos]

        return DataLoader(
            subset or self,
            batch_sampler=SimilarShapeBatcher(shapes, max_batch_size),
            collate_fn=_collate_image_batch,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def sample(self, size: int | float, *, seed: int | None = None) -> Subset:
        """Sample a random subset of the dataset.

        Args:
            size (int | float): The size of the subset. If a float, it is interpreted as a fraction of the dataset.
            seed (int | None, optional): The seed to use for the random number generator. Defaults to None.

        Returns:
            Subset: A subset of the dataset.

        Raises:
            ValueError: If `size` is not between 0 and 1 or the number of images in the dataset.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(size, float):
            if size < 0 or size > 1:
                raise ValueError("size must be between 0 and 1")
            num_samples = int(size * len(self))
        else:
            if size < 0 or size > len(self):
                raise ValueError("size must be between 0 and the number of images in the dataset")
            num_samples = size

        return Subset(self, torch.randperm(len(self))[:num_samples].tolist())

    @classmethod
    def from_directory(
        cls,
        directory: str | PathLike,
        *,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> Self:
        """Create a dataset from a directory of images.

        Args:
            directory (str | PathLike): The directory to create the dataset from.
            image_extensions (tuple[str, ...], optional): The image file extensions (case insensitive) to use to find
                image files within the directory. Defaults to `(".jpg", ".jpeg", ".png")`.

        Returns:
            Self: An instance of `ImageFilesDataset`.

        Raises:
            FileNotFoundError: If the directory does not exist or if no images are found in the directory.
        """
        if (directory := Path(directory)).is_dir():
            # Find image sources
            extension_set = {ext.lower() for ext in image_extensions}
            sources = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in extension_set]
            if not sources:
                raise FileNotFoundError(f"No images found in directory {directory}")

            # Create dataset
            return cls(sources)
        else:
            raise FileNotFoundError(f"Directory {directory} does not exist")


class SimilarShapeBatcher(Sampler):
    """Sampler for grouping images by shape and batching them.

    This sampler will:
    1. Index input image shapes
    2. Sort image shapes
    3. Group image shapes
    4. Chunk shape groups into batches of size `max_batch_size` (or less)


    Examples:
        ```python
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset=# an image dataset with variable size images,
            batch_sampler=SimilarShapeBatcher(
                image_shapes=# iterable of `(channels, height, width)` tuples for each image in the dataset
                max_batch_size=8  # or any other positive integer
            ),
            shuffle=False,
            drop_last=False,
        )
        ```
    """

    def __init__(self, image_shapes: Iterable[ImageShape], max_batch_size: int) -> None:
        """Initialize sampler.

        Args:
            image_shapes (Iterable[ImageShape]): The shapes of images to batch.
            max_batch_size (int): The maximum size (number of images) of each batch.
        """
        self.max_batch_size = max_batch_size

        # Index and sort input image shapes
        indexed_sorted_image_shapes = sorted(enumerate(image_shapes), key=lambda x: x[1])

        # Group by shape
        shape_groups = split_when(indexed_sorted_image_shapes, lambda s1, s2: s1[1] != s2[1])

        # Chunk shape groups into batches of size `max_batch_size` (or less)
        batched_indexes_per_group = (chunked((idx for idx, _ in grp), max_batch_size) for grp in shape_groups)

        # Chain together batches from each shape group & convert to list
        self.batched_indexes = list(chain.from_iterable(batched_indexes_per_group))

    def __iter__(self) -> Generator[list[int]]:
        """Iterate over the batches."""
        yield from self.batched_indexes


# Helper functions
def _collate_image_batch(batch: list[tuple[Int64[Tensor, ""], UInt8[Tensor, "3 H W"]]]) -> ImageBatch:
    """Collate a list of image tensors and their dataset indices into an `ImageBatch`."""
    indices, images = zip(*batch, strict=True)
    return ImageBatch(indices=torch.stack(indices), images=torch.stack(images))
