"""Image tools."""

from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import md5
from io import BytesIO
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Literal, Self

import torch
from jaxtyping import Float, Int64, Num, Shaped, UInt8, jaxtyped
from more_itertools import chunked, split_when
from PIL import Image
from PIL.ImageFile import ImageFile
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision.transforms.functional import pil_to_tensor
from tqdm.contrib.concurrent import thread_map

from imagescry.abstract_array import AbstractArray
from imagescry.typechecking import typechecker

ImageSource = str | PathLike | bytes | BytesIO


@pydantic_dataclass(frozen=True)
class ImageShape:
    """Image shape.

    Attributes:
        height (int): Height of the image.
        width (int): Width of the image.

    Examples:
        >>> image_shape = ImageShape(100, 100)

        # Unpack
        >>> height, width = image_shape

        # Get just image height
        >>> image_shape.height
        100

        # Serialize to JSON
        >>> from pydantic_core import to_json
        >>> to_json(image_shape)
        b'{"height":100,"width":100}'
    """

    height: int = Field(ge=0)
    width: int = Field(ge=0)

    def __eq__(self, other: object) -> bool:
        """Define equality for sorting."""
        if not isinstance(other, ImageShape):
            return NotImplemented  # pragma: no cover
        return self.to_tuple() == other.to_tuple()

    def __hash__(self) -> int:
        """Hash the image shape."""
        return hash(self.to_tuple())

    def __iter__(self) -> Generator[int]:
        """Allow unpacking with * operator."""
        yield self.height
        yield self.width

    def __lt__(self, other: object) -> bool:
        """Define less than for sorting."""
        if not isinstance(other, ImageShape):
            return NotImplemented  # pragma: no cover
        return self.to_tuple() < other.to_tuple()

    def to_tuple(self) -> tuple[int, int]:
        """Convert to (height, width) tuple."""
        return self.height, self.width


@pydantic_dataclass(frozen=True)
class ImageInfo:
    """Information about an image source.

    Attributes:
        source (Path): Path to the image file.
        shape (ImageShape): Shape of the image.
        hash (str): MD5 hash of the image.
    """

    source: Path
    shape: ImageShape
    hash: str

    @classmethod
    def from_source(cls, source: str | PathLike) -> "ImageInfo":
        """Create an ImageInfo instance from a source path.

        Args:
            source (str | PathLike): Path to the image file.

        Returns:
            ImageInfo: Information about the image source.

        Raises:
            FileNotFoundError: If the image source does not exist or is not a file.
        """
        src = Path(source).absolute()
        if not (src.exists() and src.is_file()):
            raise FileNotFoundError(f"Image source {src} does not exist or is not a file")
        return cls(source=src, shape=read_image_shape(src), hash=get_image_hash(src))


class ImageInfos(AbstractArray[ImageInfo]):
    """Array of `ImageInfo` objects."""

    ...


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

    def to(self, device: torch.device) -> "ImageBatch":
        """Move tensors to the specified device.

        Args:
            device (torch.device): The device to move the tensors to.

        Returns:
            ImageBatch: A new ImageBatch with tensors on the specified device.
        """
        return ImageBatch(
            indices=self.indices.to(device),
            images=self.images.to(device),
        )

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
        seed: int | None = None,
    ) -> DataLoader:
        """Get a `DataLoader` for the dataset using a `SimilarShapeBatcher` to batch images by shape.

        Loader will return `ImageBatch` objects.

        Args:
            max_batch_size (int): The maximum batch size for the `DataLoader`.
            sample_size (int | float | None, optional): The size of the subset to sample. If a float, it is interpreted
                as a fraction of the dataset. Defaults to None, which means no sampling.
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


def get_image_hash(image_source: ImageSource, *, buffer_size: int = 65_536) -> str:
    """Calculate the MD5 hash of an image.

    Args:
        image_source (ImageSource): The image source to get the hash of.
        buffer_size (int, optional): The buffer size to use for reading the image. Defaults to 65536.

    Returns:
        str: The MD5 hash of the image.
    """
    # Initialize MD5 hash
    md5_hash = md5(usedforsecurity=False)

    # Add image bytes to hash based on source type
    match image_source:
        case str() | PathLike():
            with Path(image_source).open("rb") as f:
                for chunk in iter(lambda: f.read(buffer_size), b""):
                    md5_hash.update(chunk)
        case bytes():
            md5_hash.update(image_source)
        case BytesIO():
            for chunk in iter(lambda: image_source.read(buffer_size), b""):
                md5_hash.update(chunk)

    # Return hexdigest
    return md5_hash.hexdigest()


@jaxtyped(typechecker=typechecker)
def normalize_per_channel(
    image_tensor: Num[Tensor, "B C H W"],
    *,
    channel_means: Float[Tensor, "#B C 1 1"] | None = None,
    channel_stds: Float[Tensor, "#B C 1 1"] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "B C H W"]:
    """Normalize image pixels (per channel) to have zero mean and unit variance.

    Args:
        image_tensor (Num[Tensor, 'B C H W']): Image tensor to normalize.
        channel_means (Float[Tensor, '#B C 1 1'] | None, optional): Channel pixel means.
            If `B=1`, will broadcast to all images. If None, will calculate channel means from image.
            Defaults to None.
        channel_stds (Float[Tensor, '#B C 1 1'] | None, optional): Channel pixels standard deviations.
            If `B=1`, will broadcast to all images. If None, will calculate channel standard deviations from image.
            Defaults to None.
        min_value (float | None, optional): Minimum value to clip normalized pixels to. Defaults to None.
        max_value (float | None, optional): Maximum value to clip normalized pixels to. Defaults to None.
        eps (float, optional): Epsilon to prevent division by zero. Defaults to 1e-6.

    Returns:
        Float[Tensor, 'B C H W']: Normalized image.


    Examples:
        >>> import torch

        Normalize single image:
        >>> image_tensor = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        >>> normalized_image = normalize_per_channel(image_tensor.unsqueeze(0), min_value=-3, max_value=3)

        Normalize batch of images with specified channel means and standard deviations:
        >>> batch = torch.rand(16, 3, 32, 32)
        >>> normalized_batch = normalize_per_channel(
        ...     batch,
        ...     channel_means=torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        ...     channel_stds=torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        ... )
    """
    # Convert image tensor to float
    image_tensor = image_tensor.float()

    # Calculate channel means and standard deviations
    if channel_means is None:
        channel_means = image_tensor.mean(dim=(0, 2, 3), keepdim=True)
    if channel_stds is None:
        channel_stds = image_tensor.std(dim=(0, 2, 3), keepdim=True)

    # Normalize image
    image_tensor = (image_tensor - channel_means) / (channel_stds + eps)

    # Clip normalized image
    if min_value is not None or max_value is not None:
        image_tensor = image_tensor.clip(min_value, max_value)

    return image_tensor


@contextmanager
def open_image_source(image_source: ImageSource) -> Generator[ImageFile]:
    """Context manager for opening a PIL Image object from an image source."""
    # Convert bytes to bytes buffer
    if isinstance(image_source, bytes):
        image_source = BytesIO(image_source)

    # Check that image source is a file path or bytes buffer
    if not isinstance(image_source, str | PathLike | BytesIO):
        raise TypeError("image_source must be a file path (str) or a bytes buffer")  # pragma: no cover

    with Image.open(image_source) as img_file:
        yield img_file


@jaxtyped(typechecker=typechecker)
def read_image_as_rgb_tensor(image_source: ImageSource, device: torch.device | None = None) -> UInt8[Tensor, "3 H W"]:
    """Read an image file or buffer, and convert it to a RGB PyTorch tensor.

    Args:
        image_source (ImageSource): File path, bytes object, or a bytes buffer containing the image data.
        device (torch.device | None, optional): The device to put the tensor on. Defaults to None, which uses CPU.

    Returns:
        UInt8[Tensor, '3 H W']: Image as a RGB tensor with shape [3, H, W] and integer values in the range [0, 255]
    """
    with open_image_source(image_source) as img:
        return pil_to_tensor(img.convert("RGB")).to(device=device)


def read_image_shape(image_source: ImageSource) -> ImageShape:
    """Read the shape of an image file or buffer.

    Args:
        image_source (ImageSource): File path, bytes object, or a bytes buffer containing the image data.

    Returns:
        ImageShape: The shape of the image.
    """
    with open_image_source(image_source) as img:
        return ImageShape(img.height, img.width)


@jaxtyped(typechecker=typechecker)
def resize(
    image_tensor: Num[Tensor, "... H1 W1"],
    output_size: int | tuple[int, int],
    *,
    side_ref: Literal["height", "width", "long", "short"] = "long",
) -> Num[Tensor, "... H2 W2"]:
    """Resize image tensor.

    Args:
        image_tensor (Num[Tensor, '... H1 W1']): Image tensor to resize.
        output_size (int | tuple[int, int]): The output size of the image.
        side_ref (Literal['height', 'width', 'long', 'short']): Image side to use for resizing when `output_size`
            is an integer. Defaults to 'long'.

    Returns:
        Num[Tensor, '... H2 W2']: Resized image tensor.
    """
    # Make sure tensor is 4D
    squeeze_dims = tuple(range(4 - image_tensor.ndim))
    image_tensor = _to_4d(image_tensor)

    # Resize image
    if isinstance(output_size, int):
        # Calculate scale factor
        height, width = image_tensor.shape[-2:]
        scale_factor = _calc_scale_factor(height, width, output_size, side_ref)

        # Resize image using scale factor
        image_tensor = interpolate(
            image_tensor,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )
    else:
        # Resize image
        image_tensor = interpolate(image_tensor, size=output_size, mode="bilinear", align_corners=False)

    # Remove phantom dimensions
    image_tensor = image_tensor.squeeze(squeeze_dims)

    return image_tensor


# Helper functions
def _calc_scale_factor(
    height: int, width: int, output_size: int, side_ref: Literal["height", "width", "long", "short"]
) -> float:
    """Calculate scale factor for resizing image tensor.

    Args:
        height (int): Height of image tensor.
        width (int): Width of image tensor.
        output_size (int): The output size of the image.
        side_ref (Literal['height', 'width', 'long', 'short']): Image side to use for resizing when `output_size`
            is an integer. Defaults to 'long'.

    Returns:
        float: Scale factor for resizing image tensor.

    Raises:
        ValueError: If `side_ref` is not 'height', 'width', 'long', or 'short'.
    """
    if side_ref == "height":
        scale_factor = output_size / height
    elif side_ref == "width":
        scale_factor = output_size / width
    elif side_ref == "long":
        scale_factor = output_size / max(height, width)
    elif side_ref == "short":
        scale_factor = output_size / min(height, width)
    else:
        raise ValueError(f"Invalid side_ref: {side_ref}")  # pragma: no cover

    return scale_factor


def _collate_image_batch(batch: list[tuple[Int64[Tensor, ""], UInt8[Tensor, "3 H W"]]]) -> ImageBatch:
    """Collate a list of image tensors and their dataset indices into an `ImageBatch`."""
    indices, images = zip(*batch, strict=True)
    return ImageBatch(indices=torch.stack(indices), images=torch.stack(images))


@jaxtyped(typechecker=typechecker)
def _to_4d(image_tensor: Shaped[Tensor, "... H W"]) -> Shaped[Tensor, "B C H W"]:
    """Convert image tensor to 4D by adding phantom dimensions to the beginning.

    Args:
        image_tensor (Shaped[Tensor, '... H W']): Image tensor to convert.

    Returns:
        Shaped[Tensor, 'B C H W']: 4D image tensor.

    Raises:
        ValueError: If `image_tensor` is not 2D, 3D, or 4D.

    Examples:
        >>> import torch

        2D tensor to 4D:
        >>> _to_4d(torch.randn(3, 4)).shape
        torch.Size([1, 1, 3, 4])

        3D to 4D:
        >>> _to_4d(torch.randn(3, 5, 7)).shape
        torch.Size([1, 3, 5, 7])

        4D to 4D:
        >>> _to_4d(torch.randn(16, 3, 5, 7)).shape
        torch.Size([16, 3, 5, 7])
    """
    if image_tensor.ndim == 2:
        return image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.ndim == 3:
        return image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4:
        return image_tensor
    else:
        raise ValueError(f"Invalid image tensor shape: {image_tensor.shape}")  # pragma: no cover
