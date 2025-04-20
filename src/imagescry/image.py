"""Image tools."""

from collections.abc import Generator
from io import BytesIO
from itertools import chain
from os import PathLike
from typing import Literal, NamedTuple

import torch
from jaxtyping import Float, Num, Shaped, UInt8, jaxtyped
from more_itertools import chunked, split_when
from PIL import Image
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import Sampler as TorchSampler
from torchvision.transforms.functional import pil_to_tensor

from imagescry.typechecking import typechecker


class ImageShape(NamedTuple):
    """Image shape.

    Args:
        channels (int): Number of channels in the image.
        height (int): Height of the image.
        width (int): Width of the image.
    """

    channels: int
    height: int
    width: int


class SimilarShapeBatcher(TorchSampler):
    """Sampler for grouping images by shape and batching them."""

    def __init__(self, image_shapes: list[ImageShape], max_batch_size: int) -> None:
        """Initialize sampler.

        1. Index input image shapes
        2. Sort image shapes
        3. Group image shapes
        4. Chunk shape groups into batches of size `max_batch_size` (or less)

        Args:
            image_shapes (list[ImageShape]): The shapes of images to batch.
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

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batched_indexes)


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


@jaxtyped(typechecker=typechecker)
def read_as_tensor(
    image_source: str | PathLike | bytes | BytesIO, device: torch.device | None = None
) -> UInt8[Tensor, "C H W"]:
    """Read an image file or buffer and convert it to a PyTorch tensor.

    Args:
        image_source (str | PathLike | bytes | BytesIO): Path to file or a bytes buffer containing the image data.
        device (torch.device | None, optional): The device to put the tensor on. Defaults to None, which uses CPU.

    Returns:
        UInt8[Tensor, 'C H W']: Image as a tensor with shape [C, H, W] and values in the range [0, 255]

    Raises:
        TypeError: If `image_source` is not a file path (str) or a bytes buffer.
    """
    if isinstance(image_source, str | PathLike):
        # Load from file path
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes | BytesIO):
        # Load from bytes buffer
        if isinstance(image_source, bytes):
            image_source = BytesIO(image_source)
        img = Image.open(image_source).convert("RGB")
    else:
        raise TypeError("image_source must be a file path (str) or a bytes buffer")  # pragma: no cover

    # Convert PIL Image to tensor
    tensor = pil_to_tensor(img)

    # Move tensor to specified device if provided
    if device is not None:
        tensor = tensor.to(device)

    return tensor


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
