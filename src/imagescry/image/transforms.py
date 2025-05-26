"""Image transforms module.

This module contains functions for transforming images.
"""

from typing import Literal

from jaxtyping import Float, Num, Shaped, jaxtyped
from torch import Tensor
from torch.nn.functional import interpolate

from imagescry.typechecking import typechecker


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
    image_tensor = to_4d(image_tensor)

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


@jaxtyped(typechecker=typechecker)
def to_4d(image_tensor: Shaped[Tensor, "... H W"]) -> Shaped[Tensor, "B C H W"]:
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
        >>> to_4d(torch.randn(3, 4)).shape
        torch.Size([1, 1, 3, 4])

        3D to 4D:
        >>> to_4d(torch.randn(3, 5, 7)).shape
        torch.Size([1, 3, 5, 7])

        4D to 4D:
        >>> to_4d(torch.randn(16, 3, 5, 7)).shape
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
