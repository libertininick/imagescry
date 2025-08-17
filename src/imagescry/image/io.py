"""Image I/O module.

This module contains functions for image I/O operations.
"""

import base64
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import md5
from io import BytesIO
from os import PathLike
from pathlib import Path

import torch
from jaxtyping import UInt8, jaxtyped
from PIL import Image
from PIL.ImageFile import ImageFile
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

from imagescry.typechecking import typechecker

ImageSource = str | PathLike | bytes | BytesIO


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
        UInt8[Tensor, '3 H W']: Image as a RGB tensor with shape [3, H, W] a&nd integer values in the range [0, 255]
    """
    with open_image_source(image_source) as img:
        return pil_to_tensor(img.convert("RGB")).to(device=device)


@jaxtyped(typechecker=typechecker)
def read_image_as_grayscale_tensor(
    image_source: ImageSource, device: torch.device | None = None
) -> UInt8[Tensor, "1 H W"]:
    """Read an image file or buffer, and convert it to a grayscale PyTorch tensor.

    Args:
        image_source (ImageSource): File path, bytes object, or a bytes buffer containing the image data.
        device (torch.device | None, optional): The device to put the tensor on. Defaults to None, which uses CPU.

    Returns:
        UInt8[Tensor, '1 H W']: Image as a grayscale tensor with shape [1, H, W] & integer values in the range [0, 255]
    """
    with open_image_source(image_source) as img:
        return pil_to_tensor(img.convert("L")).to(device=device)


def read_image_and_encode(image_source: ImageSource) -> str:
    """Read an image file or buffer, and encode it as a base64 string (suitable for HTML embedding).

    Args:
        image_source (ImageSource): File path, bytes object, or a bytes buffer containing the image data.

    Returns:
        str: Image encoded as a base64 string.
    """
    with open_image_source(image_source) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"


def validate_filepath(filepath: str | PathLike) -> Path:
    """Validate that a filepath exists and is a file.

    Args:
        filepath (str | PathLike): The filepath to validate.

    Returns:
        Path: The absolute, validated filepath.

    Raises:
        FileNotFoundError: If the filepath does not exist or is not a file.
    """
    fp = Path(filepath).absolute()
    if not (fp.exists() and fp.is_file()):
        raise FileNotFoundError(f"{fp} does not exist or is not a file")
    return fp
