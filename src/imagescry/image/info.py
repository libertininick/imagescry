"""Image info module.

This module contains classes for encapsulating image information (e.g. shape, hash, etc.).
"""

from collections.abc import Generator
from os import PathLike
from pathlib import Path

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from imagescry.abstract_array import AbstractArray
from imagescry.image.io import ImageSource, get_image_hash, open_image_source, validate_filepath


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

    @classmethod
    def from_source(cls, source: ImageSource) -> "ImageShape":
        """Read the shape of an image file or buffer.

        Args:
            source (ImageSource): File path, bytes object, or a bytes buffer containing the image data.

        Returns:
            ImageShape: The shape of the image.
        """
        with open_image_source(source) as img:
            return ImageShape(img.height, img.width)


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
        """
        src = validate_filepath(source)
        return cls(source=src, shape=ImageShape.from_source(src), hash=get_image_hash(src))


class ImageInfos(AbstractArray[ImageInfo]):
    """Array of `ImageInfo` objects."""

    ...
