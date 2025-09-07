"""Test fixtures for the testing the ImageScry Image module."""

from io import BytesIO
from pathlib import Path

import pytest
import torch
from jaxtyping import UInt8, jaxtyped
from pytest import FixtureRequest, TempPathFactory
from torch import Tensor
from torchvision.io import write_png

from imagescry.image.info import ImageShape
from imagescry.typechecking import typechecker

# Seed
SEED = 1234
torch.manual_seed(SEED)


# Fixtures
@pytest.fixture(scope="session")
def image_shape() -> ImageShape:
    """Create a image shape test fixture."""
    return ImageShape(30, 45)


@pytest.fixture(scope="session")
@jaxtyped(typechecker=typechecker)
def image_tensor(image_shape: ImageShape) -> UInt8[Tensor, "3 {image_shape.height} {image_shape.width}"]:
    """Create a uint8 image tensor test fixture."""
    torch.manual_seed(1234)
    return torch.randint(low=0, high=256, size=(3, *image_shape), dtype=torch.uint8)


@pytest.fixture(scope="session")
def image_source_file(image_tensor: UInt8[Tensor, "3 30 45"], tmp_path_factory: TempPathFactory) -> Path:
    """Create a test image source file."""
    temp_file = tmp_path_factory.mktemp("images") / "test.png"
    write_png(image_tensor, str(temp_file))
    return temp_file


@pytest.fixture(scope="session")
def image_source_bytes(image_source_file: Path) -> bytes:
    """Create a test image source bytes."""
    with image_source_file.open("rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def image_source_buffer(image_source_bytes: bytes) -> BytesIO:
    """Create a test image source buffer."""
    return BytesIO(image_source_bytes)


@pytest.fixture(params=["image_source_file", "image_source_bytes", "image_source_buffer"])
def image_source(request: FixtureRequest) -> Path | bytes | BytesIO:
    """Create a test image source."""
    # Get the fixture dynamically by name
    return request.getfixturevalue(request.param)
