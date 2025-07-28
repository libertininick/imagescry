"""Base class for storage module."""

from abc import ABC
from pathlib import Path
from typing import Any, TypeVar

from sqlmodel import Field, SQLModel, String, TypeDecorator


class BaseStorageModel(ABC, SQLModel):
    """Base model for all storage models in the application.

    This class provides a common interface for all storage models, ensuring they use an integer ID for primary keys.

    Attributes:
        id (int | None): Primary key, auto-incremented. This field is inherited by all subclasses and serves as the
            unique identifier for each record.
    """

    id: int | None = Field(default=None, primary_key=True)


StorageModel = TypeVar("StorageModel", bound=BaseStorageModel)


class PathType(TypeDecorator):
    """Custom SQLAlchemy type for python Path objects."""

    impl = String
    cache_ok = True

    @staticmethod
    def process_bind_param(value: Path | None, _: Any) -> str | None:
        """Convert Path to string when storing."""
        if value is not None:
            return str(value)
        return value

    @staticmethod
    def process_result_value(value: str | None, _: Any) -> Path | None:
        """Convert string back to Path when loading."""
        if value is not None:
            return Path(value)
        return value
