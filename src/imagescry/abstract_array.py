"""Abstract object array class.

This module provides an abstract array class that can be used to create
homogeneous collections of objects.

The class supports NumPy-like advanced indexing and slicing operations.

The class is generic and can be used to create arrays of any type.

Examples:
    Define a custom class
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    ...     def __repr__(self) -> str:
    ...         return f"Point(x={self.x}, y={self.y})"

    Define an array container for the custom class
    >>> class Points(AbstractArray[Point]): ...

    Create an array of points
    >>> points = Points([Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)])

    Indexing
    >>> points[0]
    Point(x=1, y=2)

    Slicing
    >>> points[1:3]
    Points([Point(x=3, y=4), Point(x=5, y=6)])

    Boolean indexing
    >>> points[[True, False, True]]
    Points([Point(x=1, y=2), Point(x=5, y=6)])

    Non-contiguous indexing
    >>> points[[0, 2]]
    Points([Point(x=1, y=2), Point(x=5, y=6)])

    Filtering
    >>> points.filter(lambda point: point.x > 2)
    Points([Point(x=3, y=4), Point(x=5, y=6)])

    Batching
    >>> points.batch(2)
    [Points([Point(x=1, y=2), Point(x=3, y=4)]), Points([Point(x=5, y=6)])]

    Sorting
    >>> points.sort(lambda point: point.x, reverse=True)
    Points([Point(x=5, y=6), Point(x=3, y=4), Point(x=1, y=2)])

"""

import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol, TypeGuard, TypeVar, overload

from more_itertools import chunked

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class SupportsRichComparison(Protocol[T_contra]):
    """Protocol for objects that support rich comparison operations."""

    def __lt__(self, other: T_contra) -> bool:
        """Less than comparison."""
        ...

    def __le__(self, other: T_contra) -> bool:
        """Less than or equal to comparison."""
        ...

    def __gt__(self, other: T_contra) -> bool:
        """Greater than comparison."""
        ...

    def __ge__(self, other: T_contra) -> bool:
        """Greater than or equal to comparison."""
        ...


class AbstractArray[T](Sequence):
    """An indexable and sliceable collection for homogeneous objects.

    Supports NumPy-like advanced indexing and slicing operations:
    - Single integer index
    - Slice
    - List of indices for fancy indexing
    - Boolean mask
    """

    def __init__(self, items: Iterable[T]) -> None:
        """Initialize the collection with an iterable of items."""
        self._items: list[T] = list(items)

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self._items)

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> "AbstractArray[T]": ...

    @overload
    def __getitem__(self, key: Sequence[int] | Sequence[bool]) -> "AbstractArray[T]": ...

    def __getitem__(self, key: int | slice | Sequence[int] | Sequence[bool]) -> T | "AbstractArray[T]":
        """Support flexible indexing similar to NumPy arrays."""
        if isinstance(key, int):
            return self._items[key]

        elif isinstance(key, slice):
            return self.__class__(self._items[key])

        elif isinstance(key, Sequence):
            return self._get_item_sequence(key)

        else:
            raise TypeError(f"Invalid index type: {type(key)}")  # pragma: no cover

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the items."""
        return iter(self._items)

    def __repr__(self) -> str:
        """Return a string representation of the collection."""
        item_reprs = ", ".join(repr(item) for item in itertools.islice(self._items, 5))
        if len(self) > 5:
            item_reprs += f", ... ({len(self) - 5} more)"
        return f"{self.__class__.__name__}([{item_reprs}])"

    def append(self, item: T) -> None:
        """Add an item to the end of the collection."""
        self._items.append(item)

    def batch(self, batch_size: int) -> list["AbstractArray[T]"]:
        """Split the collection into batches of specified size."""
        return [self.__class__(batch) for batch in chunked(self._items, batch_size)]

    def extend(self, items: Iterable[T]) -> None:
        """Add multiple items to the end of the collection."""
        self._items.extend(items)

    def filter(self, predicate: Callable[[T], bool]) -> "AbstractArray[T]":
        """Return a new collection with items that satisfy the predicate."""
        return self.__class__(item for item in self._items if predicate(item))

    def sort(self, key: Callable[[T], SupportsRichComparison], *, reverse: bool = False) -> "AbstractArray[T]":
        """Return a new collection with items sorted by the key function."""
        return self.__class__(sorted(self._items, key=key, reverse=reverse))

    def take(self, indices: Sequence[int]) -> "AbstractArray[T]":
        """Take items at the specified indices."""
        return self.__class__(self._items[i] for i in indices)

    def _get_item_sequence(self, key: Sequence[int] | Sequence[bool]) -> "AbstractArray[T]":
        """Get a sequence of items from the collection using a boolean mask or integer index."""
        if _is_boolean_mask(key):
            if len(key) != len(self):
                raise IndexError("Boolean index array should have same length as collection")
            return self.__class__(item for item, include in zip(self._items, key, strict=True) if include)

        elif _is_integer_index(key):
            return self.__class__(self._items[i] for i in key)

        else:
            raise TypeError(f"Invalid index sequence type: {type(key)}")  # pragma: no cover


def _is_boolean_mask(key: Sequence[int | bool]) -> TypeGuard[Sequence[bool]]:
    """Check if a sequence is a boolean mask."""
    return all(isinstance(x, bool) for x in key)


def _is_integer_index(key: Sequence[int | bool]) -> TypeGuard[Sequence[int]]:
    """Check if a sequence is an integer index."""
    return all(isinstance(x, int) for x in key)
