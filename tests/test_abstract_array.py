"""Tests for AbstractArray class."""

from dataclasses import dataclass

import pytest
from pytest_check import check

from imagescry.abstract_array import AbstractArray


@dataclass
class Point:
    """Simple point class for testing."""

    x: int
    y: int


class Points(AbstractArray[Point]):
    """Concrete implementation of AbstractArray for testing."""

    pass


# Fixtures
@pytest.fixture
def empty_points() -> Points:
    """Create an empty Points array."""
    return Points([])


@pytest.fixture
def single_point() -> Points:
    """Create a Points array with a single point."""
    return Points([Point(1, 2)])


@pytest.fixture
def multiple_points() -> Points:
    """Create a Points array with multiple points."""
    return Points([Point(1, 2), Point(3, 4), Point(5, 6), Point(7, 8), Point(9, 10)])


# Tests
def test_initialization() -> None:
    """Test initialization with different types of iterables."""
    # Empty iterable
    points = Points([])
    check.equal(len(points), 0)

    # List of Points
    points = Points([Point(1, 2), Point(3, 4)])
    check.equal(len(points), 2)

    # Tuple of Points
    points = Points((Point(1, 2), Point(3, 4)))
    check.equal(len(points), 2)

    # Generator of Points
    points = Points(Point(x, y) for x in range(1, 3) for y in range(1, 3))
    check.equal(len(points), 4)


def test_magic_methods(multiple_points: Points) -> None:
    """Test core magic methods: __len__, __iter__, __repr__."""
    # Test __len__
    check.equal(len(multiple_points), 5)

    # Test __iter__
    points_list = list(multiple_points)
    check.equal(len(points_list), 5)
    check.equal(points_list[0], Point(1, 2))

    # Test __repr__ with truncation
    repr_str = repr(multiple_points)
    check.is_true("Points([" in repr_str)
    check.is_true("Point(x=1, y=2)" in repr_str)
    check.is_true("... (0 more)" not in repr_str)  # Should show all 5 points


def test_single_integer_indexing(multiple_points: Points) -> None:
    """Test single integer indexing."""
    # Test positive indexing
    check.equal(multiple_points[0], Point(1, 2))
    check.equal(multiple_points[2], Point(5, 6))

    # Test negative indexing
    check.equal(multiple_points[-1], Point(9, 10))
    check.equal(multiple_points[-2], Point(7, 8))

    # Test out of bounds indexing
    with pytest.raises(IndexError):
        _ = multiple_points[5]
    with pytest.raises(IndexError):
        _ = multiple_points[-6]


def test_slicing(multiple_points: Points) -> None:
    """Test slicing operations."""
    # Test basic slicing
    sliced = multiple_points[1:3]
    check.equal(len(sliced), 2)
    check.equal(sliced[0], Point(3, 4))
    check.equal(sliced[1], Point(5, 6))

    # Test step slicing
    sliced = multiple_points[::2]
    check.equal(len(sliced), 3)
    check.equal(sliced[0], Point(1, 2))
    check.equal(sliced[1], Point(5, 6))

    # Test negative slicing
    sliced = multiple_points[-3:-1]
    check.equal(len(sliced), 2)
    check.equal(sliced[0], Point(5, 6))
    check.equal(sliced[1], Point(7, 8))

    # Test empty slice
    sliced = multiple_points[1:1]
    check.equal(len(sliced), 0)


def test_boolean_indexing(multiple_points: Points) -> None:
    """Test boolean mask indexing."""
    # Test valid boolean mask
    mask = [True, False, True, False, True]
    filtered = multiple_points[mask]
    check.equal(len(filtered), 3)
    check.equal(filtered[0], Point(1, 2))
    check.equal(filtered[1], Point(5, 6))
    check.equal(filtered[2], Point(9, 10))

    # Test boolean mask too short
    with pytest.raises(IndexError):
        _ = multiple_points[[True, False]]

    # Test boolean mask too long
    with pytest.raises(IndexError):
        _ = multiple_points[[True, False, True, False, True, True]]


def test_fancy_indexing(multiple_points: Points) -> None:
    """Test integer sequence indexing."""
    # Test empty index sequence
    selected = multiple_points[[]]
    check.equal(len(selected), 0)

    # Test valid indices
    indices = [0, 2, 4]
    selected = multiple_points[indices]
    check.equal(len(selected), 3)
    check.equal(selected[0], Point(1, 2))
    check.equal(selected[1], Point(5, 6))
    check.equal(selected[2], Point(9, 10))

    # Test duplicate indices
    indices = [0, 0, 2]
    selected = multiple_points[indices]
    check.equal(len(selected), 3)
    check.equal(selected[0], Point(1, 2))
    check.equal(selected[1], Point(1, 2))
    check.equal(selected[2], Point(5, 6))

    # Test out of bounds indices raises IndexError
    with pytest.raises(IndexError):
        _ = multiple_points[[0, 5]]

    # Test float indices raises TypeError
    with pytest.raises(TypeError):
        _ = multiple_points[[0.0, 2.0, 4.0]]


def test_append(empty_points: Points) -> None:
    """Test appending items."""
    # Test appending single item
    empty_points.append(Point(1, 2))
    check.equal(len(empty_points), 1)
    check.equal(empty_points[0], Point(1, 2))

    # Test invalid item type raises TypeError
    with pytest.raises(TypeError):
        empty_points.append((1, 2))  # Should fail with tuple instead of Point


def test_extend(empty_points: Points) -> None:
    """Test extending with iterables."""
    # Test extending with list
    empty_points.extend([Point(1, 2), Point(3, 4)])
    check.equal(len(empty_points), 2)
    check.equal(empty_points[0], Point(1, 2))
    check.equal(empty_points[1], Point(3, 4))

    # Test extending with empty iterable
    empty_points.extend([])
    check.equal(len(empty_points), 2)

    # Test invalid item type raises TypeError
    with pytest.raises(TypeError):
        empty_points.extend([(1, 2), (3, 4)])


def test_batch(multiple_points: Points) -> None:
    """Test batching operations."""
    # Test batching with size 2
    batches = multiple_points.batch(2)
    check.equal(len(batches), 3)
    check.equal(len(batches[0]), 2)
    check.equal(len(batches[1]), 2)
    check.equal(len(batches[2]), 1)

    # Test batching with size > len
    batches = multiple_points.batch(10)
    check.equal(len(batches), 1)
    check.equal(len(batches[0]), 5)

    # Test batching with size = 1
    batches = multiple_points.batch(1)
    check.equal(len(batches), 5)
    check.equal(len(batches[0]), 1)


def test_filter(multiple_points: Points) -> None:
    """Test filtering operations."""
    # Test filtering with predicate
    filtered = multiple_points.filter(lambda p: p.x > 3)
    check.equal(len(filtered), 3)
    check.equal(filtered[0], Point(5, 6))

    # Test filtering with always True predicate
    filtered = multiple_points.filter(lambda _: True)
    check.equal(len(filtered), 5)

    # Test filtering with always False predicate
    filtered = multiple_points.filter(lambda _: False)
    check.equal(len(filtered), 0)


def test_sort(multiple_points: Points) -> None:
    """Test sorting operations."""
    # Test sorting by x coordinate
    sorted_points = multiple_points.sort(lambda p: p.x)
    check.equal(sorted_points[0], Point(1, 2))
    check.equal(sorted_points[-1], Point(9, 10))

    # Test sorting in reverse
    sorted_points = multiple_points.sort(lambda p: p.x, reverse=True)
    check.equal(sorted_points[0], Point(9, 10))
    check.equal(sorted_points[-1], Point(1, 2))


def test_take(multiple_points: Points) -> None:
    """Test take operation."""
    # Test taking specific indices
    taken = multiple_points.take([0, 2, 4])
    check.equal(len(taken), 3)
    check.equal(taken[0], Point(1, 2))
    check.equal(taken[1], Point(5, 6))
    check.equal(taken[2], Point(9, 10))

    # Test taking duplicate indices
    taken = multiple_points.take([0, 0, 2])
    check.equal(len(taken), 3)
    check.equal(taken[0], Point(1, 2))
    check.equal(taken[1], Point(1, 2))
    check.equal(taken[2], Point(5, 6))

    # Test taking out of bounds indices raises
    with pytest.raises(IndexError):
        _ = multiple_points.take([0, 5])


def test_operation_chaining(multiple_points: Points) -> None:
    """Test chaining operations and complex combinations."""
    # Chain multiple operations
    result = multiple_points.filter(lambda p: p.x > 3).sort(lambda p: p.x, reverse=True).take([0, 1])

    check.equal(len(result), 2)
    check.equal(result[0], Point(9, 10))
    check.equal(result[1], Point(7, 8))


def test_batching_and_filtering(multiple_points: Points) -> None:
    """Test batching and filtering operations."""
    # Batch into groups of 2
    batches = multiple_points.batch(2)
    check.equal(len(batches), 3)  # Total of three batches

    # Filter each batch
    filtered_batches = [batch.filter(lambda p: p.x > 3) for batch in batches]
    check.equal(len(filtered_batches[0]), 0)  # First batch has no points with x > 3
    check.equal(len(filtered_batches[1]), 2)  # Second batch has two point with x > 3
    check.equal(len(filtered_batches[2]), 1)  # Third batch has one point with x > 3
