"""Geometry tools."""

import torch
from affine import Affine
from jaxtyping import Int64, jaxtyped
from rasterio import features
from shapely.geometry import Polygon
from torch import Tensor

from imagescry.typechecking import typechecker


@jaxtyped(typechecker=typechecker)
def create_roi_mask(
    roi: Polygon | list[Polygon],
    original_image_shape: tuple[int, int] | torch.Size,
    feature_map_shape: tuple[int, int] | torch.Size,
    class_index: int = 1,
) -> Int64[Tensor, "H W"]:
    """Create a mask of the region of interest in the feature map.

    Args:
        roi (Polygon | list[Polygon]): Polygon(s) representing region of interest on the original image
        original_image_shape (tuple[int, int] | torch.Size): (height, width) of image the roi is defined on
        feature_map_shape (tuple[int, int] | torch.Size): (height, width) of feature map to rasterize the roi(s) onto
        class_index (int): Class index to fill the mask with

    Returns:
        Int64[Tensor, 'H W']: Mask of the regions of interest in the feature map.

    Returns:
        Int64[Tensor, 'H W']: Mask of the regions of interest in the feature map

    Example:
        >>> roi = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
        >>> original_image_shape = (6, 8)
        >>> feature_map_shape = (3, 4)
        >>> mask = create_roi_mask(roi, original_image_shape, feature_map_shape)
        >>> mask
        tensor([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0]])

    """
    # Unpack spatial dimensions
    h, w = original_image_shape
    hf, wf = feature_map_shape

    # Scaling transform from feature map back to original image
    scale_transform = Affine.scale(w / wf, h / hf)

    # Region of interest mask on feature map
    mask = features.rasterize(
        shapes=[roi] if isinstance(roi, Polygon) else roi,
        out_shape=(hf, wf),
        transform=scale_transform,
        fill=0,
        all_touched=True,
    )
    mask = torch.from_numpy(mask).to(torch.int64)

    # Fill with class index
    mask *= class_index

    return mask
