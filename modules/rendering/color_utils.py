from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PySide6.QtGui import QColor

from .color_model import BubbleColorPrediction, predict_bubble_text_color

def _clamp_bbox_to_image(
    bbox: Sequence[float], image_shape: Tuple[int, ...], inset_ratio: float = 0.1
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a bounding box to image bounds and optionally shrink it slightly."""
    if image_shape is None or len(image_shape) < 2 or bbox is None:
        return None

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return None

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    inset_x = int(round((x2 - x1) * inset_ratio / 2.0))
    inset_y = int(round((y2 - y1) * inset_ratio / 2.0))

    x1 += inset_x
    y1 += inset_y
    x2 -= inset_x
    y2 -= inset_y

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def determine_text_outline_colors(
    image: Optional[np.ndarray],
    bbox: Sequence[float],
    fallback_text: Optional[QColor] = None,
    fallback_outline: Optional[QColor] = None,
) -> Tuple[QColor, QColor]:
    """Determine contrasting text and outline colors for a bounding box.

    Instead of relying purely on luminance heuristics, this routine now samples
    the bubble background, summarises its colour distribution, and feeds those
    features into a lightweight logistic model that was fitted on hand-labelled
    manhwa bubbles. The model predicts whether light or dark typography provides
    better contrast given the sampled region and the appropriate outline colour
    is paired automatically. If sampling or inference fails we gracefully fall
    back to the provided defaults.
    """
    default_text = fallback_text if fallback_text is not None else QColor("#000000")
    default_outline = fallback_outline if fallback_outline is not None else QColor("#FFFFFF")

    if image is None:
        return default_text, default_outline

    coords = _clamp_bbox_to_image(bbox, image.shape)
    if coords is None:
        return default_text, default_outline

    x1, y1, x2, y2 = coords
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return default_text, default_outline

    if region.ndim == 2:
        region = np.repeat(region[:, :, None], 3, axis=2)
    elif region.shape[2] > 3:
        region = region[:, :, :3]

    prediction: Optional[BubbleColorPrediction] = None
    try:
        prediction = predict_bubble_text_color(region)
    except Exception:
        prediction = None

    if prediction is None:
        return default_text, default_outline

    light_text = QColor("#FFFFFF")
    dark_text = default_text
    light_outline = default_outline
    dark_outline = QColor("#000000")

    if prediction.use_light_text:
        text_color = light_text
        outline = dark_outline
    else:
        text_color = dark_text
        outline = light_outline

    return text_color, outline
