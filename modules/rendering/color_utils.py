from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PySide6.QtGui import QColor


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


def _relative_luminance(rgb: Sequence[float]) -> float:
    """Compute relative luminance for RGB values in range [0, 1]."""
    def channel_lum(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (channel_lum(float(c)) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(l1: float, l2: float) -> float:
    """Return WCAG contrast ratio between two relative luminance values."""
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def determine_text_outline_colors(
    image: Optional[np.ndarray],
    bbox: Sequence[float],
    fallback_text: Optional[QColor] = None,
    fallback_outline: Optional[QColor] = None,
) -> Tuple[QColor, QColor]:
    """Determine contrasting text and outline colors for a bounding box."""
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

    region_float = region.astype(np.float32) / 255.0
    mean_rgb = region_float.reshape(-1, region_float.shape[2]).mean(axis=0)

    bg_luminance = _relative_luminance(mean_rgb)

    candidates = [QColor("#000000"), QColor("#FFFFFF")]
    best_text = default_text
    best_contrast = -1.0

    for candidate in candidates:
        candidate_lum = _relative_luminance(
            (candidate.redF(), candidate.greenF(), candidate.blueF())
        )
        contrast = _contrast_ratio(bg_luminance, candidate_lum)
        if contrast > best_contrast:
            best_contrast = contrast
            best_text = candidate

    if best_contrast < 0:
        best_text = default_text

    outline = QColor("#FFFFFF") if best_text.name().lower() == "#000000" else QColor("#000000")

    return best_text, outline
