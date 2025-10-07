"""Dynamic speech bubble styling utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from ..utils.textblock import TextBlock

logger = logging.getLogger(__name__)

RgbTuple = Tuple[int, int, int]
RgbaTuple = Tuple[int, int, int, int]


@dataclass
class BubbleRenderStyle:
    """Container describing how to draw a translated speech bubble."""

    fill_rgba: RgbaTuple
    text_rgb: RgbTuple
    outline_rgb: RgbTuple
    outline_width: float
    shadow_rgba: Optional[RgbaTuple]
    shadow_offset: Tuple[float, float]
    padding: Tuple[float, float, float, float]
    corner_radius: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fill_rgba": tuple(self.fill_rgba),
            "text_rgb": tuple(self.text_rgb),
            "outline_rgb": tuple(self.outline_rgb),
            "outline_width": float(self.outline_width),
            "shadow_rgba": tuple(self.shadow_rgba) if self.shadow_rgba else None,
            "shadow_offset": tuple(self.shadow_offset),
            "padding": tuple(self.padding),
            "corner_radius": float(self.corner_radius),
            "reason": self.reason,
        }


def _ensure_bbox_within_image(
    image: np.ndarray, bbox: Sequence[float]
) -> Optional[Tuple[int, int, int, int]]:
    if image is None or bbox is None:
        return None

    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = int(np.clip(np.floor(x0), 0, width))
    y0 = int(np.clip(np.floor(y0), 0, height))
    x1 = int(np.clip(np.ceil(x1), x0 + 1, width))
    y1 = int(np.clip(np.ceil(y1), y0 + 1, height))

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _relative_luminance(rgb: np.ndarray | Iterable[float]) -> float:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim and arr.shape[-1] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    else:
        r, g, b = arr
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def _relative_luminance_srgb(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def _mean_variance_patch(image: np.ndarray, bbox: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = bbox
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return (np.array([127.0, 127.0, 127.0], dtype=np.float32), 0.5, 0.0)

    patch_float = patch.astype(np.float32)
    mean_rgb = patch_float.reshape(-1, patch.shape[-1]).mean(axis=0)
    luminance = _relative_luminance_srgb(patch_float)
    mean_lum = float(luminance.mean()) if luminance.size else 0.5
    variance = float(luminance.var()) if luminance.size else 0.0
    return mean_rgb, mean_lum, variance


def _blend_over(bg_rgb: np.ndarray, fg_rgba: Sequence[int]) -> np.ndarray:
    bg_rgb = np.asarray(bg_rgb, dtype=np.float32)
    fr, fg, fb, fa = fg_rgba
    alpha = fa / 255.0
    blended = np.array(
        [
            fr * alpha + bg_rgb[0] * (1.0 - alpha),
            fg * alpha + bg_rgb[1] * (1.0 - alpha),
            fb * alpha + bg_rgb[2] * (1.0 - alpha),
        ],
        dtype=np.float32,
    )
    return blended


def _contrast_ratio(l1: float, l2: float) -> float:
    bright, dark = max(l1, l2), min(l1, l2)
    return (bright + 0.05) / (dark + 0.05)


def _clip_rgb(rgb: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    arr = np.clip(arr, 0.0, 255.0)
    return arr


def _as_int_tuple(values: np.ndarray | Sequence[float]) -> Tuple[int, int, int]:
    arr = _clip_rgb(values)
    return tuple(int(round(float(v))) for v in arr[:3])


def _normalise_padding(padding: Sequence[float] | float, scale: float) -> Tuple[float, float, float, float]:
    if isinstance(padding, (int, float)):
        value = float(padding) * scale
        return value, value, value, value
    if len(padding) == 2:
        px, py = padding
        return float(px) * scale, float(py) * scale, float(px) * scale, float(py) * scale
    if len(padding) == 4:
        left, top, right, bottom = padding
        return (
            float(left) * scale,
            float(top) * scale,
            float(right) * scale,
            float(bottom) * scale,
        )
    value = float(padding[0]) if padding else 12.0
    value *= scale
    return value, value, value, value


def _select_text_rgb(
    bubble_over_rgb: np.ndarray, text_min_contrast: float
) -> Tuple[RgbTuple, float, str]:
    """Pick a text colour with sufficient contrast against the blended bubble."""

    candidates = [
        np.array([255.0, 255.0, 255.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        255.0 - bubble_over_rgb,
    ]
    best_rgb = candidates[0]
    best_ratio = -1.0
    result_reason = "dynamic"
    bubble_lum = _relative_luminance(_srgb_to_linear(bubble_over_rgb))

    for cand in candidates:
        cand_lum = _relative_luminance(_srgb_to_linear(cand))
        ratio = _contrast_ratio(bubble_lum, cand_lum)
        if ratio > best_ratio:
            best_ratio = ratio
            best_rgb = cand
        if ratio >= text_min_contrast:
            best_rgb = cand
            best_ratio = ratio
            break

    if best_ratio < text_min_contrast:
        result_reason = f"dynamic_low_contrast_{best_ratio:.2f}"

    return _as_int_tuple(best_rgb), best_ratio, result_reason


def _choose_fill_and_text(
    mean_rgb: np.ndarray,
    mean_lum: float,
    variance: float,
    bubble_rgb: Sequence[int],
    dynamic_alpha: int,
    bubble_mode: str,
    plain_alpha: int,
    plain_thresh_hi: float,
    plain_thresh_lo: float,
    flat_var: float,
    text_min_contrast: float,
) -> Tuple[RgbaTuple, RgbTuple, float, str, np.ndarray]:
    """Select bubble fill, text colour, and contrast telemetry."""

    bubble_mode_normalised = (bubble_mode or "auto").lower()
    enable_plain_shortcuts = bubble_mode_normalised == "auto"

    bubble_rgb_arr = np.asarray(bubble_rgb, dtype=np.float32)
    bubble_rgba_dynamic = (*_as_int_tuple(bubble_rgb_arr), int(dynamic_alpha))
    bubble_lum = _relative_luminance(_srgb_to_linear(bubble_rgb_arr))

    if bubble_mode_normalised == "plain":
        fill_rgba = (*_as_int_tuple(bubble_rgb_arr), int(plain_alpha))
        bubble_over = _blend_over(mean_rgb, fill_rgba)
        text_rgb = (0, 0, 0) if bubble_lum >= 0.5 else (255, 255, 255)
        text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
        bubble_lum_post = _relative_luminance(_srgb_to_linear(bubble_over))
        contrast = _contrast_ratio(text_lum, bubble_lum_post)
        return fill_rgba, text_rgb, contrast, "plain_mode", bubble_over

    if enable_plain_shortcuts and dynamic_alpha >= 210:
        if bubble_lum >= plain_thresh_hi:
            bubble_over = _blend_over(mean_rgb, bubble_rgba_dynamic)
            text_rgb = (0, 0, 0)
            text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
            contrast = _contrast_ratio(
                text_lum, _relative_luminance(_srgb_to_linear(bubble_over))
            )
            return bubble_rgba_dynamic, text_rgb, contrast, "forced_black_on_white_bubble", bubble_over
        if bubble_lum <= plain_thresh_lo:
            bubble_over = _blend_over(mean_rgb, bubble_rgba_dynamic)
            text_rgb = (255, 255, 255)
            text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
            contrast = _contrast_ratio(
                text_lum, _relative_luminance(_srgb_to_linear(bubble_over))
            )
            return bubble_rgba_dynamic, text_rgb, contrast, "forced_white_on_black_bubble", bubble_over
        if bubble_lum > 0.5:
            bubble_over = _blend_over(mean_rgb, bubble_rgba_dynamic)
            text_rgb = (0, 0, 0)
            text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
            contrast = _contrast_ratio(
                text_lum, _relative_luminance(_srgb_to_linear(bubble_over))
            )
            return bubble_rgba_dynamic, text_rgb, contrast, "opaque_mid_bubble", bubble_over
        bubble_over = _blend_over(mean_rgb, bubble_rgba_dynamic)
        text_rgb = (255, 255, 255)
        text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
        contrast = _contrast_ratio(
            text_lum, _relative_luminance(_srgb_to_linear(bubble_over))
        )
        return bubble_rgba_dynamic, text_rgb, contrast, "opaque_mid_bubble", bubble_over

    if enable_plain_shortcuts and variance < flat_var:
        if mean_lum >= plain_thresh_hi:
            fill_rgba = (255, 255, 255, int(plain_alpha))
            bubble_over = _blend_over(mean_rgb, fill_rgba)
            text_rgb = (0, 0, 0)
            text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
            contrast = _contrast_ratio(
                text_lum, _relative_luminance(_srgb_to_linear(bubble_over))
            )
            return fill_rgba, text_rgb, contrast, "forced_black_on_plain_white_bg", bubble_over
        if mean_lum <= plain_thresh_lo:
            fill_rgba = (0, 0, 0, int(plain_alpha))
            bubble_over = _blend_over(mean_rgb, fill_rgba)
            text_rgb = (255, 255, 255)
            text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
            contrast = _contrast_ratio(
                text_lum, _relative_luminance(_srgb_to_linear(bubble_over))
            )
            return fill_rgba, text_rgb, contrast, "forced_white_on_plain_black_bg", bubble_over

    bubble_over = _blend_over(mean_rgb, bubble_rgba_dynamic)
    text_rgb, contrast, reason = _select_text_rgb(bubble_over, text_min_contrast)
    return bubble_rgba_dynamic, text_rgb, contrast, reason, bubble_over


def compute_dynamic_bubble_style(
    image: np.ndarray,
    blk: TextBlock,
    bubble_rgb: Sequence[int] = (35, 100, 160),
    min_alpha: int = 110,
    max_alpha: int = 205,
    corner_radius_factor: float = 0.18,
    padding: Sequence[float] | float = (12.0, 8.0),
    text_min_contrast: float = 4.5,
    max_variance_reference: float = 0.04,
    bubble_mode: str = "auto",
    plain_alpha: int = 230,
    plain_thresh_hi: float = 0.88,
    plain_thresh_lo: float = 0.12,
    flat_var: float = 8e-4,
) -> Optional[BubbleRenderStyle]:
    """Compute a dynamic bubble style for a translated text block."""

    if image is None or blk is None:
        return None

    bbox_source = getattr(blk, "bubble_xyxy", None) or getattr(blk, "xyxy", None)
    if bbox_source is None:
        return None

    bbox = _ensure_bbox_within_image(image, bbox_source)
    if bbox is None:
        return None

    min_alpha = int(np.clip(min_alpha, 0, 255))
    max_alpha = int(np.clip(max(max_alpha, min_alpha), 0, 255))
    plain_alpha = int(np.clip(plain_alpha, 0, 255))

    mean_rgb, mean_lum, variance = _mean_variance_patch(image, bbox)

    var_clamped = float(np.clip(variance, 0.0, max_variance_reference))
    t = var_clamped / max_variance_reference if max_variance_reference > 0 else 1.0
    alpha = int(round(min_alpha + (max_alpha - min_alpha) * t))
    alpha = int(np.clip(alpha, min_alpha, max_alpha))

    base_rgb = np.asarray(bubble_rgb, dtype=np.float32)
    mean_rgb = np.asarray(mean_rgb, dtype=np.float32)

    adjust = np.array([20.0 if mean_rgb[0] > base_rgb[0] else -20.0, 0.0, 0.0], dtype=np.float32)
    adjust[1] = 12.0 if mean_rgb[1] < base_rgb[1] else -12.0
    adjusted_rgb = _clip_rgb(base_rgb + adjust)
    adjusted_rgb_tuple = _as_int_tuple(adjusted_rgb)

    fill_rgba, text_rgb, contrast, reason, blended_rgb = _choose_fill_and_text(
        mean_rgb,
        mean_lum,
        variance,
        adjusted_rgb_tuple,
        alpha,
        bubble_mode,
        plain_alpha,
        plain_thresh_hi,
        plain_thresh_lo,
        flat_var,
        text_min_contrast,
    )

    blended_lum = _relative_luminance(_srgb_to_linear(blended_rgb))
    text_linear = _relative_luminance(
        _srgb_to_linear(np.array(text_rgb, dtype=np.float32))
    )
    outline_target = (0, 0, 0) if text_linear > blended_lum else (255, 255, 255)
    outline_rgb = outline_target

    outline_width = 2.0 if fill_rgba[3] >= 150 else 3.0

    shadow_alpha = 120 if text_linear >= blended_lum else 90
    shadow_rgb = (0, 0, 0) if text_linear >= blended_lum else (255, 255, 255)
    shadow_rgba = (*shadow_rgb, shadow_alpha)
    shadow_offset = (0.0, 1.0)

    x0, y0, x1, y1 = bbox
    width = float(x1 - x0)
    height = float(y1 - y0)
    scale = max(1.0, 0.08 * np.sqrt(width * height) / 50.0)
    pad_l, pad_t, pad_r, pad_b = _normalise_padding(padding, scale)
    corner_radius = max(6.0, min(width, height) * corner_radius_factor)

    logger.info(
        "bubble_style mode=%s reason=%s variance=%.6f alpha=%d contrast=%.2f",
        bubble_mode,
        reason,
        variance,
        fill_rgba[3],
        contrast,
    )
    if (
        bubble_mode.lower() == "auto"
        and not reason.startswith("dynamic")
        and variance >= 0.0015
    ):
        logger.warning(
            "auto bubble fallback triggered on busy patch (variance=%.6f, reason=%s)",
            variance,
            reason,
        )

    return BubbleRenderStyle(
        fill_rgba=fill_rgba,
        text_rgb=text_rgb,
        outline_rgb=outline_rgb,
        outline_width=outline_width,
        shadow_rgba=shadow_rgba,
        shadow_offset=shadow_offset,
        padding=(pad_l, pad_t, pad_r, pad_b),
        corner_radius=corner_radius,
        reason=reason,
    )


__all__ = [
    "BubbleRenderStyle",
    "compute_dynamic_bubble_style",
]
