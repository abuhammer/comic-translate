"""Dynamic text styling utilities for adaptive speech overlays."""

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
    """Container describing how to draw translated text with optional background box."""

    fill_rgba: RgbaTuple
    text_rgb: RgbTuple
    text_alpha: int
    outline_rgb: RgbTuple
    outline_alpha: int
    outline_width: float
    shadow_rgba: Optional[RgbaTuple]
    shadow_offset: Tuple[float, float]
    padding: Tuple[float, float, float, float]
    corner_radius: float
    reason: str = ""
    fill_gradient: Optional[dict] = None

    def to_dict(self) -> dict:
        data = {
            "fill_rgba": tuple(self.fill_rgba),
            "text_rgb": tuple(self.text_rgb),
            "text_alpha": int(self.text_alpha),
            "outline_rgb": tuple(self.outline_rgb),
            "outline_alpha": int(self.outline_alpha),
            "outline_width": float(self.outline_width),
            "shadow_rgba": tuple(self.shadow_rgba) if self.shadow_rgba else None,
            "shadow_offset": tuple(self.shadow_offset),
            "padding": tuple(self.padding),
            "corner_radius": float(self.corner_radius),
            "reason": self.reason,
        }
        if self.fill_gradient:
            gradient_dict = dict(self.fill_gradient)
            for key in ("start_rgba", "end_rgba"):
                if key in gradient_dict and gradient_dict[key] is not None:
                    gradient_dict[key] = tuple(int(v) for v in gradient_dict[key])
            data["fill_gradient"] = gradient_dict
        return data


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


def _mean_variance_patch(
    image: np.ndarray, bbox: Tuple[int, int, int, int], sample_limit: int = 16
):
    x0, y0, x1, y1 = bbox
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return (np.array([127.0, 127.0, 127.0], dtype=np.float32), 0.5, 0.0)

    if patch.ndim == 2:
        patch = np.expand_dims(patch, axis=-1)

    if sample_limit > 0:
        step_y = max(1, patch.shape[0] // sample_limit)
        step_x = max(1, patch.shape[1] // sample_limit)
        patch = patch[::step_y, ::step_x]

    patch_float = patch.astype(np.float32)
    mean_rgb = patch_float.reshape(-1, patch.shape[-1]).mean(axis=0)
    luminance = _srgb_to_linear(patch_float)
    luminance = (
        0.2126 * luminance[..., 0]
        + 0.7152 * luminance[..., 1]
        + 0.0722 * luminance[..., 2]
    )
    mean_lum = float(luminance.mean()) if luminance.size else 0.5
    variance = float(luminance.var()) if luminance.size else 0.0
    return mean_rgb, mean_lum, variance


def _clip_rgb(rgb: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return np.clip(arr, 0.0, 255.0)


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


def _contrast_ratio(l1: float, l2: float) -> float:
    bright, dark = max(l1, l2), min(l1, l2)
    return (bright + 0.05) / (dark + 0.05)


def _extract_bbox(candidate: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if candidate is None:
        return None

    try:
        arr = np.asarray(candidate, dtype=np.float32)
    except Exception:
        return None

    if arr.size < 4:
        return None

    flat = arr.reshape(-1)
    x0, y0, x1, y1 = (float(flat[i]) for i in range(4))
    return x0, y0, x1, y1


def _normalise_ratio_value(value: float) -> float:
    value = float(value)
    if value > 1.0:
        value /= 255.0
    return float(np.clip(value, 0.0, 1.0))


def compute_dynamic_bubble_style(
    image: np.ndarray,
    blk: TextBlock,
    *,
    bubble_rgb: Sequence[int] = (35, 100, 160),
    background_box_mode: str = "off",
    background_box_opacity: float = 0.25,
    text_color_mode: str = "auto",
    custom_text_rgb: Optional[Sequence[int]] = None,
    text_opacity: float = 1.0,
    stroke_enabled: bool = False,
    stroke_width: float = 2.0,
    stroke_opacity: float = 1.0,
    auto_contrast: bool = True,
    text_min_contrast: float = 4.5,
    background_plain_hi: float = 0.95,
    background_plain_lo: float = 0.05,
    flat_variance_threshold: float = 4e-4,
    auto_stroke_opacity: float = 0.6,
) -> Optional[BubbleRenderStyle]:
    """Compute the adaptive text styling for a block using local background samples."""

    if image is None or blk is None:
        return None

    bbox_source = _extract_bbox(getattr(blk, "bubble_xyxy", None))
    if bbox_source is None:
        bbox_source = _extract_bbox(getattr(blk, "xyxy", None))
    if bbox_source is None:
        return None

    bbox = _ensure_bbox_within_image(image, bbox_source)
    if bbox is None:
        return None

    mean_rgb, mean_lum, variance = _mean_variance_patch(image, bbox)
    mean_rgb = np.asarray(mean_rgb, dtype=np.float32)
    lum_std = float(np.sqrt(max(variance, 0.0)))
    flat_std_threshold = float(np.sqrt(max(flat_variance_threshold, 0.0)))

    mode = (text_color_mode or "auto").lower()
    reason = "auto"
    plain_background: Optional[str] = None
    auto_stroke_required = False
    auto_shadow_required = False
    stroke_source = "user" if stroke_enabled else "none"

    def _to_rgb_tuple(candidate: Optional[Sequence[int]], fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if candidate is None:
            return fallback
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.size < 3:
            return fallback
        return _as_int_tuple(arr[:3])

    def _contrast_for_rgb(rgb: Tuple[int, int, int]) -> float:
        lum = _relative_luminance(_srgb_to_linear(np.array(rgb, dtype=np.float32)))
        return _contrast_ratio(mean_lum, lum)

    is_plain_white = auto_contrast and mean_lum >= background_plain_hi and lum_std <= flat_std_threshold
    is_plain_black = auto_contrast and mean_lum <= background_plain_lo and lum_std <= flat_std_threshold

    if mode == "black":
        text_rgb = (0, 0, 0)
        reason = "user_black"
    elif mode == "white":
        text_rgb = (255, 255, 255)
        reason = "user_white"
    elif mode == "custom":
        text_rgb = _to_rgb_tuple(custom_text_rgb, (0, 0, 0))
        reason = "user_custom"
    else:
        if is_plain_white:
            text_rgb = (0, 0, 0)
            reason = "plain_white_bg"
            plain_background = "white"
        elif is_plain_black:
            text_rgb = (255, 255, 255)
            reason = "plain_black_bg"
            plain_background = "black"
        elif auto_contrast:
            best_color = (0, 0, 0)
            best_label = "black"
            best_ratio = -1.0
            met_target = False
            for cand_rgb, label in (((0, 0, 0), "black"), ((255, 255, 255), "white")):
                ratio = _contrast_for_rgb(cand_rgb)
                if ratio >= text_min_contrast and not met_target:
                    best_color = cand_rgb
                    best_label = label
                    best_ratio = ratio
                    met_target = True
                    break
                if ratio > best_ratio:
                    best_color = cand_rgb
                    best_label = label
                    best_ratio = ratio

            text_rgb = best_color
            if best_ratio >= text_min_contrast:
                reason = f"dynamic_{best_label}_{best_ratio:.2f}"
            else:
                reason = f"dynamic_needs_outline_{best_ratio:.2f}"
                auto_stroke_required = True
                auto_shadow_required = True
        else:
            text_rgb = (0, 0, 0)
            reason = "auto_contrast_disabled"

    text_opacity = _normalise_ratio_value(text_opacity)
    text_alpha_value = int(np.clip(round(text_opacity * 255.0), 0, 255))

    # For user-forced colours, honour the choice but still evaluate contrast if auto_contrast is enabled.
    if mode in {"black", "white", "custom"} and auto_contrast:
        ratio = _contrast_for_rgb(text_rgb)
        if ratio < text_min_contrast:
            auto_stroke_required = True
            auto_shadow_required = True
            reason = f"{reason}_needs_outline_{ratio:.2f}"
        else:
            reason = f"{reason}_{ratio:.2f}"

    text_lum = _relative_luminance(_srgb_to_linear(np.array(text_rgb, dtype=np.float32)))
    stroke_color = (0, 0, 0) if text_lum >= 0.5 else (255, 255, 255)

    stroke_active = bool(stroke_enabled)
    if plain_background and auto_contrast:
        stroke_active = True
        stroke_source = "plain"
    elif auto_stroke_required:
        stroke_active = True
        stroke_source = "auto"

    stroke_opacity = _normalise_ratio_value(stroke_opacity)
    auto_stroke_opacity = _normalise_ratio_value(auto_stroke_opacity)

    stroke_alpha_value = 0
    outline_width_value = 0.0

    if stroke_active:
        if stroke_source == "auto":
            base_opacity = auto_stroke_opacity
            outline_width_value = max(float(stroke_width), 2.0)
        elif stroke_source == "plain":
            base_opacity = auto_stroke_opacity if not stroke_enabled else max(stroke_opacity, auto_stroke_opacity)
            outline_width_value = max(float(stroke_width), 1.5)
        else:
            base_opacity = stroke_opacity
            outline_width_value = float(stroke_width)

        stroke_alpha_value = int(np.clip(round(base_opacity * 255.0), 0, 255))
        if stroke_alpha_value <= 0:
            outline_width_value = 0.0
        elif outline_width_value <= 0.0:
            outline_width_value = 1.5 if stroke_source in {"auto", "plain"} else 1.0

    shadow_rgba: Optional[RgbaTuple] = None
    if auto_shadow_required and stroke_alpha_value > 0 and not plain_background:
        shadow_alpha = int(np.clip(round(auto_stroke_opacity * 255.0 * 0.5), 0, 255))
        if shadow_alpha > 0:
            shadow_rgba = (stroke_color[0], stroke_color[1], stroke_color[2], shadow_alpha)

    bg_lum = _relative_luminance(_srgb_to_linear(mean_rgb))
    contrast_ratio = _contrast_ratio(bg_lum, text_lum)

    fill_rgba: RgbaTuple = (0, 0, 0, 0)
    background_mode = (background_box_mode or "off").lower()
    opacity_cap = float(np.clip(background_box_opacity, 0.0, 0.3))
    bubble_rgb_tuple = _as_int_tuple(bubble_rgb)

    if background_mode == "on":
        fill_rgba = (*bubble_rgb_tuple, int(round(opacity_cap * 255.0)))
        reason = f"{reason}_box_on"
    elif background_mode == "auto" and contrast_ratio < 3.0:
        fill_rgba = (*bubble_rgb_tuple, int(round(opacity_cap * 255.0)))
        reason = f"{reason}_box_auto"

    if fill_rgba[3] <= 0:
        padding = (0.0, 0.0, 0.0, 0.0)
        corner_radius = 0.0
    else:
        x0, y0, x1, y1 = bbox
        width = float(x1 - x0)
        height = float(y1 - y0)
        scale = max(1.0, 0.08 * np.sqrt(width * height) / 50.0)
        padding = _normalise_padding((6.0, 4.0), scale)
        corner_radius = max(8.0, min(width, height) * 0.12)
        shadow_rgba = None

    logger.info(
        "text_style mode=%s reason=%s variance=%.6f std=%.6f contrast=%.2f stroke=%s source=%s",
        text_color_mode,
        reason,
        variance,
        lum_std,
        contrast_ratio,
        "on" if stroke_alpha_value > 0 else "off",
        stroke_source,
    )

    return BubbleRenderStyle(
        fill_rgba=fill_rgba,
        text_rgb=text_rgb,
        text_alpha=text_alpha_value,
        outline_rgb=stroke_color,
        outline_alpha=stroke_alpha_value,
        outline_width=outline_width_value,
        shadow_rgba=shadow_rgba,
        shadow_offset=(0.0, 1.0),
        padding=padding,
        corner_radius=corner_radius,
        reason=reason,
        fill_gradient=None,
    )


__all__ = ["BubbleRenderStyle", "compute_dynamic_bubble_style"]
