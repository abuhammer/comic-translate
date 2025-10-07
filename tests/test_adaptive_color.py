import numpy as np

from modules.rendering.adaptive_color import (
    TextColorClassifier,
    sample_block_background,
    contrast_ratio,
)
from modules.utils.textblock import TextBlock


def _make_block(x1, y1, x2, y2, segm_pts=None):
    return TextBlock(
        text_bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
        text_segm_points=segm_pts,
    )


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _relative_luminance(rgb: np.ndarray) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def test_classifier_prefers_light_text_on_dark_background():
    classifier = TextColorClassifier()
    dark_patch = np.zeros((20, 20, 3), dtype=np.uint8)
    decision = classifier.decide(dark_patch)

    assert decision is not None
    assert decision.text_hex.lower() == "#ffffff"
    assert decision.outline_hex.lower() == "#000000"
    assert decision.contrast_ratio >= 4.5
    assert decision.bubble_fill_rgba is None


def test_classifier_prefers_dark_text_on_light_background():
    classifier = TextColorClassifier()
    light_patch = np.full((20, 20, 3), 245, dtype=np.uint8)
    decision = classifier.decide(light_patch)

    assert decision is not None
    assert decision.text_hex.lower() == "#000000"
    assert decision.outline_hex.lower() == "#ffffff"
    assert decision.contrast_ratio >= 3.5
    assert decision.bubble_fill_rgba is None


def test_classifier_prefers_black_on_plain_mid_tone():
    classifier = TextColorClassifier()
    mid_patch = np.full((20, 20, 3), 180, dtype=np.uint8)
    decision = classifier.decide(mid_patch)

    assert decision is not None
    assert decision.text_hex.lower() == "#000000"
    assert decision.outline_hex.lower() == "#ffffff"
    assert decision.contrast_ratio >= 4.5
    assert decision.bubble_fill_rgba is None


def test_classifier_handles_nearly_plain_white_patch_without_bubble():
    classifier = TextColorClassifier()
    patch = np.full((40, 60, 3), 245, dtype=np.uint8)
    patch[12:18, 20:26] = 220
    patch[25:30, 40:45] = 250

    decision = classifier.decide(patch)

    assert decision is not None
    assert decision.text_hex.lower() == "#000000"
    assert decision.bubble_fill_rgba is None


def test_classifier_forces_black_text_on_white_bubble_with_dark_letters():
    classifier = TextColorClassifier()
    patch = np.full((80, 80, 3), 248, dtype=np.uint8)
    patch[20:60, 30:50] = 40
    patch[30:50, 40:45] = 60

    decision = classifier.decide(patch)

    assert decision is not None
    assert decision.text_hex.lower() == "#000000"
    assert decision.bubble_fill_rgba is None


def test_classifier_forces_white_text_on_black_bubble_with_light_letters():
    classifier = TextColorClassifier()
    patch = np.zeros((80, 80, 3), dtype=np.uint8)
    patch[18:62, 28:52] = 210
    patch[24:56, 34:46] = 235

    decision = classifier.decide(patch)

    assert decision is not None
    assert decision.text_hex.lower() == "#ffffff"
    assert decision.bubble_fill_rgba is None


def test_classifier_ignores_foreground_text_when_background_is_dark():
    classifier = TextColorClassifier()
    bubble = np.full((40, 60, 3), [30, 60, 120], dtype=np.uint8)
    # Simulate bright source text in the centre of the bubble
    bubble[12:28, 15:45] = 235

    decision = classifier.decide(bubble)

    assert decision is not None
    text_rgb = _hex_to_rgb(decision.text_hex)
    outline_rgb = _hex_to_rgb(decision.outline_hex)
    assert _relative_luminance(text_rgb) > _relative_luminance(outline_rgb)
    assert decision.contrast_ratio >= 3.5
    assert decision.bubble_fill_rgba is not None


def test_classifier_handles_light_background_with_dark_foreground_noise():
    classifier = TextColorClassifier()
    bubble = np.full((40, 60, 3), 235, dtype=np.uint8)
    bubble[10:30, 20:40] = 40

    decision = classifier.decide(bubble)

    assert decision is not None
    text_rgb = _hex_to_rgb(decision.text_hex)
    outline_rgb = _hex_to_rgb(decision.outline_hex)
    assert contrast_ratio(_relative_luminance(text_rgb), _relative_luminance(outline_rgb)) >= 1.5
    assert decision.contrast_ratio >= 4.0
    assert decision.bubble_fill_rgba is None


def test_classifier_generates_translucent_bubble_on_busy_background():
    classifier = TextColorClassifier()
    # Create a high-variance patch with alternating tones
    patch = np.zeros((60, 80, 3), dtype=np.uint8)
    patch[:, :40] = [30, 70, 140]
    patch[:, 40:] = [210, 200, 220]

    decision = classifier.decide(patch)

    assert decision is not None
    assert decision.bubble_fill_rgba is not None
    r, g, b, a = decision.bubble_fill_rgba
    assert 120 <= a <= 235
    assert decision.bubble_fill_hex is not None
    assert decision.bubble_fill_hex.startswith("#")


def test_sample_block_background_fallback_when_shrink_collapses():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    blk = _make_block(2, 2, 8, 8)

    patch = sample_block_background(image, blk, shrink_ratio=0.49)

    assert patch is not None
    assert patch.shape[:2] == (6, 6)


def test_sample_block_background_uses_segmentation_bounds():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = 200
    segm = np.array([[7, 7], [13, 7], [13, 13], [7, 13]], dtype=np.int32)
    blk = _make_block(5, 5, 15, 15, segm_pts=segm)

    patch = sample_block_background(image, blk, shrink_ratio=0.0)

    assert patch is not None
    assert patch.shape[:2] == (6, 6)
    assert np.all(patch == 200)
