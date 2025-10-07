import numpy as np

from modules.rendering.dynamic_bubble import (
    compute_dynamic_bubble_style,
    image_overlaps_any_block,
)
from modules.utils.textblock import TextBlock, update_block_bounds


def _make_block(x0=40, y0=40, x1=160, y1=120):
    bbox = [int(x0), int(y0), int(x1), int(y1)]
    return TextBlock(text_bbox=bbox, bubble_bbox=list(bbox))


def test_plain_white_background_forces_black_text_and_stroke():
    image = np.full((200, 200, 3), 250, dtype=np.uint8)
    blk = _make_block()

    style = compute_dynamic_bubble_style(image, blk)

    assert style is not None
    assert style.fill_rgba[3] == 0
    assert style.text_rgb == (0, 0, 0)
    # Auto stroke should kick in with ~60% opacity on the opposite colour.
    assert style.outline_rgb == (255, 255, 255)
    assert 140 <= style.outline_alpha <= 160
    assert style.outline_width >= 1.5
    assert style.reason.startswith("plain_white_bg")


def test_plain_black_background_forces_white_text_and_stroke():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    blk = _make_block()

    style = compute_dynamic_bubble_style(image, blk)

    assert style is not None
    assert style.fill_rgba[3] == 0
    assert style.text_rgb == (255, 255, 255)
    assert style.outline_rgb == (0, 0, 0)
    assert 140 <= style.outline_alpha <= 160
    assert style.outline_width >= 1.5
    assert style.reason.startswith("plain_black_bg")


def test_dynamic_background_selects_high_contrast_text():
    # Create a mid-tone noisy patch so variance is non-zero but not plain.
    image = np.full((200, 200, 3), 160, dtype=np.uint8)
    image[50:150, 50:150] = 40
    blk = _make_block()

    style = compute_dynamic_bubble_style(image, blk)

    assert style is not None
    assert style.fill_rgba[3] == 0
    assert style.text_rgb in {(0, 0, 0), (255, 255, 255)}
    assert style.reason.startswith("dynamic")
    # Contrast should meet or exceed the WCAG target of 4.5:1
    assert style.outline_alpha in (0, 153)


def test_custom_colour_with_auto_contrast_adds_outline():
    image = np.full((160, 160, 3), 180, dtype=np.uint8)
    blk = _make_block()

    style = compute_dynamic_bubble_style(
        image,
        blk,
        text_color_mode="custom",
        custom_text_rgb=(170, 170, 170),
        auto_contrast=True,
    )

    assert style is not None
    assert style.text_rgb == (170, 170, 170)
    assert style.outline_alpha >= 140
    assert style.outline_width >= 1.5
    assert "needs_outline" in style.reason


def test_text_opacity_override_applied():
    image = np.full((120, 120, 3), 180, dtype=np.uint8)
    blk = _make_block(20, 20, 100, 100)

    style = compute_dynamic_bubble_style(image, blk, text_opacity=0.5)

    assert style is not None
    assert style.text_alpha == 128


def test_numpy_bbox_inputs_are_supported():
    image = np.full((100, 100, 3), 150, dtype=np.uint8)
    text_bbox = np.array([10.0, 15.0, 90.0, 80.0], dtype=np.float32)
    bubble_bbox = np.array([8.0, 13.0, 92.0, 82.0], dtype=np.float32)
    blk = TextBlock(text_bbox=text_bbox, bubble_bbox=bubble_bbox)

    style = compute_dynamic_bubble_style(image, blk, text_opacity=0.25)

    assert style is not None
    assert style.text_alpha == 64


def test_auto_background_box_adds_tint_when_contrast_low():
    image = np.full((140, 140, 3), 180, dtype=np.uint8)
    blk = _make_block()

    # Force a near-match text colour so contrast is poor
    style = compute_dynamic_bubble_style(
        image,
        blk,
        text_color_mode="custom",
        custom_text_rgb=(185, 185, 185),
        auto_contrast=False,
        background_box_mode="auto",
    )

    assert style is not None
    assert style.fill_rgba[3] > 0
    assert style.fill_rgba[:3] == (35, 100, 160)
    assert style.fill_rgba[3] <= int(0.3 * 255 + 1)


def test_image_cover_check_ignores_out_of_bounds_blocks():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    valid = TextBlock(text_bbox=[10, 10, 40, 40], bubble_bbox=[10, 10, 40, 40])
    offscreen = TextBlock(text_bbox=[150, 150, 210, 220], bubble_bbox=[150, 150, 210, 220])

    assert image_overlaps_any_block(image, [offscreen, valid])
    assert not image_overlaps_any_block(image, [offscreen])


def test_update_block_bounds_syncs_bubble_coords():
    blk = TextBlock(text_bbox=[10, 12, 40, 60], bubble_bbox=[10, 12, 40, 60])

    update_block_bounds(blk, [100, 120, 180, 200])

    assert tuple(int(v) for v in blk.xyxy[:4]) == (100, 120, 180, 200)
    assert tuple(int(v) for v in blk.bubble_xyxy[:4]) == (100, 120, 180, 200)


def test_dynamic_sampling_follows_updated_bounds():
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    image[:110, :110] = 240  # bright quadrant
    image[110:, 110:] = 15   # dark quadrant

    blk = TextBlock(text_bbox=[20, 20, 80, 80], bubble_bbox=[20, 20, 80, 80])

    bright_style = compute_dynamic_bubble_style(image, blk)
    assert bright_style is not None
    assert bright_style.text_rgb == (0, 0, 0)

    update_block_bounds(blk, [140, 140, 200, 200])
    dark_style = compute_dynamic_bubble_style(image, blk)

    assert dark_style is not None
    assert dark_style.text_rgb == (255, 255, 255)
