import numpy as np

from modules.rendering.dynamic_bubble import compute_dynamic_bubble_style
from modules.utils.textblock import TextBlock


def _make_block(x0=40, y0=40, x1=160, y1=120):
    bbox = [int(x0), int(y0), int(x1), int(y1)]
    return TextBlock(text_bbox=bbox, bubble_bbox=list(bbox))


def test_auto_mode_produces_tinted_bubble_on_plain_background():
    image = np.full((200, 200, 3), 240, dtype=np.uint8)
    blk = _make_block()

    style = compute_dynamic_bubble_style(image, blk, bubble_mode="auto")

    assert style is not None
    # Bubble should remain tinted (not plain white) and reasonably opaque.
    assert style.fill_rgba[:3] != (255, 255, 255)
    assert style.fill_rgba[3] >= 200
    # Prefer white text similar to the reference look.
    assert style.text_rgb == (255, 255, 255)
    assert style.reason.startswith("dynamic")


def test_translucent_mode_keeps_high_opacity():
    image = np.full((200, 200, 3), 245, dtype=np.uint8)
    blk = _make_block()

    style = compute_dynamic_bubble_style(image, blk, bubble_mode="translucent")

    assert style is not None
    assert style.fill_rgba[:3] != (255, 255, 255)
    assert style.fill_rgba[3] >= 225
    assert style.text_rgb == (255, 255, 255)
    assert style.reason.startswith("dynamic")


def test_gradient_arguments_generate_fill_gradient():
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    blk = _make_block()

    style = compute_dynamic_bubble_style(
        image,
        blk,
        bubble_mode="auto",
        gradient_enabled=True,
        gradient_start=(30, 90, 150),
        gradient_end=(220, 240, 255),
        gradient_angle=45.0,
    )

    assert style is not None
    assert style.fill_gradient is not None
    assert style.fill_gradient['angle'] == 45.0
    assert tuple(style.fill_gradient['start_rgba'][:3]) == (30, 90, 150)
    assert style.fill_gradient['start_rgba'][3] == style.fill_rgba[3]
