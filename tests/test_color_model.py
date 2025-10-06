import numpy as np

from modules.rendering.color_model import predict_bubble_text_color
from modules.rendering.color_utils import determine_text_outline_colors


def test_dark_blue_region_prefers_light_text():
    region = np.full((24, 24, 3), [35, 70, 130], dtype=np.uint8)
    prediction = predict_bubble_text_color(region)
    assert prediction is not None
    assert prediction.use_light_text


def test_pale_region_prefers_dark_text():
    region = np.full((24, 24, 3), [235, 235, 240], dtype=np.uint8)
    prediction = predict_bubble_text_color(region)
    assert prediction is not None
    assert not prediction.use_light_text


def test_determine_text_outline_colors_integrates_model():
    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    text, outline = determine_text_outline_colors(image, (10, 10, 90, 90))
    assert text.name().lower() == "#000000"
    assert outline.name().lower() == "#ffffff"

    image_dark = np.full((100, 100, 3), [30, 50, 90], dtype=np.uint8)
    text_dark, outline_dark = determine_text_outline_colors(image_dark, (10, 10, 90, 90))
    assert text_dark.name().lower() == "#ffffff"
    assert outline_dark.name().lower() == "#000000"
