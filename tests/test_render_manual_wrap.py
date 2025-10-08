from __future__ import annotations

import sys
import types

import numpy as np

if "PySide6" not in sys.modules:
    pyside6 = types.ModuleType("PySide6")
    sys.modules["PySide6"] = pyside6

    qtcore = types.ModuleType("PySide6.QtCore")

    class _AlignmentFlag:
        AlignLeft = 0
        AlignCenter = 1
        AlignRight = 2
        AlignJustify = 3

    class _LayoutDirection:
        LeftToRight = 0
        RightToLeft = 1

    qtcore.Qt = types.SimpleNamespace(AlignmentFlag=_AlignmentFlag, LayoutDirection=_LayoutDirection)
    sys.modules["PySide6.QtCore"] = qtcore

    qtgui = types.ModuleType("PySide6.QtGui")

    class _QFont:
        def __init__(self, *args, **kwargs):
            pass

        def setBold(self, *args, **kwargs):
            pass

        def setItalic(self, *args, **kwargs):
            pass

        def setUnderline(self, *args, **kwargs):
            pass

        def font_variant(self, *args, **kwargs):
            return self

    class _QTextDocument:
        def setDefaultFont(self, *args, **kwargs):
            pass

        def setPlainText(self, *args, **kwargs):
            pass

        def setDefaultTextOption(self, *args, **kwargs):
            pass

    class _QTextCursor:
        class SelectionType:
            Document = 0

        def __init__(self, *args, **kwargs):
            pass

        def select(self, *args, **kwargs):
            pass

    class _QTextBlockFormat:
        class LineHeightTypes:
            ProportionalHeight = types.SimpleNamespace(value=0)

        def setLineHeight(self, *args, **kwargs):
            pass

        def setAlignment(self, *args, **kwargs):
            pass

    class _QTextOption:
        def setTextDirection(self, *args, **kwargs):
            pass

    qtgui.QFont = _QFont
    qtgui.QTextDocument = _QTextDocument
    qtgui.QTextCursor = _QTextCursor
    qtgui.QTextBlockFormat = _QTextBlockFormat
    qtgui.QTextOption = _QTextOption
    sys.modules["PySide6.QtGui"] = qtgui

    qtwidgets = types.ModuleType("PySide6.QtWidgets")

    class _QApplication:
        @staticmethod
        def font():
            return types.SimpleNamespace(family=lambda: "StubFamily")

    qtwidgets.QApplication = _QApplication
    sys.modules["PySide6.QtWidgets"] = qtwidgets

from modules.rendering import render
from modules.rendering.auto_style import AutoStyleResult
from modules.utils.textblock import TextBlock
from modules.layout.grouping import TextGroup
from schemas.style_state import StyleState


class _DummySignal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, TextBlock]] = []

    def emit(self, text: str, font_size: int, blk: TextBlock) -> None:
        self.calls.append((text, font_size, blk))


class _DummyEngine:
    def __init__(self, result: AutoStyleResult) -> None:
        self._result = result
        self.calls: list[tuple[np.ndarray, list[TextBlock], StyleState]] = []

    def analyse_image(self, image: np.ndarray, blocks, base_state: StyleState):
        self.calls.append((image, list(blocks), base_state))
        return [self._result]


def test_manual_wrap_uses_auto_style_engine(monkeypatch):
    monkeypatch.setattr(render, "pyside_word_wrap", lambda *args, **kwargs: ("wrapped", 18))
    monkeypatch.setattr(render, "_get_text_color_classifier", lambda: None)

    block = TextBlock(text_bbox=np.array([0, 0, 120, 60]), translation="Hello world")
    background = np.zeros((64, 64, 3), dtype=np.uint8)

    style = StyleState(
        font_family="EngineFont",
        font_size=24,
        text_align="left",
        auto_color=True,
        fill=(10, 20, 30),
        stroke=(200, 210, 220),
        stroke_size=3,
        stroke_enabled=True,
    )
    group = TextGroup(
        blocks=[block],
        polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        bbox=(0, 0, 120, 60),
    )
    result = AutoStyleResult(group=group, analysis=None, style=style)
    engine = _DummyEngine(result)

    signal = _DummySignal()
    text_ctrl = types.SimpleNamespace(auto_style_engine=engine)
    main_page = types.SimpleNamespace(blk_rendered=signal, text_ctrl=text_ctrl)

    settings = render.TextRenderingSettings(
        alignment_id=0,
        font_family="TestFont",
        min_font_size=12,
        max_font_size=24,
        color="#123456",
        upper_case=False,
        outline=True,
        outline_color="#654321",
        outline_width="2",
        bold=False,
        italic=False,
        underline=False,
        line_spacing="1.0",
        direction=render.Qt.LayoutDirection.LeftToRight,
    )

    render.manual_wrap(main_page, [block], settings, render.Qt.AlignmentFlag.AlignLeft, background)

    assert len(engine.calls) == 1
    call_image, call_blocks, base_state = engine.calls[0]
    assert call_image is background
    assert call_blocks == [block]
    assert base_state.font_family == "TestFont"
    assert base_state.text_align == "left"
    assert base_state.auto_color is True

    assert block.style_state is not style
    assert block.style_state.fill == (10, 20, 30)
    assert block.style_state.stroke == (200, 210, 220)
    assert block.style_state.stroke_enabled is True
    assert block.style_state.font_size == 18
    assert block.font_color == "#0A141E"
    assert block.outline_color == "#C8D2DC"

    assert signal.calls == [("wrapped", 18, block)]
