from __future__ import annotations

import copy
import logging
import math
import numpy as np
from typing import TYPE_CHECKING, Iterable, Optional

from PySide6 import QtCore
from PySide6.QtGui import QColor

from app.ui.commands.textformat import TextFormatCommand
from app.ui.commands.box import AddTextItemCommand
from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties

from modules.utils.textblock import TextBlock
from modules.rendering.render import manual_wrap
from modules.rendering.settings import TextRenderingSettings
from modules.utils.pipeline_utils import font_selected, get_language_code, \
    get_layout_direction, is_close
from modules.utils.translator_utils import format_translations
from modules.rendering.auto_style import AutoStyleEngine
from schemas.style_state import StyleState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from controller import ComicTranslate
    from app.ui.style_panel import StylePanel

class TextController:
    def __init__(self, main: ComicTranslate):
        self.main = main

        # List of widgets to block signals for during manual rendering
        self.widgets_to_block = [
            self.main.font_dropdown,
            self.main.font_size_dropdown,
            self.main.line_spacing_dropdown,
            self.main.block_font_color_button,
            self.main.outline_font_color_button,
            self.main.outline_width_dropdown,
            self.main.outline_checkbox
        ]

        self.auto_style_engine = AutoStyleEngine()
        self.style_panel: Optional[StylePanel] = getattr(self.main, "style_panel", None)
        self._style_panel_updating = False
        self._adaptive_background_metadata: Optional[dict] = None

        if self.style_panel is not None:
            self.style_panel.styleChanged.connect(self.on_style_panel_changed)
            self.style_panel.reanalyseRequested.connect(self.reanalyse_current_block_style)
            self.style_panel.fontChanged.connect(self.on_font_dropdown_change)
            self.style_panel.fontSizeChanged.connect(lambda size: self.on_font_size_change(str(size)))
            self.style_panel.alignmentChanged.connect(self._on_style_panel_alignment)

    def connect_text_item_signals(self, text_item: TextBlockItem):
        text_item.item_selected.connect(self.on_text_item_selected)
        text_item.item_deselected.connect(self.on_text_item_deselected)
        text_item.text_changed.connect(self.update_text_block_from_item)
        text_item.text_highlighted.connect(self.set_values_from_highlight)
        text_item.change_undo.connect(self.main.rect_item_ctrl.rect_change_undo)

    def _get_current_base_image(self) -> Optional[np.ndarray]:
        """Return a copy of the original image for the active page."""

        idx = getattr(self.main, "curr_img_idx", -1)
        if idx is None or idx < 0:
            return None

        if not self.main.image_files or idx >= len(self.main.image_files):
            return None

        file_path = self.main.image_files[idx]
        base_image = self.main.image_data.get(file_path)
        if base_image is None:
            return None

        try:
            return np.array(base_image, copy=True)
        except Exception:
            logger.exception("Failed to copy base image for adaptive colours")
            return None

    @staticmethod
    def _clamp_bbox_to_image(
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> Optional[list[int]]:
        """Clamp a floating-point bbox to image bounds, returning ints or None."""

        if width <= 0 or height <= 0 or not bbox:
            return None

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, math.floor(x1)))
        y1 = max(0, min(height, math.floor(y1)))
        x2 = max(0, min(width, math.ceil(x2)))
        y2 = max(0, min(height, math.ceil(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def _build_webtoon_page_metadata(
        self,
        blocks: list[TextBlock],
        page_index: int,
        page_image: Optional[np.ndarray],
    ) -> Optional[dict]:
        """Compute per-block sampling bounds for the active webtoon page."""

        manager = getattr(self.main.image_viewer, "webtoon_manager", None)
        if (
            page_image is None
            or manager is None
            or not (0 <= page_index < len(manager.image_positions))
        ):
            return None

        height, width = page_image.shape[:2]
        page_y = manager.image_positions[page_index]
        page_height = (
            manager.image_heights[page_index]
            if page_index < len(manager.image_heights)
            else height
        )
        page_width = width
        page_x_offset = (manager.webtoon_width - page_width) / 2 if manager.webtoon_width else 0

        block_map: dict[int, dict] = {}
        for blk in blocks:
            xyxy = getattr(blk, "xyxy", None)
            if xyxy is None:
                continue

            y1_scene, y2_scene = xyxy[1], xyxy[3]
            if y2_scene <= page_y or y1_scene >= page_y + page_height:
                continue

            local_bbox = (
                xyxy[0] - page_x_offset,
                y1_scene - page_y,
                xyxy[2] - page_x_offset,
                y2_scene - page_y,
            )
            clamped_bbox = self._clamp_bbox_to_image(local_bbox, width, height)
            if clamped_bbox is None:
                continue

            entry: dict[str, list[int]] = {"bbox": clamped_bbox}
            bubble = getattr(blk, "bubble_xyxy", None)
            if bubble is not None:
                bubble_local = (
                    bubble[0] - page_x_offset,
                    bubble[1] - page_y,
                    bubble[2] - page_x_offset,
                    bubble[3] - page_y,
                )
                bubble_clamped = self._clamp_bbox_to_image(bubble_local, width, height)
                if bubble_clamped is not None:
                    entry["bubble"] = bubble_clamped

            block_map[id(blk)] = entry

        if not block_map:
            return None

        return {
            "type": "page",
            "page_index": page_index,
            "block_map": block_map,
            "image_shape": (height, width),
        }

    def _build_visible_area_metadata(
        self,
        blocks: list[TextBlock],
        mappings: list[dict],
        image_shape: tuple[int, int],
    ) -> Optional[dict]:
        """Compute sampling bounds for blocks within the combined visible capture."""

        manager = getattr(self.main.image_viewer, "webtoon_manager", None)
        if manager is None or not mappings or not image_shape:
            return None

        height, width = image_shape[:2]
        mapping_by_page: dict[int, list[dict]] = {}
        for mapping in mappings:
            page_idx = mapping.get("page_index")
            if page_idx is None:
                continue
            mapping_by_page.setdefault(page_idx, []).append(mapping)

        if not mapping_by_page:
            return None

        block_map: dict[int, dict] = {}
        layout_manager = getattr(manager, "layout_manager", None)

        for blk in blocks:
            xyxy = getattr(blk, "xyxy", None)
            if xyxy is None:
                continue

            center_y = (xyxy[1] + xyxy[3]) / 2.0
            page_idx = None
            if layout_manager is not None:
                page_idx = layout_manager.get_page_at_position(center_y)
            if page_idx is None or page_idx not in mapping_by_page:
                continue

            page_y = manager.image_positions[page_idx]
            page_height = (
                manager.image_heights[page_idx]
                if page_idx < len(manager.image_heights)
                else 0
            )
            if page_height <= 0:
                continue

            y1_local = xyxy[1] - page_y
            y2_local = xyxy[3] - page_y
            if y2_local <= 0 or y1_local >= page_height:
                continue

            if page_idx in manager.image_data:
                page_width = manager.image_data[page_idx].shape[1]
            else:
                page_width = manager.webtoon_width or width
            page_x_offset = (manager.webtoon_width - page_width) / 2 if manager.webtoon_width else 0

            x1_local = xyxy[0] - page_x_offset
            x2_local = xyxy[2] - page_x_offset

            for mapping in mapping_by_page[page_idx]:
                crop_top = mapping.get("page_crop_top", 0)
                crop_bottom = mapping.get("page_crop_bottom", page_height)
                if y2_local <= crop_top or y1_local >= crop_bottom:
                    continue

                combined_start = mapping.get("combined_y_start", 0)
                y1_combined = (y1_local - crop_top) + combined_start
                y2_combined = (y2_local - crop_top) + combined_start

                clamped_bbox = self._clamp_bbox_to_image(
                    (x1_local, y1_combined, x2_local, y2_combined), width, height
                )
                if clamped_bbox is None:
                    continue

                entry: dict[str, list[int]] = {"bbox": clamped_bbox}
                bubble = getattr(blk, "bubble_xyxy", None)
                if bubble is not None:
                    bubble_local = (
                        bubble[0] - page_x_offset,
                        bubble[1] - page_y,
                        bubble[2] - page_x_offset,
                        bubble[3] - page_y,
                    )
                    bubble_y1 = (bubble_local[1] - crop_top) + combined_start
                    bubble_y2 = (bubble_local[3] - crop_top) + combined_start
                    bubble_clamped = self._clamp_bbox_to_image(
                        (bubble_local[0], bubble_y1, bubble_local[2], bubble_y2),
                        width,
                        height,
                    )
                    if bubble_clamped is not None:
                        entry["bubble"] = bubble_clamped

                block_map[id(blk)] = entry
                break

        if not block_map:
            return None

        return {
            "type": "visible",
            "block_map": block_map,
            "image_shape": (height, width),
        }

    def _capture_standard_view_image(
        self, paint_all: bool, include_patches: bool
    ) -> Optional[np.ndarray]:
        """Capture the current view for regular mode with graceful fallbacks."""

        viewer_image = None
        try:
            viewer_image = self.main.image_viewer.get_image_array(
                paint_all=paint_all, include_patches=include_patches
            )
            if viewer_image is not None:
                viewer_image = viewer_image.copy()
        except Exception:
            logger.exception("Failed to capture background for adaptive colours")
            viewer_image = None

        if viewer_image is None and include_patches:
            try:
                viewer_image = self.main.image_viewer.get_image_array(
                    paint_all=paint_all, include_patches=False
                )
                if viewer_image is not None:
                    viewer_image = viewer_image.copy()
            except Exception:
                logger.exception("Failed to capture background for adaptive colours")
                viewer_image = None

        if viewer_image is None:
            try:
                viewer_image = self.main.image_viewer.get_image_array()
                if viewer_image is not None:
                    viewer_image = viewer_image.copy()
            except Exception:
                logger.exception("Failed to capture background for adaptive colours")
                viewer_image = None

        return viewer_image

    def _capture_webtoon_visible_area(
        self, paint_all: bool, include_patches: bool
    ) -> tuple[Optional[np.ndarray], list]:
        """Capture the combined visible area in webtoon mode with fallbacks."""

        image = None
        mappings: list = []
        try:
            image, mappings = self.main.image_viewer.get_visible_area_image(
                paint_all=paint_all, include_patches=include_patches
            )
            if image is not None:
                image = image.copy()
        except Exception:
            logger.exception("Failed to capture webtoon visible area for adaptive colours")
            image, mappings = None, []

        if image is None:
            return self._capture_standard_view_image(paint_all, include_patches), []

        return image, mappings

    def _prepare_background_image(
        self,
        blocks: Iterable[TextBlock],
        render_settings: TextRenderingSettings,
    ) -> Optional[np.ndarray]:
        """Collect a background image aligned with block coordinates."""

        blocks = list(blocks or [])

        if not getattr(render_settings, "auto_font_color", True):
            self._adaptive_background_metadata = None
            return None

        # Handle specialised capture for webtoon mode
        if getattr(self.main, "webtoon_mode", False):
            page_index = getattr(self.main, "curr_img_idx", -1)
            base_image = self._get_current_base_image()

            metadata = self._build_webtoon_page_metadata(blocks, page_index, base_image)
            if metadata is not None:
                self._adaptive_background_metadata = metadata
                return base_image

            visible_image, mappings = self._capture_webtoon_visible_area(
                paint_all=True, include_patches=True
            )
            if visible_image is not None:
                metadata = self._build_visible_area_metadata(
                    blocks, mappings, visible_image.shape
                )
                self._adaptive_background_metadata = metadata
                return visible_image

            # Fall back to the base image if all else fails
            self._adaptive_background_metadata = None
            return base_image

        # Regular mode – use viewer capture fallbacks
        viewer_image = self._capture_standard_view_image(
            paint_all=True, include_patches=True
        )
        base_image = self._get_current_base_image()

        self._adaptive_background_metadata = None

        if base_image is not None:
            return base_image

        return viewer_image

    def clear_text_edits(self):
        self.main.curr_tblock = None
        self.main.curr_tblock_item = None
        self.main.s_text_edit.clear()
        self.main.t_text_edit.clear()

    def on_blk_rendered(self, text: str, font_size: int, blk: TextBlock):
        if not self.main.image_viewer.hasPhoto():
            print("No main image to add to.")
            return

        target_lang = self.main.lang_mapping.get(self.main.t_combo.currentText(), None)
        trg_lng_cd = get_language_code(target_lang)
        if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):
            text = text.replace(' ', '')

        render_settings = self.render_settings()
        font_family = render_settings.font_family
        style_state = getattr(blk, "style_state", None)

        text_color_str = blk.font_color if getattr(blk, 'font_color', '') else render_settings.color
        if style_state and style_state.fill is not None:
            text_color = QColor(*style_state.fill)
        else:
            text_color = QColor(text_color_str)

        id = render_settings.alignment_id
        alignment = self.main.button_to_alignment[id]
        line_spacing = float(render_settings.line_spacing)
        try:
            configured_outline_width = float(render_settings.outline_width)
        except (TypeError, ValueError):
            configured_outline_width = 0.0

        outline_color = None
        outline_width = configured_outline_width
        outline_enabled = False

        if style_state and style_state.stroke is not None and style_state.stroke_enabled:
            outline_color = QColor(*style_state.stroke)
            outline_enabled = True
            if style_state.stroke_size is not None and style_state.stroke_size > 0:
                outline_width = float(style_state.stroke_size)
            elif outline_width <= 0:
                outline_width = 1.0
        elif getattr(blk, 'outline_color', ''):
            outline_color = QColor(blk.outline_color)
            outline_enabled = True
            if outline_width <= 0:
                outline_width = 1.0
        elif render_settings.outline:
            outline_color = QColor(render_settings.outline_color)
            outline_enabled = True
            if outline_width <= 0:
                outline_width = 1.0
        bold = render_settings.bold
        italic = render_settings.italic
        underline = render_settings.underline
        direction = render_settings.direction

        properties = TextItemProperties(
            text=text,
            font_family=font_family,
            font_size=font_size,
            text_color=text_color,
            alignment=alignment,
            line_spacing=line_spacing,
            outline_color=outline_color if outline_enabled else None,
            outline_width=outline_width,
            bold=bold,
            italic=italic,
            underline=underline,
            direction=direction,
            position=(blk.xyxy[0], blk.xyxy[1]),
            rotation=blk.angle,
            style_state=style_state.copy() if style_state else None,
        )

        text_item = self.main.image_viewer.add_text_item(properties)
        text_item.set_plain_text(text)
        if style_state:
            text_item.apply_style_state(style_state)

        command = AddTextItemCommand(self.main, text_item)
        self.main.push_command(command)

    def on_text_item_selected(self, text_item: TextBlockItem):
        self.main.curr_tblock_item = text_item

        x1, y1 = int(text_item.pos().x()), int(text_item.pos().y())
        rotation = text_item.rotation()

        self.main.curr_tblock = next(
            (
            blk for blk in self.main.blk_list
            if is_close(blk.xyxy[0], x1, 5) and is_close(blk.xyxy[1], y1, 5)
            and is_close(blk.angle, rotation, 1)
            ),
            None
        )

        # Update both s_text_edit and t_text_edit
        if self.main.curr_tblock:
            self.main.s_text_edit.blockSignals(True)
            self.main.s_text_edit.setPlainText(self.main.curr_tblock.text)
            self.main.s_text_edit.blockSignals(False)

        self.main.t_text_edit.blockSignals(True)
        self.main.t_text_edit.setPlainText(text_item.toPlainText())
        self.main.t_text_edit.blockSignals(False)

        self.set_values_for_blk_item(text_item)

    def on_text_item_deselected(self):
        self.clear_text_edits()
        if self.style_panel:
            self.style_panel.clear_style()

    def update_text_block(self):
        if self.main.curr_tblock:
            self.main.curr_tblock.text = self.main.s_text_edit.toPlainText()
            self.main.curr_tblock.translation = self.main.t_text_edit.toPlainText()

    def update_text_block_from_edit(self):
        new_text = self.main.t_text_edit.toPlainText()
        if self.main.curr_tblock:
            self.main.curr_tblock.translation = new_text

        if self.main.curr_tblock_item and self.main.curr_tblock_item in self.main.image_viewer._scene.items():
            cursor_position = self.main.t_text_edit.textCursor().position()
            self.main.curr_tblock_item.setPlainText(new_text)

            # Restore cursor position
            cursor = self.main.t_text_edit.textCursor()
            cursor.setPosition(cursor_position)
            self.main.t_text_edit.setTextCursor(cursor)

    def update_text_block_from_item(self, new_text: str):
        if self.main.curr_tblock and new_text:
            self.main.curr_tblock.translation = new_text
            self.main.t_text_edit.blockSignals(True)
            self.main.t_text_edit.setPlainText(new_text)
            self.main.t_text_edit.blockSignals(False)

    def save_src_trg(self):
        source_lang = self.main.s_combo.currentText()
        target_lang = self.main.t_combo.currentText()
        
        if self.main.curr_img_idx >= 0:
            current_file = self.main.image_files[self.main.curr_img_idx]
            self.main.image_states[current_file]['source_lang'] = source_lang
            self.main.image_states[current_file]['target_lang'] = target_lang

        target_en = self.main.lang_mapping.get(target_lang, None)
        t_direction = get_layout_direction(target_en)
        t_text_option = self.main.t_text_edit.document().defaultTextOption()
        t_text_option.setTextDirection(t_direction)
        self.main.t_text_edit.document().setDefaultTextOption(t_text_option)

    def set_src_trg_all(self):
        source_lang = self.main.s_combo.currentText()
        target_lang = self.main.t_combo.currentText()
        for image_path in self.main.image_files:
            self.main.image_states[image_path]['source_lang'] = source_lang
            self.main.image_states[image_path]['target_lang'] = target_lang

    def change_all_blocks_size(self, diff: int):
        if len(self.main.blk_list) == 0:
            return
        updated_blk_list = []
        for blk in self.main.blk_list:
            blk_rect = tuple(blk.xyxy)
            blk.xyxy[:] = [blk_rect[0] - diff, blk_rect[1] - diff, blk_rect[2] + diff, blk_rect[3] + diff]
            updated_blk_list.append(blk)
        self.main.blk_list = updated_blk_list
        self.main.pipeline.load_box_coords(self.main.blk_list)

    # Formatting actions
    def on_font_dropdown_change(self, font_family: str):
        if self.main.curr_tblock_item and font_family:
            if self.main.font_dropdown.currentText() != font_family:
                self.main.font_dropdown.setCurrentText(font_family)
            old_item = copy.copy(self.main.curr_tblock_item)
            font_size = int(self.main.font_size_dropdown.currentText())
            self.main.curr_tblock_item.set_font(font_family, font_size)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

            if self.main.curr_tblock_item.style_state:
                style = self.main.curr_tblock_item.style_state.copy()
                style.font_family = font_family
                style.font_size = int(font_size)
                self.main.curr_tblock_item.apply_style_state(style)
                if self.main.curr_tblock:
                    self.main.curr_tblock.style_state = style.copy()
                if self.style_panel and not self._style_panel_updating:
                    self._style_panel_updating = True
                    try:
                        self.style_panel.set_style(
                            style,
                            font_family,
                            int(font_size),
                            self._alignment_to_name(self.main.curr_tblock_item.alignment),
                        )
                    finally:
                        self._style_panel_updating = False

    def on_font_size_change(self, font_size: str):
        if self.main.curr_tblock_item and font_size:
            if self.main.font_size_dropdown.currentText() != font_size:
                self.main.font_size_dropdown.setCurrentText(font_size)
            old_item = copy.copy(self.main.curr_tblock_item)
            font_size = float(font_size)
            self.main.curr_tblock_item.set_font_size(font_size)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

            if self.main.curr_tblock_item.style_state:
                style = self.main.curr_tblock_item.style_state.copy()
                style.font_size = int(font_size)
                self.main.curr_tblock_item.apply_style_state(style)
                if self.main.curr_tblock:
                    self.main.curr_tblock.style_state = style.copy()
                if self.style_panel and not self._style_panel_updating:
                    self._style_panel_updating = True
                    try:
                        self.style_panel.set_style(
                            style,
                            self.main.curr_tblock_item.font_family,
                            int(font_size),
                            self._alignment_to_name(self.main.curr_tblock_item.alignment),
                        )
                    finally:
                        self._style_panel_updating = False

    def on_line_spacing_change(self, line_spacing: str):
        if self.main.curr_tblock_item and line_spacing:
            old_item = copy.copy(self.main.curr_tblock_item)
            spacing = float(line_spacing)
            self.main.curr_tblock_item.set_line_spacing(spacing)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def on_font_color_change(self):
        font_color = self.main.get_color()
        if font_color and font_color.isValid():
            self.main.block_font_color_button.setStyleSheet(
                f"background-color: {font_color.name()}; border: none; border-radius: 5px;"
            )
            self.main.block_font_color_button.setProperty('selected_color', font_color.name())
            if self.main.curr_tblock_item:
                old_item = copy.copy(self.main.curr_tblock_item)
                self.main.curr_tblock_item.set_color(font_color)

                command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
                self.main.push_command(command)

                if self.main.curr_tblock_item.style_state:
                    style = self.main.curr_tblock_item.style_state.copy()
                    style.fill = (font_color.red(), font_color.green(), font_color.blue())
                    style.auto_color = False
                    self.main.curr_tblock_item.apply_style_state(style)
                    if self.main.curr_tblock:
                        self.main.curr_tblock.style_state = style.copy()
                    if self.style_panel and not self._style_panel_updating:
                        self._style_panel_updating = True
                        try:
                            self.style_panel.set_style(
                                style,
                                self.main.curr_tblock_item.font_family,
                                int(self.main.curr_tblock_item.font_size),
                                self._alignment_to_name(self.main.curr_tblock_item.alignment),
                            )
                        finally:
                            self._style_panel_updating = False

    def left_align(self):
        if self.main.curr_tblock_item:
            old_item = copy.copy(self.main.curr_tblock_item)
            self.main.curr_tblock_item.set_alignment(QtCore.Qt.AlignmentFlag.AlignLeft)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def center_align(self):
        if self.main.curr_tblock_item:
            old_item = copy.copy(self.main.curr_tblock_item)
            self.main.curr_tblock_item.set_alignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def right_align(self):
        if self.main.curr_tblock_item:
            old_item = copy.copy(self.main.curr_tblock_item)
            self.main.curr_tblock_item.set_alignment(QtCore.Qt.AlignmentFlag.AlignRight)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def bold(self):
        if self.main.curr_tblock_item:
            old_item = copy.copy(self.main.curr_tblock_item)
            state = self.main.bold_button.isChecked()
            self.main.curr_tblock_item.set_bold(state)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def italic(self):
        if self.main.curr_tblock_item:
            old_item = copy.copy(self.main.curr_tblock_item)
            state = self.main.italic_button.isChecked()
            self.main.curr_tblock_item.set_italic(state)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def underline(self):
        if self.main.curr_tblock_item:
            old_item = copy.copy(self.main.curr_tblock_item)
            state = self.main.underline_button.isChecked()
            self.main.curr_tblock_item.set_underline(state)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def on_outline_color_change(self):
        outline_color = self.main.get_color()
        if outline_color and outline_color.isValid():
            self.main.outline_font_color_button.setStyleSheet(
                f"background-color: {outline_color.name()}; border: none; border-radius: 5px;"
            )
            self.main.outline_font_color_button.setProperty('selected_color', outline_color.name())
            outline_width = float(self.main.outline_width_dropdown.currentText())

            if self.main.curr_tblock_item and self.main.outline_checkbox.isChecked():
                old_item = copy.copy(self.main.curr_tblock_item)
                self.main.curr_tblock_item.set_outline(outline_color, outline_width)

                command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
                self.main.push_command(command)

                if self.main.curr_tblock_item.style_state:
                    style = self.main.curr_tblock_item.style_state.copy()
                    style.stroke = (outline_color.red(), outline_color.green(), outline_color.blue())
                    style.stroke_enabled = True
                    style.stroke_size = style.stroke_size or int(round(outline_width))
                    self.main.curr_tblock_item.apply_style_state(style)
                    if self.main.curr_tblock:
                        self.main.curr_tblock.style_state = style.copy()
                    if self.style_panel and not self._style_panel_updating:
                        self._style_panel_updating = True
                        try:
                            self.style_panel.set_style(
                                style,
                                self.main.curr_tblock_item.font_family,
                                int(self.main.curr_tblock_item.font_size),
                                self._alignment_to_name(self.main.curr_tblock_item.alignment),
                            )
                        finally:
                            self._style_panel_updating = False

    def on_outline_width_change(self, outline_width):
        if self.main.curr_tblock_item and self.main.outline_checkbox.isChecked():
            old_item = copy.copy(self.main.curr_tblock_item)
            outline_width = float(self.main.outline_width_dropdown.currentText())
            color_str = self.main.outline_font_color_button.property('selected_color')
            color = QColor(color_str)
            self.main.curr_tblock_item.set_outline(color, outline_width)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

            if self.main.curr_tblock_item.style_state:
                style = self.main.curr_tblock_item.style_state.copy()
                if color.isValid():
                    style.stroke = (color.red(), color.green(), color.blue())
                style.stroke_enabled = True
                style.stroke_size = int(round(outline_width))
                self.main.curr_tblock_item.apply_style_state(style)
                if self.main.curr_tblock:
                    self.main.curr_tblock.style_state = style.copy()
                if self.style_panel and not self._style_panel_updating:
                    self._style_panel_updating = True
                    try:
                        self.style_panel.set_style(
                            style,
                            self.main.curr_tblock_item.font_family,
                            int(self.main.curr_tblock_item.font_size),
                            self._alignment_to_name(self.main.curr_tblock_item.alignment),
                        )
                    finally:
                        self._style_panel_updating = False

    @staticmethod
    def _alignment_to_name(alignment: QtCore.Qt.AlignmentFlag) -> str:
        if alignment == QtCore.Qt.AlignmentFlag.AlignCenter:
            return "center"
        if alignment == QtCore.Qt.AlignmentFlag.AlignRight:
            return "right"
        if alignment == QtCore.Qt.AlignmentFlag.AlignJustify:
            return "justify"
        return "left"

    @staticmethod
    def _name_to_alignment(name: str) -> QtCore.Qt.AlignmentFlag:
        mapping = {
            "left": QtCore.Qt.AlignmentFlag.AlignLeft,
            "center": QtCore.Qt.AlignmentFlag.AlignCenter,
            "right": QtCore.Qt.AlignmentFlag.AlignRight,
            "justify": QtCore.Qt.AlignmentFlag.AlignJustify,
        }
        return mapping.get(name, QtCore.Qt.AlignmentFlag.AlignLeft)

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#" + "".join(f"{channel:02X}" for channel in rgb)

    def _on_style_panel_alignment(self, alignment: str) -> None:
        if self._style_panel_updating:
            return
        if not self.main.curr_tblock_item:
            return

        align_flag = self._name_to_alignment(alignment)
        old_item = copy.copy(self.main.curr_tblock_item)
        self.main.curr_tblock_item.set_alignment(align_flag)
        command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
        self.main.push_command(command)

        if self.style_panel and not self._style_panel_updating:
            self._style_panel_updating = True
            try:
                self.style_panel.set_style(
                    self.main.curr_tblock_item.style_state,
                    self.main.curr_tblock_item.font_family,
                    int(self.main.curr_tblock_item.font_size),
                    alignment,
                )
            finally:
                self._style_panel_updating = False

    def on_style_panel_changed(self, state: StyleState) -> None:
        if self._style_panel_updating:
            return
        if not self.main.curr_tblock_item:
            return

        style_copy = state.copy()
        style_copy.font_family = self.main.curr_tblock_item.font_family
        style_copy.font_size = int(self.main.curr_tblock_item.font_size)
        style_copy.text_align = self._alignment_to_name(self.main.curr_tblock_item.alignment)

        self.main.curr_tblock_item.apply_style_state(style_copy)

        if self.main.curr_tblock:
            self.main.curr_tblock.style_state = style_copy.copy()
            if style_copy.fill is not None:
                self.main.curr_tblock.font_color = self._rgb_to_hex(style_copy.fill)
            else:
                self.main.curr_tblock.font_color = ''
            if style_copy.stroke is not None and style_copy.stroke_enabled:
                self.main.curr_tblock.outline_color = self._rgb_to_hex(style_copy.stroke)
            else:
                self.main.curr_tblock.outline_color = ''

        if style_copy.fill is not None:
            color = QColor(*style_copy.fill)
            self.main.block_font_color_button.setStyleSheet(
                f"background-color: {color.name()}; border: none; border-radius: 5px;"
            )
            self.main.block_font_color_button.setProperty('selected_color', color.name())
        if style_copy.stroke is not None and style_copy.stroke_enabled:
            color = QColor(*style_copy.stroke)
            self.main.outline_font_color_button.setStyleSheet(
                f"background-color: {color.name()}; border: none; border-radius: 5px;"
            )
            self.main.outline_font_color_button.setProperty('selected_color', color.name())
            self.main.outline_checkbox.setChecked(True)
        elif not style_copy.stroke_enabled:
            self.main.outline_checkbox.setChecked(False)

        self.main.image_viewer.viewport().update()

    def reanalyse_current_block_style(self) -> None:
        if not self.main.curr_tblock or not self.main.curr_tblock_item:
            return

        render_settings = self.render_settings()
        background = self._prepare_background_image([self.main.curr_tblock], render_settings)
        if background is None:
            logger.warning("Unable to capture background for style re-analysis")
            return

        base_style = self.main.curr_tblock.style_state.copy() if self.main.curr_tblock.style_state else StyleState()
        base_style.font_family = self.main.curr_tblock_item.font_family
        base_style.font_size = int(self.main.curr_tblock_item.font_size)
        base_style.text_align = self._alignment_to_name(self.main.curr_tblock_item.alignment)

        try:
            new_state = self.auto_style_engine.style_for_block(background, self.main.curr_tblock, base_style)
        except Exception:
            logger.exception("Failed to re-analyse style for block")
            return

        self._style_panel_updating = True
        try:
            if self.style_panel:
                self.style_panel.set_style(
                    new_state,
                    self.main.curr_tblock_item.font_family,
                    int(self.main.curr_tblock_item.font_size),
                    self._alignment_to_name(self.main.curr_tblock_item.alignment),
                )
        finally:
            self._style_panel_updating = False

        self.on_style_panel_changed(new_state)

    def toggle_outline_settings(self, state):
        enabled = True if state == 2 else False
        if self.main.curr_tblock_item:
            if not enabled:
                self.main.curr_tblock_item.set_outline(None, None)
            else:
                old_item = copy.copy(self.main.curr_tblock_item)
                outline_width = float(self.main.outline_width_dropdown.currentText())
                color_str = self.main.outline_font_color_button.property('selected_color')
                color = QColor(color_str)
                self.main.curr_tblock_item.set_outline(color, outline_width)

                command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
                self.main.push_command(command)

    # Widget helpers
    def block_text_item_widgets(self, widgets):
        # Block signals
        for widget in widgets:
            widget.blockSignals(True)

        # Block Signals is buggy for these, so use disconnect/connect
        self.main.bold_button.clicked.disconnect(self.bold)
        self.main.italic_button.clicked.disconnect(self.italic)
        self.main.underline_button.clicked.disconnect(self.underline)

        self.main.alignment_tool_group.get_button_group().buttons()[0].clicked.disconnect(self.left_align)
        self.main.alignment_tool_group.get_button_group().buttons()[1].clicked.disconnect(self.center_align)
        self.main.alignment_tool_group.get_button_group().buttons()[2].clicked.disconnect(self.right_align)

    def unblock_text_item_widgets(self, widgets):
        # Unblock signals
        for widget in widgets:
            widget.blockSignals(False)

        self.main.bold_button.clicked.connect(self.bold)
        self.main.italic_button.clicked.connect(self.italic)
        self.main.underline_button.clicked.connect(self.underline)

        self.main.alignment_tool_group.get_button_group().buttons()[0].clicked.connect(self.left_align)
        self.main.alignment_tool_group.get_button_group().buttons()[1].clicked.connect(self.center_align)
        self.main.alignment_tool_group.get_button_group().buttons()[2].clicked.connect(self.right_align)

    def set_values_for_blk_item(self, text_item: TextBlockItem):

        self.block_text_item_widgets(self.widgets_to_block)

        try:
            # Set values
            self.main.font_dropdown.setCurrentText(text_item.font_family)
            self.main.font_size_dropdown.setCurrentText(str(int(text_item.font_size)))

            self.main.line_spacing_dropdown.setCurrentText(str(text_item.line_spacing))

            self.main.block_font_color_button.setStyleSheet(
                f"background-color: {text_item.text_color.name()}; border: none; border-radius: 5px;"
            )
            self.main.block_font_color_button.setProperty('selected_color', text_item.text_color.name())

            if text_item.outline_color is not None:
                self.main.outline_font_color_button.setStyleSheet(
                    f"background-color: {text_item.outline_color.name()}; border: none; border-radius: 5px;"
                )
                self.main.outline_font_color_button.setProperty('selected_color', text_item.outline_color.name())
            else:
                self.main.outline_font_color_button.setStyleSheet(
                    "background-color: white; border: none; border-radius: 5px;"
                )
                self.main.outline_font_color_button.setProperty('selected_color', '#ffffff')

            self.main.outline_width_dropdown.setCurrentText(str(text_item.outline_width))
            self.main.outline_checkbox.setChecked(text_item.outline)

            self.main.bold_button.setChecked(text_item.bold)
            self.main.italic_button.setChecked(text_item.italic)
            self.main.underline_button.setChecked(text_item.underline)

            alignment_to_button = {
                QtCore.Qt.AlignmentFlag.AlignLeft: 0,
                QtCore.Qt.AlignmentFlag.AlignCenter: 1,
                QtCore.Qt.AlignmentFlag.AlignRight: 2,
            }

            alignment = text_item.alignment
            button_group = self.main.alignment_tool_group.get_button_group()

            if alignment in alignment_to_button:
                button_index = alignment_to_button[alignment]
                button_group.buttons()[button_index].setChecked(True)

        finally:
            self.unblock_text_item_widgets(self.widgets_to_block)

        if self.style_panel and text_item:
            style = getattr(text_item, "style_state", None)
            if style is None and self.main.curr_tblock and self.main.curr_tblock.style_state:
                style = self.main.curr_tblock.style_state
            if style is None:
                style = StyleState(
                    font_family=text_item.font_family,
                    font_size=int(text_item.font_size),
                    text_align=self._alignment_to_name(text_item.alignment),
                )
            else:
                style = style.copy()
                style.font_family = text_item.font_family
                style.font_size = int(text_item.font_size)
                style.text_align = self._alignment_to_name(text_item.alignment)
            self._style_panel_updating = True
            try:
                self.style_panel.set_style(
                    style,
                    text_item.font_family,
                    int(text_item.font_size),
                    self._alignment_to_name(text_item.alignment),
                )
            finally:
                self._style_panel_updating = False

    def set_values_from_highlight(self, item_highlighted = None):

        self.block_text_item_widgets(self.widgets_to_block)

        # Attributes
        font_family = item_highlighted['font_family']
        font_size = item_highlighted['font_size']
        text_color =  item_highlighted['text_color']

        outline_color = item_highlighted['outline_color']
        outline_width =  item_highlighted['outline_width']
        outline = item_highlighted['outline']

        bold = item_highlighted['bold']
        italic =  item_highlighted['italic']
        underline = item_highlighted['underline']

        alignment = item_highlighted['alignment']

        try:
            # Set values
            self.main.font_dropdown.setCurrentText(font_family) if font_family else None
            self.main.font_size_dropdown.setCurrentText(str(int(font_size))) if font_size else None

            if text_color is not None:
                self.main.block_font_color_button.setStyleSheet(
                    f"background-color: {text_color}; border: none; border-radius: 5px;"
                )
                self.main.block_font_color_button.setProperty('selected_color', text_color)

            if outline_color is not None:
                self.main.outline_font_color_button.setStyleSheet(
                    f"background-color: {outline_color}; border: none; border-radius: 5px;"
                )
                self.main.outline_font_color_button.setProperty('selected_color', outline_color)
            else:
                self.main.outline_font_color_button.setStyleSheet(
                    "background-color: white; border: none; border-radius: 5px;"
                )
                self.main.outline_font_color_button.setProperty('selected_color', '#ffffff')

            self.main.outline_width_dropdown.setCurrentText(str(outline_width)) if outline_width else None
            self.main.outline_checkbox.setChecked(outline)

            self.main.bold_button.setChecked(bold)
            self.main.italic_button.setChecked(italic)
            self.main.underline_button.setChecked(underline)

            alignment_to_button = {
                QtCore.Qt.AlignmentFlag.AlignLeft: 0,
                QtCore.Qt.AlignmentFlag.AlignCenter: 1,
                QtCore.Qt.AlignmentFlag.AlignRight: 2,
            }

            button_group = self.main.alignment_tool_group.get_button_group()

            if alignment in alignment_to_button:
                button_index = alignment_to_button[alignment]
                button_group.buttons()[button_index].setChecked(True)

        finally:
            self.unblock_text_item_widgets(self.widgets_to_block)

        if self.style_panel and item_highlighted:
            alignment = item_highlighted.get('alignment', QtCore.Qt.AlignmentFlag.AlignLeft)
            style = StyleState(
                font_family=item_highlighted.get('font_family', ''),
                font_size=int(item_highlighted.get('font_size', 32) or 32),
                text_align=self._alignment_to_name(alignment),
            )
            text_color = item_highlighted.get('text_color')
            if text_color:
                color = QColor(text_color)
                style.fill = (color.red(), color.green(), color.blue())
            outline_color = item_highlighted.get('outline_color')
            if outline_color:
                color = QColor(outline_color)
                style.stroke = (color.red(), color.green(), color.blue())
                style.stroke_enabled = True
            self._style_panel_updating = True
            try:
                self.style_panel.set_style(
                    style,
                    style.font_family,
                    style.font_size,
                    style.text_align,
                )
            finally:
                self._style_panel_updating = False

    # Rendering
    def render_text(self):
        if self.main.image_viewer.hasPhoto() and self.main.blk_list:
            self.main.set_tool(None)
            if not font_selected(self.main):
                return
            self.clear_text_edits()
            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()

            # Add items to the scene if they're not already present
            for item in self.main.image_viewer.text_items:
                if item not in self.main.image_viewer._scene.items():
                    self.main.image_viewer._scene.addItem(item)

            # Create a dictionary to map text items to their positions and rotations
            existing_text_items = {item: (int(item.pos().x()), int(item.pos().y()), item.rotation()) for item in self.main.image_viewer.text_items}

            # Identify new blocks based on position and rotation
            new_blocks = [
                blk for blk in self.main.blk_list
                if (int(blk.xyxy[0]), int(blk.xyxy[1]), blk.angle) not in existing_text_items.values()
            ]

            self.main.image_viewer.clear_rectangles()
            self.main.curr_tblock = None
            self.main.curr_tblock_item = None

            render_settings = self.render_settings()
            upper = render_settings.upper_case
            background_image = self._prepare_background_image(new_blocks, render_settings)

            target_lang = self.main.t_combo.currentText()
            target_lang_en = self.main.lang_mapping.get(target_lang, None)
            trg_lng_cd = get_language_code(target_lang_en)

            self.main.run_threaded(
            lambda: format_translations(self.main.blk_list, trg_lng_cd, upper_case=upper)
            )

            align_id = self.main.alignment_tool_group.get_dayu_checked()
            alignment = self.main.button_to_alignment[align_id]

            self.main.undo_group.activeStack().beginMacro('text_items_rendered')
            self.main.run_threaded(manual_wrap, self.on_render_complete, self.main.default_error_handler,
                              None, self.main, new_blocks, render_settings, alignment,
                              background_image)

    def on_render_complete(self, rendered_image: np.ndarray):
        # self.main.set_image(rendered_image) 
        self.main.loading.setVisible(False)
        self.main.enable_hbutton_group()
        self.main.undo_group.activeStack().endMacro()

    def render_settings(self) -> TextRenderingSettings:
        target_lang = self.main.lang_mapping.get(self.main.t_combo.currentText(), None)
        direction = get_layout_direction(target_lang)

        return TextRenderingSettings(
            alignment_id = self.main.alignment_tool_group.get_dayu_checked(),
            font_family = self.main.font_dropdown.currentText(),
            min_font_size = int(self.main.settings_page.ui.min_font_spinbox.value()),
            max_font_size = int(self.main.settings_page.ui.max_font_spinbox.value()),
            color = self.main.block_font_color_button.property('selected_color'),
            upper_case = self.main.settings_page.ui.uppercase_checkbox.isChecked(),
            outline = self.main.outline_checkbox.isChecked(),
            outline_color = self.main.outline_font_color_button.property('selected_color'),
            outline_width = self.main.outline_width_dropdown.currentText(),
            bold = self.main.bold_button.isChecked(),
            italic = self.main.italic_button.isChecked(),
            underline = self.main.underline_button.isChecked(),
            line_spacing = self.main.line_spacing_dropdown.currentText(),
            direction = direction
        )