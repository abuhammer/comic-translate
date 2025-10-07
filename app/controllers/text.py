from __future__ import annotations

import copy
import logging
import numpy as np
from typing import TYPE_CHECKING, Iterable, Optional

from PySide6 import QtCore
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog

from app.ui.commands.textformat import TextFormatCommand
from app.ui.commands.box import AddTextItemCommand
from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties

from modules.utils.textblock import TextBlock
from modules.rendering.render import TextRenderingSettings, manual_wrap
from modules.rendering.dynamic_bubble import compute_dynamic_bubble_style
from modules.utils.pipeline_utils import font_selected, get_language_code, \
    get_layout_direction, is_close
from modules.utils.translator_utils import format_translations

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from controller import ComicTranslate

class TextController:
    def __init__(self, main: ComicTranslate):
        self.main = main

        # List of widgets to block signals for during manual rendering
        widget_candidates = [
            getattr(self.main, 'font_dropdown', None),
            getattr(self.main, 'font_size_dropdown', None),
            getattr(self.main, 'line_spacing_dropdown', None),
            getattr(self.main, 'block_font_color_button', None),
            getattr(self.main, 'outline_font_color_button', None),
            getattr(self.main, 'outline_width_dropdown', None),
            getattr(self.main, 'outline_checkbox', None),
            getattr(self.main, 'bubble_mode_combo', None),
            getattr(self.main, 'bubble_color_button', None),
            getattr(self.main, 'bubble_min_alpha_spin', None),
            getattr(self.main, 'bubble_max_alpha_spin', None),
            getattr(self.main, 'bubble_plain_alpha_spin', None),
        ]
        self.widgets_to_block = [widget for widget in widget_candidates if widget is not None]

    def connect_text_item_signals(self, text_item: TextBlockItem):
        text_item.item_selected.connect(self.on_text_item_selected)
        text_item.item_deselected.connect(self.on_text_item_deselected)
        text_item.text_changed.connect(self.update_text_block_from_item)
        text_item.text_highlighted.connect(self.set_values_from_highlight)
        text_item.change_undo.connect(self.main.rect_item_ctrl.rect_change_undo)

    def _update_bubble_color_button(self, color: QColor) -> None:
        button = getattr(self.main, 'bubble_color_button', None)
        if button is None or not color.isValid():
            return
        button.setStyleSheet(
            f"background-color: {color.name()}; border: none; border-radius: 5px;"
        )
        button.setProperty('selected_color', color.name())
        button.update()

    @staticmethod
    def _relative_luminance_rgb(rgb: tuple[int, int, int]) -> float:
        def _linearise(component: float) -> float:
            component /= 255.0
            return component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4

        r, g, b = rgb
        r_lin = _linearise(r)
        g_lin = _linearise(g)
        b_lin = _linearise(b)
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    def _item_matches_block(self, item: TextBlockItem, blk: TextBlock) -> bool:
        if getattr(blk, 'xyxy', None) is None:
            return False
        bx, by = float(blk.xyxy[0]), float(blk.xyxy[1])
        ix, iy = item.pos().x(), item.pos().y()
        if abs(ix - bx) > 3.0 or abs(iy - by) > 3.0:
            return False
        angle_blk = float(getattr(blk, 'angle', 0.0) or 0.0)
        if abs(item.rotation() - angle_blk) > 2.0:
            return False
        return True

    def _find_text_item_for_block(self, blk: TextBlock) -> Optional[TextBlockItem]:
        for item in getattr(self.main.image_viewer, 'text_items', []):
            if isinstance(item, TextBlockItem) and self._item_matches_block(item, blk):
                return item
        return None

    def _plain_style_from_config(self, render_settings: TextRenderingSettings) -> dict:
        rgb = tuple(int(v) for v in render_settings.bubble_rgb[:3])
        mode = (render_settings.bubble_mode or 'auto').lower()
        if mode == 'plain':
            alpha = int(render_settings.bubble_plain_alpha)
        elif mode == 'translucent':
            alpha = int(max(render_settings.bubble_max_alpha, render_settings.bubble_plain_alpha))
        else:
            alpha = int(render_settings.bubble_max_alpha)

        fill_rgba = (*rgb, max(0, min(255, alpha)))
        luminance = self._relative_luminance_rgb(rgb)
        text_rgb = (0, 0, 0) if luminance >= 0.5 else (255, 255, 255)
        outline_rgb = (0, 0, 0) if text_rgb == (255, 255, 255) else (255, 255, 255)

        shadow_alpha = 120 if text_rgb == (255, 255, 255) else 90

        return {
            'fill_rgba': fill_rgba,
            'text_rgb': text_rgb,
            'outline_rgb': outline_rgb,
            'outline_width': 2.0,
            'shadow_rgba': (outline_rgb[0], outline_rgb[1], outline_rgb[2], shadow_alpha),
            'shadow_offset': (0.0, 1.0),
            'padding': (12.0, 8.0, 12.0, 8.0),
            'corner_radius': 18.0,
            'reason': 'plain_fallback',
        }

    def _apply_style_to_item(self, item: TextBlockItem, blk: TextBlock, style: Optional[dict]) -> None:
        if style is None:
            item.set_bubble_style(None)
            blk.bubble_style = None
            return

        item.set_bubble_style(style)

        text_rgb = style.get('text_rgb')
        if text_rgb:
            text_hex = '#{0:02X}{1:02X}{2:02X}'.format(*text_rgb[:3])
            blk.font_color = text_hex
            item.set_color(QColor(text_hex))

        outline_rgb = style.get('outline_rgb')
        outline_width = float(style.get('outline_width', getattr(blk, 'outline_width', 1.0)))
        blk.outline_width = outline_width
        if outline_rgb:
            outline_hex = '#{0:02X}{1:02X}{2:02X}'.format(*outline_rgb[:3])
            blk.outline_color = outline_hex
            item.set_outline(QColor(outline_hex), outline_width)
        else:
            blk.outline_color = ''
            item.set_outline(None, outline_width)

        blk.bubble_style = copy.deepcopy(style)
        item.update()

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
    def _image_covers_blocks(
        image: Optional[np.ndarray], blocks: Iterable[TextBlock]
    ) -> bool:
        """Ensure every block lies within the provided image bounds."""

        if image is None or getattr(image, "size", 0) == 0:
            return False

        height, width = image.shape[:2]
        for blk in blocks:
            if getattr(blk, "xyxy", None) is None:
                continue

            try:
                x1, y1, x2, y2 = (int(v) for v in blk.xyxy)
            except Exception:
                return False

            if x1 < 0 or y1 < 0:
                return False
            if x2 > width or y2 > height:
                return False

        return True

    def _prepare_background_image(
        self,
        blocks: Iterable[TextBlock],
        render_settings: TextRenderingSettings,
    ) -> Optional[np.ndarray]:
        """Collect a background image aligned with block coordinates."""

        if not getattr(render_settings, "auto_font_color", True):
            return None

        viewer_image = None
        try:
            viewer_image = self.main.image_viewer.get_image_array(
                paint_all=True, include_patches=True
            )
            if viewer_image is None:
                viewer_image = self.main.image_viewer.get_image_array(
                    paint_all=True, include_patches=False
                )
            if viewer_image is None:
                viewer_image = self.main.image_viewer.get_image_array()
            if viewer_image is not None:
                viewer_image = viewer_image.copy()
        except Exception:
            logger.exception("Failed to capture background for adaptive colours")
            viewer_image = None

        base_image = self._get_current_base_image()

        # Prefer the unmodified base image because block coordinates are derived
        # from the original detection resolution. Fall back to the viewer capture
        # only when the base image is unavailable or does not cover the blocks.
        if base_image is not None and self._image_covers_blocks(base_image, blocks):
            return base_image

        if viewer_image is not None and self._image_covers_blocks(viewer_image, blocks):
            return viewer_image

        return base_image if base_image is not None else viewer_image

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
        text_color_str = blk.font_color if getattr(blk, 'font_color', '') else render_settings.color
        text_color = QColor(text_color_str)

        id = render_settings.alignment_id
        alignment = self.main.button_to_alignment[id]
        line_spacing = float(render_settings.line_spacing)
        outline_color_str = blk.outline_color if getattr(blk, 'outline_color', '') else render_settings.outline_color
        outline_enabled = render_settings.outline or bool(getattr(blk, 'outline_color', '')) or bool(getattr(blk, 'bubble_style', None))
        outline_color = QColor(outline_color_str) if outline_enabled else None
        outline_width = float(getattr(blk, 'outline_width', render_settings.outline_width))
        bold = render_settings.bold
        italic = render_settings.italic
        underline = render_settings.underline
        direction = render_settings.direction
        bubble_style = getattr(blk, 'bubble_style', None)

        properties = TextItemProperties(
            text=text,
            font_family=font_family,
            font_size=font_size,
            text_color=text_color,
            alignment=alignment,
            line_spacing=line_spacing,
            outline_color=outline_color,
            outline_width=outline_width,
            bold=bold,
            italic=italic,
            underline=underline,
            direction=direction,
            bubble_style=bubble_style,
            position=(blk.xyxy[0], blk.xyxy[1]),
            rotation=blk.angle,
        )
        
        text_item = self.main.image_viewer.add_text_item(properties)
        text_item.set_plain_text(text)

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

    def on_bubble_mode_changed(self, *_):
        combo = getattr(self.main, 'bubble_mode_combo', None)
        if combo is None:
            return
        data = combo.currentData()
        text = combo.currentText()
        mode = (data or text or 'auto').lower()
        if mode not in {'auto', 'plain', 'translucent'}:
            mode = 'auto'
        self.main.bubble_style_config['bubble_mode'] = mode
        self.refresh_bubble_styles(recompute=True)

    def on_bubble_color_change(self):
        cfg = getattr(self.main, 'bubble_style_config', {})
        current_rgb = tuple(int(v) for v in cfg.get('bubble_rgb', (35, 100, 160))[:3])
        initial_color = QColor(*current_rgb)
        color = QColorDialog.getColor(initial_color, self.main, self.main.tr('Bubble Color'))
        if not color.isValid():
            return
        cfg['bubble_rgb'] = (color.red(), color.green(), color.blue())
        self._update_bubble_color_button(color)
        self.refresh_bubble_styles(recompute=True)

    def on_bubble_min_alpha_change(self, value: int):
        cfg = getattr(self.main, 'bubble_style_config', {})
        value = int(value)
        cfg['bubble_min_alpha'] = value
        max_alpha = int(cfg.get('bubble_max_alpha', value))
        if value > max_alpha:
            cfg['bubble_max_alpha'] = value
            spin = getattr(self.main, 'bubble_max_alpha_spin', None)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self.refresh_bubble_styles(recompute=True)

    def on_bubble_max_alpha_change(self, value: int):
        cfg = getattr(self.main, 'bubble_style_config', {})
        value = int(value)
        cfg['bubble_max_alpha'] = value
        min_alpha = int(cfg.get('bubble_min_alpha', value))
        if value < min_alpha:
            cfg['bubble_min_alpha'] = value
            spin = getattr(self.main, 'bubble_min_alpha_spin', None)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        plain_alpha = int(cfg.get('bubble_plain_alpha', value))
        if value > plain_alpha:
            cfg['bubble_plain_alpha'] = value
            spin_plain = getattr(self.main, 'bubble_plain_alpha_spin', None)
            if spin_plain is not None:
                spin_plain.blockSignals(True)
                spin_plain.setValue(value)
                spin_plain.blockSignals(False)
        self.refresh_bubble_styles(recompute=True)

    def on_bubble_plain_alpha_change(self, value: int):
        cfg = getattr(self.main, 'bubble_style_config', {})
        value = int(value)
        cfg['bubble_plain_alpha'] = value
        max_alpha = int(cfg.get('bubble_max_alpha', value))
        if value < max_alpha:
            cfg['bubble_max_alpha'] = value
            spin = getattr(self.main, 'bubble_max_alpha_spin', None)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self.refresh_bubble_styles(recompute=True)

    def refresh_bubble_styles(self, recompute: bool = False) -> None:
        text_items = getattr(self.main.image_viewer, 'text_items', [])
        if not text_items or not getattr(self.main, 'blk_list', []):
            return

        render_settings = self.render_settings()
        background_image = None
        if recompute:
            background_image = self._prepare_background_image(self.main.blk_list, render_settings)

        for blk in self.main.blk_list:
            item = self._find_text_item_for_block(blk)
            if item is None:
                continue

            style_dict = None
            style_obj = None

            if recompute and background_image is not None:
                try:
                    style_obj = compute_dynamic_bubble_style(
                        background_image,
                        blk,
                        bubble_rgb=render_settings.bubble_rgb,
                        min_alpha=render_settings.bubble_min_alpha,
                        max_alpha=render_settings.bubble_max_alpha,
                        text_min_contrast=render_settings.text_target_contrast,
                        bubble_mode=render_settings.bubble_mode,
                        plain_alpha=render_settings.bubble_plain_alpha,
                        plain_thresh_hi=render_settings.bubble_plain_hi,
                        plain_thresh_lo=render_settings.bubble_plain_lo,
                        flat_var=render_settings.bubble_flat_var,
                    )
                except Exception:
                    logger.exception("Failed to recompute bubble style for block", exc_info=True)

            if style_obj is not None:
                style_dict = style_obj.to_dict()
                blk.font_color = '#{0:02X}{1:02X}{2:02X}'.format(*style_obj.text_rgb)
                blk.outline_color = '#{0:02X}{1:02X}{2:02X}'.format(*style_obj.outline_rgb)
                blk.outline_width = style_obj.outline_width
            elif not recompute:
                existing_style = getattr(blk, 'bubble_style', None)
                style_dict = copy.deepcopy(existing_style) if existing_style else None
            else:
                style_dict = self._plain_style_from_config(render_settings)
                blk.font_color = '#{0:02X}{1:02X}{2:02X}'.format(*style_dict['text_rgb'])
                blk.outline_color = '#{0:02X}{1:02X}{2:02X}'.format(*style_dict['outline_rgb'])
                blk.outline_width = float(style_dict.get('outline_width', getattr(blk, 'outline_width', 2.0)))

            self._apply_style_to_item(item, blk, style_dict)

    # Formatting actions
    def on_font_dropdown_change(self, font_family: str):
        if self.main.curr_tblock_item and font_family:
            old_item = copy.copy(self.main.curr_tblock_item)
            font_size = int(self.main.font_size_dropdown.currentText())
            self.main.curr_tblock_item.set_font(font_family, font_size)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

    def on_font_size_change(self, font_size: str):
        if self.main.curr_tblock_item and font_size:
            old_item = copy.copy(self.main.curr_tblock_item)
            font_size = float(font_size)
            self.main.curr_tblock_item.set_font_size(font_size)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

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

    def on_outline_width_change(self, outline_width):
        if self.main.curr_tblock_item and self.main.outline_checkbox.isChecked():
            old_item = copy.copy(self.main.curr_tblock_item)
            outline_width = float(self.main.outline_width_dropdown.currentText())
            color_str = self.main.outline_font_color_button.property('selected_color')
            color = QColor(color_str)
            self.main.curr_tblock_item.set_outline(color, outline_width)

            command = TextFormatCommand(self.main.image_viewer, old_item, self.main.curr_tblock_item)
            self.main.push_command(command)

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

        bubble_mode = 'auto'
        if hasattr(self.main, 'bubble_mode_combo'):
            combo = self.main.bubble_mode_combo
            data = combo.currentData()
            bubble_mode = (data or combo.currentText() or 'auto').lower()

        bubble_config = getattr(self.main, 'bubble_style_config', {})
        bubble_rgb = bubble_config.get('bubble_rgb', (35, 100, 160))
        if isinstance(bubble_rgb, (list, tuple)):
            bubble_rgb = tuple(int(v) for v in bubble_rgb[:3])
        else:
            bubble_rgb = (35, 100, 160)
        bubble_min_alpha = int(bubble_config.get('bubble_min_alpha', 110))
        bubble_max_alpha = int(bubble_config.get('bubble_max_alpha', 205))
        bubble_plain_hi = float(bubble_config.get('bubble_plain_hi', 0.88))
        bubble_plain_lo = float(bubble_config.get('bubble_plain_lo', 0.12))
        bubble_flat_var = float(bubble_config.get('bubble_flat_var', 8e-4))
        bubble_plain_alpha = int(bubble_config.get('bubble_plain_alpha', 230))
        text_target_contrast = float(bubble_config.get('text_target_contrast', 4.5))

        # Keep the configuration in sync so project saves persist overrides
        self.main.bubble_style_config.update(
            {
                'bubble_mode': bubble_mode,
                'bubble_rgb': bubble_rgb,
                'bubble_min_alpha': bubble_min_alpha,
                'bubble_max_alpha': bubble_max_alpha,
                'bubble_plain_hi': bubble_plain_hi,
                'bubble_plain_lo': bubble_plain_lo,
                'bubble_flat_var': bubble_flat_var,
                'bubble_plain_alpha': bubble_plain_alpha,
                'text_target_contrast': text_target_contrast,
            }
        )

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
            direction = direction,
            bubble_mode = bubble_mode,
            bubble_rgb = bubble_rgb,
            bubble_min_alpha = bubble_min_alpha,
            bubble_max_alpha = bubble_max_alpha,
            bubble_plain_hi = bubble_plain_hi,
            bubble_plain_lo = bubble_plain_lo,
            bubble_flat_var = bubble_flat_var,
            bubble_plain_alpha = bubble_plain_alpha,
            text_target_contrast = text_target_contrast,
        )