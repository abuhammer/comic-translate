from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING
from dataclasses import asdict, is_dataclass
import imkit as imk

from PySide6 import QtWidgets
from PySide6.QtCore import QSettings
from PySide6.QtGui import QUndoStack

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.projects.project_state import save_state_to_proj_file, load_state_from_proj_file
from modules.utils.archives import make

if TYPE_CHECKING:
    from controller import ComicTranslate
    

class ProjectController:
    def __init__(self, main: ComicTranslate):
        self.main = main

    def save_and_make(self, output_path: str):
        self.main.run_threaded(self.save_and_make_worker, None, self.main.default_error_handler, None, output_path)

    def save_and_make_worker(self, output_path: str):
        self.main.image_ctrl.save_current_image_state()
        temp_dir = tempfile.mkdtemp()
        try:            
            if self.main.webtoon_mode:
                #  PASS 1: Pre-build a complete, up-to-date state map for ALL pages
                all_pages_current_state = {}
                loaded_pages = self.main.image_viewer.webtoon_manager.loaded_pages

                for page_idx, file_path in enumerate(self.main.image_files):
                    if page_idx in loaded_pages:
                        # For loaded pages, create state from the live scene. This state will
                        # only contain items that "belong" to this page.
                        viewer_state = self._create_text_items_state_from_scene(page_idx)
                    else:
                        # For unloaded pages, use the already stored state.
                        viewer_state = self.main.image_states[file_path].get('viewer_state', {}).copy()
                    all_pages_current_state[file_path] = {'viewer_state': viewer_state}

                # PASS 2: Render each page using the complete state map
                for page_idx, file_path in enumerate(self.main.image_files):
                    bname = os.path.basename(file_path)
                    rgb_img = self.main.load_image(file_path)

                    renderer = ImageSaveRenderer(rgb_img)
                    
                    # Use the pre-built, up-to-date state for this page.
                    viewer_state = all_pages_current_state[file_path]['viewer_state']
                    
                    # Create a temporary context object for the renderer.
                    temp_main_page_context = type('TempMainPage', (object,), {
                        'image_files': self.main.image_files,
                        'image_states': all_pages_current_state
                    })()

                    renderer.apply_patches(self.main.image_patches.get(file_path, []))
                    renderer.add_state_to_image(viewer_state, page_idx, temp_main_page_context)
                    sv_pth = os.path.join(temp_dir, bname)
                    renderer.save_image(sv_pth)
            else:
                # Regular mode: use original logic
                for file_path in self.main.image_files:
                    bname = os.path.basename(file_path)
                    rgb_img = self.main.load_image(file_path)

                    renderer = ImageSaveRenderer(rgb_img)
                    viewer_state = self.main.image_states[file_path]['viewer_state']
                    renderer.apply_patches(self.main.image_patches.get(file_path, []))
                    renderer.add_state_to_image(viewer_state)
                    sv_pth = os.path.join(temp_dir, bname)
                    renderer.save_image(sv_pth)

            # Call make function
            make(temp_dir, output_path)
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)

    def _create_text_items_state_from_scene(self, page_idx: int) -> dict:
        """
        Create text items state from current scene items for a loaded page in webtoon mode.
        An item "belongs" to a page if its origin point is within that page's vertical bounds.
        """
        
        webtoon_manager = self.main.image_viewer.webtoon_manager
        page_y_start = webtoon_manager.image_positions[page_idx]
        
        # Calculate page bottom boundary
        if page_idx < len(webtoon_manager.image_positions) - 1:
            page_y_end = webtoon_manager.image_positions[page_idx + 1]
        else:
            # For the last page, calculate its end based on its image height
            file_path = self.main.image_files[page_idx]
            rgb_img = self.main.load_image(file_path)
            page_y_end = page_y_start + rgb_img.shape[0]
        
        text_items_data = []
        
        # Find all text items that BELONG to this page
        for item in self.main.image_viewer._scene.items():
            if isinstance(item, TextBlockItem):
                text_item = item
                text_y = text_item.pos().y()
                
                # Check if the text item's origin is on this page
                if text_y >= page_y_start and text_y < page_y_end:
                    # Convert to page-local coordinates
                    scene_pos = text_item.pos()
                    page_local_x = scene_pos.x()
                    page_local_y = scene_pos.y() - page_y_start
                    
                    # Use TextItemProperties for consistent serialization
                    text_props = TextItemProperties.from_text_item(text_item)
                    # Override position to use page-local coordinates
                    text_props.position = (page_local_x, page_local_y)
                    
                    text_items_data.append(text_props.to_dict())
        
        # Return viewer state with the collected text items
        return {
            'text_items_state': text_items_data,
            'push_to_stack': False  # Don't push to undo stack during save
        }

    def launch_save_proj_dialog(self):
        file_dialog = QtWidgets.QFileDialog()
        file_name, _ = file_dialog.getSaveFileName(
            self.main,
            "Save Project As",
            "untitled",
            "Project Files (*.ctpr);;All Files (*)"
        )

        return file_name

    def run_save_proj(self, file_name):
        self.main.project_file = file_name
        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()
        self.main.run_threaded(self.save_project, None,
                                self.main.default_error_handler, self.main.on_manual_finished, file_name)
        
    def save_current_state(self):
        if self.main.webtoon_mode:
            webtoon_manager = self.main.image_viewer.webtoon_manager
            webtoon_manager.scene_item_manager.save_all_scene_items_to_states()
            webtoon_manager.save_view_state()
        else:
            self.main.image_ctrl.save_current_image_state()

    def thread_save_project(self):
        file_name = ""
        self.save_current_state()
        if self.main.project_file:
            file_name = self.main.project_file
        else:
            file_name = self.launch_save_proj_dialog()

        if file_name:
            self.run_save_proj(file_name)

    def thread_save_as_project(self):
        file_name = self.launch_save_proj_dialog()
        if file_name:
            self.save_current_state()
            self.run_save_proj(file_name)

    def save_project(self, file_name):
        save_state_to_proj_file(self.main, file_name)

    def update_ui_from_project(self):
        index = self.main.curr_img_idx
        self.main.image_ctrl.update_image_cards()

        # highlight the row that matches the current image
        self.main.page_list.blockSignals(True)
        if 0 <= index < self.main.page_list.count():
            self.main.page_list.setCurrentRow(index)
            self.main.image_ctrl.highlight_card(index)
        self.main.page_list.blockSignals(False)

        for file in self.main.image_files:
            stack = QUndoStack(self.main)
            self.main.undo_stacks[file] = stack
            self.main.undo_group.addStack(stack)

        self.main.run_threaded(
            lambda: self.main.load_image(self.main.image_files[index]),
            lambda result: self._display_image_and_set_mode(result, index),
            self.main.default_error_handler
        )

    def _display_image_and_set_mode(self, rgb_image, index: int):
        """Display the image and then set the appropriate mode."""
        # First display the image normally
        self.main.image_ctrl.display_image_from_loaded(rgb_image, index, switch_page=False)
        
        # Now that the UI is ready, activate webtoon mode
        if self.main.webtoon_mode:
            self.main.webtoon_toggle.setChecked(True)
            self.main.webtoon_ctrl.switch_to_webtoon_mode()

    def thread_load_project(self, file_name):
        self.main.image_ctrl.clear_state()
        self.main.run_threaded(
            self.load_project, 
            self.load_state_to_ui,
            self.main.default_error_handler, 
            self.update_ui_from_project, 
            file_name
        )

    def load_project(self, file_name):
        self.main.project_file = file_name
        return load_state_from_proj_file(self.main, file_name)
    
    def load_state_to_ui(self, saved_ctx: str):
        self.main.settings_page.ui.extra_context.setPlainText(saved_ctx)

    def save_main_page_settings(self):
        settings = QSettings("ComicLabs", "ComicTranslate")

        self.process_group('text_rendering', self.main.render_settings(), settings)

        settings.beginGroup('text_rendering')
        bubble_cfg = getattr(self.main, 'bubble_style_config', {})
        settings.setValue('bubble_rgb', list(bubble_cfg.get('bubble_rgb', (35, 100, 160))))
        settings.setValue('text_color_mode', bubble_cfg.get('text_color_mode', 'auto'))
        settings.setValue('custom_text_rgb', list(bubble_cfg.get('custom_text_rgb', (0, 0, 0))))
        settings.setValue('text_fill_opacity', float(bubble_cfg.get('text_fill_opacity', 1.0)))
        settings.setValue('stroke_enabled', bool(bubble_cfg.get('stroke_enabled', False)))
        settings.setValue('stroke_width', float(bubble_cfg.get('stroke_width', 2.0)))
        settings.setValue('stroke_opacity', float(bubble_cfg.get('stroke_opacity', 1.0)))
        settings.setValue('auto_contrast', bool(bubble_cfg.get('auto_contrast', True)))
        settings.setValue('text_target_contrast', float(bubble_cfg.get('text_target_contrast', 4.5)))
        settings.setValue('background_box_mode', bubble_cfg.get('background_box_mode', 'off'))
        settings.setValue('background_box_opacity', float(bubble_cfg.get('background_box_opacity', 0.25)))
        settings.setValue('background_plain_hi', float(bubble_cfg.get('background_plain_hi', 0.95)))
        settings.setValue('background_plain_lo', float(bubble_cfg.get('background_plain_lo', 0.05)))
        settings.setValue('flat_variance_threshold', float(bubble_cfg.get('flat_variance_threshold', 8e-4)))
        settings.setValue('auto_stroke_opacity', float(bubble_cfg.get('auto_stroke_opacity', 0.6)))
        settings.endGroup()

        settings.beginGroup("main_page")
        # Save languages in English
        settings.setValue("source_language", self.main.lang_mapping[self.main.s_combo.currentText()])
        settings.setValue("target_language", self.main.lang_mapping[self.main.t_combo.currentText()])

        settings.setValue("mode", "manual" if self.main.manual_radio.isChecked() else "automatic")

        # Save brush and eraser sizes
        settings.setValue("brush_size", self.main.image_viewer.brush_size)
        settings.setValue("eraser_size", self.main.image_viewer.eraser_size)

        settings.endGroup()

        # Save window state
        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.main.saveGeometry())
        settings.setValue("state", self.main.saveState())
        settings.endGroup()

    def load_main_page_settings(self):
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("main_page")

        # Load languages and convert back to current language
        source_lang = settings.value("source_language", "Korean")
        target_lang = settings.value("target_language", "English")

        # Use reverse mapping to get the translated language names
        self.main.s_combo.setCurrentText(self.main.reverse_lang_mapping.get(source_lang, self.main.tr("Korean")))
        self.main.t_combo.setCurrentText(self.main.reverse_lang_mapping.get(target_lang, self.main.tr("English")))

        mode = settings.value("mode", "manual")
        if mode == "manual":
            self.main.manual_radio.setChecked(True)
            self.main.manual_mode_selected()
        else:
            self.main.automatic_radio.setChecked(True)
            self.main.batch_mode_selected()

        # Load brush and eraser sizes
        brush_size = int(settings.value("brush_size", 10))  # Default value is 10
        eraser_size = int(settings.value("eraser_size", 20))  # Default value is 20
        self.main.image_viewer.brush_size = brush_size
        self.main.image_viewer.eraser_size = eraser_size

        settings.endGroup()

        # Load window state
        settings.beginGroup("MainWindow")
        geometry = settings.value("geometry")
        state = settings.value("state")
        if geometry is not None:
            self.main.restoreGeometry(geometry)
        if state is not None:
            self.main.restoreState(state)
        settings.endGroup()

        # Load text rendering settings
        settings.beginGroup('text_rendering')
        alignment = settings.value('alignment_id', 1, type=int) # Default value is 1 which is Center
        self.main.alignment_tool_group.set_dayu_checked(alignment)

        self.main.font_dropdown.setCurrentText(settings.value('font_family', ''))
        min_font_size = settings.value('min_font_size', 12)  # Default value is 12
        max_font_size = settings.value('max_font_size', 40) # Default value is 40
        self.main.settings_page.ui.min_font_spinbox.setValue(int(min_font_size))
        self.main.settings_page.ui.max_font_spinbox.setValue(int(max_font_size))

        color = settings.value('color', '#000000')
        self.main.block_font_color_button.setStyleSheet(f"background-color: {color}; border: none; border-radius: 5px;")
        self.main.block_font_color_button.setProperty('selected_color', color)
        self.main.settings_page.ui.uppercase_checkbox.setChecked(settings.value('upper_case', False, type=bool))
        self.main.outline_checkbox.setChecked(settings.value('outline', True, type=bool))

        self.main.line_spacing_dropdown.setCurrentText(settings.value('line_spacing', '1.0'))
        self.main.outline_width_dropdown.setCurrentText(settings.value('outline_width', '1.0'))
        outline_color = settings.value('outline_color', '#FFFFFF')
        self.main.outline_font_color_button.setStyleSheet(f"background-color: {outline_color}; border: none; border-radius: 5px;")
        self.main.outline_font_color_button.setProperty('selected_color', outline_color)

        self.main.bold_button.setChecked(settings.value('bold', False, type=bool))
        self.main.italic_button.setChecked(settings.value('italic', False, type=bool))
        self.main.underline_button.setChecked(settings.value('underline', False, type=bool))

        def _parse_rgb(value, default):
            if isinstance(value, str):
                stripped = value.strip().lstrip('[').lstrip('(').rstrip(']').rstrip(')')
                if stripped.startswith('#') and len(stripped) in (7, 4):
                    try:
                        hex_value = stripped[1:]
                        if len(hex_value) == 3:
                            hex_value = ''.join(ch * 2 for ch in hex_value)
                        return tuple(int(hex_value[i:i + 2], 16) for i in range(0, 6, 2))
                    except ValueError:
                        return default
                parts = [p.strip() for p in stripped.split(',') if p.strip()]
                try:
                    return tuple(int(float(p)) for p in parts[:3])
                except ValueError:
                    return default
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return tuple(int(v) for v in value[:3])
            return default

        bubble_rgb = _parse_rgb(settings.value('bubble_rgb', [35, 100, 160]), (35, 100, 160))
        text_color_mode = str(settings.value('text_color_mode', 'auto') or 'auto').lower()
        if text_color_mode not in {'auto', 'black', 'white', 'custom'}:
            text_color_mode = 'auto'
        custom_text_rgb = _parse_rgb(settings.value('custom_text_rgb', [0, 0, 0]), (0, 0, 0))

        text_fill_opacity = float(settings.value('text_fill_opacity', 1.0))
        text_fill_opacity = max(0.0, min(1.0, text_fill_opacity))
        stroke_enabled = settings.value('stroke_enabled', False, type=bool)
        stroke_width = float(settings.value('stroke_width', 2.0))
        stroke_opacity = float(settings.value('stroke_opacity', 1.0))
        stroke_opacity = max(0.0, min(1.0, stroke_opacity))

        auto_contrast = settings.value('auto_contrast', True, type=bool)
        text_target_contrast = float(settings.value('text_target_contrast', 4.5))

        background_box_mode = str(settings.value('background_box_mode', 'off') or 'off').lower()
        if background_box_mode not in {'off', 'auto', 'on'}:
            background_box_mode = 'off'
        background_box_opacity = float(settings.value('background_box_opacity', 0.25))
        background_box_opacity = max(0.0, min(0.3, background_box_opacity))

        background_plain_hi = float(settings.value('background_plain_hi', 0.95))
        background_plain_lo = float(settings.value('background_plain_lo', 0.05))
        flat_variance_threshold = float(settings.value('flat_variance_threshold', 4e-4))
        auto_stroke_opacity = float(settings.value('auto_stroke_opacity', 0.6))
        auto_stroke_opacity = max(0.0, min(1.0, auto_stroke_opacity))

        self.main.bubble_style_config.update(
            {
                'bubble_rgb': bubble_rgb,
                'text_color_mode': text_color_mode,
                'custom_text_rgb': custom_text_rgb,
                'text_fill_opacity': text_fill_opacity,
                'stroke_enabled': stroke_enabled,
                'stroke_width': stroke_width,
                'stroke_opacity': stroke_opacity,
                'auto_contrast': auto_contrast,
                'text_target_contrast': text_target_contrast,
                'background_box_mode': background_box_mode,
                'background_box_opacity': background_box_opacity,
                'background_plain_hi': background_plain_hi,
                'background_plain_lo': background_plain_lo,
                'flat_variance_threshold': flat_variance_threshold,
                'auto_stroke_opacity': auto_stroke_opacity,
            }
        )

        color_button = getattr(self.main, 'bubble_color_button', None)
        if color_button is not None:
            bubble_hex = '#{0:02X}{1:02X}{2:02X}'.format(*bubble_rgb)
            color_button.blockSignals(True)
            color_button.setStyleSheet(
                f"background-color: {bubble_hex}; border: none; border-radius: 5px;"
            )
            color_button.setProperty('selected_color', bubble_hex)
            color_button.blockSignals(False)

        text_color_combo = getattr(self.main, 'text_color_mode_combo', None)
        if text_color_combo is not None:
            text_color_combo.blockSignals(True)
            index = text_color_combo.findData(text_color_mode)
            if index < 0:
                index = 0
            text_color_combo.setCurrentIndex(index)
            text_color_combo.blockSignals(False)

        custom_button = getattr(self.main, 'custom_text_color_button', None)
        if custom_button is not None:
            custom_hex = '#{0:02X}{1:02X}{2:02X}'.format(*custom_text_rgb)
            custom_button.blockSignals(True)
            custom_button.setStyleSheet(
                f"background-color: {custom_hex}; border: none; border-radius: 5px;"
            )
            custom_button.setProperty('selected_color', custom_hex)
            custom_button.blockSignals(False)

        text_opacity_pct = int(round(text_fill_opacity * 100))
        text_spin = getattr(self.main, 'text_opacity_spin', None)
        if text_spin is not None:
            text_spin.blockSignals(True)
            text_spin.setValue(text_opacity_pct)
            text_spin.blockSignals(False)
        text_slider = getattr(self.main, 'text_opacity_slider', None)
        if text_slider is not None:
            text_slider.blockSignals(True)
            text_slider.setValue(text_opacity_pct)
            text_slider.blockSignals(False)

        stroke_checkbox = getattr(self.main, 'stroke_checkbox', None)
        if stroke_checkbox is not None:
            stroke_checkbox.blockSignals(True)
            stroke_checkbox.setChecked(stroke_enabled)
            stroke_checkbox.blockSignals(False)

        stroke_width_spin = getattr(self.main, 'stroke_width_spin', None)
        if stroke_width_spin is not None:
            stroke_width_spin.blockSignals(True)
            stroke_width_spin.setValue(int(round(stroke_width)))
            stroke_width_spin.blockSignals(False)

        stroke_opacity_pct = int(round(stroke_opacity * 100))
        stroke_opacity_spin = getattr(self.main, 'stroke_opacity_spin', None)
        if stroke_opacity_spin is not None:
            stroke_opacity_spin.blockSignals(True)
            stroke_opacity_spin.setValue(stroke_opacity_pct)
            stroke_opacity_spin.blockSignals(False)
        stroke_opacity_slider = getattr(self.main, 'stroke_opacity_slider', None)
        if stroke_opacity_slider is not None:
            stroke_opacity_slider.blockSignals(True)
            stroke_opacity_slider.setValue(stroke_opacity_pct)
            stroke_opacity_slider.blockSignals(False)

        auto_contrast_checkbox = getattr(self.main, 'auto_contrast_checkbox', None)
        if auto_contrast_checkbox is not None:
            auto_contrast_checkbox.blockSignals(True)
            auto_contrast_checkbox.setChecked(auto_contrast)
            auto_contrast_checkbox.blockSignals(False)

        box_mode_combo = getattr(self.main, 'background_box_mode_combo', None)
        if box_mode_combo is not None:
            box_mode_combo.blockSignals(True)
            idx = box_mode_combo.findData(background_box_mode)
            if idx < 0:
                idx = 0
            box_mode_combo.setCurrentIndex(idx)
            box_mode_combo.blockSignals(False)

        box_opacity_pct = int(round(background_box_opacity * 100))
        box_opacity_pct = max(0, min(30, box_opacity_pct))
        box_spin = getattr(self.main, 'background_box_opacity_spin', None)
        if box_spin is not None:
            box_spin.blockSignals(True)
            box_spin.setValue(box_opacity_pct)
            box_spin.blockSignals(False)
        box_slider = getattr(self.main, 'background_box_opacity_slider', None)
        if box_slider is not None:
            box_slider.blockSignals(True)
            box_slider.setValue(box_opacity_pct)
            box_slider.blockSignals(False)

        settings.endGroup()

    def process_group(self, group_key, group_value, settings_obj: QSettings):
        """Helper function to process a group and its nested values."""
        if is_dataclass(group_value):
            group_value = asdict(group_value)
        if isinstance(group_value, dict):
            settings_obj.beginGroup(group_key)
            for sub_key, sub_value in group_value.items():
                self.process_group(sub_key, sub_value, settings_obj)
            settings_obj.endGroup()
        else:
            # Convert value to English using mappings if available
            mapped_value = self.main.settings_page.ui.value_mappings.get(group_value, group_value)
            settings_obj.setValue(group_key, mapped_value)

