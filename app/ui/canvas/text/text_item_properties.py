from dataclasses import dataclass, field
from typing import Optional, List, Any
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
import copy


def _normalize_bubble_style(style):
    if style is None:
        return None
    normalised = {}
    for key, value in style.items():
        if isinstance(value, list):
            normalised[key] = tuple(value)
        elif key == 'fill_gradient' and isinstance(value, dict):
            grad_norm = {}
            for gk, gv in value.items():
                if gk in {'start_rgba', 'end_rgba'} and isinstance(gv, (list, tuple)):
                    grad_norm[gk] = tuple(int(v) for v in gv)
                elif gk == 'angle':
                    grad_norm[gk] = float(gv)
                elif isinstance(gv, list):
                    grad_norm[gk] = tuple(gv)
                else:
                    grad_norm[gk] = gv
            normalised[key] = grad_norm
        elif key == 'text_alpha':
            normalised[key] = int(value)
        else:
            normalised[key] = value
    return normalised

@dataclass
class TextItemProperties:
    """Dataclass for TextBlockItem properties to reduce duplication in construction"""
    text: str = ""
    font_family: str = ""
    font_size: float = 20
    text_color: QColor = None
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter
    line_spacing: float = 1.2
    outline_color: Optional[QColor] = None
    outline_width: float = 1
    bold: bool = False
    italic: bool = False
    underline: bool = False
    direction: Qt.LayoutDirection = Qt.LayoutDirection.LeftToRight
    
    # Position and transformation properties
    position: tuple = (0, 0)  # (x, y)
    rotation: float = 0
    scale: float = 1.0
    transform_origin: Optional[tuple] = None  # (x, y)
    
    # Layout properties
    width: Optional[float] = None
    vertical: bool = False
    
    # Advanced properties
    selection_outlines: list = field(default_factory=list)
    bubble_style: Optional[dict] = None
            
    @classmethod
    def from_dict(cls, data: dict) -> 'TextItemProperties':
        """Create TextItemProperties from dictionary state"""
        props = cls()
        
        # Basic text properties
        props.text = data.get('text', '')
        props.font_family = data.get('font_family', '')
        props.font_size = data.get('font_size', 20)
        props.line_spacing = data.get('line_spacing', 1.2)
        props.bold = data.get('bold', False)
        props.italic = data.get('italic', False)
        props.underline = data.get('underline', False)
        
        # Color properties
        if 'text_color' in data:
            if isinstance(data['text_color'], QColor):
                props.text_color = data['text_color']
            else:
                props.text_color = QColor(data['text_color'])
        
        if 'outline_color' in data:
            if isinstance(data['outline_color'], QColor):
                props.outline_color = data['outline_color']
            elif data['outline_color']:
                props.outline_color = QColor(data['outline_color'])
                
        props.outline_width = data.get('outline_width', 1)
        
        # Alignment
        if 'alignment' in data:
            if isinstance(data['alignment'], int):
                props.alignment = Qt.AlignmentFlag(data['alignment'])
            else:
                props.alignment = data['alignment']
                
        # Direction
        if 'direction' in data:
            props.direction = data['direction']
            
        # Position and transformation
        props.position = data.get('position', (0, 0))
        props.rotation = data.get('rotation', 0)
        props.scale = data.get('scale', 1.0)
        props.transform_origin = data.get('transform_origin')
        
        # Layout
        props.width = data.get('width')
        props.vertical = data.get('vertical', False)
        
        # Advanced
        props.selection_outlines = data.get('selection_outlines', [])
        bubble_style = _normalize_bubble_style(data.get('bubble_style'))
        props.bubble_style = copy.deepcopy(bubble_style) if bubble_style is not None else None
        
        return props
    
    @classmethod
    def from_text_item(cls, item) -> 'TextItemProperties':
        """Create TextItemProperties from an existing TextBlockItem"""
        props = cls()
        
        # Basic text properties
        props.text = item.toHtml()
        props.font_family = item.font_family
        props.font_size = item.font_size
        props.text_color = item.text_color
        props.alignment = item.alignment
        props.line_spacing = item.line_spacing
        props.outline_color = item.outline_color
        props.outline_width = item.outline_width
        props.bold = item.bold
        props.italic = item.italic
        props.underline = item.underline
        props.direction = item.direction
        
        # Position and transformation
        props.position = (item.pos().x(), item.pos().y())
        props.rotation = item.rotation()
        props.scale = item.scale()
        if hasattr(item, 'transformOriginPoint'):
            origin = item.transformOriginPoint()
            props.transform_origin = (origin.x(), origin.y())
        
        # Layout properties
        props.width = item.boundingRect().width()
        props.vertical = getattr(item, 'vertical', False)
        
        # Advanced properties
        props.selection_outlines = getattr(item, 'selection_outlines', []).copy()
        bubble_style = _normalize_bubble_style(getattr(item, 'bubble_style', None))
        props.bubble_style = copy.deepcopy(bubble_style) if bubble_style is not None else None
        
        return props
    
    def to_dict(self) -> dict:
        """Convert TextItemProperties to dictionary"""
        bubble_style = None
        if self.bubble_style is not None:
            bubble_style = copy.deepcopy(self.bubble_style)
            for key in ('fill_rgba', 'text_rgb', 'outline_rgb', 'shadow_rgba', 'shadow_offset', 'padding'):
                if key in bubble_style and isinstance(bubble_style[key], tuple):
                    bubble_style[key] = list(bubble_style[key])
            gradient = bubble_style.get('fill_gradient') if bubble_style else None
            if isinstance(gradient, dict):
                gradient_copy = {}
                for gk, gv in gradient.items():
                    if isinstance(gv, tuple):
                        gradient_copy[gk] = list(gv)
                    else:
                        gradient_copy[gk] = gv
                bubble_style['fill_gradient'] = gradient_copy

        return {
            'text': self.text,
            'font_family': self.font_family,
            'font_size': self.font_size,
            'text_color': self.text_color,
            'alignment': self.alignment,
            'line_spacing': self.line_spacing,
            'outline_color': self.outline_color,
            'outline_width': self.outline_width,
            'bold': self.bold,
            'italic': self.italic,
            'underline': self.underline,
            'direction': self.direction,
            'position': self.position,
            'rotation': self.rotation,
            'scale': self.scale,
            'transform_origin': self.transform_origin,
            'width': self.width,
            'vertical': self.vertical,
            'selection_outlines': self.selection_outlines,
            'bubble_style': bubble_style,
        }
