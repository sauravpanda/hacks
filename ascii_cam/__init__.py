"""
ASCII Cam - Terminal ASCII art converter with live camera support

A powerful terminal application that converts images and live camera feeds
to ASCII art with support for:
- Color ASCII output using ANSI escape codes
- Live webcam streaming
- Edge detection mode
- Custom character sets including braille
- Interactive keyboard controls
"""

__version__ = "1.0.0"
__author__ = "ASCII Cam Developer"

from .converter import ASCIIConverter
from .camera import Camera
from .display import Display
from .effects import EdgeDetector

__all__ = ["ASCIIConverter", "Camera", "Display", "EdgeDetector"]
