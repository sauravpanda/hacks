"""
ASCII Cam - Terminal ASCII art converter with live camera support

A powerful terminal application that converts images and live camera feeds
to ASCII art with support for:
- Color ASCII output using ANSI escape codes
- Live webcam streaming
- Edge detection mode
- Custom character sets including braille
- Interactive keyboard controls
- GIF and video export
- Audio visualization
- Video playback as ASCII
- LLM context generation for AI agents
"""

__version__ = "2.0.0"
__author__ = "ASCII Cam Developer"

from .converter import ASCIIConverter, CharacterSets
from .camera import Camera, MockCamera
from .display import Display, OutputFile
from .effects import EdgeDetector, ImageEffects, Dithering
from .recorder import ASCIIRecorder, GIFExporter, VideoExporter
from .player import VideoReader, ASCIIVideoPlayer, VideoToASCII
from .llm import (
    LLMContextBuilder,
    BrowserCapture,
    StreamingVideoContext,
    video_to_llm_prompt,
    image_to_llm_context,
    browser_to_llm_context,
)

# Optional audio - may not be available
try:
    from .audio import AudioCapture, AudioVisualizer, ASCIIVisualizer
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

__all__ = [
    # Core
    "ASCIIConverter",
    "CharacterSets",
    "Camera",
    "MockCamera",
    "Display",
    "OutputFile",
    # Effects
    "EdgeDetector",
    "ImageEffects",
    "Dithering",
    # Recording
    "ASCIIRecorder",
    "GIFExporter",
    "VideoExporter",
    # Video
    "VideoReader",
    "ASCIIVideoPlayer",
    "VideoToASCII",
    # LLM Integration
    "LLMContextBuilder",
    "BrowserCapture",
    "StreamingVideoContext",
    "video_to_llm_prompt",
    "image_to_llm_context",
    "browser_to_llm_context",
]

if _AUDIO_AVAILABLE:
    __all__.extend(["AudioCapture", "AudioVisualizer", "ASCIIVisualizer"])
