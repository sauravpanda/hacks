"""
Recording and export module for ASCII art.

Supports exporting to:
- Animated GIF (with color support)
- MP4 video (renders ASCII as video frames)
- Text file sequences
"""

import os
import time
import tempfile
from typing import List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np


@dataclass
class ASCIIFrame:
    """Represents a single ASCII frame with metadata."""
    content: str
    timestamp: float
    width: int
    height: int
    has_color: bool = False


class ASCIIRecorder:
    """
    Records ASCII frames for later export.

    Captures frames in real-time and exports to various formats.
    """

    def __init__(self, max_frames: int = 1000):
        """
        Initialize recorder.

        Args:
            max_frames: Maximum frames to keep in memory
        """
        self.max_frames = max_frames
        self.frames: List[ASCIIFrame] = []
        self._recording = False
        self._start_time = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        """Get recording duration in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    def start(self):
        """Start recording."""
        self._recording = True
        self._start_time = time.time()
        self.frames = []

    def stop(self):
        """Stop recording."""
        self._recording = False

    def clear(self):
        """Clear all recorded frames."""
        self.frames = []

    def add_frame(self, content: str, has_color: bool = False):
        """
        Add a frame to the recording.

        Args:
            content: ASCII art string
            has_color: Whether frame contains ANSI color codes
        """
        if not self._recording:
            return

        if len(self.frames) >= self.max_frames:
            # Remove oldest frame
            self.frames.pop(0)

        lines = content.split('\n')
        width = max(len(self._strip_ansi(line)) for line in lines) if lines else 0
        height = len(lines)

        frame = ASCIIFrame(
            content=content,
            timestamp=time.time() - self._start_time,
            width=width,
            height=height,
            has_color=has_color
        )
        self.frames.append(frame)

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)


class GIFExporter:
    """
    Export ASCII frames to animated GIF.

    Renders each ASCII frame to an image and combines into GIF.
    """

    # Default monospace font settings
    DEFAULT_FONT_SIZE = 12
    DEFAULT_CHAR_WIDTH = 7
    DEFAULT_CHAR_HEIGHT = 14

    def __init__(
        self,
        font_size: int = 12,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        fg_color: Tuple[int, int, int] = (255, 255, 255),
        font_path: Optional[str] = None
    ):
        """
        Initialize GIF exporter.

        Args:
            font_size: Font size for rendering
            bg_color: Background RGB color
            fg_color: Default foreground RGB color
            font_path: Path to TTF font file (uses default if None)
        """
        self.font_size = font_size
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.font_path = font_path

        # Calculate character dimensions
        self.char_width = int(font_size * 0.6)
        self.char_height = int(font_size * 1.2)

        # Load font
        self._font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Load a monospace font."""
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, self.font_size)
            except Exception:
                pass

        # Try common monospace fonts
        font_names = [
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "UbuntuMono-R.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "Consolas",
            "Courier New",
        ]

        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, self.font_size)
            except Exception:
                continue

        # Fallback to default
        return ImageFont.load_default()

    def _parse_ansi_colors(self, text: str) -> List[Tuple[str, Tuple[int, int, int]]]:
        """
        Parse ANSI color codes and return list of (char, color) tuples.

        Args:
            text: Text with ANSI codes

        Returns:
            List of (character, RGB color) tuples
        """
        import re

        result = []
        current_color = self.fg_color
        i = 0

        while i < len(text):
            # Check for ANSI escape sequence
            match = re.match(r'\033\[38;2;(\d+);(\d+);(\d+)m', text[i:])
            if match:
                r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
                current_color = (r, g, b)
                i += len(match.group())
                continue

            # Check for reset
            if text[i:i+4] == '\033[0m':
                current_color = self.fg_color
                i += 4
                continue

            # Skip other ANSI codes
            if text[i:i+2] == '\033[':
                end = text.find('m', i)
                if end != -1:
                    i = end + 1
                    continue

            # Regular character
            if text[i] != '\n':
                result.append((text[i], current_color))
            else:
                result.append(('\n', current_color))
            i += 1

        return result

    def render_frame(self, frame: ASCIIFrame) -> Image.Image:
        """
        Render an ASCII frame to a PIL Image.

        Args:
            frame: ASCII frame to render

        Returns:
            PIL Image of the rendered frame
        """
        # Calculate image dimensions
        lines = frame.content.split('\n')
        width = frame.width * self.char_width + 20  # Add padding
        height = frame.height * self.char_height + 20

        # Create image
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)

        if frame.has_color:
            # Parse and render with colors
            y = 10
            for line in lines:
                chars = self._parse_ansi_colors(line + '\n')
                x = 10
                for char, color in chars:
                    if char == '\n':
                        break
                    draw.text((x, y), char, font=self._font, fill=color)
                    x += self.char_width
                y += self.char_height
        else:
            # Render without colors
            draw.text((10, 10), frame.content, font=self._font, fill=self.fg_color)

        return img

    def export(
        self,
        frames: List[ASCIIFrame],
        output_path: str,
        fps: float = 10,
        loop: int = 0,
        optimize: bool = True
    ):
        """
        Export frames to animated GIF.

        Args:
            frames: List of ASCII frames
            output_path: Output file path
            fps: Frames per second
            loop: Number of loops (0 = infinite)
            optimize: Optimize GIF size
        """
        if not frames:
            raise ValueError("No frames to export")

        # Render all frames
        images = [self.render_frame(frame) for frame in frames]

        # Calculate frame duration in milliseconds
        duration = int(1000 / fps)

        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=optimize
        )


class VideoExporter:
    """
    Export ASCII frames to video (MP4/AVI).

    Uses OpenCV to render ASCII frames as video.
    """

    def __init__(
        self,
        font_size: int = 14,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        fg_color: Tuple[int, int, int] = (255, 255, 255),
        font_path: Optional[str] = None
    ):
        """
        Initialize video exporter.

        Args:
            font_size: Font size for rendering
            bg_color: Background RGB color
            fg_color: Default foreground RGB color
            font_path: Path to TTF font file
        """
        self.gif_exporter = GIFExporter(
            font_size=font_size,
            bg_color=bg_color,
            fg_color=fg_color,
            font_path=font_path
        )

    def export(
        self,
        frames: List[ASCIIFrame],
        output_path: str,
        fps: float = 30,
        codec: str = 'mp4v'
    ):
        """
        Export frames to video file.

        Args:
            frames: List of ASCII frames
            output_path: Output file path (.mp4 or .avi)
            fps: Frames per second
            codec: Video codec (mp4v, XVID, etc.)
        """
        import cv2

        if not frames:
            raise ValueError("No frames to export")

        # Render first frame to get dimensions
        first_img = self.gif_exporter.render_frame(frames[0])
        width, height = first_img.size

        # Ensure dimensions are even (required by some codecs)
        width = width + (width % 2)
        height = height + (height % 2)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for frame in frames:
                # Render frame
                img = self.gif_exporter.render_frame(frame)

                # Resize if needed
                if img.size != (width, height):
                    img = img.resize((width, height))

                # Convert PIL to OpenCV format (RGB to BGR)
                cv_frame = np.array(img)
                cv_frame = cv_frame[:, :, ::-1]  # RGB to BGR

                out.write(cv_frame)
        finally:
            out.release()


class TextSequenceExporter:
    """
    Export ASCII frames as a sequence of text files.

    Useful for debugging or custom processing.
    """

    @staticmethod
    def export(
        frames: List[ASCIIFrame],
        output_dir: str,
        prefix: str = "frame",
        strip_ansi: bool = True
    ):
        """
        Export frames as numbered text files.

        Args:
            frames: List of ASCII frames
            output_dir: Output directory
            prefix: Filename prefix
            strip_ansi: Remove ANSI color codes
        """
        import re

        os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            content = frame.content
            if strip_ansi:
                content = re.sub(r'\033\[[0-9;]*m', '', content)

            filename = f"{prefix}_{i:05d}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
