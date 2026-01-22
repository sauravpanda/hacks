"""
ASCII Video Player module.

Plays video files as ASCII art in the terminal with:
- Playback controls (pause, seek, speed)
- Audio sync (optional)
- Frame extraction for LLM analysis
"""

import time
import threading
from typing import Optional, Generator, List, Tuple, Callable
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

from .converter import ASCIIConverter, CharacterSets
from .display import Display


@dataclass
class VideoInfo:
    """Video file metadata."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    codec: str


@dataclass
class VideoFrame:
    """A single video frame with metadata."""
    frame_number: int
    timestamp: float  # seconds
    image: np.ndarray  # BGR format
    ascii_art: Optional[str] = None


class VideoReader:
    """
    Reads video files and extracts frames.

    Supports seeking, frame extraction, and metadata retrieval.
    """

    def __init__(self, video_path: str):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None

    def open(self) -> bool:
        """Open the video file."""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            return False

        # Get video info
        self._info = VideoInfo(
            path=self.video_path,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=0.0,
            codec=self._decode_fourcc(int(self._cap.get(cv2.CAP_PROP_FOURCC)))
        )
        self._info.duration = self._info.frame_count / self._info.fps

        return True

    def close(self):
        """Close the video file."""
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def info(self) -> Optional[VideoInfo]:
        """Get video info."""
        return self._info

    @property
    def is_open(self) -> bool:
        """Check if video is open."""
        return self._cap is not None and self._cap.isOpened()

    def _decode_fourcc(self, fourcc: int) -> str:
        """Decode fourcc integer to string."""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def read_frame(self) -> Optional[VideoFrame]:
        """Read the next frame."""
        if not self.is_open:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        frame_num = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        return VideoFrame(
            frame_number=frame_num,
            timestamp=timestamp,
            image=frame
        )

    def seek(self, position: float):
        """
        Seek to position in video.

        Args:
            position: Position in seconds
        """
        if self.is_open:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000)

    def seek_frame(self, frame_number: int):
        """
        Seek to specific frame number.

        Args:
            frame_number: Frame index
        """
        if self.is_open:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_frame_at(self, position: float) -> Optional[VideoFrame]:
        """
        Get frame at specific time position.

        Args:
            position: Position in seconds

        Returns:
            VideoFrame or None
        """
        self.seek(position)
        return self.read_frame()

    def extract_frames(
        self,
        interval: float = 1.0,
        max_frames: Optional[int] = None
    ) -> Generator[VideoFrame, None, None]:
        """
        Extract frames at regular intervals.

        Args:
            interval: Time between frames in seconds
            max_frames: Maximum frames to extract

        Yields:
            VideoFrame objects
        """
        if not self.is_open or not self._info:
            return

        self.seek(0)
        count = 0
        next_time = 0.0

        while True:
            if max_frames and count >= max_frames:
                break

            self.seek(next_time)
            frame = self.read_frame()

            if frame is None:
                break

            yield frame
            count += 1
            next_time += interval

            if next_time >= self._info.duration:
                break

    def frames(self) -> Generator[VideoFrame, None, None]:
        """
        Iterate through all frames.

        Yields:
            VideoFrame objects
        """
        if not self.is_open:
            return

        self.seek(0)
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class ASCIIVideoPlayer:
    """
    Plays video files as ASCII art in the terminal.

    Features:
    - Real-time playback with frame rate control
    - Pause, seek, and speed controls
    - Enhancement options for better visibility
    - Optional audio playback
    """

    # Enhancement presets
    PRESET_DEFAULT = {"contrast": 1.0, "brightness": 1.0, "edge": False}
    PRESET_HIGH_CONTRAST = {"contrast": 1.8, "brightness": 1.1, "edge": False}
    PRESET_MOVEMENT = {"contrast": 2.0, "brightness": 1.0, "edge": True}
    PRESET_DARK = {"contrast": 1.5, "brightness": 1.4, "edge": False}
    PRESET_BRIGHT = {"contrast": 1.3, "brightness": 0.9, "edge": False}

    def __init__(
        self,
        video_path: str,
        width: Optional[int] = None,
        color: bool = True,
        charset: str = "standard",
        contrast: float = 1.0,
        brightness: float = 1.0,
        edge_enhance: bool = False,
        preset: Optional[str] = None
    ):
        """
        Initialize video player.

        Args:
            video_path: Path to video file
            width: ASCII output width (auto-detect if None)
            color: Enable color output
            charset: Character set to use
            contrast: Contrast multiplier (1.5-2.0 for movement)
            brightness: Brightness multiplier
            edge_enhance: Add edge detection for clearer shapes
            preset: Enhancement preset ('high_contrast', 'movement', 'dark', 'bright')
        """
        self.video_path = video_path
        self.reader = VideoReader(video_path)

        # Apply preset if specified
        if preset:
            presets = {
                "high_contrast": self.PRESET_HIGH_CONTRAST,
                "movement": self.PRESET_MOVEMENT,
                "dark": self.PRESET_DARK,
                "bright": self.PRESET_BRIGHT,
            }
            if preset in presets:
                p = presets[preset]
                contrast = p["contrast"]
                brightness = p["brightness"]
                edge_enhance = p["edge"]

        self.edge_enhance = edge_enhance

        # Auto-detect width
        if width is None:
            cols, _ = Display.get_terminal_size()
            width = cols - 2

        # Set up converter
        if charset == "braille":
            self.converter = ASCIIConverter(
                width=width, color=color, charset="braille",
                contrast=contrast, brightness=brightness
            )
        else:
            charset_map = {
                "standard": CharacterSets.STANDARD,
                "detailed": CharacterSets.DETAILED,
                "blocks": CharacterSets.BLOCKS,
                "minimal": CharacterSets.MINIMAL,
            }
            self.converter = ASCIIConverter(
                width=width,
                color=color,
                charset=charset_map.get(charset, CharacterSets.STANDARD),
                contrast=contrast,
                brightness=brightness
            )

        self.display = Display()
        self.width = width

        # Playback state
        self._playing = False
        self._paused = False
        self._speed = 1.0
        self._current_frame = 0
        self._auto_resize = True
        self._last_terminal_size = Display.get_terminal_size()

    def play(self, start_time: float = 0.0):
        """
        Start video playback.

        Args:
            start_time: Start position in seconds
        """
        if not self.reader.open():
            print(f"Error: Could not open video: {self.video_path}")
            return

        info = self.reader.info
        print(f"Playing: {info.path}")
        print(f"Resolution: {info.width}x{info.height}")
        print(f"Duration: {info.duration:.1f}s @ {info.fps:.1f} fps")
        print("Controls: [space] pause, [q] quit, [←/→] seek, [+/-] speed")
        time.sleep(2)

        self._playing = True
        self.reader.seek(start_time)

        try:
            from blessed import Terminal
            term = Terminal()
            with term.cbreak(), term.hidden_cursor():
                self._playback_loop(term)
        except ImportError:
            self._playback_loop_simple()

        self.reader.close()

    def _check_terminal_resize(self):
        """Check if terminal was resized and update width."""
        if not self._auto_resize:
            return

        current_size = Display.get_terminal_size()
        if current_size != self._last_terminal_size:
            self._last_terminal_size = current_size
            new_width = max(20, current_size[0] - 2)
            if new_width != self.width:
                self.width = new_width
                self.converter.width = new_width
                self.display.clear()

    def _apply_edge_enhance(self, image: Image.Image) -> Image.Image:
        """Apply edge enhancement to make shapes more visible."""
        from PIL import ImageFilter, ImageEnhance

        # Convert to grayscale for edge detection
        gray = image.convert("L")

        # Find edges
        edges = gray.filter(ImageFilter.FIND_EDGES)

        # Enhance edges
        edges = ImageEnhance.Contrast(edges).enhance(2.0)

        # Blend original grayscale with edges (70% original, 30% edges)
        blended = Image.blend(gray, edges, alpha=0.3)

        # Increase sharpness
        blended = blended.filter(ImageFilter.SHARPEN)

        return blended.convert("RGB")  # Convert back to RGB for color converter

    def _playback_loop(self, term):
        """Main playback loop with keyboard controls."""
        info = self.reader.info
        frame_duration = 1.0 / info.fps

        self.display.clear()

        while self._playing:
            start_time = time.time()

            # Check keyboard input
            key = term.inkey(timeout=0)
            if key:
                self._handle_key(key)

            # Check for terminal resize
            self._check_terminal_resize()

            if not self._paused:
                frame = self.reader.read_frame()
                if frame is None:
                    break

                # Convert to ASCII
                rgb_frame = frame.image[:, :, ::-1]  # BGR to RGB
                image = Image.fromarray(rgb_frame)

                # Apply edge enhancement if enabled
                if self.edge_enhance:
                    image = self._apply_edge_enhance(image)

                ascii_art = self.converter.convert(image)

                # Add status bar
                progress = frame.timestamp / info.duration
                status = self._get_status_bar(frame.timestamp, info.duration, progress)
                output = f"{ascii_art}\n{status}"

                self.display.render(output, clear=True)
                self._current_frame = frame.frame_number

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = (frame_duration / self._speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _playback_loop_simple(self):
        """Simple playback loop without keyboard controls."""
        info = self.reader.info
        frame_duration = 1.0 / info.fps

        self.display.clear()

        for frame in self.reader.frames():
            if not self._playing:
                break

            start_time = time.time()

            # Check for terminal resize
            self._check_terminal_resize()

            rgb_frame = frame.image[:, :, ::-1]
            image = Image.fromarray(rgb_frame)

            # Apply edge enhancement if enabled
            if self.edge_enhance:
                image = self._apply_edge_enhance(image)

            ascii_art = self.converter.convert(image)

            progress = frame.timestamp / info.duration
            status = self._get_status_bar(frame.timestamp, info.duration, progress)
            output = f"{ascii_art}\n{status}"

            self.display.render(output, clear=True)

            elapsed = time.time() - start_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_key(self, key):
        """Handle keyboard input."""
        key_str = str(key).lower() if hasattr(key, 'lower') else str(key)

        if key_str in ('q', '\x1b'):
            self._playing = False
        elif key_str == ' ':
            self._paused = not self._paused
        elif key_str == 'a':
            self._auto_resize = not self._auto_resize
            if self._auto_resize:
                self._check_terminal_resize()
        elif key.name == 'KEY_RIGHT':
            # Seek forward 5 seconds
            info = self.reader.info
            current = self._current_frame / info.fps
            self.reader.seek(min(current + 5, info.duration - 0.1))
        elif key.name == 'KEY_LEFT':
            # Seek backward 5 seconds
            info = self.reader.info
            current = self._current_frame / info.fps
            self.reader.seek(max(current - 5, 0))
        elif key_str in ('+', '='):
            self._speed = min(self._speed * 1.25, 4.0)
        elif key_str == '-':
            self._speed = max(self._speed / 1.25, 0.25)

    def _get_status_bar(self, current: float, duration: float, progress: float) -> str:
        """Generate playback status bar."""
        # Time display
        current_str = f"{int(current // 60):02d}:{int(current % 60):02d}"
        duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"

        # Progress bar
        bar_width = 30
        filled = int(progress * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Status indicators
        status = "▶" if not self._paused else "⏸"
        speed = f"{self._speed:.2f}x" if self._speed != 1.0 else ""
        auto = "AUTO" if self._auto_resize else ""

        parts = [
            f"{status} {current_str}/{duration_str}",
            f"[{bar}]",
            f"W:{self.width}",
            auto,
            speed,
            "[q]uit [space]pause [a]uto-size"
        ]

        return "\033[90m" + " ".join(filter(None, parts)) + "\033[0m"


class VideoToASCII:
    """
    Batch convert video to ASCII frames.

    Useful for creating ASCII versions of videos for:
    - LLM context
    - Text-based storage
    - Accessibility
    """

    def __init__(
        self,
        width: int = 80,
        color: bool = False,
        charset: str = "standard"
    ):
        """
        Initialize converter.

        Args:
            width: ASCII output width
            color: Include color codes
            charset: Character set to use
        """
        if charset == "braille":
            self.converter = ASCIIConverter(width=width, color=color, charset="braille")
        else:
            charset_map = {
                "standard": CharacterSets.STANDARD,
                "detailed": CharacterSets.DETAILED,
                "blocks": CharacterSets.BLOCKS,
                "minimal": CharacterSets.MINIMAL,
            }
            self.converter = ASCIIConverter(
                width=width,
                color=color,
                charset=charset_map.get(charset, CharacterSets.STANDARD)
            )

    def convert_video(
        self,
        video_path: str,
        interval: float = 1.0,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[int, str], None]] = None
    ) -> List[Tuple[float, str]]:
        """
        Convert video to ASCII frames.

        Args:
            video_path: Path to video file
            interval: Time between frames in seconds
            max_frames: Maximum frames to extract
            callback: Optional callback(frame_num, ascii_art) for progress

        Returns:
            List of (timestamp, ascii_art) tuples
        """
        results = []

        with VideoReader(video_path) as reader:
            if not reader.is_open:
                raise ValueError(f"Could not open video: {video_path}")

            for i, frame in enumerate(reader.extract_frames(interval, max_frames)):
                # Convert to ASCII
                rgb_frame = frame.image[:, :, ::-1]
                image = Image.fromarray(rgb_frame)
                ascii_art = self.converter.convert(image)

                results.append((frame.timestamp, ascii_art))

                if callback:
                    callback(i, ascii_art)

        return results

    def convert_frame(self, frame: np.ndarray) -> str:
        """
        Convert a single frame to ASCII.

        Args:
            frame: BGR image array

        Returns:
            ASCII art string
        """
        rgb_frame = frame[:, :, ::-1]
        image = Image.fromarray(rgb_frame)
        return self.converter.convert(image)
