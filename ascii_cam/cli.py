#!/usr/bin/env python3
"""
Interactive CLI for ASCII Camera Converter.

Provides a full-featured command-line interface with:
- Live webcam streaming
- Image file conversion
- Interactive keyboard controls
- Multiple output modes and effects
"""

import argparse
import sys
import os
import signal
import time
from typing import Optional
from PIL import Image

from .converter import ASCIIConverter, CharacterSets
from .camera import Camera, MockCamera
from .display import Display, OutputFile
from .effects import EdgeDetector, ImageEffects, Dithering


class InteractiveControls:
    """
    Handles keyboard input for interactive mode.

    Uses blessed library for non-blocking keyboard input
    and terminal control.
    """

    def __init__(self, app: "ASCIICamApp"):
        """Initialize controls."""
        self.app = app
        self._running = True

    def get_help_text(self) -> str:
        """Get help text for keyboard controls."""
        return """
Keyboard Controls:
  q / ESC     Quit
  c           Toggle color mode
  e           Toggle edge detection
  b           Toggle braille mode
  i           Toggle invert
  + / -       Adjust width
  [ / ]       Adjust brightness
  { / }       Adjust contrast
  1-5         Switch character set
  s           Save screenshot
  h / ?       Show this help
  SPACE       Pause/Resume
"""


class ASCIICamApp:
    """
    Main application class for ASCII Camera.

    Handles initialization, main loop, and cleanup.
    """

    # Character set presets
    CHARSETS = {
        "1": ("standard", CharacterSets.STANDARD),
        "2": ("detailed", CharacterSets.DETAILED),
        "3": ("blocks", CharacterSets.BLOCKS),
        "4": ("minimal", CharacterSets.MINIMAL),
        "5": ("braille", "braille"),
    }

    def __init__(
        self,
        width: Optional[int] = None,
        color: bool = False,
        charset: str = "standard",
        edge_mode: bool = False,
        invert: bool = False,
        brightness: float = 1.0,
        contrast: float = 1.0,
        fps: int = 15,
        camera_id: int = 0
    ):
        """
        Initialize the application.

        Args:
            width: Output width (auto-detect if None)
            color: Enable color output
            charset: Character set name
            edge_mode: Enable edge detection
            invert: Invert brightness
            brightness: Brightness adjustment
            contrast: Contrast adjustment
            fps: Target frame rate
            camera_id: Camera device ID
        """
        # Auto-detect width from terminal
        if width is None:
            cols, _ = Display.get_terminal_size()
            width = cols - 2  # Leave margin

        self.width = width
        self.color = color
        self.edge_mode = edge_mode
        self.invert = invert
        self.brightness = brightness
        self.contrast = contrast
        self.fps = fps
        self.camera_id = camera_id

        # Set up character set
        if charset == "braille":
            self.charset = "braille"
        else:
            self.charset = getattr(CharacterSets, charset.upper(), CharacterSets.STANDARD)

        # Initialize components
        self.converter = ASCIIConverter(
            width=width,
            charset=self.charset,
            color=color,
            invert=invert,
            brightness=brightness,
            contrast=contrast
        )
        self.display = Display(target_fps=fps)
        self.edge_detector = EdgeDetector(method="canny")

        # State
        self._running = False
        self._paused = False
        self._show_help = False

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def handler(signum, frame):
            self._running = False
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _process_frame(self, frame) -> str:
        """Process a frame through the pipeline."""
        # Convert numpy array to PIL Image
        if hasattr(frame, 'shape'):
            # OpenCV BGR to RGB
            frame = frame[:, :, ::-1]
            image = Image.fromarray(frame)
        else:
            image = frame

        # Apply edge detection if enabled
        if self.edge_mode:
            image = self.edge_detector.detect(image)
            # Convert back to RGB for color mode
            if self.color:
                image = image.convert("RGB")

        # Convert to ASCII
        return self.converter.convert(image)

    def run_camera(self, mock: bool = False):
        """
        Run live camera mode.

        Args:
            mock: Use mock camera for testing
        """
        self._setup_signal_handlers()

        # Initialize camera
        if mock:
            camera = MockCamera(pattern="gradient")
        else:
            camera = Camera(source=self.camera_id)

        try:
            # Import blessed for keyboard input
            from blessed import Terminal
            term = Terminal()
        except ImportError:
            term = None
            print("Note: Install 'blessed' for keyboard controls")

        with camera:
            if not camera.is_open:
                print(f"Error: Could not open camera {self.camera_id}")
                return 1

            print(f"Camera opened: {camera.resolution[0]}x{camera.resolution[1]}")
            print("Starting ASCII stream... Press 'q' to quit, 'h' for help")
            time.sleep(1)

            self._running = True
            self.display.clear()

            if term:
                with term.cbreak(), term.hidden_cursor():
                    self._camera_loop(camera, term)
            else:
                self._camera_loop_simple(camera)

        self.display.show_cursor()
        print("\nCamera stream ended.")
        return 0

    def _camera_loop(self, camera: Camera, term):
        """Main camera loop with keyboard input."""
        while self._running:
            # Check for keyboard input (non-blocking)
            key = term.inkey(timeout=0)

            if key:
                self._handle_key(key)

            if self._show_help:
                self._display_help()
                continue

            if not self._paused:
                frame = camera.read()
                if frame is None:
                    break

                ascii_art = self._process_frame(frame)

                # Add status bar
                status = self._get_status_bar()
                output = f"{ascii_art}\n{status}"

                self.display.render_frame(output)

    def _camera_loop_simple(self, camera: Camera):
        """Simple camera loop without keyboard input."""
        while self._running:
            frame = camera.read()
            if frame is None:
                break

            ascii_art = self._process_frame(frame)
            self.display.render_frame(ascii_art)

    def _handle_key(self, key):
        """Handle keyboard input."""
        key_char = str(key).lower() if hasattr(key, 'lower') else str(key)

        if key_char in ('q', '\x1b'):  # q or ESC
            self._running = False
        elif key_char == 'c':
            self.color = not self.color
            self.converter.color = self.color
        elif key_char == 'e':
            self.edge_mode = not self.edge_mode
        elif key_char == 'b':
            if self.converter.charset == "braille":
                self.converter.charset = CharacterSets.STANDARD
            else:
                self.converter.charset = "braille"
        elif key_char == 'i':
            self.invert = not self.invert
            self.converter.invert = self.invert
        elif key_char in ('+', '='):
            self.width = min(self.width + 10, 300)
            self.converter.width = self.width
        elif key_char == '-':
            self.width = max(self.width - 10, 20)
            self.converter.width = self.width
        elif key_char == '[':
            self.brightness = max(0.1, self.brightness - 0.1)
            self.converter.brightness = self.brightness
        elif key_char == ']':
            self.brightness = min(3.0, self.brightness + 0.1)
            self.converter.brightness = self.brightness
        elif key_char == '{':
            self.contrast = max(0.1, self.contrast - 0.1)
            self.converter.contrast = self.contrast
        elif key_char == '}':
            self.contrast = min(3.0, self.contrast + 0.1)
            self.converter.contrast = self.contrast
        elif key_char in self.CHARSETS:
            name, charset = self.CHARSETS[key_char]
            self.converter.charset = charset
        elif key_char == 's':
            self._save_screenshot()
        elif key_char in ('h', '?'):
            self._show_help = not self._show_help
        elif key_char == ' ':
            self._paused = not self._paused

    def _get_status_bar(self) -> str:
        """Generate status bar text."""
        parts = [
            f"W:{self.width}",
            f"B:{self.brightness:.1f}",
            f"C:{self.contrast:.1f}",
        ]

        if self.color:
            parts.append("COLOR")
        if self.edge_mode:
            parts.append("EDGE")
        if self.invert:
            parts.append("INV")
        if self._paused:
            parts.append("PAUSED")

        parts.append("[h]elp [q]uit")

        return "\033[90m" + " | ".join(parts) + "\033[0m"

    def _display_help(self):
        """Display help screen."""
        controls = InteractiveControls(self)
        self.display.clear()
        print(controls.get_help_text())
        print("\nPress any key to continue...")

    def _save_screenshot(self):
        """Save current frame as ASCII art."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ascii_screenshot_{timestamp}.txt"
        # Would need to store last frame for this
        print(f"\nScreenshot would be saved to: {filename}")

    def convert_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        html: bool = False
    ) -> int:
        """
        Convert a single image file.

        Args:
            image_path: Path to input image
            output_path: Optional output file path
            html: Save as HTML instead of text

        Returns:
            Exit code (0 for success)
        """
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return 1

        # Apply edge detection if enabled
        if self.edge_mode:
            image = self.edge_detector.detect(image)
            if self.color:
                image = image.convert("RGB")

        # Convert to ASCII
        ascii_art = self.converter.convert(image)

        if output_path:
            if html:
                OutputFile.save_html(ascii_art, output_path)
            else:
                OutputFile.save_text(ascii_art, output_path)
            print(f"Saved to: {output_path}")
        else:
            print(ascii_art)

        return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ASCII Camera - Convert images and camera feed to ASCII art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ascii-cam                     Start live camera mode
  ascii-cam image.jpg           Convert a single image
  ascii-cam -c -w 120           Camera with color, 120 chars wide
  ascii-cam image.png -o out.txt Save to file
  ascii-cam --mock              Test with mock camera (no webcam needed)

Character Sets (use -s option):
  standard  - Basic ASCII chars: .:-=+*#%@
  detailed  - Extended ASCII for better gradients
  blocks    - Block characters: ░▒▓█
  minimal   - Simple set: .-+*#
  braille   - High-res braille patterns
"""
    )

    # Positional argument for image file
    parser.add_argument(
        "image",
        nargs="?",
        help="Image file to convert (omit for camera mode)"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Save as HTML (preserves colors)"
    )

    # Display options
    parser.add_argument(
        "-w", "--width",
        type=int,
        help="Output width in characters (default: terminal width)"
    )
    parser.add_argument(
        "-c", "--color",
        action="store_true",
        help="Enable colored output"
    )
    parser.add_argument(
        "-s", "--charset",
        choices=["standard", "detailed", "blocks", "minimal", "braille"],
        default="standard",
        help="Character set to use"
    )
    parser.add_argument(
        "-i", "--invert",
        action="store_true",
        help="Invert brightness (light on dark)"
    )

    # Adjustments
    parser.add_argument(
        "-b", "--brightness",
        type=float,
        default=1.0,
        help="Brightness adjustment (default: 1.0)"
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast adjustment (default: 1.0)"
    )

    # Effects
    parser.add_argument(
        "-e", "--edge",
        action="store_true",
        help="Enable edge detection mode"
    )

    # Camera options
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target FPS for camera mode (default: 15)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock camera for testing"
    )

    args = parser.parse_args()

    # Create application
    app = ASCIICamApp(
        width=args.width,
        color=args.color,
        charset=args.charset,
        edge_mode=args.edge,
        invert=args.invert,
        brightness=args.brightness,
        contrast=args.contrast,
        fps=args.fps,
        camera_id=args.camera
    )

    # Run appropriate mode
    if args.image:
        return app.convert_image(
            args.image,
            output_path=args.output,
            html=args.html
        )
    else:
        return app.run_camera(mock=args.mock)


if __name__ == "__main__":
    sys.exit(main())
