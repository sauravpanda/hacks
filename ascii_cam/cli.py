#!/usr/bin/env python3
"""
Interactive CLI for ASCII Camera Converter.

Provides a full-featured command-line interface with:
- Live webcam streaming
- Image file conversion
- Video playback as ASCII
- Audio visualization
- GIF/Video recording and export
- LLM context generation
- Interactive keyboard controls
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
  r           Toggle recording
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

        # Recording state
        self.recorder = None
        self._recording = False

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

                # Record if enabled
                if self._recording and self.recorder:
                    self.recorder.add_frame(ascii_art, has_color=self.color)

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
        elif key_char == 'r':
            self._toggle_recording()
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

    def _toggle_recording(self):
        """Toggle recording on/off."""
        from .recorder import ASCIIRecorder

        if self._recording:
            self._recording = False
            if self.recorder and self.recorder.frame_count > 0:
                self._save_recording()
        else:
            self.recorder = ASCIIRecorder()
            self.recorder.start()
            self._recording = True

    def _save_recording(self):
        """Save recorded frames to GIF."""
        from .recorder import GIFExporter

        if not self.recorder or self.recorder.frame_count == 0:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ascii_recording_{timestamp}.gif"

        exporter = GIFExporter()
        exporter.export(self.recorder.frames, filename, fps=10)
        print(f"\nSaved recording to: {filename}")

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
        if self._recording:
            parts.append(f"●REC({self.recorder.frame_count})")
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


def cmd_camera(args):
    """Handle camera command."""
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
    return app.run_camera(mock=args.mock)


def cmd_image(args):
    """Handle image conversion command."""
    app = ASCIICamApp(
        width=args.width,
        color=args.color,
        charset=args.charset,
        edge_mode=args.edge,
        invert=args.invert,
        brightness=args.brightness,
        contrast=args.contrast,
    )
    return app.convert_image(args.image, args.output, args.html)


def cmd_video(args):
    """Handle video playback command."""
    from .player import ASCIIVideoPlayer

    player = ASCIIVideoPlayer(
        args.video,
        width=args.width,
        color=args.color,
        charset=args.charset
    )
    player.play()
    return 0


def cmd_audio(args):
    """Handle audio visualizer command."""
    try:
        from .audio import AudioVisualizer, MockAudioCapture
    except ImportError:
        print("Error: PyAudio is required for audio visualization.")
        print("Install with: pip install pyaudio")
        return 1

    from .display import Display

    # Auto-detect dimensions
    cols, rows = Display.get_terminal_size()
    width = args.width or (cols - 2)
    height = args.height or (rows - 4)

    visualizer = AudioVisualizer(
        width=width,
        height=height,
        mode=args.mode,
        color=args.color,
        mock=args.mock
    )

    display = Display(target_fps=30)

    print(f"Audio Visualizer - Mode: {args.mode}")
    print("Press Ctrl+C to quit")
    time.sleep(1)

    display.clear()

    try:
        with visualizer:
            while True:
                frame = visualizer.get_frame()
                if frame:
                    display.render_frame(frame)
                time.sleep(0.016)  # ~60fps
    except KeyboardInterrupt:
        pass

    display.show_cursor()
    print("\nAudio visualizer ended.")
    return 0


def cmd_record(args):
    """Handle recording command."""
    from .recorder import ASCIIRecorder, GIFExporter, VideoExporter

    app = ASCIICamApp(
        width=args.width,
        color=args.color,
        charset=args.charset,
        fps=args.fps,
    )

    recorder = ASCIIRecorder(max_frames=args.max_frames)

    # Initialize camera
    if args.mock:
        camera = MockCamera(pattern="gradient")
    else:
        camera = Camera(source=args.camera)

    with camera:
        if not camera.is_open:
            print(f"Error: Could not open camera")
            return 1

        print(f"Recording for {args.duration} seconds...")
        print("Press Ctrl+C to stop early")

        recorder.start()
        start_time = time.time()

        try:
            while time.time() - start_time < args.duration:
                frame = camera.read()
                if frame is None:
                    break

                ascii_art = app._process_frame(frame)
                recorder.add_frame(ascii_art, has_color=args.color)

                # Show progress
                elapsed = time.time() - start_time
                print(f"\rFrames: {recorder.frame_count} | Time: {elapsed:.1f}s", end="")
                time.sleep(1.0 / args.fps)

        except KeyboardInterrupt:
            pass

        recorder.stop()

    print(f"\n\nRecorded {recorder.frame_count} frames")

    # Export
    output = args.output or f"recording_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.format == "gif":
        if not output.endswith('.gif'):
            output += '.gif'
        exporter = GIFExporter()
        exporter.export(recorder.frames, output, fps=args.fps)
    else:
        if not output.endswith('.mp4'):
            output += '.mp4'
        exporter = VideoExporter()
        exporter.export(recorder.frames, output, fps=args.fps)

    print(f"Saved to: {output}")
    return 0


def cmd_llm(args):
    """Handle LLM context generation command."""
    from .llm import video_to_llm_prompt, image_to_llm_context

    input_path = args.input

    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        # Video file
        prompt = video_to_llm_prompt(
            input_path,
            task=args.task,
            width=args.width,
            max_frames=args.frames,
            sample_interval=args.interval
        )
    else:
        # Image file
        prompt = image_to_llm_context(
            input_path,
            width=args.width,
            task=args.task
        )

    if args.output:
        with open(args.output, 'w') as f:
            f.write(prompt)
        print(f"Saved LLM context to: {args.output}")

        # Show stats
        from .llm import LLMContextBuilder
        tokens = len(prompt) // 4
        print(f"Estimated tokens: ~{tokens:,}")
    else:
        print(prompt)

    return 0


def cmd_browse(args):
    """Handle browser rendering command."""
    try:
        from .browser import BrowserRenderer, AgentBrowserContext
        from .converter import CharacterSets
    except ImportError as e:
        print(f"Error: {e}")
        print("Install Playwright with: pip install playwright && playwright install")
        return 1

    url = args.url

    # Validate URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    mode = getattr(args, 'mode', 'visual')
    print(f"Rendering ({mode} mode): {url}")
    print("Please wait...")

    # Map charset name to charset
    charset_map = {
        'standard': CharacterSets.STANDARD,
        'detailed': CharacterSets.DETAILED,
        'blocks': CharacterSets.BLOCKS,
        'minimal': CharacterSets.MINIMAL,
        'braille': 'braille'
    }
    charset = charset_map.get(getattr(args, 'charset', 'blocks'), CharacterSets.BLOCKS)
    color = not getattr(args, 'no_color', False)
    full_page = getattr(args, 'full_page', False)

    try:
        if args.agent:
            # Agent context mode (uses semantic)
            agent = AgentBrowserContext(width=args.width, headless=not args.visible)
            output = agent.get_page_context(url=url, task=args.task)
        else:
            # Render mode
            renderer = BrowserRenderer(
                width=args.width,
                headless=not args.visible,
                browser_type=args.browser,
                mode=mode,
                charset=charset,
                color=color
            )
            output = renderer.render_url(url, full_page=full_page)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Saved to: {args.output}")

            # Show stats
            tokens = len(output) // 4
            print(f"Estimated tokens: ~{tokens:,}")
        else:
            print(output)

    except Exception as e:
        print(f"Error rendering page: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def add_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument("-w", "--width", type=int, help="Output width in characters")
    parser.add_argument("-c", "--color", action="store_true", help="Enable colored output")
    parser.add_argument(
        "-s", "--charset",
        choices=["standard", "detailed", "blocks", "minimal", "braille"],
        default="standard",
        help="Character set to use"
    )
    parser.add_argument("-i", "--invert", action="store_true", help="Invert brightness")
    parser.add_argument("-b", "--brightness", type=float, default=1.0, help="Brightness adjustment")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment")
    parser.add_argument("-e", "--edge", action="store_true", help="Enable edge detection")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ASCII Cam - Convert images, video, and camera feed to ASCII art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Camera command (default)
    cam_parser = subparsers.add_parser("camera", aliases=["cam"], help="Live camera mode")
    add_common_args(cam_parser)
    cam_parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    cam_parser.add_argument("--fps", type=int, default=15, help="Target FPS")
    cam_parser.add_argument("--mock", action="store_true", help="Use mock camera")
    cam_parser.set_defaults(func=cmd_camera)

    # Image command
    img_parser = subparsers.add_parser("image", aliases=["img"], help="Convert image to ASCII")
    img_parser.add_argument("image", help="Image file to convert")
    img_parser.add_argument("-o", "--output", help="Output file path")
    img_parser.add_argument("--html", action="store_true", help="Save as HTML")
    add_common_args(img_parser)
    img_parser.set_defaults(func=cmd_image)

    # Video command
    vid_parser = subparsers.add_parser("video", aliases=["vid", "play"], help="Play video as ASCII")
    vid_parser.add_argument("video", help="Video file to play")
    vid_parser.add_argument("-w", "--width", type=int, help="Output width")
    vid_parser.add_argument("-c", "--color", action="store_true", help="Enable color")
    vid_parser.add_argument("-s", "--charset", default="standard", help="Character set")
    vid_parser.set_defaults(func=cmd_video)

    # Audio command
    audio_parser = subparsers.add_parser("audio", aliases=["viz"], help="Audio visualizer")
    audio_parser.add_argument(
        "-m", "--mode",
        choices=["spectrum", "waveform", "vu", "oscilloscope", "circular"],
        default="spectrum",
        help="Visualization mode"
    )
    audio_parser.add_argument("-w", "--width", type=int, help="Output width")
    audio_parser.add_argument("--height", type=int, help="Output height")
    audio_parser.add_argument("-c", "--color", action="store_true", default=True, help="Enable color")
    audio_parser.add_argument("--no-color", dest="color", action="store_false", help="Disable color")
    audio_parser.add_argument("--mock", action="store_true", help="Use mock audio")
    audio_parser.set_defaults(func=cmd_audio)

    # Record command
    rec_parser = subparsers.add_parser("record", aliases=["rec"], help="Record camera to GIF/video")
    rec_parser.add_argument("-o", "--output", help="Output file path")
    rec_parser.add_argument("-d", "--duration", type=float, default=5.0, help="Recording duration (seconds)")
    rec_parser.add_argument("-f", "--format", choices=["gif", "mp4"], default="gif", help="Output format")
    rec_parser.add_argument("--max-frames", type=int, default=500, help="Maximum frames to record")
    rec_parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    rec_parser.add_argument("--fps", type=int, default=10, help="Recording FPS")
    rec_parser.add_argument("--mock", action="store_true", help="Use mock camera")
    add_common_args(rec_parser)
    rec_parser.set_defaults(func=cmd_record)

    # LLM command
    llm_parser = subparsers.add_parser("llm", help="Generate LLM context from image/video")
    llm_parser.add_argument("input", help="Image or video file")
    llm_parser.add_argument("-o", "--output", help="Output file (prints to stdout if not specified)")
    llm_parser.add_argument("-w", "--width", type=int, default=60, help="ASCII width (smaller = fewer tokens)")
    llm_parser.add_argument("-t", "--task", default="describe", help="Analysis task (describe, analyze, summarize)")
    llm_parser.add_argument("-n", "--frames", type=int, default=5, help="Max frames for video")
    llm_parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames")
    llm_parser.set_defaults(func=cmd_llm)

    # Browse command
    browse_parser = subparsers.add_parser("browse", aliases=["web"], help="Render website as ASCII art")
    browse_parser.add_argument("url", help="URL to render")
    browse_parser.add_argument("-o", "--output", help="Output file (prints to stdout if not specified)")
    browse_parser.add_argument("-w", "--width", type=int, default=120, help="ASCII width (default: 120)")
    browse_parser.add_argument("-m", "--mode", choices=["visual", "semantic"], default="visual",
                              help="Rendering mode: visual (screenshot ASCII) or semantic (DOM text)")
    browse_parser.add_argument("-c", "--charset", choices=["standard", "detailed", "blocks", "braille", "minimal"],
                              default="blocks", help="Character set for visual mode (default: blocks)")
    browse_parser.add_argument("--no-color", action="store_true", help="Disable color output")
    browse_parser.add_argument("--full-page", action="store_true", help="Capture full scrollable page")
    browse_parser.add_argument("-t", "--task", help="Task description for agent context")
    browse_parser.add_argument("--agent", action="store_true", help="Include agent action suggestions (uses semantic)")
    browse_parser.add_argument("--visible", action="store_true", help="Show browser window (not headless)")
    browse_parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default="chromium",
                              help="Browser to use")
    browse_parser.set_defaults(func=cmd_browse)

    args = parser.parse_args()

    # Handle no command (default to camera)
    if args.command is None:
        # Check if there's a positional argument that looks like an image
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            # Assume it's an image
            args = parser.parse_args(["image"] + sys.argv[1:])
        else:
            # Default to camera
            args = parser.parse_args(["camera"] + sys.argv[1:])

    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
