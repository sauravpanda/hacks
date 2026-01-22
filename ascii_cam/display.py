"""
Terminal display module for rendering ASCII art.

Handles terminal output, screen clearing, cursor positioning,
and real-time updates for live streaming.
"""

import sys
import os
import time
from typing import Optional, Callable, Tuple
from contextlib import contextmanager


class Display:
    """
    Terminal display manager for ASCII art output.

    Provides efficient screen updates, cursor control, and
    support for both static and animated output.
    """

    # ANSI escape codes
    CLEAR_SCREEN = "\033[2J"
    CURSOR_HOME = "\033[H"
    CURSOR_HIDE = "\033[?25l"
    CURSOR_SHOW = "\033[?25h"
    RESET_STYLE = "\033[0m"

    def __init__(
        self,
        target_fps: int = 30,
        show_stats: bool = True,
        output_stream=None
    ):
        """
        Initialize display.

        Args:
            target_fps: Target frames per second for live display
            show_stats: Show FPS and other stats
            output_stream: Output stream (defaults to stdout)
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.show_stats = show_stats
        self.output = output_stream or sys.stdout
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._fps_counter = 0.0
        self._fps_timer = 0.0

    def clear(self):
        """Clear the terminal screen."""
        self.output.write(self.CLEAR_SCREEN + self.CURSOR_HOME)
        self.output.flush()

    def move_cursor(self, row: int = 1, col: int = 1):
        """Move cursor to specified position (1-indexed)."""
        self.output.write(f"\033[{row};{col}H")

    def hide_cursor(self):
        """Hide the terminal cursor."""
        self.output.write(self.CURSOR_HIDE)
        self.output.flush()

    def show_cursor(self):
        """Show the terminal cursor."""
        self.output.write(self.CURSOR_SHOW)
        self.output.flush()

    @contextmanager
    def hidden_cursor(self):
        """Context manager for temporarily hiding cursor."""
        self.hide_cursor()
        try:
            yield
        finally:
            self.show_cursor()

    def render(self, content: str, clear: bool = True):
        """
        Render content to the terminal.

        Args:
            content: ASCII art string to display
            clear: Whether to clear screen first
        """
        if clear:
            # Move to home instead of clearing to reduce flicker
            self.output.write(self.CURSOR_HOME)

        self.output.write(content)

        if self.show_stats:
            self._update_fps()
            stats = self._get_stats_line()
            self.output.write(f"\n{stats}")

        self.output.write(self.RESET_STYLE)
        self.output.flush()

    def render_frame(self, content: str):
        """
        Render a frame with frame rate limiting.

        Maintains consistent frame rate by sleeping if needed.

        Args:
            content: ASCII art string to display
        """
        current_time = time.time()
        elapsed = current_time - self._last_frame_time

        # Sleep to maintain frame rate
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)

        self.render(content, clear=True)
        self._last_frame_time = time.time()
        self._frame_count += 1

    def _update_fps(self):
        """Update FPS counter."""
        current_time = time.time()

        if self._fps_timer == 0:
            self._fps_timer = current_time
            return

        elapsed = current_time - self._fps_timer

        if elapsed >= 1.0:
            self._fps_counter = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = current_time

    def _get_stats_line(self) -> str:
        """Get formatted stats line."""
        return f"\033[90mFPS: {self._fps_counter:.1f} | Frame: {self._frame_count}\033[0m"

    @staticmethod
    def get_terminal_size() -> Tuple[int, int]:
        """Get terminal dimensions (columns, rows)."""
        try:
            size = os.get_terminal_size()
            return (size.columns, size.lines)
        except OSError:
            return (80, 24)  # Default fallback

    @staticmethod
    def supports_color() -> bool:
        """Check if terminal supports color output."""
        # Check for color support
        if not sys.stdout.isatty():
            return False

        # Check TERM environment variable
        term = os.environ.get("TERM", "")
        if term in ("dumb", ""):
            return False

        # Check for color-capable terminals
        color_terms = ("xterm", "vt100", "linux", "screen", "tmux", "rxvt")
        if any(t in term.lower() for t in color_terms):
            return True

        # Check COLORTERM
        if os.environ.get("COLORTERM"):
            return True

        return True  # Assume color support by default

    @staticmethod
    def supports_truecolor() -> bool:
        """Check if terminal supports 24-bit true color."""
        colorterm = os.environ.get("COLORTERM", "")
        return colorterm in ("truecolor", "24bit")


class DoubleBuffer:
    """
    Double buffering for flicker-free animation.

    Compares frames and only updates changed regions.
    """

    def __init__(self, display: Display):
        """Initialize double buffer."""
        self.display = display
        self._front_buffer: Optional[str] = None
        self._back_buffer: Optional[str] = None

    def render(self, content: str):
        """
        Render content using double buffering.

        Only redraws lines that have changed.
        """
        self._back_buffer = content

        if self._front_buffer is None:
            # First frame - full render
            self.display.render(content, clear=True)
        else:
            # Compare and update only changed lines
            self._render_diff()

        self._front_buffer = self._back_buffer

    def _render_diff(self):
        """Render only the differences between buffers."""
        if self._front_buffer is None or self._back_buffer is None:
            return

        front_lines = self._front_buffer.split("\n")
        back_lines = self._back_buffer.split("\n")

        self.display.output.write(Display.CURSOR_HOME)

        for i, (front, back) in enumerate(zip(front_lines, back_lines)):
            if front != back:
                self.display.move_cursor(i + 1, 1)
                self.display.output.write(back)

        # Handle extra lines in back buffer
        for i in range(len(front_lines), len(back_lines)):
            self.display.move_cursor(i + 1, 1)
            self.display.output.write(back_lines[i])

        self.display.output.flush()


class OutputFile:
    """
    Save ASCII art to files.

    Supports plain text and HTML output.
    """

    @staticmethod
    def save_text(content: str, filepath: str, strip_ansi: bool = True):
        """
        Save ASCII art to a text file.

        Args:
            content: ASCII art string
            filepath: Output file path
            strip_ansi: Remove ANSI escape codes
        """
        if strip_ansi:
            import re
            content = re.sub(r'\033\[[0-9;]*m', '', content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def save_html(
        content: str,
        filepath: str,
        title: str = "ASCII Art",
        background: str = "#000000",
        font_size: int = 8
    ):
        """
        Save ASCII art as an HTML file with colors preserved.

        Args:
            content: ASCII art string with ANSI codes
            filepath: Output file path
            title: HTML page title
            background: Background color
            font_size: Font size in pixels
        """
        # Convert ANSI to HTML spans
        html_content = OutputFile._ansi_to_html(content)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            background-color: {background};
            font-family: 'Courier New', Consolas, monospace;
            font-size: {font_size}px;
            line-height: 1.0;
            white-space: pre;
            margin: 20px;
        }}
        .ascii-art {{
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <div class="ascii-art">{html_content}</div>
</body>
</html>"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    @staticmethod
    def _ansi_to_html(content: str) -> str:
        """Convert ANSI escape codes to HTML spans."""
        import re

        result = []
        current_color = None
        i = 0

        while i < len(content):
            # Check for ANSI escape sequence
            match = re.match(r'\033\[38;2;(\d+);(\d+);(\d+)m', content[i:])
            if match:
                r, g, b = match.groups()
                if current_color:
                    result.append('</span>')
                result.append(f'<span style="color: rgb({r},{g},{b})">')
                current_color = (r, g, b)
                i += len(match.group())
                continue

            # Check for reset
            if content[i:i+4] == '\033[0m':
                if current_color:
                    result.append('</span>')
                    current_color = None
                i += 4
                continue

            # Regular character - escape HTML entities
            char = content[i]
            if char == '<':
                result.append('&lt;')
            elif char == '>':
                result.append('&gt;')
            elif char == '&':
                result.append('&amp;')
            elif char == '\n':
                result.append('<br>\n')
            else:
                result.append(char)
            i += 1

        if current_color:
            result.append('</span>')

        return ''.join(result)
