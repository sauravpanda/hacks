"""
Audio visualizer module for ASCII art.

Creates real-time ASCII visualizations of audio input from:
- Microphone
- Audio files (MP3, WAV, etc.)

Visualization modes:
- Waveform
- Spectrum analyzer (frequency bars)
- Oscilloscope
- VU meter
- Circular visualizer
"""

import math
import time
import threading
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass
import numpy as np

# Audio capture is optional
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 44100
    chunk_size: int = 1024
    channels: int = 1
    format_bits: int = 16


class AudioCapture:
    """
    Captures audio from microphone or audio device.

    Uses PyAudio for cross-platform audio input.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio capture.

        Args:
            config: Audio configuration (uses defaults if None)
        """
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is required for audio capture. Install with: pip install pyaudio")

        self.config = config or AudioConfig()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._running = False
        self._buffer: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def start(self, device_index: Optional[int] = None):
        """
        Start audio capture.

        Args:
            device_index: Audio device index (None for default)
        """
        self._pa = pyaudio.PyAudio()

        # Get format
        if self.config.format_bits == 16:
            pa_format = pyaudio.paInt16
        elif self.config.format_bits == 32:
            pa_format = pyaudio.paFloat32
        else:
            pa_format = pyaudio.paInt16

        self._stream = self._pa.open(
            format=pa_format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=self._callback
        )

        self._running = True
        self._stream.start_stream()

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio data."""
        if self.config.format_bits == 16:
            data = np.frombuffer(in_data, dtype=np.int16)
            # Normalize to -1.0 to 1.0
            data = data.astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(in_data, dtype=np.float32)

        with self._lock:
            self._buffer = data

        return (None, pyaudio.paContinue)

    def get_buffer(self) -> Optional[np.ndarray]:
        """Get the current audio buffer."""
        with self._lock:
            return self._buffer.copy() if self._buffer is not None else None

    @staticmethod
    def list_devices() -> List[dict]:
        """List available audio input devices."""
        if not PYAUDIO_AVAILABLE:
            return []

        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })

        pa.terminate()
        return devices

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class MockAudioCapture:
    """
    Mock audio capture for testing without a microphone.

    Generates synthetic audio patterns.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._running = False
        self._time = 0.0
        self._pattern = "sine"

    def start(self, device_index: Optional[int] = None):
        self._running = True
        self._time = 0.0

    def stop(self):
        self._running = False

    def get_buffer(self) -> Optional[np.ndarray]:
        if not self._running:
            return None

        t = np.linspace(self._time, self._time + 0.02, self.config.chunk_size)
        self._time += 0.02

        if self._pattern == "sine":
            # Multiple frequencies for interesting visualization
            data = (
                0.3 * np.sin(2 * np.pi * 220 * t) +
                0.2 * np.sin(2 * np.pi * 440 * t) +
                0.15 * np.sin(2 * np.pi * 880 * t) +
                0.1 * np.sin(2 * np.pi * 110 * t * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)))
            )
        elif self._pattern == "noise":
            data = np.random.uniform(-0.5, 0.5, self.config.chunk_size)
        elif self._pattern == "pulse":
            data = np.sin(2 * np.pi * 2 * t) * np.sin(2 * np.pi * 440 * t)
        else:
            data = np.sin(2 * np.pi * 440 * t)

        return data.astype(np.float32)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class ASCIIVisualizer:
    """
    Generates ASCII visualizations from audio data.

    Multiple visualization modes available.
    """

    # Character sets for different intensities
    BARS_VERTICAL = " ▁▂▃▄▅▆▇█"
    BARS_HORIZONTAL = " ▏▎▍▌▋▊▉█"
    BLOCKS = " ░▒▓█"
    DOTS = " ·•●"
    SIMPLE = " -=≡#"

    def __init__(
        self,
        width: int = 80,
        height: int = 24,
        charset: str = "bars",
        color: bool = True
    ):
        """
        Initialize visualizer.

        Args:
            width: Output width in characters
            height: Output height in lines
            charset: Character set to use
            color: Enable colored output
        """
        self.width = width
        self.height = height
        self.color = color

        # Select character set
        if charset == "bars":
            self.chars = self.BARS_VERTICAL
        elif charset == "blocks":
            self.chars = self.BLOCKS
        elif charset == "dots":
            self.chars = self.DOTS
        else:
            self.chars = self.SIMPLE

        # FFT smoothing
        self._prev_spectrum: Optional[np.ndarray] = None
        self._smoothing = 0.7

    def _get_color(self, intensity: float, style: str = "spectrum") -> str:
        """Get ANSI color code based on intensity."""
        if not self.color:
            return ""

        if style == "spectrum":
            # Rainbow gradient based on frequency position
            hue = intensity * 360
            r, g, b = self._hsv_to_rgb(hue, 1.0, 1.0)
        elif style == "heat":
            # Heat map: blue -> cyan -> green -> yellow -> red
            if intensity < 0.25:
                r, g, b = 0, int(255 * intensity * 4), 255
            elif intensity < 0.5:
                r, g, b = 0, 255, int(255 * (1 - (intensity - 0.25) * 4))
            elif intensity < 0.75:
                r, g, b = int(255 * (intensity - 0.5) * 4), 255, 0
            else:
                r, g, b = 255, int(255 * (1 - (intensity - 0.75) * 4)), 0
        elif style == "fire":
            # Fire: dark red -> red -> orange -> yellow -> white
            if intensity < 0.33:
                r = int(255 * intensity * 3)
                g, b = 0, 0
            elif intensity < 0.66:
                r = 255
                g = int(255 * (intensity - 0.33) * 3)
                b = 0
            else:
                r, g = 255, 255
                b = int(255 * (intensity - 0.66) * 3)
        else:
            # Default green
            r, g, b = 0, int(255 * intensity), 0

        return f"\033[38;2;{r};{g};{b}m"

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

    def _reset_color(self) -> str:
        """Get ANSI reset code."""
        return "\033[0m" if self.color else ""

    def spectrum(self, audio_data: np.ndarray, num_bands: Optional[int] = None) -> str:
        """
        Generate spectrum analyzer visualization.

        Args:
            audio_data: Audio samples (normalized -1 to 1)
            num_bands: Number of frequency bands (defaults to width)

        Returns:
            ASCII art string
        """
        num_bands = num_bands or self.width

        # Compute FFT
        fft = np.fft.rfft(audio_data)
        spectrum = np.abs(fft)

        # Group into bands (logarithmic scaling for better visualization)
        band_size = len(spectrum) // num_bands
        if band_size < 1:
            band_size = 1

        bands = []
        for i in range(num_bands):
            start = i * band_size
            end = min(start + band_size, len(spectrum))
            if start < len(spectrum):
                bands.append(np.mean(spectrum[start:end]))
            else:
                bands.append(0)

        bands = np.array(bands)

        # Apply smoothing
        if self._prev_spectrum is not None and len(self._prev_spectrum) == len(bands):
            bands = self._smoothing * self._prev_spectrum + (1 - self._smoothing) * bands
        self._prev_spectrum = bands.copy()

        # Normalize
        max_val = np.max(bands)
        if max_val > 0:
            bands = bands / max_val

        # Build visualization
        lines = []
        for y in range(self.height):
            threshold = 1.0 - (y / self.height)
            line = ""

            for x, value in enumerate(bands):
                if value >= threshold:
                    # Calculate character intensity
                    char_idx = min(int(value * len(self.chars)), len(self.chars) - 1)
                    char = self.chars[char_idx]

                    if self.color:
                        # Color based on frequency (x position)
                        color = self._get_color(x / len(bands), "spectrum")
                        line += color + char
                    else:
                        line += char
                else:
                    line += " "

            line += self._reset_color()
            lines.append(line)

        return "\n".join(lines)

    def waveform(self, audio_data: np.ndarray) -> str:
        """
        Generate waveform visualization.

        Args:
            audio_data: Audio samples (normalized -1 to 1)

        Returns:
            ASCII art string
        """
        # Resample to fit width
        indices = np.linspace(0, len(audio_data) - 1, self.width).astype(int)
        samples = audio_data[indices]

        # Create empty canvas
        canvas = [[" " for _ in range(self.width)] for _ in range(self.height)]
        mid_y = self.height // 2

        # Draw waveform
        for x, sample in enumerate(samples):
            # Map sample to y position
            y = int(mid_y - sample * mid_y)
            y = max(0, min(self.height - 1, y))

            # Draw vertical line from center to sample
            start_y, end_y = min(mid_y, y), max(mid_y, y)
            for draw_y in range(start_y, end_y + 1):
                intensity = abs(sample)
                char_idx = min(int(intensity * len(self.chars)), len(self.chars) - 1)
                canvas[draw_y][x] = self.chars[char_idx]

        # Build output
        lines = []
        for y, row in enumerate(canvas):
            line = ""
            for x, char in enumerate(row):
                if self.color and char != " ":
                    intensity = abs(samples[x]) if x < len(samples) else 0
                    color = self._get_color(intensity, "heat")
                    line += color + char
                else:
                    line += char
            line += self._reset_color()
            lines.append(line)

        return "\n".join(lines)

    def vu_meter(self, audio_data: np.ndarray, stereo: bool = False) -> str:
        """
        Generate VU meter visualization.

        Args:
            audio_data: Audio samples
            stereo: Show stereo meters

        Returns:
            ASCII art string
        """
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_data ** 2))
        peak = np.max(np.abs(audio_data))

        # Scale to 0-1
        level = min(rms * 3, 1.0)  # Amplify for visibility
        peak_level = min(peak, 1.0)

        # Build meter
        meter_width = self.width - 10
        filled = int(level * meter_width)
        peak_pos = int(peak_level * meter_width)

        lines = []

        # Header
        lines.append("VU METER".center(self.width))
        lines.append("")

        # Scale
        scale = "".join([str(i % 10) for i in range(0, meter_width + 1, meter_width // 10)])
        lines.append(f"  dB: {scale}")

        # Meter bar
        bar = ""
        for i in range(meter_width):
            if i < filled:
                intensity = i / meter_width
                if intensity < 0.6:
                    color = self._get_color(0.3, "heat") if self.color else ""
                    char = "█"
                elif intensity < 0.8:
                    color = self._get_color(0.6, "heat") if self.color else ""
                    char = "█"
                else:
                    color = self._get_color(0.9, "heat") if self.color else ""
                    char = "█"
                bar += color + char
            elif i == peak_pos:
                bar += (self._get_color(1.0, "heat") if self.color else "") + "│"
            else:
                bar += " "

        bar += self._reset_color()
        lines.append(f"  [{bar}]")

        # Level indicator
        db = 20 * np.log10(level + 0.0001)
        lines.append(f"  Level: {db:+.1f} dB")

        # Padding
        while len(lines) < self.height:
            lines.append("")

        return "\n".join(lines[:self.height])

    def oscilloscope(self, audio_data: np.ndarray) -> str:
        """
        Generate oscilloscope-style visualization.

        Args:
            audio_data: Audio samples

        Returns:
            ASCII art string
        """
        # Similar to waveform but with dot style
        indices = np.linspace(0, len(audio_data) - 1, self.width).astype(int)
        samples = audio_data[indices]

        canvas = [[" " for _ in range(self.width)] for _ in range(self.height)]
        mid_y = self.height // 2

        # Draw grid lines
        for x in range(self.width):
            canvas[mid_y][x] = "─"
        for x in range(0, self.width, self.width // 8):
            for y in range(self.height):
                if canvas[y][x] == " ":
                    canvas[y][x] = "│"
                elif canvas[y][x] == "─":
                    canvas[y][x] = "┼"

        # Draw waveform as points
        for x, sample in enumerate(samples):
            y = int(mid_y - sample * (mid_y - 1))
            y = max(0, min(self.height - 1, y))
            canvas[y][x] = "●"

        # Build output
        lines = []
        for row in canvas:
            line = "".join(row)
            if self.color:
                line = self._get_color(0.5, "spectrum") + line + self._reset_color()
            lines.append(line)

        return "\n".join(lines)

    def circular(self, audio_data: np.ndarray) -> str:
        """
        Generate circular/radial visualization.

        Args:
            audio_data: Audio samples

        Returns:
            ASCII art string
        """
        # Compute spectrum
        fft = np.fft.rfft(audio_data)
        spectrum = np.abs(fft[:self.width])

        # Normalize
        max_val = np.max(spectrum)
        if max_val > 0:
            spectrum = spectrum / max_val

        # Create circular canvas
        canvas = [[" " for _ in range(self.width)] for _ in range(self.height)]
        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = min(center_x, center_y) - 1

        # Draw circular bars
        num_bars = min(36, len(spectrum))
        for i in range(num_bars):
            angle = (i / num_bars) * 2 * math.pi - math.pi / 2
            bar_length = spectrum[i * len(spectrum) // num_bars] * max_radius

            for r in range(1, int(bar_length) + 1):
                x = int(center_x + r * math.cos(angle) * 2)  # *2 for aspect ratio
                y = int(center_y + r * math.sin(angle))

                if 0 <= x < self.width and 0 <= y < self.height:
                    intensity = r / max_radius
                    char_idx = min(int(intensity * len(self.chars)), len(self.chars) - 1)
                    canvas[y][x] = self.chars[char_idx]

        # Build output
        lines = []
        for y, row in enumerate(canvas):
            line = ""
            for x, char in enumerate(row):
                if self.color and char != " ":
                    # Color based on angle
                    dx = x - center_x
                    dy = y - center_y
                    angle = math.atan2(dy, dx)
                    hue = (angle + math.pi) / (2 * math.pi)
                    line += self._get_color(hue, "spectrum") + char
                else:
                    line += char
            line += self._reset_color()
            lines.append(line)

        return "\n".join(lines)


class AudioVisualizer:
    """
    High-level audio visualizer that combines capture and rendering.

    Provides a simple interface for real-time audio visualization.
    """

    def __init__(
        self,
        width: int = 80,
        height: int = 24,
        mode: str = "spectrum",
        color: bool = True,
        mock: bool = False
    ):
        """
        Initialize audio visualizer.

        Args:
            width: Output width
            height: Output height
            mode: Visualization mode
            color: Enable colors
            mock: Use mock audio (no microphone)
        """
        self.width = width
        self.height = height
        self.mode = mode

        # Initialize components
        if mock:
            self.capture = MockAudioCapture()
        else:
            self.capture = AudioCapture()

        self.renderer = ASCIIVisualizer(width, height, color=color)

    def get_frame(self) -> Optional[str]:
        """
        Get a single visualization frame.

        Returns:
            ASCII art string or None if no audio data
        """
        buffer = self.capture.get_buffer()
        if buffer is None:
            return None

        if self.mode == "spectrum":
            return self.renderer.spectrum(buffer)
        elif self.mode == "waveform":
            return self.renderer.waveform(buffer)
        elif self.mode == "vu":
            return self.renderer.vu_meter(buffer)
        elif self.mode == "oscilloscope":
            return self.renderer.oscilloscope(buffer)
        elif self.mode == "circular":
            return self.renderer.circular(buffer)
        else:
            return self.renderer.spectrum(buffer)

    def __enter__(self):
        self.capture.start()
        return self

    def __exit__(self, *args):
        self.capture.stop()
