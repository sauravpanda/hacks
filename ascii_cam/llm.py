"""
LLM Context Integration module.

Converts visual content to ASCII for efficient LLM consumption:
- Video frame extraction and description
- Browser/screen capture for AI agents
- Token-efficient visual context
- Structured prompts for visual analysis

ASCII frames are ~100x more token-efficient than base64 images,
enabling vision-like capabilities with text-only models.
"""

import re
import json
import time
from typing import Optional, List, Dict, Any, Tuple, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from PIL import Image

from .converter import ASCIIConverter, CharacterSets
from .player import VideoReader, VideoToASCII


@dataclass
class FrameContext:
    """Context for a single frame sent to LLM."""
    timestamp: float
    frame_number: int
    ascii_art: str
    width: int
    height: int
    description: Optional[str] = None


@dataclass
class VideoContext:
    """Context for a video sent to LLM."""
    path: str
    duration: float
    fps: float
    total_frames: int
    sampled_frames: List[FrameContext]
    summary: Optional[str] = None


class LLMContextBuilder:
    """
    Builds structured context from visual content for LLMs.

    Optimized for token efficiency while preserving visual information.
    """

    # Recommended widths for different use cases
    WIDTH_MINIMAL = 40   # ~1.6K chars per frame, very compressed
    WIDTH_COMPACT = 60   # ~3.6K chars per frame, good balance
    WIDTH_STANDARD = 80  # ~6.4K chars per frame, detailed
    WIDTH_DETAILED = 120 # ~14K chars per frame, high detail

    # Enhancement presets for different content types
    PRESET_DEFAULT = {"contrast": 1.0, "brightness": 1.0, "edge": False}
    PRESET_HIGH_CONTRAST = {"contrast": 1.8, "brightness": 1.1, "edge": False}
    PRESET_MOVEMENT = {"contrast": 2.0, "brightness": 1.0, "edge": True}  # Good for dance, sports
    PRESET_DARK_VIDEO = {"contrast": 1.5, "brightness": 1.4, "edge": False}
    PRESET_BRIGHT_VIDEO = {"contrast": 1.3, "brightness": 0.9, "edge": False}

    def __init__(
        self,
        width: int = 60,
        charset: str = "minimal",
        include_metadata: bool = True,
        contrast: float = 1.0,
        brightness: float = 1.0,
        edge_enhance: bool = False,
        preset: Optional[str] = None
    ):
        """
        Initialize context builder.

        Args:
            width: ASCII width (smaller = fewer tokens)
            charset: Character set ('minimal' recommended for LLM)
            include_metadata: Include frame metadata
            contrast: Contrast multiplier (1.5-2.0 recommended for movement)
            brightness: Brightness multiplier
            edge_enhance: Add edge detection for clearer shapes
            preset: Use a preset ('high_contrast', 'movement', 'dark', 'bright')
        """
        self.width = width
        self.include_metadata = include_metadata
        self.edge_enhance = edge_enhance

        # Apply preset if specified
        if preset:
            presets = {
                "high_contrast": self.PRESET_HIGH_CONTRAST,
                "movement": self.PRESET_MOVEMENT,
                "dark": self.PRESET_DARK_VIDEO,
                "bright": self.PRESET_BRIGHT_VIDEO,
            }
            if preset in presets:
                p = presets[preset]
                contrast = p["contrast"]
                brightness = p["brightness"]
                self.edge_enhance = p["edge"]

        # Use minimal charset for best token efficiency
        charset_map = {
            "minimal": CharacterSets.MINIMAL,
            "standard": CharacterSets.STANDARD,
            "detailed": CharacterSets.DETAILED,
            "blocks": CharacterSets.BLOCKS,
        }
        self.converter = ASCIIConverter(
            width=width,
            color=False,  # No colors for LLM
            charset=charset_map.get(charset, CharacterSets.MINIMAL),
            contrast=contrast,
            brightness=brightness
        )

    def frame_to_context(
        self,
        image: Image.Image,
        timestamp: float = 0.0,
        frame_number: int = 0
    ) -> FrameContext:
        """
        Convert a single image to LLM context.

        Args:
            image: PIL Image
            timestamp: Frame timestamp
            frame_number: Frame index

        Returns:
            FrameContext object
        """
        # Apply edge enhancement if enabled
        if self.edge_enhance:
            image = self._apply_edge_enhance(image)

        ascii_art = self.converter.convert(image)
        lines = ascii_art.split('\n')

        return FrameContext(
            timestamp=timestamp,
            frame_number=frame_number,
            ascii_art=ascii_art,
            width=self.width,
            height=len(lines)
        )

    def _apply_edge_enhance(self, image: Image.Image) -> Image.Image:
        """
        Apply edge enhancement to make shapes more visible.

        Combines the original image with edge-detected version
        to preserve details while highlighting contours.
        """
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

        return blended

    def video_to_context(
        self,
        video_path: str,
        sample_interval: Optional[float] = None,
        max_frames: int = 10,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        fps: Optional[float] = None
    ) -> VideoContext:
        """
        Convert video to LLM context.

        Args:
            video_path: Path to video file
            sample_interval: Seconds between sampled frames (default: 2.0)
            max_frames: Maximum frames to include
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            fps: Alternative to sample_interval - frames per second to sample.
                 If provided, overrides sample_interval. E.g., fps=0.5 = 1 frame every 2 seconds

        Returns:
            VideoContext object
        """
        # Handle fps vs sample_interval
        if fps is not None:
            sample_interval = 1.0 / fps
        elif sample_interval is None:
            sample_interval = 2.0
        frames = []

        with VideoReader(video_path) as reader:
            if not reader.is_open:
                raise ValueError(f"Could not open video: {video_path}")

            info = reader.info
            end_time = end_time or info.duration

            # Calculate frame positions
            current_time = start_time
            while current_time < end_time and len(frames) < max_frames:
                frame = reader.get_frame_at(current_time)
                if frame is None:
                    break

                # Convert to ASCII
                rgb = frame.image[:, :, ::-1]
                image = Image.fromarray(rgb)
                context = self.frame_to_context(
                    image,
                    timestamp=frame.timestamp,
                    frame_number=frame.frame_number
                )
                frames.append(context)

                current_time += sample_interval

            return VideoContext(
                path=video_path,
                duration=info.duration,
                fps=info.fps,
                total_frames=info.frame_count,
                sampled_frames=frames
            )

    def format_for_prompt(
        self,
        context: VideoContext,
        task: str = "describe",
        include_all_frames: bool = True
    ) -> str:
        """
        Format video context as an LLM prompt.

        Args:
            context: VideoContext object
            task: Task description ('describe', 'analyze', 'summarize', etc.)
            include_all_frames: Include all frames or just first/last

        Returns:
            Formatted prompt string
        """
        lines = [
            f"# Video Analysis Task: {task.title()}",
            "",
            f"Video: {Path(context.path).name}",
            f"Duration: {context.duration:.1f} seconds",
            f"Frame rate: {context.fps:.1f} fps",
            f"Sampled frames: {len(context.sampled_frames)} of {context.total_frames}",
            "",
            "## ASCII Frame Representations",
            "",
            "Each frame below is an ASCII art representation of the video.",
            "Characters represent brightness: space (dark) to # (bright).",
            ""
        ]

        if include_all_frames:
            frames_to_show = context.sampled_frames
        else:
            # Just first and last
            frames_to_show = [context.sampled_frames[0], context.sampled_frames[-1]]

        for i, frame in enumerate(frames_to_show):
            timestamp = f"{int(frame.timestamp // 60):02d}:{frame.timestamp % 60:05.2f}"
            lines.append(f"### Frame {i + 1} (t={timestamp})")
            lines.append("```")
            lines.append(frame.ascii_art)
            lines.append("```")
            lines.append("")

        # Add task instructions
        task_prompts = {
            "describe": "Please describe what you see in these ASCII frames. What objects, people, or actions are visible?",
            "analyze": "Analyze the visual content of these frames. What patterns, movements, or changes do you notice?",
            "summarize": "Provide a brief summary of what happens in this video based on the ASCII frames.",
            "narrate": "Create a narration describing the events shown in these frames as if narrating a documentary.",
            "detect": "Identify and list all distinct objects, people, or elements visible in these frames.",
            "compare": "Compare the first and last frames. What has changed?",
        }

        lines.append("## Task")
        lines.append(task_prompts.get(task, task))

        return "\n".join(lines)

    def estimate_tokens(self, context: VideoContext) -> int:
        """
        Estimate token count for context.

        Args:
            context: VideoContext object

        Returns:
            Estimated token count
        """
        total_chars = sum(len(f.ascii_art) for f in context.sampled_frames)
        # Rough estimate: 1 token ≈ 4 characters for ASCII
        return total_chars // 4


class BrowserCapture:
    """
    Capture browser/screen content for AI agents.

    Converts screenshots to ASCII for token-efficient agent context.
    Useful for browser automation agents like browser-use.
    """

    def __init__(
        self,
        width: int = 120,
        charset: str = "detailed"
    ):
        """
        Initialize browser capture.

        Args:
            width: ASCII width for capture
            charset: Character set to use
        """
        charset_map = {
            "minimal": CharacterSets.MINIMAL,
            "standard": CharacterSets.STANDARD,
            "detailed": CharacterSets.DETAILED,
            "blocks": CharacterSets.BLOCKS,
        }
        self.converter = ASCIIConverter(
            width=width,
            color=False,
            charset=charset_map.get(charset, CharacterSets.DETAILED)
        )

    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Capture screen or region to ASCII.

        Args:
            region: Optional (x, y, width, height) tuple

        Returns:
            ASCII art of screen
        """
        try:
            from PIL import ImageGrab
        except ImportError:
            raise ImportError("PIL ImageGrab required. Install with: pip install pillow")

        # Capture screen
        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()

        return self.converter.convert(screenshot)

    def capture_with_selenium(self, driver) -> str:
        """
        Capture browser screenshot via Selenium WebDriver.

        Args:
            driver: Selenium WebDriver instance

        Returns:
            ASCII art of browser content
        """
        import io

        # Get screenshot as PNG bytes
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(png_bytes))

        return self.converter.convert(image)

    def capture_with_playwright(self, page) -> str:
        """
        Capture browser screenshot via Playwright.

        Args:
            page: Playwright Page instance

        Returns:
            ASCII art of browser content
        """
        import io

        # Get screenshot as bytes
        png_bytes = page.screenshot()
        image = Image.open(io.BytesIO(png_bytes))

        return self.converter.convert(image)

    def format_for_agent(
        self,
        ascii_art: str,
        url: Optional[str] = None,
        task: Optional[str] = None,
        previous_action: Optional[str] = None
    ) -> str:
        """
        Format browser capture for AI agent consumption.

        Args:
            ascii_art: ASCII representation of page
            url: Current page URL
            task: Agent's current task
            previous_action: Last action taken

        Returns:
            Formatted context for agent
        """
        lines = [
            "# Browser State",
            ""
        ]

        if url:
            lines.append(f"URL: {url}")

        if previous_action:
            lines.append(f"Previous action: {previous_action}")

        lines.extend([
            "",
            "## Page Content (ASCII representation)",
            "```",
            ascii_art,
            "```",
            ""
        ])

        if task:
            lines.extend([
                "## Current Task",
                task,
                "",
                "Based on the page content above, determine the next action to take.",
            ])

        return "\n".join(lines)


class StreamingVideoContext:
    """
    Stream video frames to LLM in real-time.

    Useful for live video analysis with streaming LLM APIs.
    """

    def __init__(
        self,
        width: int = 60,
        sample_rate: float = 1.0
    ):
        """
        Initialize streaming context.

        Args:
            width: ASCII width
            sample_rate: Frames per second to sample
        """
        self.builder = LLMContextBuilder(width=width, charset="minimal")
        self.sample_rate = sample_rate
        self._last_sample_time = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[str]:
        """
        Process a video frame, returning context if sample interval elapsed.

        Args:
            frame: BGR image array
            timestamp: Current timestamp

        Returns:
            ASCII context string or None if not yet time to sample
        """
        if timestamp - self._last_sample_time < 1.0 / self.sample_rate:
            return None

        self._last_sample_time = timestamp

        # Convert frame
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        context = self.builder.frame_to_context(image, timestamp=timestamp)

        return self._format_streaming_frame(context)

    def _format_streaming_frame(self, context: FrameContext) -> str:
        """Format a single frame for streaming context."""
        timestamp = f"{int(context.timestamp // 60):02d}:{context.timestamp % 60:05.2f}"
        return f"[Frame at {timestamp}]\n```\n{context.ascii_art}\n```"

    def stream_video(
        self,
        video_path: str
    ) -> Generator[str, None, None]:
        """
        Stream video frames as context.

        Args:
            video_path: Path to video file

        Yields:
            ASCII context strings
        """
        with VideoReader(video_path) as reader:
            for frame in reader.frames():
                context = self.process_frame(frame.image, frame.timestamp)
                if context:
                    yield context


def video_to_llm_prompt(
    video_path: str,
    task: str = "describe",
    width: int = 60,
    max_frames: int = 5,
    sample_interval: Optional[float] = None,
    fps: Optional[float] = None,
    contrast: float = 1.0,
    brightness: float = 1.0,
    edge_enhance: bool = False,
    preset: Optional[str] = None
) -> str:
    """
    Convenience function to convert video to LLM prompt.

    Args:
        video_path: Path to video file
        task: Analysis task
        width: ASCII width
        max_frames: Maximum frames to include
        sample_interval: Seconds between frames (default: 2.0)
        fps: Alternative to sample_interval - frames per second to sample.
             E.g., fps=0.5 = 1 frame every 2 seconds, fps=2 = 2 frames per second
        contrast: Contrast multiplier (1.5-2.0 for better visibility)
        brightness: Brightness multiplier
        edge_enhance: Add edge detection for clearer shapes
        preset: Use a preset ('high_contrast', 'movement', 'dark', 'bright')

    Returns:
        Ready-to-use LLM prompt
    """
    builder = LLMContextBuilder(
        width=width,
        contrast=contrast,
        brightness=brightness,
        edge_enhance=edge_enhance,
        preset=preset
    )
    context = builder.video_to_context(
        video_path,
        sample_interval=sample_interval,
        max_frames=max_frames,
        fps=fps
    )
    return builder.format_for_prompt(context, task=task)


def image_to_llm_context(
    image_path: str,
    width: int = 80,
    task: str = "describe"
) -> str:
    """
    Convert a single image to LLM context.

    Args:
        image_path: Path to image file
        width: ASCII width
        task: Analysis task

    Returns:
        LLM prompt with ASCII image
    """
    image = Image.open(image_path)
    builder = LLMContextBuilder(width=width)
    context = builder.frame_to_context(image)

    return f"""# Image Analysis Task: {task.title()}

## ASCII Representation
Characters represent brightness: space (dark) to # (bright).

```
{context.ascii_art}
```

## Task
{task.capitalize()} what you see in this ASCII image representation.
"""


def browser_to_llm_context(
    driver_or_page,
    url: Optional[str] = None,
    task: Optional[str] = None,
    width: int = 120
) -> str:
    """
    Convert browser state to LLM context.

    Works with Selenium WebDriver or Playwright Page.

    Args:
        driver_or_page: Selenium driver or Playwright page
        url: Override URL (auto-detected if None)
        task: Agent task description
        width: ASCII width

    Returns:
        LLM context string
    """
    capture = BrowserCapture(width=width)

    # Detect driver type
    driver_type = type(driver_or_page).__name__

    if 'WebDriver' in driver_type or hasattr(driver_or_page, 'get_screenshot_as_png'):
        ascii_art = capture.capture_with_selenium(driver_or_page)
        url = url or driver_or_page.current_url
    elif hasattr(driver_or_page, 'screenshot'):
        ascii_art = capture.capture_with_playwright(driver_or_page)
        url = url or driver_or_page.url
    else:
        raise ValueError(f"Unknown driver type: {driver_type}")

    return capture.format_for_agent(ascii_art, url=url, task=task)


def semantic_browser_context(
    url: Optional[str] = None,
    page=None,
    task: Optional[str] = None,
    width: int = 100
) -> str:
    """
    Get semantic ASCII browser context for AI agents.

    This provides a structured DOM-based representation that includes:
    - Page structure with semantic elements
    - Interactive elements with selectors
    - Suggested actions for agents

    Much more useful for agents than visual ASCII screenshots.

    Args:
        url: URL to render (if no page provided)
        page: Existing Playwright page object
        task: Task description for the agent
        width: Output width

    Returns:
        Semantic ASCII context for LLM

    Example:
        >>> context = semantic_browser_context(
        ...     url="https://example.com",
        ...     task="Find and click the login button"
        ... )
        >>> response = llm.complete(context)
    """
    try:
        from .browser import AgentBrowserContext
    except ImportError:
        raise ImportError(
            "Browser rendering requires Playwright. "
            "Install with: pip install playwright && playwright install"
        )

    agent = AgentBrowserContext(width=width)
    return agent.get_page_context(url=url, page=page, task=task)
