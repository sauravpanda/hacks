"""
Core ASCII conversion engine.

Converts images to ASCII art with support for:
- Multiple character sets (standard, detailed, blocks, braille)
- Color output using ANSI escape codes
- Proper aspect ratio correction for terminal display
"""

from typing import Optional, Tuple, List
import numpy as np
from PIL import Image


class CharacterSets:
    """Predefined character sets for ASCII conversion."""

    # Standard ASCII characters ordered by perceived density (light to dark)
    STANDARD = " .:-=+*#%@"

    # More detailed character set for better gradients
    DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

    # Block characters for a pixelated look
    BLOCKS = " ░▒▓█"

    # Braille characters for high resolution (2x4 dot patterns)
    # These map to 256 patterns from empty to full
    BRAILLE_BASE = 0x2800  # Unicode braille pattern blank

    # Simple/minimal set
    MINIMAL = " .-+*#"

    # Inverted standard (dark to light) - good for light backgrounds
    INVERTED = "@%#*+=-:. "

    @classmethod
    def get_braille_char(cls, dots: List[bool]) -> str:
        """
        Convert an 8-element boolean list to a braille character.

        Braille dot positions:
        1 4
        2 5
        3 6
        7 8

        The dots list should be in order [1,2,3,4,5,6,7,8]
        """
        if len(dots) != 8:
            raise ValueError("Braille requires exactly 8 dot values")

        # Braille Unicode encoding: each dot adds a power of 2
        # Dot 1=1, Dot 2=2, Dot 3=4, Dot 4=8, Dot 5=16, Dot 6=32, Dot 7=64, Dot 8=128
        value = sum(bit << i for i, bit in enumerate(dots))
        return chr(cls.BRAILLE_BASE + value)


class ASCIIConverter:
    """
    Converts images to ASCII art.

    Handles grayscale conversion, resizing with aspect ratio correction,
    and mapping pixel brightness to characters.
    """

    # Terminal characters are typically ~2x taller than wide
    ASPECT_RATIO_CORRECTION = 0.55

    def __init__(
        self,
        width: int = 80,
        charset: str = CharacterSets.STANDARD,
        color: bool = False,
        invert: bool = False,
        brightness: float = 1.0,
        contrast: float = 1.0
    ):
        """
        Initialize the ASCII converter.

        Args:
            width: Output width in characters
            charset: Character set to use for conversion
            color: Enable colored ASCII output
            invert: Invert brightness mapping
            brightness: Brightness adjustment (0.5 = darker, 2.0 = brighter)
            contrast: Contrast adjustment (0.5 = lower, 2.0 = higher)
        """
        self.width = width
        self.charset = charset
        self.color = color
        self.invert = invert
        self.brightness = brightness
        self.contrast = contrast
        self._use_braille = False

    @property
    def charset(self) -> str:
        return self._charset

    @charset.setter
    def charset(self, value: str):
        self._charset = value
        self._use_braille = (value == "braille")

    def _adjust_image(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        # Convert to float for processing
        adjusted = img.astype(np.float32)

        # Apply contrast (around midpoint 127.5)
        adjusted = (adjusted - 127.5) * self.contrast + 127.5

        # Apply brightness
        adjusted = adjusted * self.brightness

        # Clip to valid range
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _resize_image(
        self,
        image: Image.Image,
        target_width: int
    ) -> Image.Image:
        """
        Resize image to target width with aspect ratio correction.

        Terminal characters are taller than wide, so we need to
        reduce the height to maintain visual proportions.
        """
        original_width, original_height = image.size

        # Calculate aspect ratio
        aspect_ratio = original_height / original_width

        # Apply correction for terminal character dimensions
        corrected_height = int(target_width * aspect_ratio * self.ASPECT_RATIO_CORRECTION)

        # Ensure minimum height
        corrected_height = max(1, corrected_height)

        return image.resize((target_width, corrected_height), Image.Resampling.LANCZOS)

    def _pixel_to_char(self, brightness: int) -> str:
        """Map a brightness value (0-255) to an ASCII character."""
        if self.invert:
            brightness = 255 - brightness

        # Map brightness to character index
        char_index = int(brightness / 256 * len(self._charset))
        char_index = min(char_index, len(self._charset) - 1)

        return self._charset[char_index]

    def _rgb_to_ansi(self, r: int, g: int, b: int) -> str:
        """Convert RGB values to ANSI 256-color escape code."""
        # Use 24-bit true color if available (most modern terminals)
        return f"\033[38;2;{r};{g};{b}m"

    def _reset_ansi(self) -> str:
        """Return ANSI reset code."""
        return "\033[0m"

    def convert_braille(
        self,
        image: Image.Image,
        threshold: int = 128
    ) -> str:
        """
        Convert image to braille characters for higher resolution.

        Each braille character represents a 2x4 pixel block.
        """
        # Calculate dimensions (braille is 2 wide, 4 tall per character)
        char_width = self.width
        pixel_width = char_width * 2

        # Resize maintaining aspect ratio
        original_width, original_height = image.size
        aspect_ratio = original_height / original_width
        pixel_height = int(pixel_width * aspect_ratio * self.ASPECT_RATIO_CORRECTION * 2)

        # Ensure height is divisible by 4
        pixel_height = max(4, (pixel_height // 4) * 4)

        # Resize and convert to grayscale
        image = image.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
        gray = np.array(image.convert("L"))
        gray = self._adjust_image(gray)

        if self.invert:
            gray = 255 - gray

        # Get color data if needed
        color_data = None
        if self.color:
            color_img = image.convert("RGB")
            color_data = np.array(color_img)

        lines = []
        for y in range(0, pixel_height, 4):
            line = ""
            for x in range(0, pixel_width, 2):
                # Get 2x4 block of pixels
                dots = []
                for dy in range(4):
                    for dx in range(2):
                        py, px = y + dy, x + dx
                        if py < pixel_height and px < pixel_width:
                            dots.append(gray[py, px] > threshold)
                        else:
                            dots.append(False)

                # Reorder dots for braille encoding
                # Input order: (0,0)(0,1)(1,0)(1,1)(2,0)(2,1)(3,0)(3,1)
                # Braille order: 1,2,3,7 (left col), 4,5,6,8 (right col)
                braille_dots = [
                    dots[0], dots[2], dots[4], dots[1],
                    dots[3], dots[5], dots[6], dots[7]
                ]

                char = CharacterSets.get_braille_char(braille_dots)

                if self.color and color_data is not None:
                    # Average color of the block
                    block = color_data[y:y+4, x:x+2]
                    if block.size > 0:
                        avg_color = block.mean(axis=(0, 1)).astype(int)
                        char = self._rgb_to_ansi(*avg_color) + char

                line += char

            if self.color:
                line += self._reset_ansi()
            lines.append(line)

        return "\n".join(lines)

    def convert(
        self,
        image: Image.Image,
        width: Optional[int] = None
    ) -> str:
        """
        Convert a PIL Image to ASCII art.

        Args:
            image: PIL Image to convert
            width: Override default width (optional)

        Returns:
            String containing the ASCII art
        """
        target_width = width or self.width

        # Handle braille mode separately
        if self._use_braille:
            return self.convert_braille(image)

        # Resize image
        image = self._resize_image(image, target_width)

        # Convert to grayscale for character mapping
        gray = np.array(image.convert("L"))
        gray = self._adjust_image(gray)

        # Get color data if needed
        color_data = None
        if self.color:
            color_img = image.convert("RGB")
            color_data = np.array(color_img)

        # Build ASCII art
        lines = []
        height = gray.shape[0]

        for y in range(height):
            line = ""
            for x in range(target_width):
                brightness = gray[y, x]
                char = self._pixel_to_char(brightness)

                if self.color and color_data is not None:
                    r, g, b = color_data[y, x]
                    char = self._rgb_to_ansi(r, g, b) + char

                line += char

            if self.color:
                line += self._reset_ansi()
            lines.append(line)

        return "\n".join(lines)

    def convert_from_file(self, filepath: str, width: Optional[int] = None) -> str:
        """
        Load and convert an image file to ASCII art.

        Args:
            filepath: Path to image file
            width: Override default width (optional)

        Returns:
            String containing the ASCII art
        """
        image = Image.open(filepath)
        return self.convert(image, width)

    def convert_from_array(
        self,
        array: np.ndarray,
        width: Optional[int] = None
    ) -> str:
        """
        Convert a numpy array (e.g., from OpenCV) to ASCII art.

        Args:
            array: Numpy array in BGR or RGB format
            width: Override default width (optional)

        Returns:
            String containing the ASCII art
        """
        # OpenCV uses BGR, convert to RGB
        if len(array.shape) == 3 and array.shape[2] == 3:
            array = array[:, :, ::-1]  # BGR to RGB

        image = Image.fromarray(array)
        return self.convert(image, width)
