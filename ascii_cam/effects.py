"""
Image effects and filters for ASCII conversion.

Includes edge detection, dithering, and other visual effects
to create different ASCII art styles.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class EdgeDetector:
    """
    Edge detection for line-art style ASCII output.

    Supports Sobel, Canny, and Laplacian edge detection methods.
    """

    def __init__(
        self,
        method: str = "sobel",
        threshold: int = 50,
        low_threshold: int = 50,
        high_threshold: int = 150
    ):
        """
        Initialize edge detector.

        Args:
            method: Detection method ('sobel', 'canny', 'laplacian', 'prewitt')
            threshold: Edge threshold for Sobel/Laplacian (0-255)
            low_threshold: Low threshold for Canny
            high_threshold: High threshold for Canny
        """
        self.method = method
        self.threshold = threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect(self, image: Image.Image) -> Image.Image:
        """
        Apply edge detection to an image.

        Args:
            image: PIL Image to process

        Returns:
            Edge-detected image
        """
        # Convert to grayscale
        gray = image.convert("L")

        if self.method == "sobel":
            return self._sobel(gray)
        elif self.method == "canny":
            return self._canny(gray)
        elif self.method == "laplacian":
            return self._laplacian(gray)
        elif self.method == "prewitt":
            return self._prewitt(gray)
        else:
            return self._sobel(gray)

    def _sobel(self, image: Image.Image) -> Image.Image:
        """Apply Sobel edge detection."""
        arr = np.array(image, dtype=np.float32)

        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Apply convolution
        gx = self._convolve(arr, sobel_x)
        gy = self._convolve(arr, sobel_y)

        # Compute gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize and threshold
        magnitude = np.clip(magnitude, 0, 255)
        magnitude = np.where(magnitude > self.threshold, 255, 0)

        return Image.fromarray(magnitude.astype(np.uint8))

    def _canny(self, image: Image.Image) -> Image.Image:
        """Apply Canny edge detection using OpenCV."""
        try:
            import cv2
            arr = np.array(image)
            edges = cv2.Canny(arr, self.low_threshold, self.high_threshold)
            return Image.fromarray(edges)
        except ImportError:
            # Fallback to Sobel if OpenCV not available
            return self._sobel(image)

    def _laplacian(self, image: Image.Image) -> Image.Image:
        """Apply Laplacian edge detection."""
        arr = np.array(image, dtype=np.float32)

        # Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # Apply convolution
        result = self._convolve(arr, kernel)

        # Take absolute value and threshold
        result = np.abs(result)
        result = np.clip(result, 0, 255)
        result = np.where(result > self.threshold, 255, 0)

        return Image.fromarray(result.astype(np.uint8))

    def _prewitt(self, image: Image.Image) -> Image.Image:
        """Apply Prewitt edge detection."""
        arr = np.array(image, dtype=np.float32)

        # Prewitt kernels
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        # Apply convolution
        gx = self._convolve(arr, prewitt_x)
        gy = self._convolve(arr, prewitt_y)

        # Compute gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize and threshold
        magnitude = np.clip(magnitude, 0, 255)
        magnitude = np.where(magnitude > self.threshold, 255, 0)

        return Image.fromarray(magnitude.astype(np.uint8))

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution manually (for pure numpy implementation)."""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        # Output array
        output = np.zeros_like(image)

        # Apply convolution
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

        return output


class Dithering:
    """
    Dithering algorithms for creating halftone-like effects.

    Useful for creating stylized ASCII art with limited character sets.
    """

    @staticmethod
    def floyd_steinberg(image: Image.Image, levels: int = 2) -> Image.Image:
        """
        Apply Floyd-Steinberg dithering.

        Args:
            image: PIL Image to dither
            levels: Number of output levels (2 for pure black/white)

        Returns:
            Dithered image
        """
        arr = np.array(image.convert("L"), dtype=np.float32)
        h, w = arr.shape

        # Quantization step
        step = 255 / (levels - 1)

        for y in range(h):
            for x in range(w):
                old_pixel = arr[y, x]
                new_pixel = round(old_pixel / step) * step
                arr[y, x] = new_pixel
                error = old_pixel - new_pixel

                # Distribute error to neighbors
                if x + 1 < w:
                    arr[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        arr[y + 1, x - 1] += error * 3 / 16
                    arr[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        arr[y + 1, x + 1] += error * 1 / 16

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def ordered_dither(
        image: Image.Image,
        matrix_size: int = 4
    ) -> Image.Image:
        """
        Apply ordered (Bayer) dithering.

        Args:
            image: PIL Image to dither
            matrix_size: Size of Bayer matrix (2, 4, or 8)

        Returns:
            Dithered image
        """
        # Bayer matrices
        bayer_2 = np.array([[0, 2], [3, 1]]) / 4
        bayer_4 = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]) / 16
        bayer_8 = np.array([
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ]) / 64

        if matrix_size == 2:
            matrix = bayer_2
        elif matrix_size == 8:
            matrix = bayer_8
        else:
            matrix = bayer_4

        arr = np.array(image.convert("L"), dtype=np.float32) / 255
        h, w = arr.shape
        mh, mw = matrix.shape

        # Tile matrix to image size
        tiled = np.tile(matrix, (h // mh + 1, w // mw + 1))[:h, :w]

        # Apply threshold
        result = (arr > tiled).astype(np.uint8) * 255

        return Image.fromarray(result)


class ImageEffects:
    """Collection of image effects for ASCII art preprocessing."""

    @staticmethod
    def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
        """Sharpen the image."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def blur(image: Image.Image, radius: int = 2) -> Image.Image:
        """Apply Gaussian blur."""
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def posterize(image: Image.Image, bits: int = 4) -> Image.Image:
        """Reduce color depth for a posterized look."""
        from PIL import ImageOps
        return ImageOps.posterize(image, bits)

    @staticmethod
    def emboss(image: Image.Image) -> Image.Image:
        """Apply emboss effect."""
        return image.filter(ImageFilter.EMBOSS)

    @staticmethod
    def invert(image: Image.Image) -> Image.Image:
        """Invert image colors."""
        from PIL import ImageOps
        return ImageOps.invert(image.convert("RGB"))

    @staticmethod
    def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
        """Adjust image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
        """Adjust image brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
