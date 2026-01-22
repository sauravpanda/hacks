"""
Camera capture module for live video streaming.

Provides a simple interface for capturing frames from webcams
and video files using OpenCV.
"""

from typing import Optional, Generator, Tuple, Union
import cv2
import numpy as np


class Camera:
    """
    Camera capture class for live video streaming.

    Supports webcams and video files with configurable resolution
    and frame rate.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Initialize the camera.

        Args:
            source: Camera index (0 for default) or video file path
            width: Desired capture width (optional)
            height: Desired capture height (optional)
            fps: Desired frame rate (optional)
        """
        self.source = source
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def open(self) -> bool:
        """
        Open the camera/video source.

        Returns:
            True if successfully opened, False otherwise
        """
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            return False

        # Set resolution if specified
        if self.desired_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        if self.desired_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
        if self.desired_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        self._is_open = True
        return True

    def close(self):
        """Release the camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if camera is open and ready."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current capture resolution (width, height)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    @property
    def fps(self) -> float:
        """Get current frame rate."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    def read(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.

        Returns:
            Frame as numpy array (BGR format), or None if read failed
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        return frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.

        Yields:
            Frames as numpy arrays (BGR format)
        """
        while self.is_open:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def __enter__(self):
        """Context manager entry - opens the camera."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the camera."""
        self.close()
        return False


class MockCamera(Camera):
    """
    Mock camera for testing without a real webcam.

    Generates animated test patterns for development and testing.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pattern: str = "gradient"
    ):
        """
        Initialize mock camera.

        Args:
            width: Frame width
            height: Frame height
            fps: Simulated frame rate
            pattern: Test pattern type ('gradient', 'noise', 'checkerboard')
        """
        super().__init__(0, width, height, fps)
        self._frame_width = width
        self._frame_height = height
        self._frame_count = 0
        self._pattern = pattern

    def open(self) -> bool:
        """Open the mock camera."""
        self._is_open = True
        return True

    def close(self):
        """Close the mock camera."""
        self._is_open = False

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._frame_width, self._frame_height)

    @property
    def fps(self) -> float:
        return float(self.desired_fps or 30)

    def read(self) -> Optional[np.ndarray]:
        """Generate a test pattern frame."""
        if not self._is_open:
            return None

        self._frame_count += 1

        if self._pattern == "gradient":
            return self._generate_gradient()
        elif self._pattern == "noise":
            return self._generate_noise()
        elif self._pattern == "checkerboard":
            return self._generate_checkerboard()
        else:
            return self._generate_gradient()

    def _generate_gradient(self) -> np.ndarray:
        """Generate an animated gradient pattern."""
        frame = np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)

        # Animated horizontal gradient
        offset = (self._frame_count * 2) % 256
        for x in range(self._frame_width):
            value = (x * 256 // self._frame_width + offset) % 256
            frame[:, x, 0] = value  # Blue
            frame[:, x, 1] = (value + 85) % 256  # Green
            frame[:, x, 2] = (value + 170) % 256  # Red

        return frame

    def _generate_noise(self) -> np.ndarray:
        """Generate random noise pattern."""
        return np.random.randint(
            0, 256,
            (self._frame_height, self._frame_width, 3),
            dtype=np.uint8
        )

    def _generate_checkerboard(self) -> np.ndarray:
        """Generate an animated checkerboard pattern."""
        frame = np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)

        block_size = 32
        offset = (self._frame_count // 10) % 2

        for y in range(0, self._frame_height, block_size):
            for x in range(0, self._frame_width, block_size):
                if ((x // block_size) + (y // block_size) + offset) % 2:
                    frame[y:y+block_size, x:x+block_size] = 255

        return frame
