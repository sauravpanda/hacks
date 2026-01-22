#!/usr/bin/env python3
"""
ASCII Cam - Terminal ASCII art converter with live camera support.

Quick start:
    python main.py                    # Start camera mode
    python main.py image.jpg          # Convert an image
    python main.py --mock             # Test without camera
    python main.py -c -w 100          # Color mode, 100 chars wide

For more options: python main.py --help
"""

from ascii_cam.cli import main

if __name__ == "__main__":
    exit(main())
