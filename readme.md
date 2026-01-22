# ASCII Cam

A terminal application that converts images and live camera feeds to ASCII art.

```
                               @@@@@@@@@@@@@@@@@@@@
                           @@@@@@@@@@@@@@@@@@@@@@@@@@@@
                        @@@@@@@%#**++==--::..::--==++*#%@@@
                      @@@@@#*+=:.                    .:=+*#@@@
                    @@@@#+=:.                            .:=+#@@
                   @@@#+=.                                  .=+#@@
                  @@#+=.                                      .=+#@
                 @@#=:                    @@                    :=#@
                @@#=.                    @@@@                    .=#@
                @#=.                      @@                      .=#@
               @@+:                                                :+@@
               @#=                                                  =#@
               @#:                                                  :#@
               @#:                                                  :#@
               @#=                                                  =#@
               @@+:                                                :+@@
                @#=.                                              .=#@
                @@#=.                                            .=#@@
                 @@#=:                                          :=#@@
                  @@#+=.                                      .=+#@@
                   @@@#+=.                                  .=+#@@@
                    @@@@#+=:.                            .:=+#@@@@
                      @@@@@#*+=:.                    .:=+*#@@@@@
                        @@@@@@@%#**++==--::..::--==++*#%@@@@@@@
                           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                               @@@@@@@@@@@@@@@@@@@@@@@@
```

## Features

- **Live Camera Streaming** - Real-time ASCII art from your webcam
- **Color ASCII Output** - Full ANSI true color support for stunning visuals
- **Multiple Character Sets** - Standard, detailed, blocks, minimal, and braille
- **Edge Detection** - Canny/Sobel edge detection for line-art style
- **Interactive Controls** - Adjust settings on-the-fly with keyboard
- **High Resolution Braille** - Use braille characters for 2x4 sub-character resolution
- **Image File Support** - Convert any image format (PNG, JPG, etc.)
- **HTML Export** - Save colored ASCII art as HTML files

## Installation

```bash
# Clone and install
pip install -e .

# Or install dependencies directly
pip install pillow opencv-python numpy blessed
```

## Quick Start

```bash
# Start live camera mode
python main.py

# Convert an image
python main.py image.jpg

# Color mode with custom width
python main.py -c -w 120

# Test without a camera (mock mode)
python main.py --mock

# Edge detection mode
python main.py -e image.jpg
```

## Usage

```
python main.py [OPTIONS] [IMAGE]

Arguments:
  IMAGE                  Image file to convert (omit for camera mode)

Options:
  -o, --output PATH      Save output to file
  --html                 Save as HTML (preserves colors)
  -w, --width INT        Output width in characters
  -c, --color            Enable colored output
  -s, --charset NAME     Character set: standard, detailed, blocks, minimal, braille
  -i, --invert           Invert brightness
  -b, --brightness FLOAT Brightness adjustment (default: 1.0)
  --contrast FLOAT       Contrast adjustment (default: 1.0)
  -e, --edge             Enable edge detection mode
  --camera INT           Camera device ID (default: 0)
  --fps INT              Target FPS for camera mode (default: 15)
  --mock                 Use mock camera for testing
```

## Interactive Keyboard Controls

When in camera mode:

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `c` | Toggle color mode |
| `e` | Toggle edge detection |
| `b` | Toggle braille mode |
| `i` | Toggle invert |
| `+` / `-` | Adjust width |
| `[` / `]` | Adjust brightness |
| `{` / `}` | Adjust contrast |
| `1-5` | Switch character set |
| `s` | Save screenshot |
| `h` / `?` | Show help |
| `SPACE` | Pause/Resume |

## Character Sets

| Set | Characters | Best For |
|-----|------------|----------|
| `standard` | ` .:-=+*#%@` | General use |
| `detailed` | Extended ASCII gradient | Photo-realistic |
| `blocks` | `░▒▓█` | Pixelated look |
| `minimal` | ` .-+*#` | Clean output |
| `braille` | `⠁⠂⠄...` (256 patterns) | High resolution |

## Examples

### Convert image with color
```bash
python main.py photo.jpg -c -w 80
```

### Save as HTML
```bash
python main.py photo.jpg -c -o output.html --html
```

### Edge detection
```bash
python main.py photo.jpg -e -w 100
```

### Braille mode (high resolution)
```bash
python main.py photo.jpg -s braille -w 60
```

## API Usage

```python
from ascii_cam import ASCIIConverter, Camera, CharacterSets
from PIL import Image

# Convert an image
converter = ASCIIConverter(width=80, color=True)
image = Image.open("photo.jpg")
ascii_art = converter.convert(image)
print(ascii_art)

# Live camera
with Camera() as cam:
    for frame in cam.stream():
        ascii_art = converter.convert_from_array(frame)
        print(ascii_art)
```

## Requirements

- Python 3.8+
- Pillow (image processing)
- OpenCV (camera capture)
- NumPy (array operations)
- Blessed (keyboard input) - optional but recommended

## License

MIT
