# ASCII Cam

A terminal application that converts images, video, and live camera feeds to ASCII art - with LLM integration for AI-powered visual analysis.

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
- **Video Playback** - Play any video file as ASCII art in the terminal
- **Audio Visualizer** - Spectrum analyzer, waveform, VU meter, and more
- **GIF/Video Export** - Record and export ASCII animations
- **Color ASCII Output** - Full ANSI true color support
- **Edge Detection** - Canny/Sobel edge detection for line-art style
- **Multiple Character Sets** - Standard, detailed, blocks, minimal, and braille
- **LLM Context Generation** - Convert video/images to token-efficient ASCII for AI analysis
- **Semantic Browser Rendering** - Render web pages as structured ASCII with interactive elements for AI agents

## Installation

```bash
# Basic installation
pip install -e .

# With audio visualization support
pip install -e ".[audio]"

# With browser capture support
pip install -e ".[browser]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```bash
# Live camera
ascii-cam camera

# Convert an image
ascii-cam image photo.jpg

# Play a video as ASCII
ascii-cam video movie.mp4

# Audio visualizer
ascii-cam audio --mode spectrum

# Record camera to GIF
ascii-cam record -d 5 -o my_recording.gif

# Generate LLM context from video
ascii-cam llm video.mp4 -t describe -o context.txt

# Render website as semantic ASCII
ascii-cam browse https://example.com --agent
```

## Commands

### `camera` - Live Webcam
```bash
ascii-cam camera              # Default camera
ascii-cam camera -c           # With color
ascii-cam camera --mock       # Test without camera
ascii-cam cam -w 120 -c       # 120 chars wide, color
```

### `image` - Convert Images
```bash
ascii-cam image photo.jpg           # Print to terminal
ascii-cam image photo.png -o out.txt  # Save to file
ascii-cam image photo.jpg -c --html -o out.html  # Color HTML
```

### `video` - Play Videos as ASCII
```bash
ascii-cam video movie.mp4           # Play video
ascii-cam video movie.mp4 -c        # With color
ascii-cam play clip.webm -w 100     # Custom width
```

Controls: `space` pause, `←/→` seek, `+/-` speed, `q` quit

### `audio` - Audio Visualizer
```bash
ascii-cam audio                     # Spectrum analyzer (default)
ascii-cam audio -m waveform         # Waveform display
ascii-cam audio -m vu               # VU meter
ascii-cam audio -m oscilloscope     # Oscilloscope
ascii-cam audio -m circular         # Circular visualizer
ascii-cam audio --mock              # Test with synthetic audio
```

### `record` - Record to GIF/Video
```bash
ascii-cam record -d 10              # Record 10 seconds
ascii-cam record -d 5 -f gif        # Export as GIF
ascii-cam record -d 30 -f mp4       # Export as MP4
ascii-cam record -c --fps 15        # Color at 15 FPS
```

### `llm` - Generate LLM Context
```bash
# Convert video for LLM analysis
ascii-cam llm video.mp4 -t describe

# Convert image
ascii-cam llm screenshot.png -t analyze

# Save to file with stats
ascii-cam llm video.mp4 -o context.txt -w 60 -n 10
```

### `browse` - Semantic Web Rendering
```bash
# Render a website as semantic ASCII
ascii-cam browse https://example.com

# With agent context (includes action suggestions)
ascii-cam browse https://example.com --agent

# With a task description
ascii-cam browse https://example.com --agent -t "Find the login button"

# Save to file
ascii-cam browse https://example.com -o page.txt

# Use different browser
ascii-cam browse https://example.com --browser firefox
```

Output includes:
- Page structure with semantic elements (header, nav, main, footer)
- Interactive elements with CSS selectors
- Suggested actions for AI agents

## LLM Integration

ASCII frames are ~100x more token-efficient than base64 images. Use this to give vision-like capabilities to text-only models, or reduce costs with vision models.

### Video Analysis Example

```python
from ascii_cam import video_to_llm_prompt

# Generate prompt from video
prompt = video_to_llm_prompt(
    "demo.mp4",
    task="describe",
    width=60,           # Smaller = fewer tokens
    max_frames=5,       # Sample 5 frames
    sample_interval=2.0 # Every 2 seconds
)

# Send to any LLM
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": prompt}]
)
```

### Semantic Browser for AI Agents

Render web pages as structured ASCII with interactive elements - perfect for browser automation agents:

```python
from ascii_cam import semantic_browser_context

# Get semantic ASCII representation of a webpage
context = semantic_browser_context(
    url="https://example.com",
    task="Find and click the login button"
)

# Context includes:
# - Page structure (header, nav, main, forms, etc.)
# - Interactive elements with CSS selectors
# - Suggested actions for the agent

response = llm.complete(context)
```

Example output:
```
╔════════════════════════════════════════════════════════════╗
║ Example Domain                                              ║
║ https://example.com                                         ║
╚════════════════════════════════════════════════════════════╝

## Page Structure

├─ ┌─ HEADER ─┐
│  ├─ → Home  #nav-home
│  ├─ → Products  #nav-products
│  ├─ [ Login ]  #login-btn
├─ ┌─ MAIN ─┐
│  ├─ # Welcome to Example
│  ├─ [text: Enter email]  #email-input
│  ├─ [password: ___]  #password-input
│  ├─ [ Submit ]  #submit-btn

## Interactive Elements

### BUTTONS
  1. [btn] Login                 #login-btn
  2. [btn] Submit                #submit-btn

### LINKS
  3. [→] Home                    #nav-home
  4. [→] Products                #nav-products

### INPUTS
  5. [___] Email                 #email-input
  6. [___] Password              #password-input
```

### Visual Browser Capture

For simple screenshot-based ASCII (less structured but faster):

```python
from ascii_cam import browser_to_llm_context
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com")

    # Get visual ASCII screenshot
    context = browser_to_llm_context(
        page,
        task="Find the login button and click it"
    )

    response = agent.process(context)
```

### Streaming Video Analysis

```python
from ascii_cam import StreamingVideoContext

streamer = StreamingVideoContext(width=60, sample_rate=1.0)

for ascii_frame in streamer.stream_video("video.mp4"):
    # Send each frame to LLM for real-time analysis
    response = client.messages.create(
        model="claude-3-haiku",
        messages=[{"role": "user", "content": ascii_frame}]
    )
    print(response.content)
```

## Python API

```python
from ascii_cam import (
    ASCIIConverter,
    Camera,
    ASCIIVideoPlayer,
    ASCIIRecorder,
    GIFExporter,
    video_to_llm_prompt,
)
from PIL import Image

# Convert image
converter = ASCIIConverter(width=80, color=True)
image = Image.open("photo.jpg")
ascii_art = converter.convert(image)
print(ascii_art)

# Live camera with recording
recorder = ASCIIRecorder()
recorder.start()

with Camera() as cam:
    for frame in cam.stream():
        ascii_art = converter.convert_from_array(frame)
        recorder.add_frame(ascii_art, has_color=True)
        print(ascii_art)

recorder.stop()

# Export to GIF
exporter = GIFExporter()
exporter.export(recorder.frames, "output.gif", fps=10)
```

## Keyboard Controls

### Camera/Video Mode
| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `c` | Toggle color |
| `e` | Toggle edge detection |
| `b` | Toggle braille mode |
| `i` | Toggle invert |
| `r` | Toggle recording |
| `+` / `-` | Adjust width |
| `[` / `]` | Adjust brightness |
| `{` / `}` | Adjust contrast |
| `1-5` | Switch character set |
| `SPACE` | Pause/Resume |

## Character Sets

| Set | Characters | Best For |
|-----|------------|----------|
| `standard` | ` .:-=+*#%@` | General use |
| `detailed` | Extended ASCII | Photo-realistic |
| `blocks` | `░▒▓█` | Pixelated look |
| `minimal` | ` .-+*#` | LLM context |
| `braille` | `⠁⠂⠄...` | High resolution |

## Requirements

- Python 3.8+
- Pillow (image processing)
- OpenCV (camera/video)
- NumPy (array operations)
- Blessed (keyboard input)

Optional:
- PyAudio (audio visualization)
- Selenium/Playwright (browser capture)

## License

MIT
