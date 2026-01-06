# Inverse Tool (Video Analyzer)

Python tool that analyzes a 60-second video recording of the web clock and determines when the clock was originally started.

## Installation

```bash
cd inverse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py video.MOV
```

Options:
- `--timestamp "YYYY-MM-DD HH:MM:SS"` - Override video timestamp
- `--tolerance N` - Color detection tolerance (default: 80)
- `--verbose` - Show detailed output
- `--visualize` - Generate debug video

## Example Output

```
Video timestamp: 2026-01-03 11:55:00 (from exiftool)

Analyzing video...
Extracted 60 seconds of cell observations

Searching for valid clock state...
Found valid state! (60/60 seconds matched)
  Video started at second: 0
  Minute identifier (k): 29457295

==================================================
RESULT
==================================================

Elapsed time since clock started:
  1,767,437,700 seconds
  (29,457,295 minutes, 20,456.5 days, 56.0 years)

Clock origin (UTC):
  1970-01-01 00:00:00

Note: The clock period is 46,221,064,723,759,104 minutes
      (>88 billion years)
```

## How Inversion Works

1. **Video Analysis**: Extract frames and detect which cells are visible each second
2. **Rotation Search**: Try all 60 possible starting seconds to align observations
3. **Bit Extraction**: Reconstruct the minute identifier (k) from observed patterns
4. **Origin Calculation**: Subtract k minutes from video timestamp

## Detection Features

- **Hybrid detection**: Simple edge-based detection for quality videos, Hough-based corner tracking for tilted/moving videos
- **Color-independent**: Works with any color scheme (standard colors, monochrome red, etc.)
- **Sum-based seconds**: Cells shown at second S always sum to S, making detection FPS-independent
- **Error correction**: Can fix single-cell detection errors using valid combination lookup

## Dependencies

Python 3, OpenCV, NumPy, ffprobe/exiftool
