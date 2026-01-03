# Mondrian Clock

A visual clock inspired by Piet Mondrian's geometric art. Each minute displays a unique pattern of colored rectangles, with the sequence not repeating for over 88 billion years.

## Live Demo

Visit [hraj.si/now](https://hraj.si/now) to see the clock in action.

## How It Works

The clock face consists of 7 rectangular cells with areas 1, 2, 4, 6, 12, 15, and 20 units. Each second, a specific combination of cells is shown or hidden. The pattern for each minute is determined by a permutation function that encodes the minute number (k) into display choices.

### The Period

The clock's period is **(2^30) × (3^16) = 46,221,064,723,759,104 minutes**, which equals approximately **88 billion years**. Every minute within this period shows a unique sequence of 60 patterns.

### Cell Layout

```
┌────────────────┬──────┐
│                │      │
│      20        │  12  │
│   (yellow)     │(blue)│
│                │      │
├────────┬───┬───┤      │
│        │ 1 ├───┴──────┤
│   15   ├───┤    4     │
│  (red) │ 2 │ (black)  │
│        │───┼──────────┤
│        │      6       │
└────────┴──────────────┘
```

## Project Structure

```
now/
├── web/
│   └── index.html      # The clock webpage
├── inverse/
│   ├── main.py         # CLI tool to decode clock videos
│   ├── clock_inverse.py    # Core inversion algorithm
│   ├── video_analyzer.py   # OpenCV video processing
│   ├── metadata.py     # Video timestamp extraction
│   ├── requirements.txt
│   └── STREAMING.md    # Future iOS app design notes
└── README.md
```

## Inverse Tool

The `inverse/` directory contains a Python tool that analyzes a 60-second video recording of the clock and determines when the clock was originally started.

### Installation

```bash
cd inverse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
python main.py video.MOV
```

Options:
- `--timestamp "YYYY-MM-DD HH:MM:SS"` - Override video timestamp
- `--tolerance N` - Color detection tolerance (default: 80)
- `--verbose` - Show detailed output
- `--visualize` - Generate debug video

### Example Output

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

### How Inversion Works

1. **Video Analysis**: Extract frames and detect which cells are visible each second
2. **Rotation Search**: Try all 60 possible starting seconds to align observations
3. **Bit Extraction**: Reconstruct the minute identifier (k) from observed patterns
4. **Origin Calculation**: Subtract k minutes from video timestamp

## URL Parameters

The clock webpage accepts URL parameters:

- `origin` - Custom epoch (ISO 8601 format, e.g., `2000-01-01T00:00:00Z`)
- `offset` - Year offset from origin
- `period` - Time unit in minutes (1=minutes, 60=hours, 3600=days)
- `mod` - Modulus for display (default: 60)

Example: `https://hraj.si/now?origin=2000-01-01T00:00:00Z`

## Click Interaction

Click the clock to cycle through display modes:
- **Minutes** (default): Pattern changes every second
- **Hours**: Pattern changes every minute
- **Days**: Pattern changes every hour

## Technical Details

### Permutation Function

The `perm(k)` function maps a minute number to 60 display indices:
- 20 seconds have 1 option (deterministic)
- 18 seconds have 2 options (18 bits)
- 16 seconds have 3 options (~25 bits)
- 6 seconds have 4 options (12 bits)

Total entropy: ~55 bits per minute, sufficient for the 46-quadrillion-minute period.

### Dependencies

- **Web**: Pure HTML/CSS/JavaScript, no dependencies
- **Inverse tool**: Python 3, OpenCV, NumPy, ffprobe/exiftool

## License

CC BY-NC 4.0

## Author


