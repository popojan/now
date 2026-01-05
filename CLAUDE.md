# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Now" is an animated clock with Mondrian-style design where no two minutes repeat within the 88-billion-year period. Each minute uses a unique combination of filled rectangles to express each second. The display uses seven colored cells (values 1, 2, 4, 6, 12, 15, 20) whose visibility cycles through different combinations.

The actual period of the clock is (2^30)(3^16) minutes (~46 quadrillion minutes), which is more than 6 times the estimated age of the Universe. Uses system time rather than International Atomic Time (TAI) to avoid network calls.

Also released on Wallpaper Engine with adjustable colors.

## Architecture

- **Single-page static website**: No build system or server-side code
- **Pure HTML/CSS/JS**: All logic contained in `web/index.html`
- **Time representation**: Uses a permutation-based system to map time values to combinations of visible cells
- **URL parameters**: Supports `origin`, `offset`, `period`, and `mod` query parameters to customize the time display

## Inverse Video Analyzer (`inverse/`)

The `inverse/` directory contains a video analyzer that determines when the clock was originally started by analyzing a video recording of the clock.

### Key Files

- **clock_inverse.py**: Core algorithm with `perm()` (forward) and `inverse_perm()` (reverse) functions
- **video_analyzer.py**: Video processing with OpenCV - cell detection, perspective correction, corner tracking
- **main.py**: CLI entry point
- **metadata.py**: Video timestamp extraction

### Key Concepts

- **Sum-based second detection**: Cells shown at second S always sum to S (`cells_to_second(cells) = sum(cells) % 60`)
- **Cell areas**: 1, 2, 4, 6, 12, 15, 20 (sum = 60)
- **Grid structure**: 6 columns × 10 rows with black grid lines
- **Perspective correction**: Warp detected quadrilateral to standard rectangle
- **Per-frame corner tracking**: For videos with camera motion

### Running the Video Analyzer

**IMPORTANT: Always activate the venv first!**

```bash
cd inverse
source venv/bin/activate
python main.py /path/to/video.MOV -v
```

## Development

### Web Clock

Serve the `web/` directory with any static file server:
```bash
python3 -m http.server 8000 --directory web
```

### Video Analyzer

```bash
cd inverse
source venv/bin/activate
pip install -r requirements.txt  # if needed
python main.py <video_path> [--verbose] [--tolerance N]
```

## Testing

### Unit Tests

```bash
cd inverse
source venv/bin/activate
python -m unittest test_inverse -v
```

### Video Integration Tests

Test all videos in `test_videos/` directory:

```bash
cd inverse
source venv/bin/activate
for video in ../test_videos/*.MOV; do
  echo "=== $(basename "$video") ==="
  python main.py "$video" 2>&1 | grep -A1 "Clock origin"
done
```

See `test_videos/README.md` for expected origins and video characteristics.

## Key Implementation Details

- The `ordering` object maps numbers 0-59 to arrays of cell combinations
- `perm()` function generates pseudo-random permutations based on time
- Click anywhere to cycle through display modes (minutes → hours → minutes)
- DST handling is built into the timezone offset calculation
- Clock origin is stored in UTC to be timezone-independent
