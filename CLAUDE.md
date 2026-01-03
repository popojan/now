# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Now" is an animated clock with Mondrian-style design where no two minutes are ever the same. Each minute uses a unique combination of filled rectangles to express each second. The display uses seven colored cells (values 1, 2, 4, 6, 12, 15, 20) whose visibility cycles through different combinations.

The actual period of the clock is (2^30)(3^16) minutes (~46 quadrillion minutes), which is more than 6 times the estimated age of the Universe. Uses system time rather than International Atomic Time (TAI) to avoid network calls.

Also released on Wallpaper Engine with adjustable colors.

## Architecture

- **Single-page static website**: No build system or server-side code
- **Pure HTML/CSS/JS**: All logic contained in `web/index.html`
- **Time representation**: Uses a permutation-based system to map time values to combinations of visible cells
- **URL parameters**: Supports `origin`, `offset`, `period`, and `mod` query parameters to customize the time display

## Development

To run locally, serve the `web/` directory with any static file server:
```bash
python3 -m http.server 8000 --directory web
```

## Key Implementation Details

- The `ordering` object maps numbers 0-59 to arrays of cell combinations
- `perm()` function generates pseudo-random permutations based on time
- Click anywhere to cycle through display modes (minutes → hours → minutes)
- DST handling is built into the timezone offset calculation
