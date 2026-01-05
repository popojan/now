# Mondrian Terminal Clock

A terminal-based Mondrian clock that displays the same patterns as the [web version](../web/index.html). Each minute has a unique visual representation that doesn't repeat for 88 billion years.

## Features

- Beautiful CJK default (日月火水木金土 elements)
- Multiple presets: cjk, blocks, distinct, kanji, emoji
- Custom fill characters
- ASCII and Unicode display modes
- Inverse mode to decode frames back to minute number
- Cross-platform (Linux, macOS, Windows)

## Build

```bash
make
```

Or directly:
```bash
gcc -Os -Wall -s -o now now.c
```

## Usage

```
./now [options]

Modes:
  (default)   Live clock, display frames (1/sec)
  -s          Simulate: fast output (no delay), timestamp as-if-live
  -l          In-place update (no scroll, requires TTY)
  -i          Inverse: read frames from stdin, output k and origin
  -n N        Output N frames then exit

Display:
  -a          ASCII borders (.|'-)
  -u          Unicode borders (box drawing) [default]
  -f PRESET   Preset: cjk [default], blocks, blocks1, distinct, kanji, emoji
     CHARS    Or 7 custom UTF-8 fill characters
  -1          Half-width: 1 column per cell (compact 8-col output)
  -w          Wide fills: -f glyphs are full-width (CJK, 2 cols each)

Time:
  -o ORIGIN   Custom origin (ISO 8601, e.g. 2000-01-01T00:00:00Z)
  -t T        Use absolute time T seconds from origin
```

## Examples

Live clock (CJK default):
```bash
./now
```

In-place updates (no scroll):
```bash
./now -l
```

Emoji preset:
```bash
./now -f emoji -l
```

Classic monochrome blocks:
```bash
./now -f blocks
```

Distinct 4-color shading:
```bash
./now -f distinct
```

Half-width compact mode:
```bash
./now -f blocks1
```

Custom UTF-8 fills:
```bash
./now -f "░▒▓█○●◐"
```

Generate 60 frames starting at t=740700 (minute 12345):
```bash
./now -t 740700 -n 60 -s
```

Round-trip test:
```bash
./now -s -n 60 | ./now -i
```

## How It Works

The clock uses 7 cells with values 1, 2, 4, 6, 12, 15, 20. Each second displays a unique combination of cells that sum to that second number (0-59). Multiple combinations exist for most seconds, and the specific choice depends on the minute number `k`.

The period is (2^30)(3^16) minutes (~46 quadrillion), more than 6 times the estimated age of the Universe.

### Inverse Mode

The `-i` flag reads 60 frames from stdin and reconstructs the minute number. It:
1. Parses each frame to detect which cells are filled
2. Uses the sum of visible cells to determine each second
3. Extracts the combo choice for multi-option seconds
4. Reconstructs `k` from the extracted bits
5. Computes and displays the origin timestamp

Note: All 60 frames must be from the same minute for accurate reconstruction.
