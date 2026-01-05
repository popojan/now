# Mondrian Terminal Clock

A terminal-based Mondrian clock that displays the same patterns as the [web version](../web/index.html). Each minute has a unique visual representation that doesn't repeat for 88 billion years.

## Features

- ASCII and Unicode display modes
- 4-color graph coloring for distinguishable cell fills
- Custom fill characters
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
  -a          ASCII mode (.|'#)
  -u          Unicode mode (box drawing + blocks) [default]
  -d          Distinct fills (4-color graph coloring)
  -f CHARS    Custom fill chars for cells 1,2,4,6,12,15,20 (7 chars)

Time:
  -o ORIGIN   Custom origin (ISO 8601, e.g. 2000-01-01T00:00:00Z)
  -k K        Use minute K directly (ignores system time)
```

## Examples

Live clock with Unicode (scrolling):
```bash
./now
```

Live clock with in-place updates:
```bash
./now -l
```

Generate 60 frames for minute 12345 in ASCII:
```bash
./now -a -k 12345 -n 60
```

Round-trip test (generate frames, decode back):
```bash
./now -s -n 60 | ./now -i
```

Round-trip with custom origin (12 hours of frames):
```bash
./now -o 2026-01-01T12:00:00Z -s -n 43260 | tail -n 782 | ./now -i
# Output: 2026-01-01T12:00:00Z
```

**Invariant**: Running with `-s` (simulate) produces the same origin reconstruction as running live and waiting for the frames to complete. The termination timestamp reflects the virtual time of the last frame.

Distinct colors mode (4-color graph coloring):
```bash
./now -a -d
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
