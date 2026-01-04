# Mondrian Terminal Clock

A terminal-based Mondrian clock that displays the same patterns as the [web version](../web/index.html). Each minute has a unique visual representation, and no two minutes are ever the same.

## Features

- ASCII and Unicode display modes
- 3-color graph coloring for distinguishable cell fills
- Custom fill characters
- Inverse mode to decode frames back to minute number
- Cross-platform (Linux, macOS, Windows)

## Build

```bash
make
```

Or directly:
```bash
gcc -Os -Wall -s -o clock clock.c
```

## Usage

```
./clock [options]

Modes:
  (default)   Run clock, display frames (1/sec)
  -i          Inverse: read 60 frames from stdin, output k
  -n N        Output N frames fast (no delay), then exit

Display:
  -a          ASCII mode (.|'#)
  -u          Unicode mode (box drawing + blocks) [default]
  -d          Distinct fills (3-color graph coloring)
  -f CHARS    Custom fill chars for cells 1,2,4,6,12,15,20 (7 chars)

Time:
  -o ORIGIN   Custom origin (ISO 8601, e.g. 2000-01-01T00:00:00Z)
  -k K        Use minute K directly (ignores system time)
```

## Examples

Live clock with Unicode:
```bash
./clock
```

Generate 60 frames for minute 12345 in ASCII:
```bash
./clock -a -k 12345 -n 60
```

Round-trip test (generate frames, decode back):
```bash
./clock -a -k 12345 -n 60 | ./clock -i
```

Output:
```
Minute (k): 12345
Rotation: 0 (first frame was second 0)
Origin: 1970-01-09 10:45:00 (local) = 1970-01-09T09:45:00Z
```

Distinct colors mode (3-color graph coloring):
```bash
./clock -a -d
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
