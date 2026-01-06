# Now

A clock cycling through 60 states per minute, each a subset sum of {1, 2, 4, 6, 12, 15, 20}.

These weights cover [Farey sequence](https://en.wikipedia.org/wiki/Farey_sequence) $F_6$ exactly.

Period: $2^{30} \times 3^{16}$ min ≈ 88 billion years

## How It Works

The clock face consists of 7 rectangular cells with areas 1, 2, 4, 6, 12, 15, and 20 units. Each second displays a unique combination of cells that sum to that second (0–59). The specific combination chosen each second depends on the minute number (k), creating a unique 60-pattern sequence for every minute in the period.

## Project Structure

- `src/` - C terminal implementation
- `web/` - Clock webpage (HTML/CSS/JS)
- `inverse/` - Python video analyzer (reconstructs clock origin from video recordings)
- `tests/` - Test scripts

## Terminal Clock

Build from repository root:

```bash
make
bin/now            # Live clock
bin/now -l         # In-place updates (no scroll)
bin/now -p emoji   # Emoji preset
```

### Display Options

```
-a          ASCII borders
-p PRESET   Preset: cjk (default), blocks, blocks1, distinct, kanji, emoji
-f CHARS    Custom 7 UTF-8 fill characters
-1          Half-width mode
-w          Wide fills (for CJK/emoji)
-l          In-place update (TTY only)
-s          Simulate (fast, no delay)
-i          Inverse: read frames, detect signature, output origin
```

### Round-trip Test

```bash
bin/now -s -n 120 | bin/now -i -s
```

Note: Use `-s` for both encoder and decoder when testing with simulated data.

### Signatures

Encode a unique identifier into your clock that can be auto-detected from recordings:

```bash
bin/now -P 7 -l                       # Live clock with signature 7
bin/now -P 7 -n 180 -s | bin/now -i -s # Encode and auto-detect
```

Options:
- `-P VALUE` - Clock signature (coprime with 60 recommended, but any value works)
- `-N SALT` - Optional salt for era cycling (default: 0)

The signature is encoded in the permutation pattern and auto-detected from any 2-minute recording, while preserving the full 88-billion-year period through era cycling.

## Technical Details

### Permutation Function

The `perm(k)` function maps a minute number to 60 display indices:
- 20 seconds have 1 option (deterministic)
- 18 seconds have 2 options (18 bits)
- 16 seconds have 3 options (~25 bits)
- 6 seconds have 4 options (12 bits)

Total entropy: ~55 bits per minute, sufficient for the 46-quadrillion-minute period.

### Dependencies

- **Terminal clock**: C compiler (gcc/clang)
- **Web clock**: None (pure HTML/CSS/JS)

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - Non-commercial use only
