# Streaming Algorithm Analysis

This document analyzes how to adapt the batch-based clock inversion algorithm into a real-time streaming approach suitable for an iOS app with live camera overlay.

## Key Insight: Sum-Based Second Detection

The clock cells have areas 1, 2, 4, 6, 12, 15, 20 (sum = 60). **Each second S displays cells that sum to exactly S.** This enables:

1. **Instant second identification**: `clock_second = sum(visible_cells) % 60`
2. **No rotation search needed**: We know which second each frame represents
3. **Error detection**: If observed cells don't form a valid combination, detection failed
4. **Error correction**: Try flipping one cell to find a valid match

## Current Batch Approach

1. Collect all ~120 frames (60 seconds at ~2 FPS)
2. Detect empty color via frequency clustering across ALL frames
3. Extract cell observations for each clock second
4. Use sum-based detection to identify which second each frame represents
5. Compute clock origin from k and video timestamp

## Streaming Blockers

### 1. Empty Color Detection (Primary Blocker)

The current approach needs all frames to determine which color appears most frequently (the "empty" cell color). Alternatives for streaming:

| Approach | Pros | Cons |
|----------|------|------|
| **Calibration tap** | Accurate, simple | Requires user interaction |
| **Heuristic (white)** | Zero setup | Fails for dark themes |
| **Adaptive histogram** | Automatic | Needs 5-10 seconds to stabilize |
| **Relative contrast** | Frame-independent | Complex, less reliable |

**Recommended**: Ask user to tap an empty cell region once at start. Minimal friction, maximum reliability.

### 2. Rotation Ambiguity

We don't know which second (0-59) the observation started on.

**Solutions**:
- Track 60 parallel hypotheses, prune as evidence accumulates
- Some seconds have unique patterns (second 0: all cells off)
- After ~10-15 seconds, most rotations become impossible
- Show "candidates remaining" as confidence indicator

## Bit Structure (Streaming-Friendly)

The perm() function encodes k using different bit densities per second:

| Seconds | Options | Information | Collection |
|---------|---------|-------------|------------|
| 20 | 1 (deterministic) | 0 bits | No data needed |
| 18 | 2 options | 18 bits (k2) | As encountered |
| 16 | 3 options | ~25.4 bits (k3) | As encountered |
| 6 | 4 options | 12 bits (k4) | As encountered |

Total: ~55.4 bits per minute, encoding k up to 46,221,064,723,759,104.

## Proposed iOS Overlay UI

```
┌─────────────────────────────┐
│  Mondrian Clock Decoder     │
├─────────────────────────────┤
│ Seconds: 23/60              │
│ ████████████░░░░░░░░░░░░░░  │
│                             │
│ k2: 14/18 bits    ████████░ │
│ k3: 10/16 digits  ██████░░░ │
│ k4: 3/6 values    ████░░░░░ │
│                             │
│ Rotation candidates: 3      │
│ Confidence: 78%             │
│                             │
│ Time remaining: ~37s        │
└─────────────────────────────┘
```

## Streaming Decoder Architecture

```python
class StreamingDecoder:
    def __init__(self, empty_color=None):
        self.empty_color = empty_color  # From calibration
        self.observations = []          # Cell sets per frame
        self.candidates = list(range(60))  # Possible rotations

        # Bit collection slots
        self.k2_slots = [None] * 18  # 2-option seconds
        self.k3_slots = [None] * 16  # 3-option seconds
        self.k4_slots = [None] * 6   # 4-option seconds

    def add_frame(self, cells_visible: set) -> dict:
        """Process one frame, update state, return progress."""
        self.observations.append(cells_visible)
        self._prune_impossible_rotations()
        self._fill_bit_slots()
        return self.get_progress()

    def _prune_impossible_rotations(self):
        """Remove rotation candidates that don't match observations."""
        valid = []
        for rot in self.candidates:
            if self._rotation_matches(rot):
                valid.append(rot)
        self.candidates = valid

    def _fill_bit_slots(self):
        """Fill k2/k3/k4 slots based on current best rotation."""
        if len(self.candidates) == 1:
            # Unambiguous - fill slots definitively
            self._fill_slots_for_rotation(self.candidates[0])

    def get_progress(self) -> dict:
        return {
            'seconds_observed': len(set(self._get_clock_seconds())),
            'k2_complete': sum(1 for x in self.k2_slots if x is not None),
            'k3_complete': sum(1 for x in self.k3_slots if x is not None),
            'k4_complete': sum(1 for x in self.k4_slots if x is not None),
            'rotation_candidates': len(self.candidates),
            'is_complete': self._is_complete(),
            'k': self._compute_k() if self._is_complete() else None,
            'confidence': self._compute_confidence()
        }

    def _compute_confidence(self) -> float:
        """Confidence based on candidates and filled slots."""
        rotation_conf = 1.0 / max(len(self.candidates), 1)
        slots_filled = (
            sum(1 for x in self.k2_slots if x is not None) +
            sum(1 for x in self.k3_slots if x is not None) +
            sum(1 for x in self.k4_slots if x is not None)
        )
        slots_total = 18 + 16 + 6
        slot_conf = slots_filled / slots_total
        return (rotation_conf + slot_conf) / 2
```

## Early Termination Opportunities

In some cases, we can determine k before 60 seconds:

1. **Unique patterns**: Second 0 has all cells empty - if observed, rotation is known
2. **Constraint propagation**: Each observation eliminates impossible k values
3. **Lucky bit coverage**: If we happen to see all 40 variable seconds early

Typical case: Full determination requires 55-60 seconds.

## iOS Implementation Notes

### Camera Integration
- Use AVFoundation for real-time camera feed
- Process at 2-4 FPS (matching clock update rate)
- Run detection on background thread

### Clock Region Detection
- Use Core ML or Vision framework for quadrilateral detection
- Alternatively: manual corner selection for reliability
- Apply perspective transform to normalize view

### Performance
- Cell color sampling: ~7 point samples, trivial cost
- Rotation pruning: 60 candidates × 60 seconds = 3600 comparisons max
- Runs easily at 30+ FPS on modern iOS devices

## Future Enhancements

1. **AR overlay**: Show detected cells highlighted in real-time
2. **History**: Save decoded origins with timestamps
3. **Share**: Generate shareable result cards
4. **Widget**: Show current k value as home screen widget

## Signature Support (TODO for iOS)

The C terminal utility (`src/now.c`) supports clock signatures via `-P` and `-N` flags:

### Encoding Mode
- `-P VALUE`: Clock signature/base (must be coprime with 60: no factors 2, 3, 5)
- `-N SALT`: Value to encode (0 to P-1)

Example: `-P 7 -N 3` encodes value 3 with base 7.

### Decoding Mode
From 2-minute recording, auto-detect P from k_combined deltas between minutes:
```
k_combined[minute+1] - k_combined[minute] = P
N_era = k_combined % P
```

### iOS Integration Ideas
1. **Personalized clocks**: Each user picks their signature P
2. **Verification**: Decode P and N from screen recording to verify authenticity
3. **QR alternative**: Clock pattern encodes data visually
4. **Web app sync**: Add `?P=7&N=3` URL params to web version

### Implementation Notes
- Requires 2 complete minutes (120 frames) for auto-detection
- With known P (user setting), only 60 frames needed
- Era cycling ensures full 88B-year period preserved regardless of P
- Coprime-60 constraint ensures N doesn't correlate with visual seconds
