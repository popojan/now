# Test Videos for Clock Origin Detector

## Supported Format

Currently supported: **iPhone Time-lapse videos**

The analyzer is designed to be FPS-independent. iPhone time-lapse videos vary in compression ratio based on recording duration:
- Short recordings: ~30fps playback (less compression)
- Longer recordings: Lower fps playback (more compression)

The analyzer detects the actual clock seconds by summing visible cell values (cells shown at second S always sum to S), making it robust to varying frame rates.

## Video Files

### IMG_6703.MOV (Quality Video - Unix Epoch Clock)
- **Characteristics**: High quality, static shot, clock fills most of frame
- **Recorded**: 2026-01-03 06:37:01 CET (05:37:01 UTC)
- **Clock Origin**: 1970-01-01 00:00:00 UTC (Unix epoch)
- **Expected k**: 29,457,037 minutes
- **Expected Result**: 60/60 seconds matched
- **Notes**: Standard test case - axis-aligned clock, clean background

### IMG_6719.MOV (Difficult Video - Y2K Clock)
- **Characteristics**: Filmed from distance, tilted clock, camera motion, cluttered background (notebook, laptop visible)
- **Recorded**: 2026-01-03 22:26:27 CET (21:26:27 UTC)
- **Clock Origin**: 2000-01-01 00:00:00 UTC (Y2K)
- **Expected k**: 13,679,846 minutes
- **Expected Result**: 60/60 seconds matched
- **Notes**: Stress test - requires perspective correction and per-frame corner tracking

## Running Tests

```bash
cd inverse
source venv/bin/activate
python main.py ../test_videos/IMG_6703.MOV -v
python main.py ../test_videos/IMG_6719.MOV -v
```

## Validation Criteria

A successful detection must:
1. Match 60/60 or at least 54/60 (90%) seconds
2. Compute the correct clock origin (matches expected for each video)
3. Handle both axis-aligned and tilted video orientations

## Detection Approaches

The analyzer uses a hybrid approach:
1. **Simple approach**: Per-frame contour detection (fast, works for quality axis-aligned videos)
2. **Hough approach**: Hough line-based corner refinement with per-frame tracking (handles tilted/moving videos)

The analyzer tries the simple approach first, and falls back to Hough if needed.
