# Test Videos for Clock Origin Detector

## Supported Format

Currently supported: **iPhone Time-lapse videos**

The analyzer is designed to be FPS-independent. iPhone time-lapse videos vary in compression ratio based on recording duration:
- Short recordings: ~30fps playback (less compression)
- Longer recordings: Lower fps playback (more compression)

The analyzer detects the actual clock seconds by summing visible cell values (cells shown at second S always sum to S), making it robust to varying frame rates.

## Clock Origin History

On 2026-01-03 at 11:51 CET, the web clock was changed from local time origin to UTC origin (commit 7edf77b). This affects the detected clock origin:

- **Before 11:51 CET**: Clock used local time origin (1970-01-01 00:00:00 CET = 1969-12-31 23:00:00 UTC)
- **After 11:51 CET**: Clock uses UTC origin (1970-01-01 00:00:00 UTC)

Videos recorded before this change will show the local time origin when analyzed.

## Video Files

### IMG_6703.MOV (Quality Video - Local Time Origin)
- **Characteristics**: High quality, static shot, clock fills most of frame
- **Recorded**: 2026-01-03 06:37:01 CET (before UTC change)
- **Clock Origin**: 1969-12-31 23:00:00 UTC (= 1970-01-01 00:00:00 CET, local time epoch)
- **Expected k**: 29,457,037 minutes
- **Expected Result**: 60/60 seconds matched
- **Notes**: Standard test case - axis-aligned clock, clean background

### IMG_6707.MOV (Local Time Origin)
- **Recorded**: 2026-01-03 10:41:31 CET (before UTC change)
- **Clock Origin**: 1969-12-31 23:00:00 UTC (local time epoch)
- **Expected k**: 29,457,281 minutes
- **Expected Result**: 60/60 seconds matched

### IMG_6709.MOV (Local Time Origin)
- **Recorded**: 2026-01-03 11:20:53 CET (before UTC change)
- **Clock Origin**: 1969-12-31 23:00:00 UTC (local time epoch)
- **Expected k**: 29,457,321 minutes
- **Expected Result**: 60/60 seconds matched

### IMG_6712.MOV (UTC Origin)
- **Recorded**: 2026-01-03 11:55:00 CET (just after UTC change)
- **Clock Origin**: 1970-01-01 00:00:00 UTC (Unix epoch)
- **Expected k**: 29,457,295 minutes
- **Expected Result**: 60/60 seconds matched

### IMG_6717.MOV (Y2K Clock - UTC Origin)
- **Recorded**: 2026-01-03 21:37:00 CET (after UTC change)
- **Clock Origin**: 2000-01-01 00:00:00 UTC (Y2K)
- **Expected k**: 13,679,797 minutes
- **Expected Result**: 60/60 seconds matched

### IMG_6719.MOV (Difficult Video - Y2K Clock)
- **Characteristics**: Filmed from distance, tilted clock, camera motion, cluttered background
- **Recorded**: 2026-01-03 22:26:27 CET (after UTC change)
- **Clock Origin**: 2000-01-01 00:00:00 UTC (Y2K)
- **Expected k**: 13,679,846 minutes
- **Expected Result**: 60/60 seconds matched
- **Notes**: Stress test - requires Hough-based corner tracking

### IMG_6743.MOV (Red Monochrome - UTC Origin)
- **Characteristics**: Monochrome red color scheme (filled=red, empty=white)
- **Recorded**: 2026-01-04 14:14:26 CET
- **Clock Origin**: 1970-01-01 00:00:00 UTC (Unix epoch)
- **Expected k**: 29,458,874 minutes
- **Expected Result**: 60/60 seconds matched
- **Notes**: Tests color-independent detection (simple approach works)

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
