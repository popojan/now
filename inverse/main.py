#!/usr/bin/env python3
"""
Clock Origin Detector - Main CLI

Analyzes a video recording of the Mondrian clock to determine
when the clock was originally started.

Usage:
    python main.py <video_path> [--timestamp "YYYY-MM-DD HH:MM:SS"] [--verbose]
"""

import argparse
import sys
from datetime import datetime, timedelta

from clock_inverse import (
    find_k_from_observations,
    get_all_cells_for_minute,
    PERIOD,
)
from video_analyzer import analyze_video, VideoTooShortError
from metadata import get_video_timestamp, format_timestamp


def parse_timestamp(timestamp_str):
    """Parse user-provided timestamp string."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse timestamp: {timestamp_str}")


def compute_clock_origin(video_timestamp, rotation_offset, k, minute_boundary_offset_ms=None):
    """
    Compute when the clock was originally started.

    Args:
        video_timestamp: datetime when video was filmed
        rotation_offset: which second (0-59) the video started at
        k: the minute identifier recovered from inversion
        minute_boundary_offset_ms: if available, precise offset from video start to minute boundary

    Returns:
        (clock_origin datetime, sub_second_offset_ms for precision indication)
    """
    if minute_boundary_offset_ms is not None:
        # Use precise minute boundary from second 0 detection
        minute_start = video_timestamp + timedelta(milliseconds=minute_boundary_offset_ms)
        # Track sub-second precision for display
        sub_second_ms = minute_boundary_offset_ms % 1000
        if sub_second_ms > 500:
            sub_second_ms -= 1000  # Normalize to -500 to +500
    else:
        # Fallback: estimate from rotation offset
        minute_start = video_timestamp - timedelta(seconds=rotation_offset)
        sub_second_ms = 0

    # The clock has been running for k minutes since epoch
    # Use days to avoid overflow with large k values
    days = k // (60 * 24)
    remaining_minutes = k % (60 * 24)
    clock_origin = minute_start - timedelta(days=days, minutes=remaining_minutes)

    # Round to nearest second if sub-second offset is significant
    if abs(sub_second_ms) >= 400:
        # Round to nearest second
        if sub_second_ms > 0:
            clock_origin = clock_origin + timedelta(milliseconds=1000 - sub_second_ms)
        else:
            clock_origin = clock_origin - timedelta(milliseconds=abs(sub_second_ms))
        sub_second_ms = 0

    return clock_origin, sub_second_ms


def main():
    parser = argparse.ArgumentParser(
        description="Determine when the Mondrian clock was started from a video recording."
    )
    parser.add_argument(
        "video_path",
        help="Path to the video file (60-second recording of the clock)"
    )
    parser.add_argument(
        "--timestamp", "-t",
        help="Override video timestamp (format: YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=80,
        help="Color distance tolerance for cell detection (default: 80)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate debug visualization video"
    )

    args = parser.parse_args()

    # Get video timestamp
    if args.timestamp:
        try:
            video_timestamp = parse_timestamp(args.timestamp)
            timestamp_source = "command line"
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        video_timestamp, timestamp_source = get_video_timestamp(
            args.video_path, verbose=args.verbose
        )
        if video_timestamp is None:
            print("Error: Could not determine video timestamp.", file=sys.stderr)
            print("Please provide timestamp with --timestamp option.", file=sys.stderr)
            sys.exit(1)

    print(f"Video timestamp: {format_timestamp(video_timestamp)} (from {timestamp_source})")

    # Analyze video
    print("\nAnalyzing video...")
    try:
        observations, metadata = analyze_video(
            args.video_path,
            tolerance=args.tolerance,
            verbose=args.verbose
        )
    except VideoTooShortError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing video: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(observations)} seconds of cell observations")

    # Report corrections
    if metadata.get('corrected_frames', 0) > 0:
        print(f"  ({metadata['corrected_frames']} frames corrected via error detection)")

    # Get minute boundary offset (if second 0 was found)
    minute_boundary_offset_ms = metadata.get('minute_boundary_offset_ms')
    if minute_boundary_offset_ms is not None:
        print(f"  Minute boundary: {minute_boundary_offset_ms}ms after video start")

    if args.verbose:
        print("\nFirst 5 observations:")
        for i, obs in enumerate(observations[:5]):
            print(f"  Second {i}: {sorted(obs)}")

    # Find k using sum-based detection - try progressively lower thresholds if needed
    print("\nSearching for valid clock state...")
    start_second, k, match_count = find_k_from_observations(observations)

    # If 90% threshold fails, try lower thresholds
    if k is None:
        for threshold in [0.85, 0.80, 0.75]:
            start_second, k, match_count = find_k_from_observations(observations, min_match_ratio=threshold)
            if k is not None:
                print(f"  (Used {int(threshold*100)}% match threshold)")
                break

    if k is None:
        print(f"\nError: Could not find valid clock state ({match_count}/60 matches).", file=sys.stderr)
        print("Possible causes:", file=sys.stderr)
        print("  - Video does not show the Mondrian clock", file=sys.stderr)
        print("  - Video quality too low for detection", file=sys.stderr)
        print("  - Incorrect tolerance value (try adjusting with --tolerance)", file=sys.stderr)
        sys.exit(1)

    print(f"Found valid state! ({match_count}/60 seconds matched)")
    print(f"  Video started at second: {start_second}")
    print(f"  Minute identifier (k): {k}")

    # Verify with main observations
    if args.verbose:
        expected = get_all_cells_for_minute(k)
        print("\nVerification (first 10 seconds):")
        for i in range(min(10, len(observations))):
            clock_second = (start_second + i) % 60
            expected_cells = sorted(expected[clock_second])
            observed_cells = sorted(observations[i])
            match = "OK" if set(expected_cells) == set(observed_cells) else "MISMATCH"
            print(f"  Second {clock_second}: expected {expected_cells}, observed {observed_cells} [{match}]")

    # Compute origin
    clock_origin, sub_second_ms = compute_clock_origin(video_timestamp, start_second, k, minute_boundary_offset_ms)

    # Compute elapsed time
    seconds_elapsed = k * 60 + start_second
    minutes_elapsed = k
    days_elapsed = minutes_elapsed / (60 * 24)
    years_elapsed = days_elapsed / 365.25

    print(f"\n{'='*50}")
    print(f"RESULT")
    print(f"{'='*50}")
    print(f"\nElapsed time since clock started:")
    print(f"  {seconds_elapsed:,} seconds")
    print(f"  ({minutes_elapsed:,} minutes, {days_elapsed:,.1f} days", end="")
    if years_elapsed >= 1:
        print(f", {years_elapsed:,.1f} years", end="")
    print(")")
    # Convert to UTC for display (clock now uses UTC origin)
    if hasattr(clock_origin, 'tzinfo') and clock_origin.tzinfo is not None:
        from datetime import timezone
        clock_origin_utc = clock_origin.astimezone(timezone.utc)
    else:
        clock_origin_utc = clock_origin
    print(f"\nClock origin (UTC):")
    if sub_second_ms != 0:
        print(f"  {format_timestamp(clock_origin_utc)} ({sub_second_ms:+d}ms)")
    else:
        print(f"  {format_timestamp(clock_origin_utc)}")

    print(f"\nNote: The clock period is {PERIOD:,} minutes")
    print(f"      (>{PERIOD / (60 * 24 * 365.25) / 1e9:.0f} billion years)")

    # Generate visualization if requested
    if args.visualize:
        from video_analyzer import visualize_detection
        output_path = args.video_path.rsplit('.', 1)[0] + '_debug.mp4'
        visualize_detection(args.video_path, output_path, args.tolerance)

    return 0


if __name__ == "__main__":
    sys.exit(main())
