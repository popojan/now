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
    find_rotation_and_k,
    verify_inversion,
    get_all_cells_for_minute,
    rotate_list,
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


def compute_clock_origin(video_timestamp, rotation_offset, k):
    """
    Compute when the clock was originally started.

    Args:
        video_timestamp: datetime when video was filmed
        rotation_offset: which second (0-59) the video started at
        k: the minute identifier recovered from inversion

    Returns:
        datetime of clock origin (epoch)
    """
    # The video starts at second `rotation_offset` of some minute
    # First, find the start of that minute
    video_minute_start = video_timestamp - timedelta(seconds=rotation_offset)

    # The clock has been running for k minutes since epoch
    clock_origin = video_minute_start - timedelta(minutes=k)

    return clock_origin


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

    if args.verbose:
        print("\nFirst 5 observations:")
        for i, obs in enumerate(observations[:5]):
            print(f"  Second {i}: {sorted(obs)}")

    # Find rotation and k - try progressively lower thresholds if needed
    print("\nSearching for valid clock state...")
    rotation, k, match_count = find_rotation_and_k(observations)

    # If 90% threshold fails, try lower thresholds
    if rotation is None:
        for threshold in [0.85, 0.80, 0.75]:
            rotation, k, match_count = find_rotation_and_k(observations, min_match_ratio=threshold)
            if rotation is not None:
                print(f"  (Used {int(threshold*100)}% match threshold)")
                break

    if rotation is None:
        print(f"\nError: Could not find valid clock state ({match_count}/60 matches).", file=sys.stderr)
        print("Possible causes:", file=sys.stderr)
        print("  - Video does not show the Mondrian clock", file=sys.stderr)
        print("  - Video quality too low for detection", file=sys.stderr)
        print("  - Incorrect tolerance value (try adjusting with --tolerance)", file=sys.stderr)
        sys.exit(1)

    print(f"Found valid state! ({match_count}/60 seconds matched)")
    print(f"  Video started at second: {rotation}")
    print(f"  Minute identifier (k): {k}")

    # Verify with main observations
    if args.verbose:
        expected = get_all_cells_for_minute(k)
        rotated_obs = rotate_list(observations, rotation)
        print("\nVerification (first 10 seconds):")
        for i in range(10):
            expected_cells = sorted(expected[i])
            observed_cells = sorted(rotated_obs[i])
            match = "OK" if set(expected_cells) == set(observed_cells) else "MISMATCH"
            print(f"  Second {i}: expected {expected_cells}, observed {observed_cells} [{match}]")

    # Cross-validate with extra observations (for videos > 60 seconds)
    extra_obs = metadata.get('extra_observations', [])
    if extra_obs:
        # Extra observations start where the main ones ended
        # After 60 seconds, we're in minute k+1, k+2, etc.
        # The starting second in the new minute depends on rotation
        extra_verified = 0
        extra_total = len(extra_obs)

        for i, obs in enumerate(extra_obs):
            # Calculate which minute and second this observation represents
            total_second = 60 + i  # seconds since video start
            # After rotation alignment: which absolute second is this?
            abs_second = (rotation + total_second) % 60
            minute_offset = (rotation + total_second) // 60
            expected_k = k + minute_offset

            expected_cells = get_all_cells_for_minute(expected_k)
            if set(expected_cells[abs_second]) == set(obs):
                extra_verified += 1

        print(f"\nCross-validation with extra {extra_total} seconds: {extra_verified}/{extra_total} matched")
        if extra_verified < extra_total * 0.9:
            print("  Warning: Low cross-validation match rate", file=sys.stderr)

    # Compute origin
    clock_origin = compute_clock_origin(video_timestamp, rotation, k)

    # Compute elapsed time
    seconds_elapsed = k * 60 + rotation
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
    print(f"\nClock origin (datetime):")
    print(f"  {format_timestamp(clock_origin)}")

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
