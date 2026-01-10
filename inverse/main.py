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
    find_k_with_autodetect,
    find_k_multi_minute,
    detect_sig_period_from_multi_minute,
    get_all_cells_for_minute,
    inverse_with_correction,
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


def format_ancient_date(year, month, day, hour, minute, second):
    """Format a date that may be before year 1, using ISO 8601 extended format.
    Year 0 = 1 B.C., Year -1 = 2 B.C., etc.
    Negative years use the format: -YYYY-MM-DD (ISO 8601 extended)
    """
    if year >= 1:
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    elif year == 0:
        return f"0000-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    else:
        # Negative year: -1 -> -0001, -10 -> -0010, etc.
        return f"-{abs(year):04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def compute_clock_origin(video_timestamp, rotation_offset, k, minute_boundary_offset_ms=None, sig_P=1, sig_N=0):
    """
    Compute when the clock was originally started.

    Args:
        video_timestamp: datetime when video was filmed
        rotation_offset: which second (0-59) the video started at
        k: the minute identifier recovered from inversion
        minute_boundary_offset_ms: if available, precise offset from video start to minute boundary
        sig_P: signature period P (for signature encoding k = minute * P + N)
        sig_N: signature N value

    Returns:
        (clock_origin datetime or string for ancient dates, sub_second_offset_ms for precision indication)
    """
    if minute_boundary_offset_ms is not None:
        # Use precise minute boundary from second 0 detection
        minute_start = video_timestamp + timedelta(milliseconds=minute_boundary_offset_ms)
        # Track sub-second precision for display
        sub_second_ms = minute_boundary_offset_ms % 1000
        if sub_second_ms > 500:
            sub_second_ms -= 1000  # Normalize to -500 to +500
    else:
        # Fallback: estimate when current minute's second 0 occurred
        # If video starts at second S, second 0 was S seconds ago
        minute_start = video_timestamp - timedelta(seconds=rotation_offset)
        sub_second_ms = 0

    # When video spans two minutes (rotation_offset > 0), k is the NEXT minute
    # (containing second 0 we observe), so adjust k for the current minute
    if rotation_offset > 0:
        effective_k = k - sig_P  # Previous minute in signature encoding
    else:
        effective_k = k

    # With signature encoding (k = minute * P + N), compute actual minutes elapsed
    if sig_P > 1:
        actual_minutes = (effective_k - sig_N) // sig_P
    else:
        actual_minutes = effective_k

    # The clock has been running for actual_minutes since epoch
    # Use days to avoid overflow with large k values
    days = actual_minutes // (60 * 24)
    remaining_minutes = actual_minutes % (60 * 24)

    try:
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
    except OverflowError:
        # Date is before year 1 (B.C.) - compute manually
        # Convert minute_start to total minutes since year 1
        # Make minute_start timezone-naive for comparison
        if hasattr(minute_start, 'tzinfo') and minute_start.tzinfo is not None:
            # Convert to UTC and make naive
            from datetime import timezone
            minute_start_naive = minute_start.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            minute_start_naive = minute_start

        year1_epoch = datetime(1, 1, 1, 0, 0, 0)
        minutes_since_year1 = int((minute_start_naive - year1_epoch).total_seconds() / 60)
        origin_minutes = minutes_since_year1 - effective_k

        # Convert back to date components
        # Negative origin_minutes means we're before year 1
        if origin_minutes >= 0:
            # Should not happen if we got OverflowError, but handle it
            origin_dt = year1_epoch + timedelta(minutes=origin_minutes)
            return origin_dt, sub_second_ms

        # Calculate date before year 1 using exact day calculation
        minutes_per_day = 60 * 24  # 1440

        def is_leap_year(y):
            """Proleptic Gregorian leap year (works for year <= 0)
            In ISO 8601 / astronomical year numbering:
            year 0 = 1 BC, year -1 = 2 BC, etc.
            Leap years: divisible by 4, except centuries unless divisible by 400
            """
            # For negative years, use the astronomical convention directly
            # Year 0 mod 4 = 0, so year 0 is a leap year
            return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

        def days_in_year(y):
            return 366 if is_leap_year(y) else 365

        # Calculate target year directly using average year length
        # (avoids iterating billions of times for ancient dates)
        abs_minutes = abs(origin_minutes)
        minutes_per_year_avg = 365.2425 * minutes_per_day  # Gregorian average

        # Estimate how many years back from year 1 (not year 0)
        # Year 0 is 1 year before year 1, year -1 is 2 years before, etc.
        years_back_estimate = int(abs_minutes / minutes_per_year_avg)

        # Calculate exact minutes for the estimated years using 400-year cycles
        full_400_cycles = years_back_estimate // 400
        remaining_years = years_back_estimate % 400

        # Minutes in full 400-year cycles (146097 days per 400 years)
        minutes_in_cycles = full_400_cycles * 146097 * minutes_per_day

        # Minutes in remaining years (year by year, but at most 399 iterations)
        # Count from year 0 going backwards
        minutes_in_remaining = 0
        for y in range(remaining_years):
            check_year = -(full_400_cycles * 400 + y)
            minutes_in_remaining += days_in_year(check_year) * minutes_per_day

        total_minutes_consumed = minutes_in_cycles + minutes_in_remaining

        # Start at year 0 and adjust based on how many minutes we need
        origin_year = 0
        remaining_minutes = abs_minutes

        # Subtract full 400-year cycles
        for _ in range(full_400_cycles):
            origin_year -= 400
            remaining_minutes -= 146097 * minutes_per_day

        # Subtract remaining years one at a time
        # Use > not >= because if remaining equals exactly one year, we're at the start of that year
        while remaining_minutes > days_in_year(origin_year) * minutes_per_day:
            remaining_minutes -= days_in_year(origin_year) * minutes_per_day
            origin_year -= 1

        # remaining_minutes is time going BACKWARD from year 1
        # Convert to time going FORWARD from start of origin_year
        minutes_from_start = days_in_year(origin_year) * minutes_per_day - remaining_minutes

        # Calculate day/time within the year
        days_into_year = int(minutes_from_start // minutes_per_day)
        remaining = int(minutes_from_start % minutes_per_day)
        hours = remaining // 60
        minutes = remaining % 60

        # Convert day-of-year to month/day (proleptic Gregorian)
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if is_leap_year(origin_year):
            month_days[1] = 29  # February has 29 days in leap year
        day_of_year = days_into_year
        month = 1
        for m, mdays in enumerate(month_days, 1):
            if day_of_year < mdays:
                month = m
                day = day_of_year + 1
                break
            day_of_year -= mdays
        else:
            month = 12
            day = 31

        # Return as formatted string for ancient dates
        origin_str = format_ancient_date(origin_year, month, day, hours, minutes, 0)
        return origin_str, sub_second_ms


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
        default=50,
        help="Color distance tolerance for cell detection (default: 50)"
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
    parser.add_argument(
        "--sig-period", "-P",
        type=int,
        default=None,
        help="Signature period P for k encoding (auto-detected if not given)"
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

    real_obs = metadata.get('real_observations', len(observations))
    print(f"Extracted {real_obs}/60 seconds of cell observations")

    # Warn if coverage is too low
    if real_obs < 54:
        print(f"  WARNING: Low coverage ({real_obs}/60) - results may be unreliable", file=sys.stderr)

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

    # Detect signature period P and best k from multi-minute data (if available)
    all_minutes = metadata.get('all_minutes', {})
    detected_sig_period = args.sig_period  # User override takes precedence
    multi_best_k = None
    multi_best_coverage = 0

    if all_minutes:
        multi_P, multi_N, multi_k_values, multi_best_k, multi_best_coverage = detect_sig_period_from_multi_minute(all_minutes)
        if detected_sig_period is None and multi_P > 1:
            detected_sig_period = multi_P
            print(f"\nDetected signature from multi-minute analysis: P={multi_P}, N={multi_N}")
            if args.verbose:
                print(f"  k values per minute: {multi_k_values}")

    # Find k using sum-based detection with error correction (always on)
    print("\nSearching for valid clock state...")
    result = inverse_with_correction(observations, apply_correction=True, sig_period=detected_sig_period)

    start_second = result['start_second']
    k = result['k']
    match_count = result['match_count']
    sig_P = result['P']
    sig_N = result['N']
    sig_confidence = result['signature_confidence']
    anchor_count = result['anchor_count']
    corrections = result['corrections']

    # Report error correction stats
    if corrections:
        num_corrected = sum(1 for c in corrections if not c.is_anchor)
        if num_corrected > 0:
            distances = [c.distance for c in corrections if not c.is_anchor]
            print(f"  Error correction: {anchor_count} anchors, {num_corrected} corrected (distances: {distances})")

    # Use multi-minute best k only if it has PERFECT coverage (60/60)
    # This handles collisions where spanning detection finds an ambiguous k
    # but avoids using incorrect k from partial observations with errors
    if multi_best_k is not None and multi_best_coverage == 60 and match_count < 60:
        if args.verbose:
            print(f"  Using multi-minute k (perfect {multi_best_coverage}/60 vs spanning {match_count}/60)")
        k = multi_best_k
        match_count = multi_best_coverage
        # For multi-minute k, start_second is 0 (aligned minute)
        start_second = 0

    # If no valid state found, try lower thresholds with autodetection
    if k is None:
        for threshold in [0.85, 0.80, 0.75]:
            start_second, k, match_count, detected_P = find_k_with_autodetect(
                observations, min_match_ratio=threshold, sig_period=args.sig_period
            )
            if k is not None:
                print(f"  (Used {int(threshold*100)}% match threshold, correction disabled)")
                sig_P = detected_P
                sig_N = k % sig_P if sig_P > 0 else 0
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
    if sig_P > 1:
        print(f"  Detected signature: P={sig_P}, N={sig_N}")

    # For multi-minute videos, use multi-minute verification
    # This catches detection errors by cross-checking k values across minutes
    all_minutes = metadata.get('all_minutes', {})
    if len(all_minutes) >= 2:
        multi_result = find_k_multi_minute(all_minutes, verbose=args.verbose, sig_period=sig_P)
        if multi_result['k'] is not None and len(multi_result['k_values']) >= 2:
            # Check if multi-minute analysis found a different k
            if multi_result['corrected_minute'] is not None:
                original_k = multi_result['original_k']
                corrected_k = multi_result['k']
                print(f"  Multi-minute verification: corrected k={original_k} -> k={corrected_k}")
                k = corrected_k
            # Use signature detection from multi-minute analysis
            if multi_result['P'] > 1:
                sig_P = multi_result['P']
                sig_N = multi_result['N']
                sig_confidence = multi_result['confidence']
                if args.verbose:
                    print(f"  Detected signature: P={sig_P}, N={sig_N} (confidence={sig_confidence*100:.0f}%)")

    # Correct video timestamp using detected second (±30s tolerance)
    # The detected second tells us exactly where in the minute cycle we are
    ts_second = video_timestamp.second
    ts_microsecond = video_timestamp.microsecond
    correction_seconds = start_second - ts_second
    if correction_seconds > 30:
        correction_seconds -= 60  # wrap backward
    elif correction_seconds < -30:
        correction_seconds += 60  # wrap forward

    # Total correction includes zeroing sub-second part
    correction_ms = correction_seconds * 1000 - ts_microsecond // 1000

    if correction_seconds != 0 or ts_microsecond > 0:
        video_timestamp = video_timestamp + timedelta(seconds=correction_seconds)
        # Zero out microseconds since we're snapping to second boundary
        video_timestamp = video_timestamp.replace(microsecond=0)
        # After correction, video_timestamp now matches detected second exactly,
        # so we don't need minute_boundary_offset_ms (which was relative to original timestamp)
        minute_boundary_offset_ms = None
        print(f"  Timestamp corrected by {correction_ms:+d}ms (detected second: {start_second})")

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
    clock_origin, sub_second_ms = compute_clock_origin(video_timestamp, start_second, k, minute_boundary_offset_ms, sig_P, sig_N)

    # Compute elapsed time (using actual minutes for signature encoding)
    # When spanning (start_second > 0), k is for the "next" minute, so adjust
    if sig_P > 1:
        effective_k = k - sig_P if start_second > 0 else k
        actual_minutes = (effective_k - sig_N) // sig_P
    else:
        actual_minutes = k - 1 if start_second > 0 else k
    seconds_elapsed = actual_minutes * 60 + start_second
    minutes_elapsed = actual_minutes
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
    # Display clock origin
    print(f"\nClock origin (UTC):")
    if isinstance(clock_origin, str):
        # Ancient date - already formatted as string with B.C. notation
        if sub_second_ms != 0:
            print(f"  {clock_origin} ({sub_second_ms:+d}ms)")
        else:
            print(f"  {clock_origin}")
    else:
        # Normal datetime - convert to UTC for display
        if hasattr(clock_origin, 'tzinfo') and clock_origin.tzinfo is not None:
            from datetime import timezone
            clock_origin_utc = clock_origin.astimezone(timezone.utc)
        else:
            clock_origin_utc = clock_origin
        if sub_second_ms != 0:
            print(f"  {format_timestamp(clock_origin_utc)} ({sub_second_ms:+d}ms)")
        else:
            print(f"  {format_timestamp(clock_origin_utc)}")

    # Display signature (P/N) right after origin
    print(f"\nSignature:")
    print(f"  P: {sig_P}")
    print(f"  N: {sig_N}")

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
