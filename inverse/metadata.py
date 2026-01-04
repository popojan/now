"""
Video Metadata Extraction

Extracts creation/recording timestamp from video files.
Tries multiple methods: ffprobe, exiftool, file system.
"""

import subprocess
import os
import json
from datetime import datetime


def parse_datetime(date_str):
    """
    Parse various datetime formats from video metadata.
    Returns datetime object or None.

    If no timezone is present, assumes local system timezone.
    """
    if not date_str:
        return None

    # Clean up the string
    date_str = date_str.strip()

    # Common formats - try timezone-aware formats first
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",      # ISO with microseconds and Z
        "%Y-%m-%dT%H:%M:%SZ",          # ISO with Z
        "%Y-%m-%dT%H:%M:%S.%f%z",      # ISO with microseconds and timezone
        "%Y-%m-%dT%H:%M:%S%z",         # ISO with timezone
        "%Y:%m:%d %H:%M:%S%z",         # EXIF format with timezone
        "%Y-%m-%dT%H:%M:%S.%f",        # ISO with microseconds (no tz)
        "%Y-%m-%dT%H:%M:%S",           # ISO basic (no tz)
        "%Y:%m:%d %H:%M:%S",           # EXIF format (no tz)
        "%Y-%m-%d %H:%M:%S",           # Common format (no tz)
        "%Y/%m/%d %H:%M:%S",           # Alternate (no tz)
        "%d/%m/%Y %H:%M:%S",           # European (no tz)
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # If naive (no timezone), assume local system timezone
            if dt.tzinfo is None:
                import sys
                print(f"Warning: No timezone in metadata, assuming local timezone", file=sys.stderr)
                dt = dt.astimezone()  # Adds local timezone
            return dt
        except ValueError:
            continue

    # Try removing timezone suffix and parsing
    for suffix in ["+00:00", "-00:00", "+0000", "-0000"]:
        if date_str.endswith(suffix):
            return parse_datetime(date_str[:-len(suffix)])

    return None


def get_timestamp_ffprobe(video_path):
    """
    Extract timestamp using ffprobe.
    Returns datetime or None.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})

        # Try different tag names - prefer Apple tag (has timezone)
        for key in ["com.apple.quicktime.creationdate", "creation_time", "date"]:
            if key in tags:
                dt = parse_datetime(tags[key])
                if dt:
                    return dt

        return None

    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return None


def get_timestamp_exiftool(video_path):
    """
    Extract timestamp using exiftool.
    Returns datetime or None.
    """
    try:
        cmd = [
            "exiftool", "-j",
            "-CreationDate", "-CreateDate", "-DateTimeOriginal", "-MediaCreateDate",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        if not data:
            return None

        metadata = data[0]

        # Try different tag names - CreationDate (with timezone) preferred for iPhone videos
        for key in ["CreationDate", "DateTimeOriginal", "CreateDate", "MediaCreateDate"]:
            if key in metadata:
                dt = parse_datetime(metadata[key])
                if dt:
                    return dt

        return None

    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return None


def get_timestamp_filesystem(video_path):
    """
    Get file modification time as fallback.
    Returns datetime or None.
    """
    try:
        mtime = os.path.getmtime(video_path)
        return datetime.fromtimestamp(mtime)
    except OSError:
        return None


def get_video_timestamp(video_path, verbose=False):
    """
    Get the recording timestamp of a video file.

    Tries multiple methods in order:
    1. ffprobe (if available)
    2. exiftool (if available)
    3. File system modification time (fallback)

    Returns (datetime, source_method) or (None, None).
    """
    # Try ffprobe first
    if verbose:
        print("Trying ffprobe...")
    dt = get_timestamp_ffprobe(video_path)
    if dt:
        if verbose:
            print(f"Found timestamp via ffprobe: {dt}")
        return dt, "ffprobe"

    # Try exiftool
    if verbose:
        print("Trying exiftool...")
    dt = get_timestamp_exiftool(video_path)
    if dt:
        if verbose:
            print(f"Found timestamp via exiftool: {dt}")
        return dt, "exiftool"

    # Fall back to filesystem
    if verbose:
        print("Falling back to filesystem timestamp...")
    dt = get_timestamp_filesystem(video_path)
    if dt:
        if verbose:
            print(f"Using filesystem timestamp: {dt}")
        return dt, "filesystem"

    return None, None


def format_timestamp(dt, include_ms=False):
    """Format datetime for display."""
    if dt is None:
        return "Unknown"

    if include_ms:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metadata.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    dt, source = get_video_timestamp(video_path, verbose=True)

    print(f"\nVideo: {video_path}")
    print(f"Timestamp: {format_timestamp(dt)}")
    print(f"Source: {source}")
