"""
Video Analyzer - Clock Detection from Video

Extracts cell visibility data from video recordings of the Mondrian clock.
Uses OpenCV for video processing.

Grid Layout (6 columns x 10 rows):
    Columns: 0  1  2  3  4  5
    ┌────────────────┬──────┐
    │                │      │ Row 0
    │      20        │  12  │ Row 1
    │    (4×5)       │(2×6) │ Row 2
    │                │      │ Row 3
    │                │      │ Row 4
    ├────────┬───┬───┤      │ Row 5
    │        │ 1 ├───┴──────┤ Row 6
    │   15   ├───┤    4     │ Row 7
    │  (3×5) │ 2 │   (2×2)  │
    │        │───┼──────────┤ Row 8
    │        │      6       │ Row 9
    └────────┴──────────────┘

Cell positions (column_start, row_start, columns, rows):
    20: (0, 0, 4, 5)
    12: (4, 0, 2, 6)
    15: (0, 5, 3, 5)
     1: (3, 5, 1, 1)
     2: (3, 6, 1, 2)
     4: (4, 6, 2, 2)
     6: (3, 8, 3, 2)
"""

import cv2
import numpy as np
from collections import Counter


# Cell definitions: id -> (col_start, row_start, col_span, row_span)
CELL_LAYOUT = {
    20: (0, 0, 4, 5),
    12: (4, 0, 2, 6),
    15: (0, 5, 3, 5),
    1:  (3, 5, 1, 1),
    2:  (3, 6, 1, 2),
    4:  (4, 6, 2, 2),
    6:  (3, 8, 3, 2),
}

GRID_COLS = 6
GRID_ROWS = 10

# Standard output size for perspective-corrected clock
WARPED_WIDTH = 600
WARPED_HEIGHT = 1000


def extract_frames(video_path):
    """
    Extract all frames from video.

    Returns list of frames (BGR numpy arrays) and metadata dict with:
    - fps: frames per second
    - frame_count: total number of frames
    - duration: video duration in seconds
    - real_duration: estimated real-world duration (for timelapses)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # For iPhone timelapse: ~120 frames = 60 seconds real time
    # Timelapse records at ~2 FPS real-world, plays back at 30 FPS
    video_duration = frame_count / fps if fps > 0 else 0
    real_duration = frame_count / 2.0  # Assume 2 FPS real-world capture

    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "duration": video_duration,
        "real_duration": real_duration,
    }

    return frames, metadata


def order_corners(pts):
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.reshape(4, 2)

    # Sort by y-coordinate (top vs bottom)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # Sort top by x (left vs right)
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def detect_clock_quadrilateral(frame):
    """
    Detect the clock rectangle by finding edges and selecting best rectangle.

    Strategy:
    1. Detect edges using Canny
    2. Find all rectangular contours
    3. Select the one with aspect ratio closest to 6:10

    Returns 4 corner points ordered as: TL, TR, BR, BL
    or None if not detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to connect nearby lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find rectangle with best aspect ratio match to 6:10
    target_ratio = 6.0 / 10.0
    best_rect = None
    best_score = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            continue

        # Get bounding rectangle
        x, y, rw, rh = cv2.boundingRect(cnt)

        if rw < 200 or rh < 200:
            continue

        ratio = rw / rh
        ratio_error = abs(ratio - target_ratio)

        # Score: prefer correct aspect ratio and larger area
        score = ratio_error * 10 - area / 100000

        if score < best_score:
            best_score = score
            best_rect = (x, y, rw, rh)

    if best_rect is None:
        return None

    x, y, rw, rh = best_rect

    corners = np.array([
        [x, y],            # TL
        [x + rw, y],       # TR
        [x + rw, y + rh],  # BR
        [x, y + rh]        # BL
    ], dtype=np.float32)

    return corners


def warp_to_rectangle(frame, quad_pts):
    """
    Apply perspective transform to get a straight-on view of the clock.

    Returns the warped frame (WARPED_WIDTH x WARPED_HEIGHT).
    """
    # Destination points (straight rectangle)
    dst_pts = np.array([
        [0, 0],
        [WARPED_WIDTH - 1, 0],
        [WARPED_WIDTH - 1, WARPED_HEIGHT - 1],
        [0, WARPED_HEIGHT - 1]
    ], dtype=np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(quad_pts, dst_pts)

    # Apply transform
    warped = cv2.warpPerspective(frame, M, (WARPED_WIDTH, WARPED_HEIGHT))

    return warped


def find_empty_color_in_warped(warped_frame):
    """
    Find the "empty" (unfilled cell) color in a perspective-corrected frame.

    The empty color should be the most common bright color within cells.
    """
    # Sample from cell interiors (avoiding borders)
    cell_w = WARPED_WIDTH / GRID_COLS
    cell_h = WARPED_HEIGHT / GRID_ROWS

    samples = []
    for cell_id, (col, row, cspan, rspan) in CELL_LAYOUT.items():
        # Sample center of cell
        cx = int((col + cspan / 2) * cell_w)
        cy = int((row + rspan / 2) * cell_h)

        # Sample a small region
        margin = 5
        region = warped_frame[
            max(0, cy - margin):cy + margin,
            max(0, cx - margin):cx + margin
        ]
        if region.size > 0:
            samples.append(np.mean(region, axis=(0, 1)))

    if not samples:
        return np.array([200, 200, 200], dtype=np.uint8)

    # The empty color is likely the brightest (white-ish) among samples
    samples = np.array(samples)
    brightness = np.sum(samples, axis=1)
    brightest_idx = np.argmax(brightness)

    return samples[brightest_idx].astype(np.uint8)


def get_cell_sample_region_warped(cell_id):
    """
    Get the pixel region for sampling a cell in a warped (perspective-corrected) frame.

    Samples from a small central region to avoid black borders.
    Returns (x, y, width, height) in warped frame coordinates.
    """
    col_start, row_start, col_span, row_span = CELL_LAYOUT[cell_id]

    cell_w = WARPED_WIDTH / GRID_COLS
    cell_h = WARPED_HEIGHT / GRID_ROWS

    # Calculate cell center
    center_x = (col_start + col_span / 2) * cell_w
    center_y = (row_start + row_span / 2) * cell_h

    # Sample a small region around the center (20% of cell size)
    sample_w = max(10, int(col_span * cell_w * 0.2))
    sample_h = max(10, int(row_span * cell_h * 0.2))

    x = int(center_x - sample_w / 2)
    y = int(center_y - sample_h / 2)

    return (x, y, sample_w, sample_h)


def sample_cell_color_warped(warped_frame, cell_id):
    """
    Sample the average color of a cell in a warped frame.
    """
    x, y, w, h = get_cell_sample_region_warped(cell_id)

    # Clamp to frame bounds
    fh, fw = warped_frame.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))

    region = warped_frame[y:y+h, x:x+w]
    avg_color = np.mean(region, axis=(0, 1))

    return avg_color.astype(np.uint8)


def color_distance(c1, c2):
    """Euclidean distance between two BGR colors."""
    return np.sqrt(np.sum((c1.astype(np.float32) - c2.astype(np.float32)) ** 2))


def is_cell_filled(cell_color, empty_color, tolerance=50):
    """
    Determine if a cell is filled (visible) or empty.

    A cell is "filled" if its color differs significantly from the empty color.
    Filled cells can be ANY color (yellow, blue, red, black, etc.) -
    we just check that it's NOT the empty/background color.

    Args:
        cell_color: BGR color of the cell
        empty_color: BGR color of empty (unfilled) cells
        tolerance: color distance threshold (Euclidean distance in BGR space)
    """
    dist = color_distance(cell_color, empty_color)

    # A cell is filled if it differs enough from the empty color
    # Using Euclidean distance in BGR space
    # Typical threshold: 50-80 works well for distinguishing colors
    return dist > tolerance


def get_visible_cells_warped(warped_frame, empty_color, tolerance=80):
    """
    Detect which cells are visible (filled) in a warped frame.

    Args:
        warped_frame: perspective-corrected frame
        empty_color: BGR color of empty cells
        tolerance: color distance threshold (default 80 to handle shadows)

    Returns a set of cell IDs (1, 2, 4, 6, 12, 15, 20).
    """
    visible = set()

    for cell_id in CELL_LAYOUT.keys():
        cell_color = sample_cell_color_warped(warped_frame, cell_id)
        if is_cell_filled(cell_color, empty_color, tolerance):
            visible.add(cell_id)

    return visible


class VideoTooShortError(Exception):
    """Raised when video is shorter than 60 seconds."""
    pass


def analyze_video(video_path, tolerance=80, verbose=False):
    """
    Analyze a video and extract cell visibility for each second.

    Handles iPhone timelapse format: ~120 frames = 60 real-world seconds.
    Uses perspective correction to handle angled shots.

    Returns:
    - observations: list of 60 sets, each containing visible cell IDs
    - metadata: video metadata dict (includes 'extra_observations' for >60s videos)

    Raises:
    - VideoTooShortError: if video is shorter than 60 seconds
    """
    # Extract frames
    frames, metadata = extract_frames(video_path)

    if verbose:
        print(f"Loaded {len(frames)} frames, {metadata['fps']:.2f} FPS, "
              f"{metadata['duration']:.1f}s video, ~{metadata['real_duration']:.1f}s real")

    # Check video duration
    real_duration = metadata['real_duration']
    if real_duration < 59.5:  # Allow small tolerance for 60s videos
        raise VideoTooShortError(
            f"Video is too short ({real_duration:.1f}s). "
            f"Need at least 60 seconds for unique clock identification."
        )

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Detect clock quadrilateral from first frame
    quad_pts = detect_clock_quadrilateral(frames[0])

    if quad_pts is None:
        raise ValueError("Could not detect clock in video")

    if verbose:
        print(f"Detected clock corners: {quad_pts.astype(int).tolist()}")

    # First pass: warp all frames and collect cell colors
    warped_frames = []
    all_cell_colors = []  # List of (BGR) colors from all cells, all frames

    for i, frame in enumerate(frames):
        frame_quad = detect_clock_quadrilateral(frame)
        if frame_quad is None:
            frame_quad = quad_pts

        warped = warp_to_rectangle(frame, frame_quad)
        warped_frames.append(warped)

        # Collect colors from all cells
        for cell_id in CELL_LAYOUT.keys():
            color = sample_cell_color_warped(warped, cell_id)
            all_cell_colors.append(color)

    # Find empty color: most frequent color cluster across all samples
    # Quantize colors to reduce variations, then find mode
    all_colors_array = np.array(all_cell_colors)
    quantized = (all_colors_array // 20) * 20  # Quantize to 20-unit bins

    # Count color occurrences
    color_counts = {}
    for c in quantized:
        key = tuple(c)
        color_counts[key] = color_counts.get(key, 0) + 1

    # Most frequent quantized color - use bin center (+10) for more robust matching
    # Averaging exact colors can shift towards filled colors
    most_frequent = max(color_counts, key=color_counts.get)
    empty_color = np.array([v + 10 for v in most_frequent], dtype=np.uint8)

    if verbose:
        print(f"Detected empty color: RGB({empty_color[2]}, {empty_color[1]}, {empty_color[0]})")

    # Second pass: detect visible cells using the global empty color
    frame_observations = []
    for i, warped in enumerate(warped_frames):
        visible = get_visible_cells_warped(warped, empty_color, tolerance)
        frame_observations.append(visible)

        if verbose and i < 5:
            print(f"Frame {i}: {sorted(visible)}")

    # Match each frame to its clock second using sum-based detection
    # The sum of visible cell areas directly encodes the clock second (0-59)
    from clock_inverse import cells_to_second, find_combination_index, validate_observation

    # Build mapping: clock second -> (frame_idx, observation)
    # Use validate_observation for error correction on invalid detections
    second_to_frame = {}
    frame_to_second = []
    corrected_count = 0
    second_0_frames = []  # Track frames showing second 0 for timestamp sync

    for frame_idx, obs in enumerate(frame_observations):
        clock_second = cells_to_second(obs)
        frame_to_second.append(clock_second)

        # Track second 0 frames for timestamp calibration
        if clock_second == 0:
            second_0_frames.append(frame_idx)

        if clock_second not in second_to_frame:
            # Check if this is a valid combination
            idx = find_combination_index(clock_second, obs)
            if idx >= 0:
                # Valid - use as is
                second_to_frame[clock_second] = (frame_idx, obs)
            else:
                # Invalid - try error correction
                _, is_valid, corrected = validate_observation(obs)
                if corrected is not None:
                    corrected_second = cells_to_second(corrected)
                    if corrected_second not in second_to_frame:
                        second_to_frame[corrected_second] = (frame_idx, corrected)
                        corrected_count += 1
                        if verbose:
                            print(f"Frame {frame_idx}: corrected {sorted(obs)} -> {sorted(corrected)}")

    if verbose and corrected_count > 0:
        print(f"Applied {corrected_count} error corrections")

    # Calculate minute boundary offset using second 0 as sync point
    # Second 0 marks the exact minute boundary
    # This allows precise origin calculation regardless of when recording started
    minute_boundary_offset_ms = None
    if second_0_frames and 0 in second_to_frame:
        # Find first valid occurrence of second 0
        second_0_frame = second_0_frames[0]
        first_second = frame_to_second[0] if frame_to_second else 0

        # Frame F shows second 0 at time: video_start + F * 0.5 seconds
        # But if first_second > 0, this is second 0 of the NEXT minute (k+1)
        # We want the start of minute k, which is 60 seconds earlier
        offset_to_second_0 = second_0_frame * 500  # ms

        if first_second > 0:
            # Second 0 is from minute k+1, subtract 60 seconds
            minute_boundary_offset_ms = offset_to_second_0 - 60000
        else:
            # First frame is second 0, so this is minute k
            minute_boundary_offset_ms = offset_to_second_0

        if verbose:
            print(f"Second 0 at frame {second_0_frame}, "
                  f"minute k starts at {minute_boundary_offset_ms}ms from video start")

    # Find the starting clock second (first valid observation)
    start_second = frame_to_second[0] if frame_to_second else 0

    # Build observation sequence starting from start_second
    all_observations = []
    for i in range(len(second_to_frame)):  # Up to number of unique seconds found
        target_second = (start_second + i) % 60
        if target_second in second_to_frame:
            frame_idx, obs = second_to_frame[target_second]
            all_observations.append(obs)
        else:
            all_observations.append(set())  # Missing second

        if verbose and i < 5:
            print(f"Second {i}: {sorted(all_observations[-1])}")

    # Split into main observations (first 60) and extra (for verification)
    observations = all_observations[:60]
    extra_observations = all_observations[60:] if len(all_observations) > 60 else []

    # Pad to 60 if needed (shouldn't happen with duration check, but safety)
    while len(observations) < 60:
        observations.append(set())

    # Store extra observations and corrections in metadata
    metadata['extra_observations'] = extra_observations
    metadata['minute_boundary_offset_ms'] = minute_boundary_offset_ms
    metadata['corrected_frames'] = corrected_count
    metadata['start_second'] = start_second

    if extra_observations and verbose:
        print(f"Video has {len(extra_observations)} extra seconds for verification")

    return observations, metadata


def visualize_detection(video_path, output_path=None, tolerance=30):
    """
    Create a visualization showing detected cells on warped frames.
    Useful for debugging.
    """
    frames, metadata = extract_frames(video_path)

    if len(frames) == 0:
        return None

    # Detect clock
    quad_pts = detect_clock_quadrilateral(frames[0])
    if quad_pts is None:
        print("Could not detect clock quadrilateral")
        return None

    # Warp and find empty color
    warped_sample = warp_to_rectangle(frames[0], quad_pts)
    empty_color = find_empty_color_in_warped(warped_sample)

    # Cell colors for visualization
    cell_colors = {
        1: (30, 30, 30),
        2: (0, 255, 255),
        4: (30, 30, 30),
        6: (255, 0, 0),
        12: (255, 0, 0),
        15: (0, 0, 255),
        20: (0, 255, 255),
    }

    annotated = []
    for frame in frames:
        warped = warp_to_rectangle(frame, quad_pts)
        vis = warped.copy()

        visible = get_visible_cells_warped(warped, empty_color, tolerance)

        for cell_id in CELL_LAYOUT.keys():
            x, y, w, h = get_cell_sample_region_warped(cell_id)
            color = cell_colors.get(cell_id, (128, 128, 128))

            if cell_id in visible:
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 3)
                cv2.putText(vis, str(cell_id), (x+5, y+h-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (180, 180, 180), 1)

        # Add frame number
        cv2.putText(vis, f"Frame {len(annotated)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis, f"Visible: {sorted(visible)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        annotated.append(vis)

    if output_path:
        fps = metadata['fps'] if metadata['fps'] > 0 else 2
        h, w = annotated[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in annotated:
            out.write(frame)
        out.release()
        print(f"Wrote visualization to {output_path}")

    return annotated


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_analyzer.py <video_path> [--visualize]")
        sys.exit(1)

    video_path = sys.argv[1]
    visualize = "--visualize" in sys.argv

    observations, metadata = analyze_video(video_path, verbose=True)

    print(f"\nExtracted {len(observations)} seconds of observations")

    if visualize:
        output_path = video_path.rsplit('.', 1)[0] + '_debug.mp4'
        visualize_detection(video_path, output_path)
