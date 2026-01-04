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


def create_grid_mask(width=120, height=200, line_thickness=3):
    """
    Create a binary mask of the clock's grid lines.

    Grid lines are always black regardless of which cells are filled,
    making this mask ideal for template matching.

    Grid structure (normalized coordinates):
        x=0   x=0.5  x=0.67  x=1
        ┌────────────────┬──────┐ y=0
        │      20        │  12  │
        ├────────┬───┬───┤      │ y=0.5
        │   15   │ 1 ├───┴──────┤ y=0.6
        │        ├───┤    4     │
        │        │ 2 ├──────────┤ y=0.8
        │        │      6       │
        └────────┴──────────────┘ y=1.0

    Returns white background (255) with black lines (0).
    """
    mask = np.ones((height, width), dtype=np.uint8) * 255

    def vline(x_ratio, y_start, y_end):
        x = int(x_ratio * width)
        y1 = int(y_start * height)
        y2 = int(y_end * height)
        cv2.line(mask, (x, y1), (x, y2), 0, line_thickness)

    def hline(y_ratio, x_start, x_end):
        y = int(y_ratio * height)
        x1 = int(x_start * width)
        x2 = int(x_end * width)
        cv2.line(mask, (x1, y), (x2, y), 0, line_thickness)

    # Outer border
    cv2.rectangle(mask, (0, 0), (width-1, height-1), 0, line_thickness)

    # Vertical lines
    vline(4/6, 0, 0.6)      # Between 20|12
    vline(3/6, 0.5, 1.0)    # Between 15|1,2,6 (extends to bottom)
    vline(4/6, 0.6, 0.8)    # Between 1,2|4

    # Horizontal lines
    hline(0.5, 0, 4/6)      # Below 20
    hline(0.6, 3/6, 1.0)    # Below 1, separates 12|4
    hline(0.8, 3/6, 1.0)    # Below 2 and 4

    return mask


def validate_grid_structure(warped_frame, threshold=0.5):
    """
    Validate that a warped frame contains clock grid structure.

    After perspective correction, the grid lines should align with
    expected positions. This checks for dark pixels along grid lines.

    Args:
        warped_frame: perspective-corrected frame (WARPED_WIDTH x WARPED_HEIGHT)
        threshold: minimum ratio of dark pixels on grid lines (0-1)

    Returns:
        (is_valid, score) where score is the grid line match quality
    """
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Expected grid line positions (normalized)
    # Vertical lines
    vlines = [
        (4/6, 0, 0.6),      # Between 20|12
        (3/6, 0.5, 1.0),    # Between 15|1,2,6
        (4/6, 0.6, 0.8),    # Between 1,2|4
    ]
    # Horizontal lines
    hlines = [
        (0.5, 0, 4/6),      # Below 20
        (0.6, 3/6, 1.0),    # Below 1
        (0.8, 3/6, 1.0),    # Below 2 and 4
    ]

    dark_threshold = 100  # Pixels darker than this are "dark"
    line_width = max(3, int(w * 0.02))  # Sample width around line

    total_samples = 0
    dark_samples = 0

    # Check vertical lines
    for x_ratio, y_start, y_end in vlines:
        x = int(x_ratio * w)
        y1 = int(y_start * h)
        y2 = int(y_end * h)

        for y in range(y1, y2, 5):  # Sample every 5 pixels
            x_start = max(0, x - line_width // 2)
            x_end = min(w, x + line_width // 2)
            region = gray[y:y+1, x_start:x_end]
            if region.size > 0:
                total_samples += 1
                if np.min(region) < dark_threshold:
                    dark_samples += 1

    # Check horizontal lines
    for y_ratio, x_start_ratio, x_end_ratio in hlines:
        y = int(y_ratio * h)
        x1 = int(x_start_ratio * w)
        x2 = int(x_end_ratio * w)

        for x in range(x1, x2, 5):
            y_start = max(0, y - line_width // 2)
            y_end = min(h, y + line_width // 2)
            region = gray[y_start:y_end, x:x+1]
            if region.size > 0:
                total_samples += 1
                if np.min(region) < dark_threshold:
                    dark_samples += 1

    # Check outer border
    border_samples = 0
    border_dark = 0
    for i in range(0, w, 5):
        # Top edge
        if gray[0:line_width, i:i+1].size > 0:
            border_samples += 1
            if np.min(gray[0:line_width, i:i+1]) < dark_threshold:
                border_dark += 1
        # Bottom edge
        if gray[h-line_width:h, i:i+1].size > 0:
            border_samples += 1
            if np.min(gray[h-line_width:h, i:i+1]) < dark_threshold:
                border_dark += 1
    for i in range(0, h, 5):
        # Left edge
        if gray[i:i+1, 0:line_width].size > 0:
            border_samples += 1
            if np.min(gray[i:i+1, 0:line_width]) < dark_threshold:
                border_dark += 1
        # Right edge
        if gray[i:i+1, w-line_width:w].size > 0:
            border_samples += 1
            if np.min(gray[i:i+1, w-line_width:w]) < dark_threshold:
                border_dark += 1

    total_samples += border_samples
    dark_samples += border_dark

    if total_samples == 0:
        return False, 0

    score = dark_samples / total_samples
    return score >= threshold, score


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


def refine_corners_with_edges(frame, initial_corners, margin=30):
    """
    Refine corner positions by detecting the clock's black border lines.

    Uses Hough line detection within a region around the initial corners
    to find the actual border lines and their intersections.

    Args:
        frame: BGR image
        initial_corners: rough corner estimates (TL, TR, BR, BL)
        margin: pixels to expand search region

    Returns:
        Refined corners or original if refinement fails
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Get bounding box of initial corners with margin
    x_min = max(0, int(np.min(initial_corners[:, 0]) - margin))
    x_max = min(w, int(np.max(initial_corners[:, 0]) + margin))
    y_min = max(0, int(np.min(initial_corners[:, 1]) - margin))
    y_max = min(h, int(np.max(initial_corners[:, 1]) + margin))

    roi = gray[y_min:y_max, x_min:x_max]

    # Edge detection
    edges = cv2.Canny(roi, 50, 150)

    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) < 4:
        return initial_corners

    # Classify lines as horizontal or vertical
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if angle < 30 or angle > 150:  # Horizontal-ish
            horizontal_lines.append((x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min))
        elif 60 < angle < 120:  # Vertical-ish
            vertical_lines.append((x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return initial_corners

    # Find top, bottom, left, right border lines
    # Sort horizontal by average y (top to bottom)
    horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    top_line = horizontal_lines[0]
    bottom_line = horizontal_lines[-1]

    # Sort vertical by average x (left to right)
    vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    left_line = vertical_lines[0]
    right_line = vertical_lines[-1]

    def line_intersection(l1, l2):
        """Find intersection of two lines given as (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    # Find corner intersections
    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    br = line_intersection(bottom_line, right_line)
    bl = line_intersection(bottom_line, left_line)

    if None in [tl, tr, br, bl]:
        return initial_corners

    refined = np.array([tl, tr, br, bl], dtype=np.float32)

    # Sanity check: refined should be close to initial
    distances = np.linalg.norm(refined - initial_corners, axis=1)
    if np.max(distances) > margin * 2:
        return initial_corners

    return refined


def detect_clock_candidates(frame, max_candidates=5):
    """
    Detect multiple candidate clock rectangles.

    Strategy: Look for bright rectangular regions (clock on light background)
    with dark internal borders (the black grid lines between cells).

    Returns list of (corners, score) tuples, sorted by score (best first).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    candidates = []
    target_ratio = 6.0 / 10.0

    # Method 1: Find bright regions (clock on light background like a webpage)
    # Threshold to find bright areas
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    contours_bright, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_bright:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw < 80 or rh < 80:
            continue

        ratio = rw / rh
        ratio_error = abs(ratio - target_ratio)
        if ratio_error > 0.4:
            continue

        # Check for dark lines inside (grid borders)
        roi = gray[y:y+rh, x:x+rw]
        dark_pixels = np.sum(roi < 80) / roi.size
        # Clock should have some dark pixels (borders) but not too many
        if dark_pixels < 0.02 or dark_pixels > 0.5:
            continue

        corners = np.array([
            [x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]
        ], dtype=np.float32)

        # Score: prefer correct aspect ratio, larger area, and moderate dark content
        score = ratio_error * 10 - area / 100000 - dark_pixels * 5
        candidates.append((corners, score, "bright"))

    # Method 2: Edge-based detection (original method as fallback)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_edge:
        area = cv2.contourArea(cnt)
        if area < 20000:
            continue

        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw < 100 or rh < 100:
            continue

        ratio = rw / rh
        ratio_error = abs(ratio - target_ratio)
        if ratio_error > 0.3:
            continue

        corners = np.array([
            [x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]
        ], dtype=np.float32)

        score = ratio_error * 10 - area / 100000
        candidates.append((corners, score, "edge"))

    # Sort by score (lower is better) and deduplicate similar regions
    candidates.sort(key=lambda x: x[1])

    # Remove duplicates (overlapping regions)
    unique = []
    for corners, score, method in candidates:
        is_dup = False
        for uc, us, um in unique:
            # Check if centers are close
            c1 = np.mean(corners, axis=0)
            c2 = np.mean(uc, axis=0)
            if np.linalg.norm(c1 - c2) < 50:
                is_dup = True
                break
        if not is_dup:
            unique.append((corners, score, method))

    return [(c, s) for c, s, m in unique[:max_candidates]]


def detect_clock_simple(frame):
    """
    Simple clock detection - edge-based contour detection.
    Works well for axis-aligned, quality videos.

    Returns 4 corner points ordered as: TL, TR, BR, BL
    or None if not detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw < 200 or rh < 200:
            continue

        ratio = rw / rh
        ratio_error = abs(ratio - target_ratio)
        score = ratio_error * 10 - area / 100000

        if score < best_score:
            best_score = score
            best_rect = (x, y, rw, rh)

    if best_rect is None:
        return None

    x, y, rw, rh = best_rect
    corners = np.array([
        [x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]
    ], dtype=np.float32)

    return corners


def detect_clock_quadrilateral(frame):
    """
    Detect the clock rectangle by finding edges and selecting best rectangle.

    Returns 4 corner points ordered as: TL, TR, BR, BL
    or None if not detected.
    """
    candidates = detect_clock_candidates(frame, max_candidates=1)
    if candidates:
        return candidates[0][0]
    return None


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


def shrink_quad(quad_pts, margin_ratio=0.02):
    """Shrink a quadrilateral by a margin ratio to avoid including background."""
    center = np.mean(quad_pts, axis=0)
    shrunk = []
    for pt in quad_pts:
        direction = center - pt
        new_pt = pt + direction * margin_ratio
        shrunk.append(new_pt)
    return np.array(shrunk, dtype=np.float32)


def detect_empty_color(cell_colors):
    """
    Detect the empty (background) color from a list of cell color samples.

    Returns the most frequent NEUTRAL (grayish/white) color, or brightest
    color if no neutral found.
    """
    all_colors_array = np.array(cell_colors)
    quantized = (all_colors_array // 20) * 20

    color_counts = {}
    for c in quantized:
        key = tuple(c)
        color_counts[key] = color_counts.get(key, 0) + 1

    # Find neutral colors (R≈G≈B) and bright colors
    neutral_candidates = []
    bright_candidates = []

    for key, count in color_counts.items():
        b, g, r = int(key[0]), int(key[1]), int(key[2])
        max_diff = max(abs(r-g), abs(g-b), abs(r-b))
        brightness = (r + g + b) / 3

        if max_diff < 60 and brightness > 150:  # Neutral and bright
            neutral_candidates.append((key, count, brightness))
        elif brightness > 180:  # Just bright (might be off-white)
            bright_candidates.append((key, count, brightness))

    if neutral_candidates:
        neutral_candidates.sort(key=lambda x: (-x[1], -x[2]))
        most_frequent = neutral_candidates[0][0]
    elif bright_candidates:
        bright_candidates.sort(key=lambda x: (-x[2], -x[1]))  # Prefer brightest
        most_frequent = bright_candidates[0][0]
    else:
        # Fallback: most frequent overall
        all_sorted = sorted(color_counts.items(), key=lambda x: -x[1])
        most_frequent = all_sorted[0][0]

    return np.array([v + 10 for v in most_frequent], dtype=np.uint8)


def get_empty_color_candidates(cell_colors, max_candidates=3):
    """Get multiple empty color candidates for validation testing."""
    all_colors_array = np.array(cell_colors)
    quantized = (all_colors_array // 20) * 20

    color_counts = {}
    for c in quantized:
        key = tuple(c)
        color_counts[key] = color_counts.get(key, 0) + 1

    candidates = []
    for key, count in color_counts.items():
        b, g, r = int(key[0]), int(key[1]), int(key[2])
        brightness = (r + g + b) / 3
        max_diff = max(abs(r-g), abs(g-b), abs(r-b))
        # Score: prefer bright, neutral colors
        neutrality = 100 - max_diff
        score = brightness + neutrality * 0.5 + count * 0.1
        candidates.append((key, score))

    candidates.sort(key=lambda x: -x[1])
    return [np.array([v + 10 for v in c[0]], dtype=np.uint8)
            for c in candidates[:max_candidates]]


def validate_clock_candidate(frames, quad_pts, tolerance=80, sample_frames=20):
    """
    Validate a candidate clock region by checking if it produces valid clock readings
    and monotonically increasing seconds.

    Tries the original region and slightly shrunk versions to find best fit.
    Returns (score, empty_color, best_quad) where score is validation quality.
    Higher score = better candidate.
    """
    from clock_inverse import cells_to_second, find_combination_index, ORDERING

    # Sample frames evenly across the video (more samples for better monotonicity check)
    step = max(1, len(frames) // sample_frames)
    sample_indices = list(range(0, len(frames), step))[:sample_frames]

    best_score = -1
    best_empty_color = None
    best_quad = quad_pts

    # Try original and shrunk versions
    for margin in [0.0, 0.02, 0.04, 0.06]:
        test_quad = shrink_quad(quad_pts, margin) if margin > 0 else quad_pts

        # Warp sample frames
        warped_samples = []
        for i in sample_indices:
            warped = warp_to_rectangle(frames[i], test_quad)
            warped_samples.append(warped)

        # Detect empty color from samples
        all_cell_colors = []
        for warped in warped_samples:
            for cell_id in CELL_LAYOUT.keys():
                color = sample_cell_color_warped(warped, cell_id)
                all_cell_colors.append(color)

        # Try multiple empty color candidates
        empty_candidates = get_empty_color_candidates(all_cell_colors, max_candidates=3)

        for empty_color in empty_candidates:
            valid_count = 0
            seen_seconds = set()
            seconds_sequence = []

            for warped in warped_samples:
                visible = get_visible_cells_warped(warped, empty_color, tolerance)
                clock_second = cells_to_second(visible)
                idx = find_combination_index(clock_second, visible)

                seconds_sequence.append(clock_second)
                if idx >= 0:  # Valid combination
                    valid_count += 1
                    seen_seconds.add(clock_second)

            # Count monotonicity (seconds should generally increase)
            mono_count = 0
            for i in range(1, len(seconds_sequence)):
                prev, curr = seconds_sequence[i-1], seconds_sequence[i]
                # Allow same (multiple frames per second) or +1 (next second)
                # Also handle wrap-around at 60
                if curr == prev or curr == (prev + 1) % 60:
                    mono_count += 1

            # Score: valid readings + monotonicity bonus + diversity
            # Monotonicity is important - a real clock should advance in order
            diversity_bonus = len(seen_seconds) * 0.3
            mono_ratio = mono_count / max(1, len(seconds_sequence) - 1)
            mono_bonus = mono_ratio * 5  # Strong weight for monotonicity

            score = valid_count + diversity_bonus + mono_bonus

            if score > best_score:
                best_score = score
                best_empty_color = empty_color
                best_quad = test_quad

    return best_score, best_empty_color, best_quad


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

    # Note: We don't check duration here - sum-based detection is FPS-independent.
    # The actual check happens after detection: do we have observations for all 60 seconds?

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Detect multiple clock candidates and validate each
    candidates = detect_clock_candidates(frames[0], max_candidates=5)

    if not candidates:
        raise ValueError("Could not detect clock in video")

    # Validate candidates to find the best one
    best_quad_pts = None
    best_score = -1
    best_empty_color = None

    if verbose:
        print(f"Found {len(candidates)} candidate regions, validating...")

    for i, (quad_pts, _) in enumerate(candidates):
        score, empty_color, adjusted_quad = validate_clock_candidate(frames, quad_pts, tolerance)
        if verbose:
            print(f"  Candidate {i+1}: score={score:.1f}, corners={quad_pts.astype(int).tolist()}")
        if score > best_score:
            best_score = score
            best_quad_pts = adjusted_quad  # Use the adjusted quad from validation
            best_empty_color = empty_color

    if best_quad_pts is None:
        raise ValueError("Could not detect clock in video")

    quad_pts = best_quad_pts

    if verbose:
        print(f"Selected candidate with score {best_score:.1f}")
        print(f"Detected clock corners: {quad_pts.astype(int).tolist()}")

    from clock_inverse import cells_to_second, find_combination_index, validate_observation

    # Store simple-detected corners for simple detection fallback
    original_quad = detect_clock_simple(frames[0])
    if original_quad is None:
        original_quad = candidates[0][0] if candidates else quad_pts

    def run_simple_detection(frames, tolerance, verbose=False):
        """
        Simple detection approach: per-frame contour detection with fallback.
        This works well for quality, axis-aligned videos.
        Returns (second_to_frame, unique_count, empty_color, corner_history).
        """
        # Re-detect corners on each frame (like old version)
        corner_history = []
        warped_frames = []
        all_cell_colors = []

        for i, frame in enumerate(frames):
            frame_quad = detect_clock_simple(frame)
            if frame_quad is None:
                frame_quad = original_quad
            corner_history.append(frame_quad.copy())

            warped = warp_to_rectangle(frame, frame_quad)
            warped_frames.append(warped)

            # Collect colors from all cells
            for cell_id in CELL_LAYOUT.keys():
                color = sample_cell_color_warped(warped, cell_id)
                all_cell_colors.append(color)

        # Use quantized color detection across all frames (like old version)
        all_colors_array = np.array(all_cell_colors)
        quantized = (all_colors_array // 20) * 20
        color_counts = {}
        for c in quantized:
            key = tuple(c)
            color_counts[key] = color_counts.get(key, 0) + 1
        most_frequent = max(color_counts, key=color_counts.get)
        empty_color = np.array([v + 10 for v in most_frequent], dtype=np.uint8)

        # Build second_to_frame with error correction
        second_to_frame = {}
        for frame_idx, warped in enumerate(warped_frames):
            visible = get_visible_cells_warped(warped, empty_color, tolerance)
            clock_second = cells_to_second(visible)

            if clock_second not in second_to_frame:
                idx = find_combination_index(clock_second, visible)
                if idx >= 0:
                    second_to_frame[clock_second] = (frame_idx, visible)
                else:
                    _, _, corrected = validate_observation(visible)
                    if corrected is not None:
                        corrected_second = cells_to_second(corrected)
                        if corrected_second not in second_to_frame:
                            second_to_frame[corrected_second] = (frame_idx, corrected)

        return second_to_frame, len(second_to_frame), empty_color, corner_history

    def run_hough_detection(frames, quad_pts, tolerance, verbose=False):
        """
        Hough-based detection: corner refinement + per-frame tracking.
        This works well for tilted videos with camera motion.
        Returns (second_to_frame, unique_count, empty_color, corner_history).
        """
        # Refine corners using Hough line detection
        refined = refine_corners_with_edges(frames[0], quad_pts, margin=50)
        corner_history = [refined.copy()]
        prev = refined

        # Track corners across all frames
        for i in range(1, len(frames)):
            refined = refine_corners_with_edges(frames[i], prev, margin=30)
            corner_history.append(refined.copy())
            prev = refined

        # Warp all frames
        warped_frames = []
        all_cell_colors = []
        for i, frame in enumerate(frames):
            warped = warp_to_rectangle(frame, corner_history[i])
            warped_frames.append(warped)
            for cell_id in CELL_LAYOUT.keys():
                all_cell_colors.append(sample_cell_color_warped(warped, cell_id))

        # Detect empty color
        empty_color = detect_empty_color(all_cell_colors)

        # Build second_to_frame with error correction
        second_to_frame = {}
        for frame_idx, warped in enumerate(warped_frames):
            visible = get_visible_cells_warped(warped, empty_color, tolerance)
            clock_second = cells_to_second(visible)

            if clock_second not in second_to_frame:
                idx = find_combination_index(clock_second, visible)
                if idx >= 0:
                    second_to_frame[clock_second] = (frame_idx, visible)
                else:
                    _, _, corrected = validate_observation(visible)
                    if corrected is not None:
                        corrected_second = cells_to_second(corrected)
                        if corrected_second not in second_to_frame:
                            second_to_frame[corrected_second] = (frame_idx, corrected)

        return second_to_frame, len(second_to_frame), empty_color, corner_history

    # Try simple detection first (works for most videos)
    simple_result = run_simple_detection(frames, tolerance, verbose)
    simple_unique = simple_result[1]

    if verbose:
        print(f"Simple approach: {simple_unique}/60 unique seconds")

    # If simple approach gets >= 54 seconds (90%), use it
    if simple_unique >= 54:
        second_to_frame, _, empty_color, corner_history = simple_result
        if verbose:
            print("Using simple approach (good results)")
    else:
        # Try Hough-based approach for difficult videos
        hough_result = run_hough_detection(frames, quad_pts, tolerance, verbose)
        hough_unique = hough_result[1]

        if verbose:
            print(f"Hough approach: {hough_unique}/60 unique seconds")

        if hough_unique > simple_unique:
            second_to_frame, _, empty_color, corner_history = hough_result
            if verbose:
                print("Using Hough approach (better results)")
        else:
            second_to_frame, _, empty_color, corner_history = simple_result
            if verbose:
                print("Using simple approach (Hough not better)")

    # Check if we have enough unique seconds (need ~40 info-carrying seconds)
    unique_seconds = len(second_to_frame)
    if unique_seconds < 40:
        raise VideoTooShortError(
            f"Only detected {unique_seconds}/60 unique seconds. "
            f"Need at least 40 for clock identification. "
            f"Video may be too short or detection failed."
        )

    # Warp frames for verbose output and timestamp sync
    warped_frames = []
    for i, frame in enumerate(frames):
        warped = warp_to_rectangle(frame, corner_history[i])
        warped_frames.append(warped)

    if verbose:
        print(f"Detected empty color: RGB({empty_color[2]}, {empty_color[1]}, {empty_color[0]})")
        for i in range(min(5, len(warped_frames))):
            visible = get_visible_cells_warped(warped_frames[i], empty_color, tolerance)
            print(f"Frame {i}: {sorted(visible)}")

    # Build frame_to_second and second_0_frames for timestamp sync
    frame_to_second = []
    second_0_frames = []
    corrected_count = 0  # Already counted in detection functions

    for frame_idx, warped in enumerate(warped_frames):
        visible = get_visible_cells_warped(warped, empty_color, tolerance)
        clock_second = cells_to_second(visible)
        frame_to_second.append(clock_second)
        if clock_second == 0:
            second_0_frames.append(frame_idx)

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

    # Count real (non-empty) observations before padding
    real_observations = sum(1 for obs in observations if obs)

    # Pad to 60 if needed (shouldn't happen with duration check, but safety)
    while len(observations) < 60:
        observations.append(set())

    # Store extra observations and corrections in metadata
    metadata['extra_observations'] = extra_observations
    metadata['minute_boundary_offset_ms'] = minute_boundary_offset_ms
    metadata['corrected_frames'] = corrected_count
    metadata['start_second'] = start_second
    metadata['real_observations'] = real_observations  # Non-empty observation count

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
