"""
Clock Origin Detector - Core Algorithm

This module contains the pure algorithm for:
1. The perm() function (forward direction)
2. The inverse_perm() function (reverse direction)

Written for portability - uses only basic data structures.
"""

# The ordering dictionary defines which cell combinations are valid for each second.
# Each second maps to a list of valid combinations.
# The perm() function selects which combination to use based on the minute identifier k.
ORDERING = {
    0: [[1, 2, 4, 6, 12, 15, 20], []],
    1: [[1]],
    2: [[2]],
    3: [[1, 2]],
    4: [[4]],
    5: [[1, 4]],
    6: [[6], [2, 4]],
    7: [[1, 6], [1, 2, 4]],
    8: [[2, 6]],
    9: [[1, 2, 6]],
    10: [[4, 6]],
    11: [[1, 4, 6]],
    12: [[12], [2, 4, 6]],
    13: [[1, 12], [1, 2, 4, 6]],
    14: [[2, 12]],
    15: [[15], [1, 2, 12]],
    16: [[1, 15], [4, 12]],
    17: [[2, 15], [1, 4, 12]],
    18: [[1, 2, 15], [6, 12], [2, 4, 12]],
    19: [[4, 15], [1, 6, 12], [1, 2, 4, 12]],
    20: [[20], [1, 4, 15], [2, 6, 12]],
    21: [[1, 20], [6, 15], [2, 4, 15], [1, 2, 6, 12]],
    22: [[2, 20], [1, 6, 15], [1, 2, 4, 15], [4, 6, 12]],
    23: [[1, 2, 20], [2, 6, 15], [1, 4, 6, 12]],
    24: [[4, 20], [1, 2, 6, 15], [2, 4, 6, 12]],
    25: [[1, 4, 20], [4, 6, 15], [1, 2, 4, 6, 12]],
    26: [[6, 20], [2, 4, 20], [1, 4, 6, 15]],
    27: [[1, 6, 20], [1, 2, 4, 20], [12, 15], [2, 4, 6, 15]],
    28: [[1, 12, 15], [2, 6, 20], [1, 2, 4, 6, 15]],
    29: [[1, 2, 6, 20], [2, 12, 15]],
    30: [[1, 2, 12, 15], [4, 6, 20]],
    31: [[1, 4, 6, 20], [4, 12, 15]],
    32: [[12, 20], [1, 4, 12, 15], [2, 4, 6, 20]],
    33: [[1, 12, 20], [1, 2, 4, 6, 20], [6, 12, 15], [2, 4, 12, 15]],
    34: [[2, 12, 20], [1, 6, 12, 15], [1, 2, 4, 12, 15]],
    35: [[15, 20], [1, 2, 12, 20], [2, 6, 12, 15]],
    36: [[1, 15, 20], [4, 12, 20], [1, 2, 6, 12, 15]],
    37: [[2, 15, 20], [1, 4, 12, 20], [4, 6, 12, 15]],
    38: [[1, 2, 15, 20], [6, 12, 20], [2, 4, 12, 20], [1, 4, 6, 12, 15]],
    39: [[4, 15, 20], [1, 6, 12, 20], [1, 2, 4, 12, 20], [2, 4, 6, 12, 15]],
    40: [[1, 4, 15, 20], [2, 6, 12, 20], [1, 2, 4, 6, 12, 15]],
    41: [[6, 15, 20], [2, 4, 15, 20], [1, 2, 6, 12, 20]],
    42: [[1, 6, 15, 20], [1, 2, 4, 15, 20], [4, 6, 12, 20]],
    43: [[2, 6, 15, 20], [1, 4, 6, 12, 20]],
    44: [[1, 2, 6, 15, 20], [2, 4, 6, 12, 20]],
    45: [[4, 6, 15, 20], [1, 2, 4, 6, 12, 20]],
    46: [[1, 4, 6, 15, 20]],
    47: [[12, 15, 20], [2, 4, 6, 15, 20]],
    48: [[1, 12, 15, 20], [1, 2, 4, 6, 15, 20]],
    49: [[2, 12, 15, 20]],
    50: [[1, 2, 12, 15, 20]],
    51: [[4, 12, 15, 20]],
    52: [[1, 4, 12, 15, 20]],
    53: [[6, 12, 15, 20], [2, 4, 12, 15, 20]],
    54: [[1, 6, 12, 15, 20], [1, 2, 4, 12, 15, 20]],
    55: [[2, 6, 12, 15, 20]],
    56: [[1, 2, 6, 12, 15, 20]],
    57: [[4, 6, 12, 15, 20]],
    58: [[1, 4, 6, 12, 15, 20]],
    59: [[2, 4, 6, 12, 15, 20]],
}

# Cell values (areas)
CELL_VALUES = [1, 2, 4, 6, 12, 15, 20]

# Cell value to bit index mapping: cell 1 -> bit 0, cell 2 -> bit 1, etc.
CELL_TO_BIT = {1: 0, 2: 1, 4: 2, 6: 3, 12: 4, 15: 5, 20: 6}
BIT_TO_CELL = {0: 1, 1: 2, 2: 4, 3: 6, 4: 12, 5: 15, 6: 20}

# Period of the clock: 2^30 * 3^16 minutes
PERIOD = (1 << 30) * (3 ** 16)  # 46221064723759104

# Constants for bit manipulation
POW_2_30 = 1 << 30  # 1073741824


def cells_to_second(visible_cells):
    """
    Determine clock second (0-59) from visible cell areas.

    The clock is designed so that cells shown at second S always sum to S.
    This allows instant identification of the clock second from any frame.
    """
    total = sum(visible_cells)
    return total % 60  # Handle second 0 which can show sum=0 or sum=60


# ============ Mask/Bitmask Utilities ============

def cells_to_mask(cells):
    """Convert list of cell values to 7-bit mask."""
    mask = 0
    for cell in cells:
        if cell in CELL_TO_BIT:
            mask |= (1 << CELL_TO_BIT[cell])
    return mask


def mask_to_cells(mask):
    """Convert 7-bit mask to list of cell values."""
    cells = []
    for bit, cell in BIT_TO_CELL.items():
        if mask & (1 << bit):
            cells.append(cell)
    return cells


def mask_to_sum(mask):
    """Calculate sum of cells represented by mask."""
    return sum(mask_to_cells(mask))


def hamming_distance(a, b):
    """Count differing bits between two masks."""
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


# Precompute valid masks for each second (for fast lookup)
MASKS_FOR_SECOND = {}
for sec in range(60):
    MASKS_FOR_SECOND[sec] = [cells_to_mask(combo) for combo in ORDERING[sec]]


def closest_valid_mask(received_mask, target_second, return_ambiguity=False):
    """
    Find the valid mask for target_second with minimum Hamming distance to received_mask.

    Args:
        received_mask: The observed cell bitmask
        target_second: Expected second (0-59)
        return_ambiguity: If True, also return number of masks at minimum distance

    Returns:
        If return_ambiguity=False: (best_mask, distance)
        If return_ambiguity=True: (best_mask, distance, num_at_min_dist)

    The ambiguity count helps detect when correction is unreliable:
    - num_at_min_dist=1: unambiguous correction
    - num_at_min_dist>1: multiple equally-close options (correction may be wrong)
    """
    target_second = target_second % 60
    valid_masks = MASKS_FOR_SECOND[target_second]

    # First pass: find minimum distance
    distances = [(mask, hamming_distance(received_mask, mask)) for mask in valid_masks]
    min_dist = min(d for _, d in distances)

    # Find all masks at minimum distance
    masks_at_min = [m for m, d in distances if d == min_dist]
    best_mask = masks_at_min[0]

    if return_ambiguity:
        return best_mask, min_dist, len(masks_at_min)
    return best_mask, min_dist


# ============ Error Correction ============

class FrameCorrection:
    """Stores correction info for a single frame."""
    def __init__(self, original, received_sum, expected_sum, corrected, distance, is_anchor, ambiguity=1):
        self.original = original
        self.received_sum = received_sum
        self.expected_sum = expected_sum
        self.corrected = corrected
        self.distance = distance
        self.is_anchor = is_anchor
        self.ambiguity = ambiguity  # Number of masks at same minimum distance (1=unambiguous)


def correct_errors(masks):
    """
    Apply flip-aware error correction to a sequence of masks.

    Uses virtual_start voting to find true alignment, then corrects
    non-anchor frames using minimum Hamming distance.

    Args:
        masks: list of 7-bit masks (one per frame)

    Returns:
        (corrected_masks, corrections, anchor_count) where:
        - corrected_masks: list of corrected masks
        - corrections: list of FrameCorrection objects
        - anchor_count: number of frames that matched majority virtual_start
        Returns (None, None, 0) if correction fails (no clear winner)
    """
    num_frames = len(masks)
    if num_frames < 1:
        return None, None, 0

    # Step 1: Compute virtual_start for each frame
    # virtual_start = (sum - frame_index) mod 60
    virtual_starts = []
    vstart_counts = [0] * 60

    for i, mask in enumerate(masks):
        s = mask_to_sum(mask) % 60
        vs = (s - i) % 60
        virtual_starts.append(vs)
        vstart_counts[vs] += 1

    # Step 2: Find plurality winner (most common virtual_start)
    majority_vs = 0
    majority_count = vstart_counts[0]
    second_count = 0

    for vs in range(1, 60):
        if vstart_counts[vs] > majority_count:
            second_count = majority_count
            majority_count = vstart_counts[vs]
            majority_vs = vs
        elif vstart_counts[vs] > second_count:
            second_count = vstart_counts[vs]

    # Require clear winner: majority > second place
    if majority_count <= second_count:
        return None, None, 0

    # Step 3: Mark anchors and correct errors
    corrected_masks = []
    corrections = []
    num_anchors = 0

    for i, mask in enumerate(masks):
        received_sum = mask_to_sum(mask) % 60
        expected_sum = (majority_vs + i) % 60

        if virtual_starts[i] == majority_vs:
            # This frame is an anchor - sum matches expected
            corrected_masks.append(mask)
            corrections.append(FrameCorrection(
                original=mask,
                received_sum=received_sum,
                expected_sum=expected_sum,
                corrected=mask,
                distance=0,
                is_anchor=True
            ))
            num_anchors += 1
        else:
            # This frame needs correction
            corrected, dist = closest_valid_mask(mask, expected_sum)
            corrected_masks.append(corrected)
            corrections.append(FrameCorrection(
                original=mask,
                received_sum=received_sum,
                expected_sum=expected_sum,
                corrected=corrected,
                distance=dist,
                is_anchor=False
            ))

    return corrected_masks, corrections, num_anchors


# ============ Signature Detection (P and N) ============

def detect_signature(k_values):
    """
    Detect signature P from a sequence of k values from consecutive minutes.

    Uses majority voting on k-deltas: for consecutive minutes, k[i+1] - k[i] = P.

    Args:
        k_values: list of k values from consecutive aligned minutes

    Returns:
        (P, N, confidence) where:
        - P: detected period (1 if no signature)
        - N: detected signature value
        - confidence: fraction of deltas that agreed with majority
        Returns (1, 0, 1.0) for single minute (no detection possible)
    """
    if len(k_values) < 2:
        return 1, 0, 1.0

    # Compute deltas between consecutive minutes
    deltas = []
    for i in range(len(k_values) - 1):
        delta = k_values[i + 1] - k_values[i]
        # Handle wrap-around (should be rare)
        if delta < 0:
            delta += PERIOD
        deltas.append(delta)

    # Count occurrences of each delta
    delta_counts = {}
    for d in deltas:
        delta_counts[d] = delta_counts.get(d, 0) + 1

    # Find majority delta
    majority_delta = max(delta_counts, key=delta_counts.get)
    majority_count = delta_counts[majority_delta]

    # Confidence is fraction of deltas matching majority
    confidence = majority_count / len(deltas)

    P = majority_delta
    # N = k[0] mod P (for first minute)
    N = k_values[0] % P if P > 0 else 0

    return P, N, confidence


def inverse_with_correction(observed_cells_list, apply_correction=True, sig_period=None):
    """
    Inverse with optional error correction and signature detection.

    Args:
        observed_cells_list: list of observed cell sets (can span multiple minutes)
        apply_correction: whether to apply error correction
        sig_period: signature period P (None for autodetection)

    Returns dict with:
        - start_second: clock second of first frame
        - k: minute identifier (first minute)
        - k_values: list of k values for each minute (if multiple)
        - P: detected signature period
        - N: detected signature value
        - corrections: list of FrameCorrection (if correction applied)
        - anchor_count: number of anchor frames
        - match_count: seconds matched in k recovery
    """
    result = {
        'start_second': None,
        'k': None,
        'k_values': [],
        'P': 1,
        'N': 0,
        'corrections': None,
        'anchor_count': 0,
        'match_count': 0,
        'signature_confidence': 1.0
    }

    # Convert cells to masks
    masks = [cells_to_mask(cells) for cells in observed_cells_list]
    num_frames = len(masks)

    # IMPORTANT: Find k from RAW observations first with autodetection
    # Error correction can "correct" to a different valid k, which would be wrong
    start_second, k, match_count, detected_P = find_k_with_autodetect(
        observed_cells_list, sig_period=sig_period
    )

    # Apply error correction AFTER finding k (for stats/display only, not k search)
    if apply_correction and num_frames >= 30:
        corrected_masks, corrections, anchor_count = correct_errors(masks)
        if corrected_masks is not None:
            result['corrections'] = corrections
            result['anchor_count'] = anchor_count

    result['start_second'] = start_second
    result['k'] = k
    result['match_count'] = match_count
    result['P'] = detected_P
    result['N'] = k % detected_P if k is not None and detected_P > 0 else 0

    if k is None:
        return result

    result['k_values'] = [k]

    # For multi-minute recordings, try to extract k for each aligned minute
    # and detect signature from k-deltas
    if start_second is not None:
        align_offset = (60 - start_second) % 60 if start_second > 0 else 0
        aligned_frames = num_frames - align_offset

        # Need at least 2 full aligned minutes for signature detection
        if aligned_frames >= 120:
            k_values = []
            for minute_idx in range(aligned_frames // 60):
                frame_start = align_offset + minute_idx * 60
                frame_end = frame_start + 60
                minute_cells = observed_cells_list[frame_start:frame_end]

                _, minute_k, _ = find_k_from_observations(minute_cells)
                if minute_k is not None:
                    k_values.append(minute_k)

            if len(k_values) >= 2:
                result['k_values'] = k_values
                P, N, confidence = detect_signature(k_values)
                result['P'] = P
                result['N'] = N
                result['signature_confidence'] = confidence

    return result


def find_k_multi_minute(all_minutes_data, verbose=False, sig_period=1):
    """
    Find k using multiple minutes of data for increased robustness.

    When multiple minutes of observations are available, we can:
    1. Compute k for each minute independently
    2. Check consistency across minutes (detect outliers/detection errors)
    3. Use k-deltas to detect and verify signature (P value)

    Args:
        all_minutes_data: dict mapping minute_idx -> (observations_list, start_second)
        verbose: print debug info
        sig_period: signature period P for k encoding (default 1)

    Returns:
        dict with k, k_values, P, N, confidence, and correction info
    """
    result = {
        'k': None,
        'k_values': [],
        'P': 1,
        'N': 0,
        'confidence': 0.0,
        'corrected_minute': None,
        'original_k': None
    }

    if not all_minutes_data:
        return result

    # Compute k for each minute
    k_values = []
    minute_data = []

    for min_idx in sorted(all_minutes_data.keys()):
        obs_list, start_sec = all_minutes_data[min_idx]
        # Count non-empty observations
        non_empty = sum(1 for o in obs_list if o)
        if non_empty < 54:  # Need at least 90% coverage
            continue

        start_second, k, matches = find_k_from_observations(obs_list, sig_period=sig_period)
        if k is not None and matches >= 54:
            k_values.append(k)
            minute_data.append((min_idx, k, matches, start_second))

    if not k_values:
        return result

    result['k_values'] = k_values

    if len(k_values) == 1:
        result['k'] = k_values[0]
        result['confidence'] = 1.0
        return result

    # With multiple minutes, compute deltas to detect P
    deltas = []
    for i in range(len(k_values) - 1):
        delta = k_values[i + 1] - k_values[i]
        if delta < 0:
            delta += PERIOD
        deltas.append(delta)

    # Count occurrences of each delta
    delta_counts = {}
    for d in deltas:
        delta_counts[d] = delta_counts.get(d, 0) + 1

    # Find majority delta (= P)
    majority_delta = max(delta_counts, key=delta_counts.get)
    majority_count = delta_counts[majority_delta]

    P = majority_delta
    confidence = majority_count / len(deltas) if deltas else 1.0

    result['P'] = P
    result['confidence'] = confidence

    # Find N from k values
    if P > 0:
        n_values = [k % P for k in k_values]
        # Use majority vote for N (in case one minute has error)
        n_counts = {}
        for n in n_values:
            n_counts[n] = n_counts.get(n, 0) + 1
        N = max(n_counts, key=n_counts.get)
        result['N'] = N

        # Check if any k value is inconsistent with majority N
        # This indicates a detection error in that minute
        for i, k in enumerate(k_values):
            if k % P != N:
                result['corrected_minute'] = minute_data[i][0]
                result['original_k'] = k
                # Correct k to match N
                k_corrected = ((k // P) * P + N) if k % P > N else ((k // P) * P + N)
                # Alternatively: find k value that's ±1 from original and matches N
                for adj in [0, -1, 1]:
                    test_k = k + adj
                    if test_k % P == N:
                        k_values[i] = test_k
                        if verbose:
                            print(f"  Multi-minute correction: minute {minute_data[i][0]} "
                                  f"k={k} -> k={test_k} (N={N})")
                        break

    # Return the k value for the first complete minute
    result['k'] = k_values[0]

    return result


def validate_observation(visible_cells):
    """
    Validate that observed cells form a valid combination for their implied second.

    Returns (clock_second, is_valid, suggested_correction) where:
    - clock_second: the second implied by the cell sum
    - is_valid: True if the cells match a valid combination for that second
    - suggested_correction: if invalid, a corrected cell set that would be valid (or None)
    """
    clock_second = cells_to_second(visible_cells)
    observed_set = set(visible_cells)

    # Check if observation matches any valid combination for this second
    valid_combinations = ORDERING[clock_second]
    for combo in valid_combinations:
        if set(combo) == observed_set:
            return clock_second, True, None

    # Invalid - try to find a correction by flipping one cell
    for cell in CELL_VALUES:
        # Try adding a missing cell
        if cell not in observed_set:
            test_set = observed_set | {cell}
            test_second = sum(test_set) % 60
            for combo in ORDERING[test_second]:
                if set(combo) == test_set:
                    return clock_second, False, test_set

        # Try removing an extra cell
        if cell in observed_set:
            test_set = observed_set - {cell}
            test_second = sum(test_set) % 60
            for combo in ORDERING[test_second]:
                if set(combo) == test_set:
                    return clock_second, False, test_set

    return clock_second, False, None


def int_to_base_digits(n, base, width):
    """Convert integer n to list of digits in given base, padded to width."""
    digits = []
    for _ in range(width):
        digits.append(n % base)
        n = n // base
    digits.reverse()
    return digits


def base_digits_to_int(digits, base):
    """Convert list of digits in given base back to integer."""
    result = 0
    for d in digits:
        result = result * base + d
    return result


def perm(k):
    """
    Forward permutation function.

    Given minute identifier k, returns a list of 60 indices.
    Each index selects which combination from ORDERING[second] to display.
    """
    k = k % PERIOD

    # Split k into components:
    # - k3_value: upper part (k // 2^30), represented as 16 base-3 digits
    # - lower 30 bits: split into k2 (18 bits) and k4 (12 bits as 6 pairs)
    k3_value = k // POW_2_30
    lower_30 = k % POW_2_30

    # k3: 16 base-3 digits
    k3 = int_to_base_digits(k3_value, 3, 16)

    # k2: first 18 bits (bits 12-29 of lower_30)
    k2_bits = (lower_30 >> 12) & ((1 << 18) - 1)
    k2 = int_to_base_digits(k2_bits, 2, 18)

    # k4: last 12 bits as 6 pairs (bits 0-11 of lower_30)
    k4_bits = lower_30 & ((1 << 12) - 1)
    k4 = []
    for i in range(6):
        k4.append((k4_bits >> (10 - 2*i)) & 3)

    # Build result by consuming from k2, k3, k4 based on option count
    result = []
    k2_idx = 0
    k3_idx = 0
    k4_idx = 0

    for second in range(60):
        num_options = len(ORDERING[second])
        if num_options == 1:
            result.append(0)
        elif num_options == 2:
            result.append(k2[k2_idx])
            k2_idx += 1
        elif num_options == 3:
            result.append(k3[k3_idx])
            k3_idx += 1
        elif num_options == 4:
            result.append(k4[k4_idx])
            k4_idx += 1

    return result


def get_all_cells_for_minute(k):
    """Get list of 60 cell-sets, one for each second of the minute."""
    indices = perm(k)
    result = []
    for second in range(60):
        idx = indices[second]
        result.append(set(ORDERING[second][idx]))
    return result


def find_combination_index(second, observed_cells):
    """
    Find which index in ORDERING[second] matches the observed cells.
    Returns the index, or -1 if no match found.
    """
    observed_set = set(observed_cells)
    options = ORDERING[second]
    for idx, combination in enumerate(options):
        if set(combination) == observed_set:
            return idx
    return -1


def inverse_perm(observed_indices):
    """
    Inverse permutation function.

    Given a list of 60 indices (one per second), reconstruct the minute identifier k.
    Each index indicates which combination from ORDERING[second] was observed.

    Returns k, or None if the indices are invalid.
    """
    # Collect k2, k3, k4 values from the indices
    k2_values = []
    k3_values = []
    k4_values = []

    for second in range(60):
        num_options = len(ORDERING[second])
        idx = observed_indices[second]

        if idx < 0 or idx >= num_options:
            return None  # Invalid index

        if num_options == 1:
            pass  # No information encoded
        elif num_options == 2:
            k2_values.append(idx)
        elif num_options == 3:
            k3_values.append(idx)
        elif num_options == 4:
            k4_values.append(idx)

    # Verify we got the expected counts
    if len(k2_values) != 18 or len(k3_values) != 16 or len(k4_values) != 6:
        return None

    # Reconstruct k3_value from base-3 digits
    k3_value = base_digits_to_int(k3_values, 3)

    # Reconstruct k2_bits from binary digits
    k2_bits = base_digits_to_int(k2_values, 2)

    # Reconstruct k4_bits from pairs
    k4_bits = 0
    for i, val in enumerate(k4_values):
        k4_bits = k4_bits | (val << (10 - 2*i))

    # Combine: lower_30 = (k2_bits << 12) | k4_bits
    lower_30 = (k2_bits << 12) | k4_bits

    # k = k3_value * 2^30 + lower_30
    k = k3_value * POW_2_30 + lower_30

    return k


def inverse_from_observations(observed_cells_list):
    """
    Given a list of 60 observed cell sets, compute the minute identifier k.

    Returns (k, indices) or (None, None) if no valid inversion found.
    """
    indices = []
    for second in range(60):
        idx = find_combination_index(second, observed_cells_list[second])
        if idx < 0:
            return None, None
        indices.append(idx)

    k = inverse_perm(indices)
    return k, indices


def verify_k_spanning(k, seconds_observed, start_second, sig_period=1):
    """
    Verify k against observations that span two minutes.

    When start_second > 0:
    - Seconds 0..start_second-1 come from minute k
    - Seconds start_second..59 come from minute k-P (previous minute)

    For signature encoding (k = minute * P + N), consecutive minutes differ by P.
    sig_period defaults to 1 for standard encoding.

    Returns match count (0-60).
    """
    indices_k = perm(k)
    k_prev = (k - sig_period) % PERIOD
    indices_k_prev = perm(k_prev)

    matches = 0
    for second in range(60):
        if second not in seconds_observed:
            continue
        cells = seconds_observed[second]
        idx = find_combination_index(second, cells)
        if idx < 0:
            continue

        # Determine which minute this second belongs to
        if start_second == 0:
            # All from same minute k
            expected_idx = indices_k[second]
        elif second < start_second:
            # Seconds 0..start_second-1 are from minute k
            expected_idx = indices_k[second]
        else:
            # Seconds start_second..59 are from minute k-P (previous minute)
            expected_idx = indices_k_prev[second]

        if idx == expected_idx:
            matches += 1

    return matches


def find_k_from_observations(observed_cells_list, min_match_ratio=0.9, sig_period=1):
    """
    Given observed cell sets, compute the minute identifier k.

    Uses sum-based second detection: cells shown at second S always sum to S.
    This eliminates the need to try 60 rotations.

    When observations span two minutes (start_second > 0):
    - Seconds start_second..59 come from minute k-P (previous minute)
    - Seconds 0..start_second-1 come from minute k
    k is the minute containing second 0.

    For signature encoding (k = minute * P + N), consecutive minutes differ by P.
    sig_period defaults to 1 for standard encoding.

    Args:
        observed_cells_list: list of observed cell sets (one per frame/sample)
        min_match_ratio: minimum fraction of seconds that must match (default 0.9)
        sig_period: signature period P for k encoding (default 1)

    Returns (start_second, k, match_count) where start_second is the clock second
    of the first observation, or (None, None, 0) if no valid solution found.
    """
    min_matches = int(60 * min_match_ratio)

    # Build observations indexed by clock second (using sum-based detection)
    # Note: empty set [] is valid for second 0 (all cells off), so don't skip it
    seconds_observed = {}  # clock_second -> observed_cells
    start_second = None
    seen_second_0 = False  # Track if we've seen a real second 0

    for i, cells in enumerate(observed_cells_list):
        clock_second = cells_to_second(cells)
        if i == 0:
            start_second = clock_second
        if clock_second not in seconds_observed:
            # For second 0: only accept first occurrence (avoid padding pollution)
            if clock_second == 0:
                if seen_second_0:
                    continue  # Skip duplicate second 0 (likely padding)
                seen_second_0 = True
            seconds_observed[clock_second] = cells

    # Check how many of the 60 seconds we have
    if len(seconds_observed) < min_matches:
        return None, None, len(seconds_observed)

    # Build indices array for inversion (mixed from two minutes if start_second > 0)
    indices = []
    for second in range(60):
        if second in seconds_observed:
            cells = seconds_observed[second]
            idx = find_combination_index(second, cells)
            indices.append(idx if idx >= 0 else 0)
        else:
            indices.append(0)

    # Get initial candidate from mixed reconstruction
    k_candidate = inverse_perm(indices)
    if k_candidate is None:
        return None, None, 0

    # If start_second == 0, all observations are from same minute - simple case
    if start_second == 0:
        matches = verify_k_spanning(k_candidate, seconds_observed, start_second, sig_period)
        if matches >= min_matches:
            return start_second, k_candidate, matches
        return None, None, matches

    # When start_second > 0, observations span two minutes
    # The mixed reconstruction gives approximately k (minute of second 0)
    # Try candidates around it to find the best match
    best_k = None
    best_matches = 0
    best_candidates = []  # Track all candidates with max matches

    # Check k_candidate and immediate neighbors (±P) first
    # Spanning mix can cause inverse_perm to return k-P instead of true k
    k_candidate_matches = verify_k_spanning(k_candidate, seconds_observed, start_second, sig_period)
    k_plus_P_matches = verify_k_spanning((k_candidate + sig_period) % PERIOD, seconds_observed, start_second, sig_period)
    k_minus_P_matches = verify_k_spanning((k_candidate - sig_period) % PERIOD, seconds_observed, start_second, sig_period)

    # Find best among k_candidate and its immediate neighbors
    best_immediate = k_candidate
    best_immediate_matches = k_candidate_matches
    if k_plus_P_matches > best_immediate_matches:
        best_immediate = (k_candidate + sig_period) % PERIOD
        best_immediate_matches = k_plus_P_matches
    if k_minus_P_matches > best_immediate_matches:
        best_immediate = (k_candidate - sig_period) % PERIOD
        best_immediate_matches = k_minus_P_matches

    # If best immediate neighbor has PERFECT matches, return it (no need to search further)
    if best_immediate_matches == len(seconds_observed):
        return start_second, best_immediate, best_immediate_matches

    # Search around k_candidate, and also around 0 (for wrap-around edge case)
    # Even with 59/60 matches, check 0 in case of PERIOD wraparound
    # Search in steps of sig_period (P) to find k values for consecutive minutes
    search_centers = [k_candidate, 0]

    for center in search_centers:
        for delta_mult in range(100):
            delta = delta_mult * sig_period
            for candidate in [center + delta, center - delta]:
                if delta == 0 and candidate != center:
                    continue
                candidate = candidate % PERIOD
                matches = verify_k_spanning(candidate, seconds_observed, start_second, sig_period)
                if matches > best_matches:
                    best_matches = matches
                    best_candidates = [(candidate, abs(candidate - k_candidate))]
                elif matches == best_matches:
                    best_candidates.append((candidate, abs(candidate - k_candidate)))
            # If we've found perfect matches, stop expanding search
            if best_matches == len(seconds_observed):
                break
        if best_matches == len(seconds_observed):
            break

    # Among candidates with max matches, prefer the one closest to inverse_perm result
    if best_candidates:
        best_candidates.sort(key=lambda x: x[1])  # Sort by distance to k_candidate
        best_k = best_candidates[0][0]

    if best_k is not None and best_matches >= min_matches:
        return start_second, best_k, best_matches

    return None, None, best_matches


def detect_sig_period_from_multi_minute(all_minutes_data):
    """
    Detect signature period P from multiple minutes of data.

    Computes k for each minute independently, then detects P from the deltas
    between consecutive minutes' k values.

    Args:
        all_minutes_data: dict mapping minute_idx -> (observations_list, start_second)
                         where observations_list[sec] = observation for second sec

    Returns (detected_P, detected_N, k_values) where:
        - detected_P: signature period (1 if not detected)
        - detected_N: signature value
        - k_values: list of k values for each minute (for verification)
    """
    if not all_minutes_data or len(all_minutes_data) < 2:
        return 1, 0, []

    # Compute k for each minute with good coverage
    k_values = []
    minute_indices = []

    for min_idx in sorted(all_minutes_data.keys()):
        obs_list, start_sec = all_minutes_data[min_idx]

        # Build indices array - obs_list[sec] is observation for second sec
        indices = []
        non_empty = 0
        for sec in range(60):
            if sec < len(obs_list) and obs_list[sec]:
                idx = find_combination_index(sec, obs_list[sec])
                indices.append(idx if idx >= 0 else 0)
                if idx >= 0:
                    non_empty += 1
            else:
                indices.append(0)

        if non_empty < 50:  # Need good coverage
            continue

        k = inverse_perm(indices)
        if k is not None:
            k_values.append(k)
            minute_indices.append(min_idx)

    if len(k_values) < 2:
        return 1, 0, k_values

    # Compute deltas between consecutive minutes
    deltas = []
    for i in range(len(k_values) - 1):
        delta = k_values[i + 1] - k_values[i]
        if delta < 0:
            delta += PERIOD
        deltas.append(delta)

    # Use majority voting for P (in case one delta is wrong due to detection error)
    delta_counts = {}
    for d in deltas:
        delta_counts[d] = delta_counts.get(d, 0) + 1

    detected_P = max(delta_counts, key=delta_counts.get)
    if detected_P == 0:
        detected_P = 1

    # Compute N from k values using majority voting
    if detected_P > 1:
        n_values = [k % detected_P for k in k_values]
        n_counts = {}
        for n in n_values:
            n_counts[n] = n_counts.get(n, 0) + 1
        detected_N = max(n_counts, key=n_counts.get)
    else:
        detected_N = 0

    return detected_P, detected_N, k_values


def detect_sig_period_from_spanning(observed_cells_list):
    """
    Detect signature period P from spanning observations within a single minute.

    When video starts mid-minute (start_second > 0), we observe two partial minutes.
    By reconstructing k separately for each part, we can compute P = k_next - k_prev.

    Returns (detected_P, confidence) where confidence is how well P explains the data.
    Returns (1, 0.0) if detection fails or observations don't span.
    """
    # First, determine start_second from the observations
    if not observed_cells_list:
        return 1, 0.0

    start_second = cells_to_second(observed_cells_list[0])
    if start_second == 0:
        return 1, 1.0  # No spanning, P=1 works

    # Separate observations into "next minute" (seconds 0..start_second-1) and
    # "prev minute" (seconds start_second..59)
    next_minute_obs = {}  # second -> cells
    prev_minute_obs = {}  # second -> cells

    for i, cells in enumerate(observed_cells_list):
        clock_second = cells_to_second(cells)
        if clock_second < start_second:
            if clock_second not in next_minute_obs:
                next_minute_obs[clock_second] = cells
        else:
            if clock_second not in prev_minute_obs:
                prev_minute_obs[clock_second] = cells

    # Need reasonable coverage of both parts
    # Also need minimum observations from each part for reliable P detection
    MIN_OBS_FOR_P_DETECTION = 20  # Need at least 20 seconds from each part
    if (len(next_minute_obs) < MIN_OBS_FOR_P_DETECTION or
        len(prev_minute_obs) < MIN_OBS_FOR_P_DETECTION or
        len(next_minute_obs) < start_second * 0.8 or
        len(prev_minute_obs) < (60 - start_second) * 0.8):
        return 1, 0.0

    # Build full observation arrays (pad missing seconds with zeros)
    # For k_next: use next_minute_obs for seconds 0..start_second-1, padding for rest
    # For k_prev: use prev_minute_obs for seconds start_second..59, padding for rest
    full_next = []
    full_prev = []

    for sec in range(60):
        if sec in next_minute_obs:
            idx = find_combination_index(sec, next_minute_obs[sec])
            full_next.append(idx if idx >= 0 else 0)
        else:
            full_next.append(0)

        if sec in prev_minute_obs:
            idx = find_combination_index(sec, prev_minute_obs[sec])
            full_prev.append(idx if idx >= 0 else 0)
        else:
            full_prev.append(0)

    # Get k candidates from each partial observation
    k_next = inverse_perm(full_next)
    k_prev = inverse_perm(full_prev)

    if k_next is None or k_prev is None:
        return 1, 0.0

    # P = k_next - k_prev (with wraparound handling)
    P = (k_next - k_prev) % PERIOD
    if P == 0:
        P = 1  # Fallback

    return P, 1.0


def find_k_with_autodetect(observed_cells_list, min_match_ratio=0.9, sig_period=None):
    """
    Find k with automatic signature period detection.

    If sig_period is None, detects P from spanning observations.
    For non-spanning observations (start_second=0), P=1 is used.

    Args:
        observed_cells_list: list of observed cell sets
        min_match_ratio: minimum fraction of seconds that must match
        sig_period: if provided, use this P; if None, autodetect

    Returns (start_second, k, match_count, detected_P)
    """
    if sig_period is not None:
        # User specified P, use it directly
        start_sec, k, matches = find_k_from_observations(
            observed_cells_list, min_match_ratio, sig_period
        )
        return start_sec, k, matches, sig_period

    # First try P=1 (standard encoding)
    start_sec, k, matches = find_k_from_observations(
        observed_cells_list, min_match_ratio, sig_period=1
    )

    # If perfect match or no spanning, P=1 is correct
    if k is not None and (matches == 60 or start_sec == 0):
        return start_sec, k, matches, 1

    # For spanning with imperfect matches, try to detect P
    detected_P, confidence = detect_sig_period_from_spanning(observed_cells_list)

    if detected_P > 1 and confidence > 0:
        # Try with detected P
        test_start, test_k, test_matches = find_k_from_observations(
            observed_cells_list, min_match_ratio, sig_period=detected_P
        )

        if test_k is not None and test_matches > matches:
            return test_start, test_k, test_matches, detected_P

    # Fallback to P=1 result
    return start_sec, k, matches, 1


# For testing
if __name__ == "__main__":
    import random

    print("Testing perm/inverse round-trip...")

    for test_num in range(100):
        test_k = random.randint(0, PERIOD - 1)
        cells = get_all_cells_for_minute(test_k)
        recovered_k, _ = inverse_from_observations(cells)

        if recovered_k != test_k:
            print(f"FAIL: k={test_k}, recovered={recovered_k}")
        elif test_num < 5:
            print(f"OK: k={test_k}")

    print("Round-trip test complete.")

    print("\nTesting sum-based second detection...")

    test_k = random.randint(0, PERIOD - 1)
    cells = get_all_cells_for_minute(test_k)

    # Test that cells_to_second works for all 60 seconds
    for second in range(60):
        detected = cells_to_second(cells[second])
        if detected != second:
            print(f"FAIL: second {second}, detected {detected}")
        elif second < 5:
            print(f"OK: second {second} -> sum {sum(cells[second])}")

    print("Sum-based detection test complete.")

    print("\nTesting find_k_from_observations...")

    start_second, found_k, matches = find_k_from_observations(cells)
    if found_k == test_k:
        print(f"OK: k={test_k}, matches={matches}/60")
    else:
        print(f"FAIL: expected k={test_k}, got k={found_k}")

    print("All tests complete.")
