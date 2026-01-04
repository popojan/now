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


def verify_k_spanning(k, seconds_observed, start_second):
    """
    Verify k against observations that span two minutes.

    When start_second > 0:
    - Seconds 0..start_second-1 come from minute k
    - Seconds start_second..59 come from minute k-1

    Returns match count (0-60).
    """
    indices_k = perm(k)
    indices_k_minus_1 = perm(k - 1) if k > 0 else perm(PERIOD - 1)

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
            # Seconds start_second..59 are from minute k-1
            expected_idx = indices_k_minus_1[second]

        if idx == expected_idx:
            matches += 1

    return matches


def find_k_from_observations(observed_cells_list, min_match_ratio=0.9):
    """
    Given observed cell sets, compute the minute identifier k.

    Uses sum-based second detection: cells shown at second S always sum to S.
    This eliminates the need to try 60 rotations.

    When observations span two minutes (start_second > 0):
    - Seconds start_second..59 come from minute k-1
    - Seconds 0..start_second-1 come from minute k
    k is the minute containing second 0.

    Args:
        observed_cells_list: list of observed cell sets (one per frame/sample)
        min_match_ratio: minimum fraction of seconds that must match (default 0.9)

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
        matches = verify_k_spanning(k_candidate, seconds_observed, start_second)
        if matches >= min_matches:
            return start_second, k_candidate, matches
        return None, None, matches

    # When start_second > 0, observations span two minutes
    # The mixed reconstruction gives approximately k (minute of second 0)
    # Try candidates around it to find the best match
    best_k = None
    best_matches = 0

    # Search around k_candidate, and also around 0 (for wrap-around edge case)
    search_centers = [k_candidate, 0]

    for center in search_centers:
        for delta in range(100):
            for candidate in [center + delta, center - delta]:
                if delta == 0 and candidate != center:
                    continue
                candidate = candidate % PERIOD
                matches = verify_k_spanning(candidate, seconds_observed, start_second)
                if matches > best_matches:
                    best_matches = matches
                    best_k = candidate
                # Perfect match - return immediately
                if matches == len(seconds_observed):
                    return start_second, candidate, matches
            # If we've found a perfect match searching this center, stop
            if best_matches == len(seconds_observed):
                break

    if best_k is not None and best_matches >= min_matches:
        return start_second, best_k, best_matches

    return None, None, best_matches


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
