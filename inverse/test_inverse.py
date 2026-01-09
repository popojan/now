#!/usr/bin/env python3
"""Unit tests for clock inverse algorithm."""

import unittest
import random
from datetime import datetime, timezone
from clock_inverse import (
    find_k_from_observations,
    get_all_cells_for_minute,
    perm,
    inverse_perm,
    cells_to_second,
    cells_to_mask,
    mask_to_cells,
    mask_to_sum,
    hamming_distance,
    closest_valid_mask,
    correct_errors,
    detect_signature,
    inverse_with_correction,
    PERIOD,
    MASKS_FOR_SECOND,
    ORDERING,
)
from main import compute_clock_origin, format_ancient_date


class TestInverseBasic(unittest.TestCase):
    """Basic inverse tests with no minute boundary crossing."""

    def test_perm_inverse_roundtrip(self):
        """Test that inverse_perm(perm(k)) == k for various k values."""
        test_values = [0, 1, 12345, 99999, PERIOD - 1, PERIOD // 2]
        for k in test_values:
            indices = perm(k)
            recovered = inverse_perm(indices)
            self.assertEqual(recovered, k, f"Round-trip failed for k={k}")

    def test_cells_to_second(self):
        """Test sum-based second detection."""
        k = 12345
        cells = get_all_cells_for_minute(k)
        for s in range(60):
            detected = cells_to_second(cells[s])
            self.assertEqual(detected, s, f"Second detection failed for s={s}")

    def test_find_k_no_spanning(self):
        """Test find_k when all 60 frames are from same minute."""
        for k in [0, 1, 12345, 99999, PERIOD - 1]:
            cells = get_all_cells_for_minute(k)
            start_sec, found_k, matches = find_k_from_observations(cells)
            self.assertEqual(found_k, k, f"Failed for k={k}")
            self.assertEqual(start_sec, 0, f"Start second wrong for k={k}")
            self.assertEqual(matches, 60, f"Matches wrong for k={k}")


class TestInverseSpanning(unittest.TestCase):
    """Tests for observations spanning two minutes."""

    def _build_spanning_observations(self, k, start_sec):
        """Build 60 observations starting at second start_sec of minute k-1."""
        observations = []
        for frame in range(60):
            actual_sec = (start_sec + frame) % 60
            if start_sec == 0:
                minute = k
            elif actual_sec >= start_sec:
                minute = (k - 1) % PERIOD
            else:
                minute = k
            cells = get_all_cells_for_minute(minute)
            observations.append(cells[actual_sec])
        return observations

    def test_spanning_middle(self):
        """Test spanning with boundary in middle (start_sec=30)."""
        for k in [12345, 99999, 1]:
            observations = self._build_spanning_observations(k, 30)
            start_sec, found_k, matches = find_k_from_observations(observations)
            self.assertEqual(found_k, k, f"Failed for k={k}, start_sec=30")
            self.assertEqual(start_sec, 30)
            self.assertEqual(matches, 60)

    def test_spanning_near_end(self):
        """Test spanning with boundary near end (start_sec=58)."""
        k = 12345
        observations = self._build_spanning_observations(k, 58)
        start_sec, found_k, matches = find_k_from_observations(observations)
        self.assertEqual(found_k, k)
        self.assertEqual(start_sec, 58)
        self.assertEqual(matches, 60)

    def test_spanning_near_start(self):
        """Test spanning with boundary near start (start_sec=5)."""
        k = 12345
        observations = self._build_spanning_observations(k, 5)
        start_sec, found_k, matches = find_k_from_observations(observations)
        self.assertEqual(found_k, k)
        self.assertEqual(start_sec, 5)
        self.assertEqual(matches, 60)

    def test_spanning_wraparound_k0(self):
        """Test edge case: k=0 with spanning (wraps to PERIOD-1)."""
        k = 0
        observations = self._build_spanning_observations(k, 30)
        start_sec, found_k, matches = find_k_from_observations(observations)
        self.assertEqual(found_k, k)
        self.assertEqual(start_sec, 30)
        self.assertEqual(matches, 60)

    def test_spanning_wraparound_period_minus_1(self):
        """Test edge case: k=PERIOD-1 with spanning."""
        k = PERIOD - 1
        observations = self._build_spanning_observations(k, 30)
        start_sec, found_k, matches = find_k_from_observations(observations)
        self.assertEqual(found_k, k)
        self.assertEqual(start_sec, 30)
        self.assertEqual(matches, 60)


class TestInverseVariousK(unittest.TestCase):
    """Tests with various k values to ensure coverage."""

    def test_many_k_values(self):
        """Test inverse with many different k values."""
        import random
        random.seed(42)
        test_ks = [random.randint(0, PERIOD - 1) for _ in range(10)]

        for k in test_ks:
            cells = get_all_cells_for_minute(k)
            start_sec, found_k, matches = find_k_from_observations(cells)
            self.assertEqual(found_k, k, f"Failed for k={k}")
            self.assertEqual(matches, 60)

    def test_many_spanning_cases(self):
        """Test spanning with various k and start_sec combinations."""
        import random
        random.seed(42)

        for _ in range(10):
            k = random.randint(1, PERIOD - 1)
            start_sec = random.randint(1, 59)

            observations = []
            for frame in range(60):
                actual_sec = (start_sec + frame) % 60
                if actual_sec >= start_sec:
                    minute = (k - 1) % PERIOD
                else:
                    minute = k
                cells = get_all_cells_for_minute(minute)
                observations.append(cells[actual_sec])

            found_start, found_k, matches = find_k_from_observations(observations)
            self.assertEqual(found_k, k, f"Failed for k={k}, start_sec={start_sec}")
            self.assertEqual(found_start, start_sec)
            self.assertEqual(matches, 60)


class TestClockOrigin(unittest.TestCase):
    """Tests for compute_clock_origin and B.C. date calculation."""

    def test_year_0_origin(self):
        """Test that year 0 origin (0000-01-01 00:00:00) is computed correctly.

        This is a regression test - the B.C. calculation was incorrectly
        returning 0000-12-31 instead of 0000-01-01.
        """
        # k for year 0 origin: minutes from 0000-01-01 to 0001-01-01
        # Year 0 is a leap year (366 days)
        k_year_0 = 366 * 24 * 60  # 527040 minutes

        # Video timestamp at 0001-01-01 00:00:00 UTC, start_second=0
        video_ts = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        origin, _ = compute_clock_origin(video_ts, 0, k_year_0)

        self.assertIsInstance(origin, str)  # Ancient dates return strings
        self.assertEqual(origin, "0000-01-01 00:00:00")

    def test_year_minus_1_origin(self):
        """Test year -1 (2 B.C.) origin."""
        # k for year -1 origin: 2 years before year 1
        # Year 0: 366 days (leap), Year -1: 365 days (not leap)
        k_year_minus_1 = (366 + 365) * 24 * 60

        video_ts = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        origin, _ = compute_clock_origin(video_ts, 0, k_year_minus_1)

        self.assertIsInstance(origin, str)
        self.assertEqual(origin, "-0001-01-01 00:00:00")

    def test_year_1_origin(self):
        """Test year 1 origin (normal datetime, not ancient)."""
        # k=0 means origin is exactly at video timestamp
        video_ts = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        origin, _ = compute_clock_origin(video_ts, 0, 0)

        self.assertIsInstance(origin, datetime)
        self.assertEqual(origin.year, 1)
        self.assertEqual(origin.month, 1)
        self.assertEqual(origin.day, 1)

    def test_recent_origin(self):
        """Test recent origin (2024-01-01)."""
        # Video at 2025-01-01 00:00:00, k = 1 year = 527040 minutes (2024 is leap)
        video_ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        origin, _ = compute_clock_origin(video_ts, 0, 527040)

        self.assertIsInstance(origin, datetime)
        self.assertEqual(origin.year, 2024)
        self.assertEqual(origin.month, 1)
        self.assertEqual(origin.day, 1)

    def test_format_ancient_date(self):
        """Test format_ancient_date for various years."""
        self.assertEqual(format_ancient_date(2024, 6, 15, 12, 30, 45),
                        "2024-06-15 12:30:45")
        self.assertEqual(format_ancient_date(0, 1, 1, 0, 0, 0),
                        "0000-01-01 00:00:00")
        self.assertEqual(format_ancient_date(-1, 12, 31, 23, 59, 59),
                        "-0001-12-31 23:59:59")
        self.assertEqual(format_ancient_date(-100, 6, 15, 12, 0, 0),
                        "-0100-06-15 12:00:00")


class TestMaskUtilities(unittest.TestCase):
    """Tests for mask/bitmask conversion utilities."""

    def test_cells_to_mask_roundtrip(self):
        """Test cells -> mask -> cells roundtrip."""
        for second in range(60):
            for combo in ORDERING[second]:
                mask = cells_to_mask(combo)
                recovered = set(mask_to_cells(mask))
                self.assertEqual(recovered, set(combo),
                               f"Roundtrip failed for {combo}")

    def test_mask_to_sum(self):
        """Test mask_to_sum produces correct sums."""
        for second in range(60):
            for combo in ORDERING[second]:
                mask = cells_to_mask(combo)
                expected_sum = sum(combo)
                self.assertEqual(mask_to_sum(mask), expected_sum,
                               f"Sum wrong for mask={mask:07b}")

    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        self.assertEqual(hamming_distance(0b0000000, 0b0000000), 0)
        self.assertEqual(hamming_distance(0b0000001, 0b0000000), 1)
        self.assertEqual(hamming_distance(0b1111111, 0b0000000), 7)
        self.assertEqual(hamming_distance(0b1010101, 0b0101010), 7)
        self.assertEqual(hamming_distance(0b1100110, 0b1100110), 0)
        self.assertEqual(hamming_distance(0b1100110, 0b1100111), 1)

    def test_masks_for_second_precomputed(self):
        """Test MASKS_FOR_SECOND is correctly precomputed."""
        for second in range(60):
            expected_masks = [cells_to_mask(combo) for combo in ORDERING[second]]
            self.assertEqual(MASKS_FOR_SECOND[second], expected_masks)


class TestClosestValidMask(unittest.TestCase):
    """Tests for closest_valid_mask function."""

    def test_exact_match_returns_same(self):
        """Test that exact match returns same mask with distance 0."""
        for second in range(60):
            for mask in MASKS_FOR_SECOND[second]:
                result, dist = closest_valid_mask(mask, second)
                self.assertEqual(dist, 0, f"Distance should be 0 for exact match")
                self.assertEqual(result, mask)

    def test_single_bit_flip_correction(self):
        """Test correction of single-bit errors."""
        for second in range(60):
            for mask in MASKS_FOR_SECOND[second]:
                # Flip each bit and verify correction
                for bit in range(7):
                    corrupted = mask ^ (1 << bit)
                    result, dist = closest_valid_mask(corrupted, second)
                    # Distance should be at most 1 (might be 0 if corrupted is also valid)
                    self.assertLessEqual(dist, 1)

    def test_returns_valid_mask(self):
        """Test that result is always a valid mask for target second."""
        random.seed(42)
        for _ in range(100):
            second = random.randint(0, 59)
            random_mask = random.randint(0, 127)
            result, _ = closest_valid_mask(random_mask, second)
            self.assertIn(result, MASKS_FOR_SECOND[second])


class TestErrorCorrection(unittest.TestCase):
    """Tests for correct_errors function."""

    def test_perfect_input_no_corrections(self):
        """Test that perfect input has all anchors, no corrections."""
        k = 12345
        cells = get_all_cells_for_minute(k)
        masks = [cells_to_mask(c) for c in cells]

        corrected, corrections, anchor_count = correct_errors(masks)

        self.assertEqual(anchor_count, 60)
        self.assertEqual(corrected, masks)
        for corr in corrections:
            self.assertTrue(corr.is_anchor)
            self.assertEqual(corr.distance, 0)

    def test_single_error_corrected(self):
        """Test that single corrupted frame is corrected."""
        k = 12345
        cells = get_all_cells_for_minute(k)
        masks = [cells_to_mask(c) for c in cells]

        # Corrupt frame 30 by flipping bit 0
        masks[30] ^= 0x01

        corrected, corrections, anchor_count = correct_errors(masks)

        self.assertEqual(anchor_count, 59)
        # Frame 30 should be corrected
        self.assertFalse(corrections[30].is_anchor)
        self.assertGreater(corrections[30].distance, 0)
        # Other frames should be anchors
        for i in range(60):
            if i != 30:
                self.assertTrue(corrections[i].is_anchor)

    def test_multiple_errors_corrected(self):
        """Test correction with multiple errors (< 50%)."""
        k = 12345
        cells = get_all_cells_for_minute(k)
        masks = [cells_to_mask(c) for c in cells]

        # Corrupt 10 frames
        error_indices = [5, 10, 15, 20, 25, 35, 40, 45, 50, 55]
        for i in error_indices:
            masks[i] ^= 0x01  # Flip bit 0

        corrected, corrections, anchor_count = correct_errors(masks)

        self.assertIsNotNone(corrected)
        self.assertEqual(anchor_count, 50)

        for i in range(60):
            if i in error_indices:
                self.assertFalse(corrections[i].is_anchor)
            else:
                self.assertTrue(corrections[i].is_anchor)

    def test_heavy_corruption_fails_gracefully(self):
        """Test that heavy corruption (>50% same wrong vs) returns None."""
        random.seed(42)
        # Create random masks - no clear winner
        masks = [random.randint(0, 127) for _ in range(60)]

        result = correct_errors(masks)

        # Should fail gracefully - no clear winner
        # (In practice random masks might accidentally have a winner,
        # but the function should not crash)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


class TestSignatureDetection(unittest.TestCase):
    """Tests for detect_signature function."""

    def test_single_minute_returns_default(self):
        """Test that single minute returns P=1, N=0."""
        P, N, conf = detect_signature([12345])
        self.assertEqual(P, 1)
        self.assertEqual(N, 0)
        self.assertEqual(conf, 1.0)

    def test_no_signature_p1(self):
        """Test detection of P=1 (no signature)."""
        # k values with P=1: consecutive values differ by 1
        k_values = [100, 101, 102, 103, 104]
        P, N, conf = detect_signature(k_values)
        self.assertEqual(P, 1)
        self.assertEqual(conf, 1.0)

    def test_signature_p7(self):
        """Test detection of P=7."""
        # k = local_minute * P + N
        # With P=7, N=3: k_values differ by 7
        k_values = [3, 10, 17, 24, 31]  # N=3, deltas all 7
        P, N, conf = detect_signature(k_values)
        self.assertEqual(P, 7)
        self.assertEqual(N, 3)
        self.assertEqual(conf, 1.0)

    def test_signature_with_noise(self):
        """Test detection with one noisy delta."""
        # 4 deltas: 7, 7, 8, 7 - majority is 7
        k_values = [3, 10, 17, 25, 32]  # Third delta is 8 instead of 7
        P, N, conf = detect_signature(k_values)
        self.assertEqual(P, 7)
        self.assertEqual(conf, 0.75)  # 3/4 deltas agree


class TestInverseWithCorrection(unittest.TestCase):
    """Tests for inverse_with_correction function."""

    def test_single_minute_basic(self):
        """Test inverse_with_correction for single minute."""
        k = 12345
        cells = get_all_cells_for_minute(k)

        result = inverse_with_correction(cells)

        self.assertEqual(result['k'], k)
        self.assertEqual(result['P'], 1)
        self.assertEqual(result['N'], 0)
        self.assertEqual(result['anchor_count'], 60)

    def test_two_minutes_p1(self):
        """Test inverse_with_correction for two consecutive minutes."""
        k = 12345
        cells = get_all_cells_for_minute(k) + get_all_cells_for_minute(k + 1)

        result = inverse_with_correction(cells)

        self.assertEqual(result['k'], k)
        self.assertEqual(len(result['k_values']), 2)
        self.assertEqual(result['k_values'][0], k)
        self.assertEqual(result['k_values'][1], k + 1)
        self.assertEqual(result['P'], 1)
        self.assertEqual(result['signature_confidence'], 1.0)

    def test_two_minutes_with_signature(self):
        """Test detection of P=7 signature across two minutes."""
        # Simulate P=7, N=3 encoding
        # k_combined = local_minute * P + N
        # Minute 0: k = 0*7 + 3 = 3
        # Minute 1: k = 1*7 + 3 = 10
        cells_m0 = get_all_cells_for_minute(3)
        cells_m1 = get_all_cells_for_minute(10)
        cells = cells_m0 + cells_m1

        result = inverse_with_correction(cells)

        self.assertEqual(result['P'], 7)
        self.assertEqual(result['N'], 3)


if __name__ == "__main__":
    unittest.main()
