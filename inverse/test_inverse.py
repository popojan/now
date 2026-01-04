#!/usr/bin/env python3
"""Unit tests for clock inverse algorithm."""

import unittest
from clock_inverse import (
    find_k_from_observations,
    get_all_cells_for_minute,
    perm,
    inverse_perm,
    cells_to_second,
    PERIOD,
)


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


if __name__ == "__main__":
    unittest.main()
