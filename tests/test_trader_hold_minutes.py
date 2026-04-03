from __future__ import annotations

import unittest

from apps.trader.positions import compute_hold_minutes


class TraderHoldMinutesTests(unittest.TestCase):
    def test_mixed_naive_and_utc_timestamps_compute_hold_minutes(self) -> None:
        hold_minutes = compute_hold_minutes(
            "2026-02-22 16:19:33.82",
            "2026-03-22T04:35:20+00:00",
        )
        self.assertGreater(hold_minutes, 1000.0)


if __name__ == "__main__":
    unittest.main()
