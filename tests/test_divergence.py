import unittest

from core.features.divergence import (
    bearish_divergence,
    bullish_divergence,
    detect_rsi_divergence,
    divergence_age_bars,
    divergence_strength,
)
from core.policy.final_score import compute_final_score


class DivergenceFeatureTests(unittest.TestCase):
    def test_bullish_divergence(self) -> None:
        closes = [110.0, 108.0, 105.0, 102.0, 100.0, 103.0, 106.0, 104.0, 101.0, 98.0, 100.0, 103.0, 105.0]
        rsi = [55.0, 48.0, 40.0, 32.0, 28.0, 42.0, 50.0, 46.0, 39.0, 35.0, 44.0, 51.0, 54.0]

        signal = detect_rsi_divergence(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2, scan_bars=20)

        self.assertTrue(signal.bullish_divergence)
        self.assertFalse(signal.bearish_divergence)
        self.assertTrue(bullish_divergence(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2))
        self.assertGreater(divergence_strength(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2), 0.0)
        self.assertEqual(divergence_age_bars(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2), 3)

    def test_bearish_divergence(self) -> None:
        closes = [100.0, 103.0, 107.0, 109.0, 112.0, 110.0, 108.0, 111.0, 113.0, 115.0, 112.0, 109.0, 107.0]
        rsi = [50.0, 58.0, 67.0, 72.0, 76.0, 70.0, 63.0, 68.0, 69.0, 66.0, 58.0, 49.0, 44.0]

        signal = detect_rsi_divergence(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2, scan_bars=20)

        self.assertFalse(signal.bullish_divergence)
        self.assertTrue(signal.bearish_divergence)
        self.assertTrue(bearish_divergence(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2))
        self.assertGreater(signal.divergence_strength, 0.0)
        self.assertEqual(signal.divergence_age_bars, 3)

    def test_no_divergence(self) -> None:
        closes = [100.0, 99.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0]
        rsi = [50.0, 46.0, 42.0, 38.0, 34.0, 40.0, 46.0, 52.0, 58.0, 64.0, 70.0]

        signal = detect_rsi_divergence(closes, rsi_values=rsi, rsi_period=2, swing_lookback=2, scan_bars=20)

        self.assertFalse(signal.bullish_divergence)
        self.assertFalse(signal.bearish_divergence)
        self.assertEqual(signal.divergence_strength, 0.0)
        self.assertEqual(signal.divergence_age_bars, 99)

    def test_stale_divergence_decay(self) -> None:
        fresh = compute_final_score(
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "entry_score": 60.0,
                "bullish_divergence": True,
                "divergence_strength": 1.0,
                "divergence_age_bars": 1,
            }
        )
        stale = compute_final_score(
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "entry_score": 60.0,
                "bullish_divergence": True,
                "divergence_strength": 1.0,
                "divergence_age_bars": 14,
            }
        )

        self.assertGreater(fresh.divergence_bonus, stale.divergence_bonus)
        self.assertLessEqual(fresh.divergence_bonus, 6.0)
        self.assertGreater(fresh.final_score, stale.final_score)


if __name__ == "__main__":
    unittest.main()
