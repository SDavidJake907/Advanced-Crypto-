import unittest

from apps.trader.features import compute_features


class ComputeFeaturesTests(unittest.TestCase):
    def test_breakout_uses_prior_donchian_window(self) -> None:
        closes_1h = [100.0 + i for i in range(60)]

        closes_15m = [100.0 + i * 0.2 for i in range(60)]
        highs_15m = [close + 1.0 for close in closes_15m]
        lows_15m = [close - 1.0 for close in closes_15m]
        volumes_15m = [100.0 for _ in range(59)] + [160.0]

        prior_20_high = max(highs_15m[-21:-1])
        closes_15m[-1] = prior_20_high + 0.5
        highs_15m[-1] = closes_15m[-1] + 0.5
        lows_15m[-1] = closes_15m[-1] - 1.0

        features = compute_features(
            closes_15m=closes_15m,
            highs_15m=highs_15m,
            lows_15m=lows_15m,
            volumes_15m=volumes_15m,
            closes_1h=closes_1h,
        )

        self.assertIsNotNone(features)
        assert features is not None
        self.assertTrue(features.breakout)
        self.assertTrue(features.volume_confirm)
        self.assertEqual(features.trend, "up")


if __name__ == "__main__":
    unittest.main()
