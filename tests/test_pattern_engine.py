from __future__ import annotations

import unittest

import pandas as pd

from core.features.pattern_engine import detect_top_pattern_from_frame


class PatternEngineTests(unittest.TestCase):
    def test_detects_double_bottom(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [10.4, 10.1, 9.6, 9.1, 8.8, 9.2, 9.8, 10.3, 9.9, 9.4, 8.85, 9.3, 10.1, 10.7, 11.0],
                "high": [10.6, 10.3, 9.8, 9.3, 9.0, 9.4, 10.0, 10.5, 10.1, 9.6, 9.05, 9.5, 10.3, 10.9, 11.2],
                "low": [10.2, 9.9, 9.4, 8.9, 8.6, 9.0, 9.6, 10.1, 9.7, 9.2, 8.65, 9.1, 9.9, 10.5, 10.8],
                "volume": [100, 102, 105, 110, 120, 118, 122, 130, 128, 126, 135, 138, 150, 170, 210],
            }
        )

        result = detect_top_pattern_from_frame(symbol="TEST/USD", timeframe="15m", frame=frame)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["pattern"], "double_bottom")
        self.assertEqual(result["bias"], "bullish")
        self.assertIn("structure_candidate", result)
        self.assertIn("candle_context", result)
        self.assertIn("location_context", result)
        self.assertIn("conflict_checks", result)
        self.assertIn("extension_risk_score", result["structure_candidate"])
        self.assertGreaterEqual(result["structure_candidate"]["extension_risk_score"], 0.0)

    def test_detects_double_top(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [9.8, 10.2, 10.7, 11.1, 11.35, 10.9, 10.2, 9.8, 10.3, 10.85, 11.3, 10.7, 10.0, 9.4, 9.0],
                "high": [10.0, 10.4, 10.9, 11.3, 11.55, 11.1, 10.4, 10.0, 10.5, 11.05, 11.5, 10.9, 10.2, 9.6, 9.2],
                "low": [9.6, 10.0, 10.5, 10.9, 11.15, 10.7, 10.0, 9.6, 10.1, 10.65, 11.1, 10.5, 9.8, 9.2, 8.8],
                "volume": [100, 102, 108, 112, 130, 125, 120, 118, 122, 128, 140, 150, 165, 180, 220],
            }
        )

        result = detect_top_pattern_from_frame(symbol="TEST/USD", timeframe="15m", frame=frame)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["pattern"], "double_top")
        self.assertEqual(result["bias"], "bearish")

    def test_includes_candle_confirmation_context(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [10.5, 10.2, 9.8, 9.2, 8.95, 9.2, 9.8, 10.4, 10.05, 9.55, 8.95, 9.35, 10.0, 10.9, 10.6],
                "close": [10.4, 10.1, 9.6, 9.1, 8.8, 9.2, 9.8, 10.3, 9.9, 9.4, 8.85, 9.3, 10.1, 10.7, 11.0],
                "high": [10.6, 10.3, 9.8, 9.3, 9.0, 9.4, 10.0, 10.5, 10.1, 9.6, 9.05, 9.5, 10.3, 10.9, 11.2],
                "low": [10.2, 9.9, 9.4, 8.9, 8.6, 9.0, 9.6, 10.1, 9.7, 9.2, 8.65, 9.1, 9.9, 10.5, 10.8],
                "volume": [100, 102, 105, 108, 112, 118, 122, 128, 130, 132, 138, 142, 150, 160, 210],
            }
        )

        result = detect_top_pattern_from_frame(symbol="TEST/USD", timeframe="15m", frame=frame)

        self.assertIsNotNone(result)
        assert result is not None
        candle_names = [item["name"] for item in result["candle_context"]["recent_candles"]]
        self.assertTrue(candle_names)
        self.assertIn("location", result["candle_context"]["recent_candles"][0])

    def test_flags_late_breakout_extension_risk(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [10.5, 10.2, 9.8, 9.2, 8.95, 9.2, 9.8, 10.4, 10.05, 9.55, 8.95, 9.35, 10.0, 10.9, 10.9],
                "close": [10.4, 10.1, 9.6, 9.1, 8.8, 9.2, 9.8, 10.3, 9.9, 9.4, 8.85, 9.3, 10.1, 10.7, 11.2],
                "high": [10.6, 10.3, 9.8, 9.3, 9.0, 9.4, 10.0, 10.5, 10.1, 9.6, 9.05, 9.5, 10.3, 10.9, 11.25],
                "low": [10.2, 9.9, 9.4, 8.9, 8.6, 9.0, 9.6, 10.1, 9.7, 9.2, 8.65, 9.1, 9.9, 10.5, 10.85],
                "volume": [100, 102, 105, 108, 112, 118, 122, 128, 130, 132, 138, 142, 150, 160, 210],
            }
        )

        result = detect_top_pattern_from_frame(symbol="TEST/USD", timeframe="15m", frame=frame)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result["structure_candidate"]["extension_risk_score"], 0.6)
        self.assertTrue(result["location_context"]["overextended_from_entry_zone"])
        self.assertIn("Late breakout risk", " ".join(result["notes"]))


if __name__ == "__main__":
    unittest.main()
