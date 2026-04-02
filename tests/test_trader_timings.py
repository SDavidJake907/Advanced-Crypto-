from __future__ import annotations

import unittest

from apps.trader.main import _build_phi3_reflex_cache_key, _format_decision_timings


class TraderTimingsTests(unittest.TestCase):
    def test_format_decision_timings_separates_decision_and_cycle_total(self) -> None:
        payload = _format_decision_timings(
            70.0,
            {
                "phi3_ms": 337.0,
                "advisory_ms": 0.0,
                "nemotron_ms": 5897.0,
                "execution_ms": 2.0,
                "total_ms": 64447.0,
            },
        )
        self.assertEqual(payload["features_ms"], 70.0)
        self.assertEqual(payload["decision_ms"], 64447.0)
        self.assertEqual(payload["cycle_total_ms"], 64517.0)
        self.assertNotIn("total_ms", payload)

    def test_phi3_reflex_cache_key_ignores_micro_book_noise_within_bar(self) -> None:
        features_a = {
            "bar_ts": "2026-03-31T21:30:00Z",
            "spread_pct": 0.12,
            "book_valid": True,
        }
        features_b = {
            "bar_ts": "2026-03-31T21:30:00Z",
            "spread_pct": 0.48,
            "book_valid": False,
        }
        key_a = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-123", features=features_a)
        key_b = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-123", features=features_b)
        self.assertEqual(key_a, key_b)

    def test_phi3_reflex_cache_key_changes_when_bar_changes(self) -> None:
        features = {
            "bar_ts": "2026-03-31T21:30:00Z",
            "spread_pct": 0.12,
            "book_valid": True,
        }
        key_a = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-123", features=features)
        key_b = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-124", features=features)
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
