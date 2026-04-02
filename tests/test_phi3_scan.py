import unittest
from unittest.mock import patch

from core.llm.phi3_scan import ScanCandidate, _heuristic_scan, phi3_scan_market


class Phi3ScanTests(unittest.TestCase):
    def test_heuristic_scan_ranks_by_final_score_not_golden_profile(self) -> None:
        candidates = [
            ScanCandidate(
                symbol="LOWER/USD",
                lane="L3",
                momentum_5=0.01,
                momentum_14=0.01,
                momentum_30=0.01,
                rotation_score=0.4,
                rsi=56.0,
                volume=500_000.0,
                trend_1h=1,
                regime_7d="trending",
                macro_30d="bull",
                entry_score=58.0,
                entry_recommendation="BUY",
                reversal_risk="LOW",
                net_edge_pct=1.0,
                tp_after_cost_valid=True,
                final_score=60.0,
                golden_profile_score=25.0,
                golden_profile_tag="golden",
            ),
            ScanCandidate(
                symbol="HIGHER/USD",
                lane="L3",
                momentum_5=0.008,
                momentum_14=0.009,
                momentum_30=0.01,
                rotation_score=0.35,
                rsi=54.0,
                volume=400_000.0,
                trend_1h=1,
                regime_7d="trending",
                macro_30d="bull",
                entry_score=56.0,
                entry_recommendation="BUY",
                reversal_risk="LOW",
                net_edge_pct=0.5,
                tp_after_cost_valid=True,
                final_score=68.0,
                golden_profile_score=2.0,
                golden_profile_tag="fragile",
            ),
        ]

        result = _heuristic_scan(candidates)
        self.assertEqual(result.watchlist[0].symbol, "HIGHER/USD")
        self.assertEqual(result.watchlist[0].reason, "heuristic_scan:final_score")

    def test_heuristic_scan_blocks_bad_edge_even_with_high_golden_profile(self) -> None:
        candidates = [
            ScanCandidate(
                symbol="BLOCKED/USD",
                lane="L3",
                momentum_5=0.01,
                momentum_14=0.01,
                momentum_30=0.01,
                rotation_score=0.4,
                rsi=56.0,
                volume=500_000.0,
                trend_1h=1,
                regime_7d="trending",
                macro_30d="bull",
                entry_score=70.0,
                entry_recommendation="BUY",
                reversal_risk="LOW",
                net_edge_pct=0.2,
                tp_after_cost_valid=True,
                final_score=75.0,
                golden_profile_score=30.0,
                golden_profile_tag="golden",
            )
        ]

        result = _heuristic_scan(candidates)
        self.assertEqual(result.watchlist, [])

    def test_scan_candidate_payload_omits_golden_profile_fields(self) -> None:
        candidate = ScanCandidate(
            symbol="TEST/USD",
            lane="L3",
            momentum_5=0.01,
            momentum_14=0.01,
            momentum_30=0.01,
            rotation_score=0.4,
            rsi=56.0,
            volume=500_000.0,
            trend_1h=1,
            regime_7d="trending",
            macro_30d="bull",
            golden_profile_score=30.0,
            golden_profile_tag="golden",
        )

        payload = candidate.to_dict()
        self.assertNotIn("golden_profile_score", payload)
        self.assertNotIn("golden_profile_tag", payload)

    def test_phi3_scan_market_prefers_lesson_summary_payload_field(self) -> None:
        candidate = ScanCandidate(
            symbol="TEST/USD",
            lane="L3",
            momentum_5=0.01,
            momentum_14=0.01,
            momentum_30=0.01,
            rotation_score=0.4,
            rsi=56.0,
            volume=500_000.0,
            trend_1h=1,
            regime_7d="trending",
            macro_30d="bull",
        )

        def fake_phi3_advisory_chat(payload, *, system, max_tokens):
            self.assertIn("lesson_summary", payload)
            self.assertEqual(payload["lesson_summary"], ["keep sizes smaller", "avoid late chase"])
            self.assertNotIn("learned_lessons", payload)
            self.assertNotIn("golden_profile_score", payload["candidates"][0])
            self.assertNotIn("golden_profile_tag", payload["candidates"][0])
            return '{"watchlist":[],"market_note":"ok"}'

        with patch("core.llm.phi3_scan.phi3_advisory_chat", side_effect=fake_phi3_advisory_chat):
            result = phi3_scan_market(
                [candidate],
                lesson_summary=["keep sizes smaller", "avoid late chase"],
                learned_lessons="=== RECENT ACCOUNT LESSONS ===\n- old format\n",
            )

        self.assertEqual(result.watchlist, [])
        self.assertEqual(result.market_note, "ok")


if __name__ == "__main__":
    unittest.main()
