import unittest
from unittest.mock import patch

from apps.universe_manager.main import Candidate, build_lane_shortlists
from core.policy.nemotron_gate import passes_deterministic_candidate_gate
from core.risk.portfolio import PositionState


class LaneScannerTests(unittest.TestCase):
    def test_build_lane_shortlists_preserves_lane_diversity(self) -> None:
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=70000, lane="L1", candidate_score=72.0, momentum_5=0.01, momentum_14=0.02, momentum_30=0.03, trend_1h=1, volume_ratio=1.8, spread_bps=4.0),
            Candidate(pair="QNT/USD", score=0.02, volume_usd=300_000, last=70, lane="L1", candidate_score=66.0, momentum_5=0.008, momentum_14=0.015, momentum_30=0.02, trend_1h=1, volume_ratio=1.2, spread_bps=8.0),
            Candidate(pair="APT/USD", score=0.02, volume_usd=450_000, last=1.0, lane="L2", candidate_score=61.0, momentum_5=0.009, momentum_14=0.01, price_zscore=0.9, spread_bps=7.0),
            Candidate(pair="ALGO/USD", score=0.015, volume_usd=350_000, last=0.09, lane="L2", candidate_score=57.0, momentum_5=0.007, momentum_14=0.006, price_zscore=0.82, spread_bps=9.0),
            Candidate(pair="LINK/USD", score=0.015, volume_usd=500_000, last=12.0, lane="L3", candidate_score=64.0, momentum_5=0.006, momentum_14=0.012, trend_1h=1, rsi=58.0, spread_bps=6.0),
            Candidate(pair="SOL/USD", score=0.018, volume_usd=900_000, last=150.0, lane="L3", candidate_score=62.0, momentum_5=0.007, momentum_14=0.011, trend_1h=1, rsi=55.0, spread_bps=5.0),
            Candidate(pair="WIF/USD", score=0.04, volume_usd=250_000, last=2.5, lane="L4", candidate_score=58.0, momentum_5=0.02, momentum_14=0.012, volume_ratio=1.5, spread_bps=18.0),
            Candidate(pair="BONK/USD", score=0.05, volume_usd=225_000, last=0.0012, lane="L4", candidate_score=55.0, momentum_5=0.025, momentum_14=0.01, volume_ratio=1.6, spread_bps=20.0),
        ]
        candidates[4].trade_quality = 60.0
        candidates[4].risk_quality = 58.0
        candidates[4].volume_ratio = 0.95
        candidates[5].trade_quality = 58.0
        candidates[5].risk_quality = 56.0
        candidates[5].volume_ratio = 0.9
        lane_shortlists, merged, lane_meta = build_lane_shortlists(candidates, shortlist_size=8)

        self.assertTrue(lane_shortlists["L1"])
        self.assertTrue(lane_shortlists["L2"])
        self.assertTrue(lane_shortlists["L3"])
        self.assertTrue(lane_shortlists["L4"])
        self.assertEqual(len({candidate.pair for candidate in merged}), len(merged))
        self.assertIn("BTC/USD", lane_meta["L1"])
        self.assertIn("APT/USD", lane_meta["L2"])
        self.assertIn("LINK/USD", lane_meta["L3"])
        self.assertTrue(lane_meta["L4"])
        self.assertTrue(set(lane_meta["L4"]).issubset({"WIF/USD", "BONK/USD"}))
        self.assertIn("BTC/USD", lane_meta["merged"])
        self.assertTrue(any(symbol in lane_meta["merged"] for symbol in {"WIF/USD", "BONK/USD"}))

    def test_lane_shortlisted_symbol_can_pass_gate_without_flat_top_rank(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={"STABILIZATION_STRICT_ENTRY_ENABLED": False}):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="APT/USD",
                positions_state=PositionState(),
                universe_context={
                    "current_symbol_is_top_candidate": False,
                    "lane_shortlists": {
                        "L2": [
                            {
                                "symbol": "APT/USD",
                                "lane": "L2",
                                "candidate_score": 61.0,
                                "recommendation": "BUY",
                                "risk": "MEDIUM",
                                "reasons": [],
                            }
                        ]
                    },
                },
                features={
                    "symbol": "APT/USD",
                    "lane": "L2",
                    "indicators_ready": True,
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 61.0,
                    "rotation_score": 0.12,
                    "momentum_5": 0.01,
                    "volume_ratio": 1.1,
                    "volume_surge": 0.2,
                    "trend_confirmed": True,
                    "ranging_market": False,
                    "sentiment_symbol_trending": False,
                    "macro_30d": "sideways",
                    "regime_7d": "choppy",
                    "xgb_score": 65.0,
                    "short_tf_ready_5m": True,
                    "short_tf_ready_15m": True,
                },
            )
        self.assertTrue(passed)
        self.assertEqual(reason, "passed")

    def test_l2_scanner_can_pull_near_l2_rotation_name_from_l3(self) -> None:
        candidates = [
            Candidate(
                pair="APT/USD",
                score=0.03,
                volume_usd=400_000,
                last=5.0,
                lane="L3",
                candidate_score=58.0,
                candidate_recommendation="BUY",
                momentum_5=0.002,
                momentum_14=0.005,
                volume_ratio=0.8,
                rsi=57.0,
                spread_bps=12.0,
            )
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertIn("APT/USD", lane_meta["L2"])
        self.assertTrue(lane_shortlists["L2"])

    def test_l3_scanner_prefers_stable_names_over_hot_balanced_fallbacks(self) -> None:
        candidates = [
            Candidate(
                pair="TRX/USD",
                score=0.02,
                volume_usd=500_000,
                last=0.14,
                lane="L3",
                candidate_score=59.0,
                candidate_recommendation="BUY",
                momentum_5=0.003,
                momentum_14=0.006,
                momentum_30=0.004,
                volume_ratio=0.9,
                rsi=56.0,
                spread_bps=8.0,
                trade_quality=58.0,
                risk_quality=55.0,
            ),
            Candidate(
                pair="HOT/USD",
                score=0.04,
                volume_usd=350_000,
                last=1.2,
                lane="L3",
                candidate_score=61.0,
                candidate_recommendation="BUY",
                momentum_5=0.013,
                momentum_14=0.007,
                momentum_30=0.005,
                volume_ratio=1.1,
                rsi=60.0,
                spread_bps=7.0,
                trade_quality=58.0,
                risk_quality=55.0,
            ),
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertIn("TRX/USD", lane_meta["L3"])
        self.assertNotIn("HOT/USD", lane_meta["L3"])

    def test_lane_scanner_rewards_fast_improving_leader(self) -> None:
        candidates = [
            Candidate(
                pair="ALGO/USD",
                score=0.02,
                volume_usd=300_000,
                last=0.12,
                lane="L2",
                candidate_score=58.0,
                candidate_recommendation="BUY",
                momentum_5=0.004,
                momentum_14=0.006,
                price_zscore=0.6,
                spread_bps=8.0,
                rank_delta=0,
                momentum_delta=0.0,
                volume_ratio_delta=0.0,
                rank_score=58.0,
            ),
            Candidate(
                pair="APT/USD",
                score=0.02,
                volume_usd=290_000,
                last=5.0,
                lane="L2",
                candidate_score=57.5,
                candidate_recommendation="BUY",
                momentum_5=0.005,
                momentum_14=0.006,
                price_zscore=0.5,
                spread_bps=8.0,
                rank_delta=6,
                momentum_delta=0.003,
                volume_ratio_delta=0.25,
                rank_score=68.0,
            ),
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertEqual(lane_meta["L2"][0], "APT/USD")

    def test_l3_scanner_can_prefer_cleaner_profile_over_raw_heat(self) -> None:
        candidates = [
            Candidate(
                pair="LINK/USD",
                score=0.02,
                volume_usd=500_000,
                last=12.0,
                lane="L3",
                candidate_score=60.0,
                candidate_recommendation="BUY",
                momentum_5=0.004,
                momentum_14=0.008,
                momentum_30=0.006,
                volume_ratio=1.0,
                rsi=56.0,
                spread_bps=6.0,
                structure_quality=82.0,
                trade_quality=85.0,
                continuation_quality=76.0,
                risk_quality=72.0,
            ),
            Candidate(
                pair="ALT/USD",
                score=0.03,
                volume_usd=520_000,
                last=4.0,
                lane="L3",
                candidate_score=60.0,
                candidate_recommendation="BUY",
                momentum_5=0.009,
                momentum_14=0.009,
                momentum_30=0.006,
                volume_ratio=1.1,
                rsi=60.0,
                spread_bps=6.0,
                structure_quality=58.0,
                trade_quality=57.0,
                continuation_quality=60.0,
                risk_quality=55.0,
            ),
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertEqual(lane_meta["L3"][0], "LINK/USD")

    def test_l3_scanner_rejects_weak_low_quality_watch_names(self) -> None:
        candidates = [
            Candidate(
                pair="WEAK/USD",
                score=0.01,
                volume_usd=300_000,
                last=1.0,
                lane="L3",
                candidate_score=53.0,
                candidate_recommendation="WATCH",
                candidate_risk="MEDIUM",
                momentum_5=0.003,
                momentum_14=0.004,
                volume_ratio=0.78,
                rsi=54.0,
                spread_bps=12.0,
                trade_quality=50.0,
                risk_quality=49.0,
            ),
            Candidate(
                pair="CLEAN/USD",
                score=0.02,
                volume_usd=500_000,
                last=2.0,
                lane="L3",
                candidate_score=58.0,
                candidate_recommendation="BUY",
                candidate_risk="LOW",
                momentum_5=0.004,
                momentum_14=0.007,
                volume_ratio=0.95,
                rsi=56.0,
                spread_bps=8.0,
                trade_quality=62.0,
                risk_quality=58.0,
            ),
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertIn("CLEAN/USD", lane_meta["L3"])
        self.assertNotIn("WEAK/USD", lane_meta["L3"])

    def test_l3_scanner_rejects_negative_net_edge_setup(self) -> None:
        candidates = [
            Candidate(
                pair="GOOD/USD",
                score=0.02,
                volume_usd=500_000,
                last=2.0,
                lane="L3",
                candidate_score=58.0,
                candidate_recommendation="BUY",
                candidate_risk="LOW",
                momentum_5=0.004,
                momentum_14=0.007,
                volume_ratio=0.95,
                rsi=56.0,
                spread_bps=8.0,
                trade_quality=62.0,
                risk_quality=58.0,
                net_edge_pct=1.2,
                tp_after_cost_valid=True,
            ),
            Candidate(
                pair="BAD/USD",
                score=0.03,
                volume_usd=550_000,
                last=3.0,
                lane="L3",
                candidate_score=62.0,
                candidate_recommendation="BUY",
                candidate_risk="LOW",
                momentum_5=0.005,
                momentum_14=0.008,
                volume_ratio=1.0,
                rsi=57.0,
                spread_bps=8.0,
                trade_quality=65.0,
                risk_quality=60.0,
                net_edge_pct=0.1,
                tp_after_cost_valid=False,
            ),
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertIn("GOOD/USD", lane_meta["L3"])
        self.assertNotIn("BAD/USD", lane_meta["L3"])

    def test_l2_scanner_prefers_golden_profile_over_raw_heat(self) -> None:
        candidates = [
            Candidate(
                pair="CLEAN/USD",
                score=0.02,
                volume_usd=650_000,
                last=4.0,
                lane="L2",
                candidate_score=60.0,
                candidate_recommendation="BUY",
                candidate_risk="LOW",
                momentum_5=0.006,
                momentum_14=0.010,
                volume_ratio=1.08,
                spread_bps=7.0,
                trade_quality=84.0,
                continuation_quality=78.0,
                risk_quality=74.0,
                structure_quality=72.0,
                trend_1h=1,
                momentum_alignment=4,
                net_edge_pct=1.0,
                total_cost_pct=0.7,
                expected_move_pct=2.0,
                tp_after_cost_valid=True,
                golden_profile_score=20.0,
                golden_profile_tag="golden",
            ),
            Candidate(
                pair="HOT/USD",
                score=0.03,
                volume_usd=700_000,
                last=4.5,
                lane="L2",
                candidate_score=62.0,
                candidate_recommendation="BUY",
                candidate_risk="MEDIUM",
                momentum_5=0.010,
                momentum_14=0.008,
                volume_ratio=0.9,
                spread_bps=15.0,
                trade_quality=62.0,
                continuation_quality=60.0,
                risk_quality=58.0,
                structure_quality=58.0,
                trend_1h=0,
                momentum_alignment=1,
                net_edge_pct=0.4,
                total_cost_pct=1.0,
                expected_move_pct=1.6,
                tp_after_cost_valid=True,
                golden_profile_score=3.0,
                golden_profile_tag="fragile",
            ),
        ]
        lane_shortlists, _, lane_meta = build_lane_shortlists(candidates, shortlist_size=4)
        self.assertEqual(lane_meta["L2"][0], "CLEAN/USD")


if __name__ == "__main__":
    unittest.main()
