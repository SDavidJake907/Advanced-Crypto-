import os
import unittest
from unittest.mock import patch
import asyncio

from apps.universe_manager.main import (
    Candidate,
    _apply_candidate_final_score,
    _candidate_final_score_features,
    _lane_rank_score,
    _refresh_interval_sec,
    _universe_symbol_allowed,
    build_scan_meta,
    rebalance_universe,
)
from core.policy.universe_policy import UniversePolicy
from core.policy.final_score import compute_final_score


def _fake_save_universe(active_pairs, reason, meta=None):
    return {
        "active_pairs": active_pairs,
        "reason": reason,
        "meta": meta or {},
    }


async def _fake_build_scan_meta(*_args, **_kwargs):
    return {}


class _FakeNewsFeed:
    def __init__(self, *args, **kwargs):
        self.snapshot = self

    async def maybe_update(self):
        return None

    def to_dict(self):
        return {}


class _FakeDexFeed:
    async def fetch_market_summary(self, *_args, **_kwargs):
        return {}


class UniverseSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.copy()
        self._runtime_patch = patch("core.config.runtime.load_runtime_overrides", return_value={})
        self._runtime_patch.start()

    def tearDown(self) -> None:
        self._runtime_patch.stop()
        os.environ.clear()
        os.environ.update(self._env)

    def test_universe_symbol_allowed_blocks_memes_when_disabled(self) -> None:
        os.environ["MEME_UNIVERSE_ENABLED"] = "false"
        os.environ["MEME_SYMBOLS"] = "WIF/USD"
        self.assertFalse(_universe_symbol_allowed("WIF/USD"))
        self.assertTrue(_universe_symbol_allowed("BTC/USD"))

    def test_universe_symbol_allowed_blocks_memes_when_runtime_override_disabled(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={"MEME_UNIVERSE_ENABLED": False}):
            os.environ["MEME_SYMBOLS"] = "WIF/USD"
            self.assertFalse(_universe_symbol_allowed("WIF/USD"))
            self.assertTrue(_universe_symbol_allowed("BTC/USD"))

    def test_universe_symbol_allowed_restricts_to_core_and_conditional_seed(self) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,SOL/USD,LINK/USD"
        os.environ["CONDITIONAL_UNIVERSE"] = "XRP/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "true"

        self.assertTrue(_universe_symbol_allowed("BTC/USD"))
        self.assertTrue(_universe_symbol_allowed("XRP/USD"))
        self.assertFalse(_universe_symbol_allowed("ADA/USD"))

    def test_refresh_interval_uses_floor(self) -> None:
        os.environ["UNIVERSE_REFRESH_INTERVAL_MIN"] = "2"
        os.environ["UNIVERSE_MIN_REFRESH_INTERVAL_MIN"] = "5"
        self.assertEqual(_refresh_interval_sec(), 300)

    def test_candidate_final_score_snapshot_matches_shared_final_score_logic(self) -> None:
        candidate = Candidate(
            pair="ETH/USD",
            score=0.03,
            volume_usd=1_000_000,
            last=3000.0,
            lane="L3",
            candidate_score=62.5,
            spread_bps=40.0,
            net_edge_pct=0.45,
            bullish_divergence=True,
            divergence_strength=0.5,
            divergence_age_bars=2,
        )

        expected = compute_final_score(
            _candidate_final_score_features(candidate),
            reliability_map={"ETH/USD": {"win_rate": 0.61, "trade_count": 10}},
        )
        _apply_candidate_final_score(candidate, reliability_map={"ETH/USD": {"win_rate": 0.61, "trade_count": 10}})

        self.assertEqual(candidate.final_score, expected.final_score)
        self.assertEqual(candidate.reliability_bonus, expected.reliability_bonus)
        self.assertEqual(candidate.basket_fit_bonus, expected.basket_fit_bonus)
        self.assertEqual(candidate.score_breakdown, expected.score_breakdown)

    def test_lane_rank_score_ignores_golden_profile_bonus(self) -> None:
        base = Candidate(
            pair="ETH/USD",
            score=0.08,
            volume_usd=1_000_000,
            last=3000.0,
            lane="L3",
            candidate_score=64.0,
            net_edge_pct=0.6,
            trade_quality=72.0,
            structure_quality=70.0,
            continuation_quality=68.0,
            risk_quality=66.0,
            momentum_quality=69.0,
            volume_quality=62.0,
            market_support=61.0,
            momentum_14=0.02,
            momentum_30=0.015,
            m8h=0.01,
            m24h=0.02,
            volume_ratio=1.2,
            rsi=58.0,
            spread_bps=6.0,
            tp_after_cost_valid=True,
        )
        boosted = Candidate(**{**base.__dict__, "golden_profile_score": 25.0})

        self.assertEqual(_lane_rank_score(base), _lane_rank_score(boosted))

    @patch("apps.universe_manager.main.phi3_supervise_lanes", return_value=[])
    @patch("apps.universe_manager.main.phi3_scan_market")
    @patch("apps.universe_manager.main.DexScreenerFeed", return_value=_FakeDexFeed())
    @patch("apps.universe_manager.main.NewsSentimentFeed", return_value=_FakeNewsFeed())
    def test_build_scan_meta_uses_candidate_rotation_score_for_phi3_packet(
        self,
        _news_feed,
        _dex_feed,
        mock_phi3_scan,
        _phi3_supervise,
    ) -> None:
        mock_phi3_scan.return_value = type("ScanResult", (), {"to_dict": lambda self: {"watchlist": [], "market_note": "ok"}})()
        candidate = Candidate(
            pair="ETH/USD",
            score=0.08,
            volume_usd=1_000_000,
            last=3000.0,
            lane="L3",
            candidate_score=67.0,
            candidate_recommendation="BUY",
            candidate_risk="LOW",
            net_edge_pct=0.7,
            tp_after_cost_valid=True,
            structure_quality=70.0,
            continuation_quality=68.0,
            risk_quality=66.0,
            trade_quality=71.0,
            final_score=69.0,
            reliability_bonus=2.0,
            basket_fit_bonus=0.0,
            score_breakdown={"entry_score": 67.0},
        )

        asyncio.run(build_scan_meta([candidate], {"L3": ["ETH/USD"]}))

        scan_candidates = mock_phi3_scan.call_args.args[0]
        self.assertEqual(len(scan_candidates), 1)
        self.assertAlmostEqual(scan_candidates[0].rotation_score, 0.08)

    @patch("apps.universe_manager.main.build_scan_meta", side_effect=_fake_build_scan_meta)
    @patch("apps.universe_manager.main.load_synced_position_symbols", return_value=[])
    @patch("apps.universe_manager.main.load_universe", return_value={"active_pairs": [], "meta": {}})
    @patch("apps.universe_manager.main.save_universe", side_effect=_fake_save_universe)
    def test_rebalance_universe_caps_meme_slots(
        self,
        _save_universe,
        _load_universe,
        _load_synced,
        _build_scan_meta,
    ) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,LINK/USD,WIF/USD,BONK/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "false"
        os.environ["MEME_UNIVERSE_ENABLED"] = "true"
        os.environ["MEME_ACTIVE_UNIVERSE_MAX"] = "1"
        os.environ["MEME_SYMBOLS"] = "WIF/USD,BONK/USD"
        os.environ["ACTIVE_MIN"] = "2"
        os.environ["ACTIVE_MAX"] = "4"
        os.environ["ROTATION_SHORTLIST_SIZE"] = "4"

        policy = UniversePolicy(
            active_min=2,
            active_max=4,
            max_adds=5,
            max_removes=5,
            cooldown_minutes=180,
            min_volume_usd=200_000,
            max_spread_bps=12.0,
            min_price=0.05,
            churn_threshold=1.2,
        )
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=60000, lane="L1", candidate_score=75.0, rank_score=75.0),
            Candidate(pair="ETH/USD", score=0.025, volume_usd=900_000, last=3000, lane="L1", candidate_score=72.0, rank_score=72.0),
            Candidate(pair="LINK/USD", score=0.02, volume_usd=400_000, last=12.0, lane="L3", candidate_score=68.0, rank_score=68.0),
            Candidate(pair="WIF/USD", score=0.04, volume_usd=350_000, last=2.5, lane="L4", candidate_score=80.0, rank_score=80.0),
            Candidate(pair="BONK/USD", score=0.05, volume_usd=325_000, last=0.001, lane="L4", candidate_score=78.0, rank_score=78.0),
        ]

        result = rebalance_universe(candidates, policy)
        meme_pairs = [pair for pair in result["active_pairs"] if pair in {"WIF/USD", "BONK/USD"}]
        self.assertEqual(len(meme_pairs), 1)

    @patch("apps.universe_manager.main.build_scan_meta", side_effect=_fake_build_scan_meta)
    @patch("apps.universe_manager.main.load_synced_position_symbols", return_value=[])
    @patch("apps.universe_manager.main.load_universe", return_value={"active_pairs": [], "meta": {}})
    @patch("apps.universe_manager.main.save_universe", side_effect=_fake_save_universe)
    def test_rebalance_treats_l4_candidate_as_meme_even_if_not_listed(
        self,
        _save_universe,
        _load_universe,
        _load_synced,
        _build_scan_meta,
    ) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,PLAY/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "false"
        os.environ["MEME_UNIVERSE_ENABLED"] = "false"
        os.environ["MEME_SYMBOLS"] = "WIF/USD"
        os.environ["ACTIVE_MIN"] = "2"
        os.environ["ACTIVE_MAX"] = "3"
        os.environ["ROTATION_SHORTLIST_SIZE"] = "3"

        policy = UniversePolicy(
            active_min=2,
            active_max=3,
            max_adds=5,
            max_removes=5,
            cooldown_minutes=180,
            min_volume_usd=200_000,
            max_spread_bps=12.0,
            min_price=0.05,
            churn_threshold=1.2,
        )
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=60000, lane="L1", candidate_score=80.0, rank_score=80.0),
            Candidate(pair="ETH/USD", score=0.025, volume_usd=900_000, last=3000, lane="L3", candidate_score=75.0, rank_score=75.0),
            Candidate(pair="PLAY/USD", score=0.04, volume_usd=350_000, last=0.25, lane="L4", candidate_score=82.0, rank_score=82.0),
        ]

        result = rebalance_universe(candidates, policy)
        self.assertNotIn("PLAY/USD", result["active_pairs"])
        self.assertEqual(result["meta"].get("segment_tags", {}).get("PLAY/USD"), "meme")

    @patch("apps.universe_manager.main.build_scan_meta", side_effect=_fake_build_scan_meta)
    @patch("apps.universe_manager.main.load_synced_position_symbols", return_value=[])
    @patch("apps.universe_manager.main.load_universe", return_value={"active_pairs": ["ADA/USD"], "meta": {}})
    @patch("apps.universe_manager.main.save_universe", side_effect=_fake_save_universe)
    def test_rebalance_universe_retains_buffered_incumbent(
        self,
        _save_universe,
        _load_universe,
        _load_synced,
        _build_scan_meta,
    ) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,SOL/USD,ADA/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "false"
        os.environ["MEME_UNIVERSE_ENABLED"] = "false"
        os.environ["UNIVERSE_RETAIN_BUFFER_MULT"] = "1.5"
        os.environ["ACTIVE_MIN"] = "3"
        os.environ["ACTIVE_MAX"] = "3"
        os.environ["ROTATION_SHORTLIST_SIZE"] = "3"

        policy = UniversePolicy(
            active_min=3,
            active_max=3,
            max_adds=5,
            max_removes=5,
            cooldown_minutes=180,
            min_volume_usd=200_000,
            max_spread_bps=12.0,
            min_price=0.05,
            churn_threshold=1.2,
        )
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=60000, lane="L1", candidate_score=80.0, rank_score=80.0),
            Candidate(pair="ETH/USD", score=0.025, volume_usd=900_000, last=3000, lane="L1", candidate_score=78.0, rank_score=78.0),
            Candidate(pair="SOL/USD", score=0.024, volume_usd=850_000, last=150.0, lane="L3", candidate_score=76.0, rank_score=76.0),
            Candidate(pair="ADA/USD", score=0.022, volume_usd=700_000, last=0.7, lane="L3", candidate_score=74.0, rank_score=74.0),
        ]

        result = rebalance_universe(candidates, policy)
        self.assertIn("ADA/USD", result["active_pairs"])

    @patch("apps.universe_manager.main.build_scan_meta", side_effect=_fake_build_scan_meta)
    @patch("apps.universe_manager.main.load_synced_position_symbols", return_value=[])
    @patch("apps.universe_manager.main.load_universe", return_value={"active_pairs": [], "meta": {}})
    @patch("apps.universe_manager.main.save_universe", side_effect=_fake_save_universe)
    def test_rebalance_universe_restricts_to_core_symbols_by_default(
        self,
        _save_universe,
        _load_universe,
        _load_synced,
        _build_scan_meta,
    ) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,SOL/USD,LINK/USD"
        os.environ["CONDITIONAL_UNIVERSE"] = "XRP/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "false"
        os.environ["ACTIVE_MIN"] = "4"
        os.environ["ACTIVE_MAX"] = "4"
        os.environ["ROTATION_SHORTLIST_SIZE"] = "4"

        policy = UniversePolicy(
            active_min=4,
            active_max=4,
            max_adds=5,
            max_removes=5,
            cooldown_minutes=180,
            min_volume_usd=200_000,
            max_spread_bps=12.0,
            min_price=0.05,
            churn_threshold=1.2,
        )
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=60000, lane="L1", candidate_score=80.0, rank_score=80.0),
            Candidate(pair="ETH/USD", score=0.025, volume_usd=900_000, last=3000, lane="L1", candidate_score=78.0, rank_score=78.0),
            Candidate(pair="SOL/USD", score=0.024, volume_usd=850_000, last=150.0, lane="L3", candidate_score=76.0, rank_score=76.0),
            Candidate(pair="LINK/USD", score=0.02, volume_usd=400_000, last=12.0, lane="L3", candidate_score=74.0, rank_score=74.0),
            Candidate(pair="ADA/USD", score=0.021, volume_usd=500_000, last=0.7, lane="L3", candidate_score=82.0, rank_score=82.0),
        ]

        result = rebalance_universe(candidates, policy)

        self.assertEqual(result["active_pairs"], ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD"])

    @patch("apps.universe_manager.main.build_scan_meta", side_effect=_fake_build_scan_meta)
    @patch("apps.universe_manager.main.load_synced_position_symbols", return_value=[])
    @patch("apps.universe_manager.main.load_universe", return_value={"active_pairs": [], "meta": {}})
    @patch("apps.universe_manager.main.save_universe", side_effect=_fake_save_universe)
    def test_rebalance_excludes_xrp_by_default_when_conditional_checks_fail(
        self,
        _save_universe,
        _load_universe,
        _load_synced,
        _build_scan_meta,
    ) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,SOL/USD,LINK/USD"
        os.environ["CONDITIONAL_UNIVERSE"] = "XRP/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "true"
        os.environ["ACTIVE_MIN"] = "4"
        os.environ["ACTIVE_MAX"] = "5"
        os.environ["ROTATION_SHORTLIST_SIZE"] = "5"

        policy = UniversePolicy(
            active_min=4,
            active_max=5,
            max_adds=5,
            max_removes=5,
            cooldown_minutes=180,
            min_volume_usd=200_000,
            max_spread_bps=12.0,
            min_price=0.05,
            churn_threshold=1.2,
        )
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=60000, lane="L1", candidate_score=80.0, rank_score=80.0, structure_quality=78.0, net_edge_pct=1.4, spread_bps=10.0, tp_after_cost_valid=True),
            Candidate(pair="ETH/USD", score=0.025, volume_usd=900_000, last=3000, lane="L1", candidate_score=78.0, rank_score=78.0, structure_quality=76.0, net_edge_pct=1.3, spread_bps=11.0, tp_after_cost_valid=True),
            Candidate(pair="SOL/USD", score=0.024, volume_usd=850_000, last=150.0, lane="L3", candidate_score=76.0, rank_score=76.0, structure_quality=75.0, net_edge_pct=1.2, spread_bps=12.0, tp_after_cost_valid=True),
            Candidate(pair="LINK/USD", score=0.02, volume_usd=400_000, last=12.0, lane="L3", candidate_score=74.0, rank_score=74.0, structure_quality=72.0, net_edge_pct=1.1, spread_bps=13.0, tp_after_cost_valid=True),
            Candidate(pair="XRP/USD", score=0.021, volume_usd=700_000, last=0.7, lane="L3", candidate_score=79.0, rank_score=79.0, structure_quality=55.0, net_edge_pct=0.4, spread_bps=40.0, tp_after_cost_valid=False),
        ]

        result = rebalance_universe(candidates, policy)

        self.assertNotIn("XRP/USD", result["active_pairs"])

    @patch("apps.universe_manager.main.build_scan_meta", side_effect=_fake_build_scan_meta)
    @patch("apps.universe_manager.main.load_synced_position_symbols", return_value=[])
    @patch("apps.universe_manager.main.load_universe", return_value={"active_pairs": [], "meta": {}})
    @patch("apps.universe_manager.main.save_universe", side_effect=_fake_save_universe)
    def test_rebalance_includes_xrp_when_conditional_checks_pass(
        self,
        _save_universe,
        _load_universe,
        _load_synced,
        _build_scan_meta,
    ) -> None:
        os.environ["CORE_ACTIVE_UNIVERSE"] = "BTC/USD,ETH/USD,SOL/USD,LINK/USD"
        os.environ["CONDITIONAL_UNIVERSE"] = "XRP/USD"
        os.environ["ENABLE_CONDITIONAL_UNIVERSE"] = "true"
        os.environ["ACTIVE_MIN"] = "4"
        os.environ["ACTIVE_MAX"] = "5"
        os.environ["ROTATION_SHORTLIST_SIZE"] = "5"
        os.environ["XRP_MIN_STRUCTURE_QUALITY"] = "0.60"
        os.environ["XRP_MIN_NET_EDGE_PCT"] = "1.0"
        os.environ["XRP_MAX_SPREAD_PCT"] = "0.35"
        os.environ["XRP_MAX_DUPLICATE_CORR"] = "0.85"

        policy = UniversePolicy(
            active_min=4,
            active_max=5,
            max_adds=5,
            max_removes=5,
            cooldown_minutes=180,
            min_volume_usd=200_000,
            max_spread_bps=12.0,
            min_price=0.05,
            churn_threshold=1.2,
        )
        candidates = [
            Candidate(pair="BTC/USD", score=0.03, volume_usd=1_000_000, last=60000, lane="L1", candidate_score=80.0, rank_score=80.0, trend_1h=1, momentum_14=0.02, price_zscore=0.8, rsi=60.0, structure_quality=78.0, net_edge_pct=1.4, spread_bps=10.0, tp_after_cost_valid=True),
            Candidate(pair="ETH/USD", score=0.025, volume_usd=900_000, last=3000, lane="L1", candidate_score=78.0, rank_score=78.0, trend_1h=1, momentum_14=0.018, price_zscore=0.7, rsi=58.0, structure_quality=76.0, net_edge_pct=1.3, spread_bps=11.0, tp_after_cost_valid=True),
            Candidate(pair="SOL/USD", score=0.024, volume_usd=850_000, last=150.0, lane="L3", candidate_score=76.0, rank_score=76.0, trend_1h=1, momentum_14=0.015, price_zscore=0.6, rsi=57.0, structure_quality=75.0, net_edge_pct=1.2, spread_bps=16.0, tp_after_cost_valid=True),
            Candidate(pair="LINK/USD", score=0.02, volume_usd=400_000, last=12.0, lane="L3", candidate_score=74.0, rank_score=74.0, trend_1h=1, momentum_14=0.014, price_zscore=0.55, rsi=56.0, structure_quality=72.0, net_edge_pct=1.1, spread_bps=18.0, tp_after_cost_valid=True),
            Candidate(pair="XRP/USD", score=0.021, volume_usd=700_000, last=0.7, lane="L3", candidate_score=79.0, rank_score=79.0, trend_1h=1, momentum_14=0.016, price_zscore=0.6, rsi=57.0, structure_quality=82.0, net_edge_pct=1.6, spread_bps=20.0, tp_after_cost_valid=True),
        ]

        result = rebalance_universe(candidates, policy)

        self.assertIn("XRP/USD", result["active_pairs"])


if __name__ == "__main__":
    unittest.main()
