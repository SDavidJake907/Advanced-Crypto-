import unittest

from core.data.account_sync import _compute_entry_basis
from core.policy.nemotron_gate import should_run_nemotron
from core.policy.regime_state import _MACHINES, update_regime_state
from core.risk.portfolio import PositionState


class _FakeClient:
    def __init__(self, trades: dict[str, dict[str, object]]) -> None:
        self._trades = trades

    def get_trades_history(self) -> dict[str, object]:
        return {"result": {"trades": self._trades}}


class HighPriorityRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        _MACHINES.clear()

    def test_regime_state_resets_confirmation_when_raw_matches_state(self) -> None:
        bullish = {"trend_1h": 1, "regime_7d": "trending", "macro_30d": "bull", "market_regime": {"breadth": 0.5}}
        bearish = {"trend_1h": -1, "macro_30d": "bear", "market_regime": {"breadth": 0.5}}

        update_regime_state("TEST/USD", bullish)
        update_regime_state("TEST/USD", bullish)
        drift = update_regime_state("TEST/USD", bearish)
        self.assertEqual(drift["regime_confirm_count"], 1)

        recovered = update_regime_state("TEST/USD", bullish)
        self.assertEqual(recovered["regime_state"], "bullish")
        self.assertEqual(recovered["regime_confirm_count"], 0)

    def test_should_run_nemotron_allows_promoted_watch_candidates(self) -> None:
        allowed = should_run_nemotron(
            symbol="TEST/USD",
            features={
                "entry_recommendation": "WATCH",
                "reversal_risk": "MEDIUM",
                "entry_score": 70.0,
                "rotation_score": 0.2,
                "momentum_5": 0.01,
                "trend_confirmed": True,
            },
            positions_state=PositionState(),
            universe_context={"current_symbol_is_top_candidate": False},
            candidate_review={"promotion_decision": "promote", "priority": 0.8},
        )
        self.assertTrue(allowed)

    def test_account_sync_zeroes_cost_on_full_sell(self) -> None:
        client = _FakeClient(
            {
                "1": {"pair": "XBTUSD", "type": "buy", "vol": "1.0", "price": "100.0", "time": 1},
                "2": {"pair": "XBTUSD", "type": "sell", "vol": "2.0", "price": "120.0", "time": 2},
            }
        )
        entry = _compute_entry_basis(
            client=client,
            symbols=["BTC/USD"],
            asset_pairs={"XXBTZUSD": {"wsname": "BTC/USD", "altname": "XBTUSD"}},
            diagnostics={},
        )
        self.assertEqual(entry, {})


if __name__ == "__main__":
    unittest.main()
