import unittest
from unittest.mock import patch

from core.llm.nemotron import NemotronStrategist
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, Position, PositionState
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy


class NemotronSummaryTests(unittest.TestCase):
    def test_portfolio_summary_exposes_risk_budget_and_gross_exposure(self) -> None:
        strategist = NemotronStrategist(
            strategy=SimpleMomentumStrategy(),
            risk_engine=BasicRiskEngine(max_position_notional=1_000_000.0, max_leverage=10.0, cooldown_bars=0),
            portfolio_config=PortfolioConfig(max_open_positions=6, max_total_gross_exposure=0.95),
            executor=object(),  # type: ignore[arg-type]
        )
        positions_state = PositionState()
        positions_state.add_or_update(Position(symbol="BTC/USD", side="LONG", weight=0.15, entry_price=100.0))
        positions_state.add_or_update(Position(symbol="ETH/USD", side="LONG", weight=0.2, entry_price=200.0))
        portfolio_state = PortfolioState(
            cash=500.0,
            positions={"BTC/USD": 1.0, "ETH/USD": 2.0},
            position_marks={"BTC/USD": 100.0, "ETH/USD": 200.0},
        )

        summary = strategist._build_portfolio_summary(
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            current_price=100.0,
            current_symbol="BTC/USD",
        )

        self.assertEqual(summary["gross_exposure_pct"], 0.35)
        self.assertEqual(summary["gross_exposure_headroom_pct"], 0.6)
        self.assertEqual(summary["max_total_gross_exposure_pct"], 0.95)
        self.assertGreater(summary["risk_budget_usd"], 0.0)

    def test_specific_hold_reason_no_longer_flags_standard_lane_volume_too_light_at_point8x(self) -> None:
        strategist = NemotronStrategist(
            strategy=SimpleMomentumStrategy(),
            risk_engine=BasicRiskEngine(max_position_notional=1_000_000.0, max_leverage=10.0, cooldown_bars=0),
            portfolio_config=PortfolioConfig(max_open_positions=6, max_total_gross_exposure=0.95),
            executor=object(),  # type: ignore[arg-type]
        )
        with patch("core.config.runtime.load_runtime_overrides", return_value={"NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7}):
            reason = strategist._specific_hold_reason(
                features={
                    "symbol": "ONDO/USD",
                    "lane": "L2",
                    "reversal_risk": "MEDIUM",
                    "net_edge_pct": 0.35,
                    "volume_ratio": 0.82,
                    "ranging_market": True,
                    "trend_confirmed": True,
                    "momentum_5": 0.004,
                    "short_tf_ready_15m": True,
                },
                market_state_review={
                    "breakout_state": "inside_range",
                    "trend_stage": "confirmed",
                    "pattern_explanation": {"structure_phase": "unclear"},
                },
                portfolio_decision={"decision": "allow"},
                current_reason="reason_missing",
            )
        self.assertNotEqual(reason, "volume_too_light")
        self.assertEqual(reason, "pattern_not_confirmed")


if __name__ == "__main__":
    unittest.main()
