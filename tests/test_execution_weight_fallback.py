from __future__ import annotations

import unittest
from unittest.mock import patch

from core.config.runtime import get_effective_min_notional_usd, get_proposed_weight
from core.execution.mock_exec import MockExecutor


class ExecutionWeightFallbackTests(unittest.TestCase):
    def test_mock_executor_uses_runtime_weight_when_proposed_weight_missing(self) -> None:
        executor = MockExecutor()
        features = {
            "price": 10.0,
            "lane": "L3",
            "spread_pct": 0.01,
            "atr": 0.1,
            "trend_confirmed": True,
            "tp_after_cost_valid": True,
            "expected_move_pct": 3.0,
            "fees_pct": 0.1,
            "slippage_pct": 0.05,
            "stop_pct": 4.0,
        }
        state = {"cash": 100.0}

        with patch("core.execution.mock_exec.get_runtime_setting") as mock_get_runtime_setting:
            def _runtime_value(name: str):
                values = {
                    "EXEC_MIN_NOTIONAL_USD": 15.0,
                    "EXEC_RISK_PER_TRADE_PCT": 25.0,
                    "EXEC_MIN_TRADE_RISK_BUDGET_MULT": 1.5,
                }
                return values[name]

            mock_get_runtime_setting.side_effect = _runtime_value
            with patch("core.execution.mock_exec.get_proposed_weight", return_value=0.5):
                result = executor.execute("LONG", "ALGO/USD", features, state)

        self.assertEqual(result["status"], "filled")
        self.assertAlmostEqual(float(result["notional"]), 50.0, places=6)

    def test_get_proposed_weight_uses_lane_specific_risk_pct(self) -> None:
        with patch("core.config.runtime.get_runtime_setting") as mock_get_runtime_setting:
            def _runtime_value(name: str):
                values = {
                    "L1_EXEC_RISK_PER_TRADE_PCT": 40.0,
                    "L2_EXEC_RISK_PER_TRADE_PCT": 25.0,
                    "EXEC_RISK_PER_TRADE_PCT": 10.0,
                    "MEME_EXEC_RISK_PER_TRADE_PCT": 8.0,
                    "TRADER_PROPOSED_WEIGHT": 0.15,
                    "MEME_PROPOSED_WEIGHT": 0.03,
                }
                return values[name]

            mock_get_runtime_setting.side_effect = _runtime_value
            self.assertAlmostEqual(get_proposed_weight(lane="L1"), 0.40, places=6)
            self.assertAlmostEqual(get_proposed_weight(lane="L2"), 0.25, places=6)
            self.assertAlmostEqual(get_proposed_weight(lane="L3"), 0.10, places=6)
            self.assertAlmostEqual(get_proposed_weight(lane="L4"), 0.08, places=6)

    def test_get_effective_min_notional_uses_lane_equity_floor(self) -> None:
        with patch("core.config.runtime.get_runtime_setting") as mock_get_runtime_setting:
            def _runtime_value(name: str):
                values = {
                    "EXEC_MIN_NOTIONAL_USD": 15.0,
                    "MEME_EXEC_MIN_NOTIONAL_USD": 15.0,
                    "L1_EXEC_MIN_NOTIONAL_PCT_EQUITY": 0.08,
                    "L2_EXEC_MIN_NOTIONAL_PCT_EQUITY": 0.07,
                    "L3_EXEC_MIN_NOTIONAL_PCT_EQUITY": 0.06,
                    "MEME_EXEC_MIN_NOTIONAL_PCT_EQUITY": 0.03,
                }
                return values[name]

            mock_get_runtime_setting.side_effect = _runtime_value
            self.assertAlmostEqual(get_effective_min_notional_usd(equity_usd=657.0, lane="L1"), 52.56, places=2)
            self.assertAlmostEqual(get_effective_min_notional_usd(equity_usd=657.0, lane="L2"), 45.99, places=2)
            self.assertAlmostEqual(get_effective_min_notional_usd(equity_usd=657.0, lane="L3"), 39.42, places=2)
            self.assertAlmostEqual(get_effective_min_notional_usd(equity_usd=657.0, lane="L4"), 19.71, places=2)

    def test_mock_executor_clamps_up_to_lane_scaled_min_notional(self) -> None:
        executor = MockExecutor()
        features = {
            "price": 10.0,
            "lane": "L3",
            "spread_pct": 0.01,
            "atr": 0.1,
            "trend_confirmed": True,
            "tp_after_cost_valid": True,
            "expected_move_pct": 3.0,
            "fees_pct": 0.1,
            "slippage_pct": 0.05,
            "stop_pct": 4.0,
        }
        state = {"cash": 657.0}

        with patch("core.execution.mock_exec.get_runtime_setting") as mock_get_runtime_setting:
            def _runtime_value(name: str):
                values = {
                    "EXEC_RISK_PER_TRADE_PCT": 10.0,
                    "EXEC_MIN_TRADE_RISK_BUDGET_MULT": 1.5,
                }
                return values[name]

            mock_get_runtime_setting.side_effect = _runtime_value
            with patch("core.execution.mock_exec.get_proposed_weight", return_value=0.01):
                with patch("core.execution.mock_exec.get_effective_min_notional_usd", return_value=39.42):
                    result = executor.execute("LONG", "ALGO/USD", features, state)

        self.assertEqual(result["status"], "filled")
        self.assertAlmostEqual(float(result["notional"]), 39.42, places=2)


if __name__ == "__main__":
    unittest.main()
