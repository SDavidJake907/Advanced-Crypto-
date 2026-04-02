from __future__ import annotations

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
