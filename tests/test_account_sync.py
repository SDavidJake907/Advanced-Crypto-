import os
import unittest
from unittest.mock import patch

from core.data.account_sync import bootstrap_account_state
from core.config.runtime import get_runtime_setting as _real_get_runtime_setting


class _FakeKrakenRestClient:
    def get_asset_pairs(self) -> dict:
        return {
            "result": {
                "XXBTZUSD": {
                    "altname": "XBTUSD",
                    "wsname": "XBT/USD",
                    "base": "XXBT",
                    "quote": "ZUSD",
                },
                "XETHZUSD": {
                    "altname": "ETHUSD",
                    "wsname": "ETH/USD",
                    "base": "XETH",
                    "quote": "ZUSD",
                },
            }
        }

    def get_ticker(self, pairs: list[str]) -> dict:
        return {
            "result": {
                "XBTUSD": {"c": ["100.0"]},
                "ETHUSD": {"c": ["200.0"]},
            }
        }

    def get_balance_ex(self) -> dict:
        return {
            "result": {
                "ZUSD": "10.0",
                "XXBT": "0.1",
                "XETH": "0.2",
            }
        }

    def get_balance(self) -> dict:
        raise AssertionError("get_balance should not be called when BalanceEx succeeds")

    def get_trade_balance(self, asset: str) -> dict:
        return {"result": {"e": "12.34"}}

    def get_trades_history(self) -> dict:
        raise RuntimeError("EAPI:Invalid key")

    def get_ledgers(self, type_filter: str = "all") -> dict:
        return {"result": {"ledger": {}}}


class AccountSyncTests(unittest.TestCase):
    @patch.dict(os.environ, {"KRAKEN_TRADE_HISTORY_ZIP": ""})
    def test_bootstrap_prefers_spot_equity_and_seeds_position_marks(self) -> None:
        with patch("core.data.account_sync.get_runtime_setting") as mock_setting:
            mock_setting.side_effect = lambda key, default=None: True if key == "ACCOUNT_SYNC_USE_TRADES_HISTORY" else _real_get_runtime_setting(key, default)
            bootstrap = bootstrap_account_state(
                client=_FakeKrakenRestClient(),
                symbols=["BTC/USD", "ETH/USD"],
                dust_usd_threshold=0.0,
            )

        self.assertAlmostEqual(bootstrap.cash_usd, 10.0)
        self.assertAlmostEqual(bootstrap.initial_equity_usd, 60.0)
        self.assertEqual(
            bootstrap.portfolio_state.position_marks,
            {"BTC/USD": 100.0, "ETH/USD": 200.0},
        )
        self.assertEqual(bootstrap.diagnostics["trade_balance_equity_usd"], 12.34)
        self.assertEqual(bootstrap.diagnostics["spot_equity_usd"], 60.0)
        self.assertIn("trades_history_error", bootstrap.diagnostics)

    @patch.dict(os.environ, {"KRAKEN_TRADE_HISTORY_ZIP": ""})
    def test_bootstrap_can_run_in_ledger_only_mode_without_trades_history_error(self) -> None:
        with patch("core.data.account_sync.get_runtime_setting") as mock_setting:
            mock_setting.side_effect = lambda key, default=None: False if key == "ACCOUNT_SYNC_USE_TRADES_HISTORY" else _real_get_runtime_setting(key, default)
            bootstrap = bootstrap_account_state(
                client=_FakeKrakenRestClient(),
                symbols=["BTC/USD", "ETH/USD"],
                dust_usd_threshold=0.0,
            )

        self.assertEqual(bootstrap.diagnostics["trades_history_source"], "disabled_ledger_only")
        self.assertNotIn("trades_history_error", bootstrap.diagnostics)


if __name__ == "__main__":
    unittest.main()
