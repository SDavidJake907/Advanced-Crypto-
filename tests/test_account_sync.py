import unittest

from core.data.account_sync import bootstrap_account_state


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
    def test_bootstrap_prefers_spot_equity_and_seeds_position_marks(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
