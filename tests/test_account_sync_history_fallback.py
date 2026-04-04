from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from core.data.account_sync import _compute_entry_basis, _normalize_entry_ts
from core.config.runtime import get_runtime_setting as _real_get_runtime_setting


class AccountSyncHistoryFallbackTests(unittest.TestCase):
    def test_entry_basis_uses_ledgers_directly_when_trades_history_disabled(self) -> None:
        diagnostics: dict[str, object] = {}
        with patch("core.data.account_sync.get_runtime_setting") as mock_setting:
            mock_setting.side_effect = lambda key, default=None: False if key == "ACCOUNT_SYNC_USE_TRADES_HISTORY" else _real_get_runtime_setting(key, default)
            entry_meta = _compute_entry_basis(
                client=_FailingTradesLedgerClient(),
                symbols=["NIGHT/USD"],
                asset_pairs={"NIGHTUSD": {"wsname": "NIGHT/USD", "altname": "NIGHTUSD", "base": "NIGHT", "quote": "ZUSD"}},
                diagnostics=diagnostics,
            )

        self.assertIn("NIGHT/USD", entry_meta)
        self.assertEqual(entry_meta["NIGHT/USD"]["source"], "kraken_ledgers")
        self.assertEqual(diagnostics.get("trades_history_source"), "ledgers_fallback")
        self.assertNotIn("trades_history_error", diagnostics)

    def test_entry_basis_falls_back_to_ledgers_before_zip(self) -> None:
        diagnostics: dict[str, object] = {}
        with patch("core.data.account_sync.get_runtime_setting") as mock_setting:
            mock_setting.side_effect = lambda key, default=None: True if key == "ACCOUNT_SYNC_USE_TRADES_HISTORY" else _real_get_runtime_setting(key, default)
            entry_meta = _compute_entry_basis(
                client=_FailingTradesLedgerClient(),
                symbols=["NIGHT/USD"],
                asset_pairs={"NIGHTUSD": {"wsname": "NIGHT/USD", "altname": "NIGHTUSD", "base": "NIGHT", "quote": "ZUSD"}},
                diagnostics=diagnostics,
            )

        self.assertIn("NIGHT/USD", entry_meta)
        self.assertEqual(entry_meta["NIGHT/USD"]["source"], "kraken_ledgers")
        self.assertAlmostEqual(float(entry_meta["NIGHT/USD"]["entry_price"]), (7.0 + 0.0161) / 164.2807, places=8)
        self.assertEqual(diagnostics.get("trades_history_source"), "ledgers_fallback")
        self.assertIn("trades_history_error", diagnostics)

    def test_entry_basis_falls_back_to_zip_when_trades_history_unavailable(self) -> None:
        csv_data = "\n".join(
            [
                '"txid","ordertxid","pair","aclass","subclass","time","type","ordertype","price","cost","fee","vol","margin","misc","ledgers","posttxid","posstatuscode","cprice","ccost","cfee","cvol","cmargin","net","costusd","trades"',
                '"a","oa","ADA/USD","forex","crypto","2026-03-20 01:00:00.0000","buy","market",0.25,10.0,0.03,40.0,0.0,"","","","","","","","","","","10.0",""',
                '"b","ob","ADA/USD","forex","crypto","2026-03-20 03:00:00.0000","sell","market",0.26,2.6,0.01,10.0,0.0,"","","","","","","","","","","2.6",""',
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "kraken_history.zip")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("trades.csv", csv_data)

            diagnostics: dict[str, object] = {}
            with patch.dict(os.environ, {"KRAKEN_TRADE_HISTORY_ZIP": zip_path}, clear=False):
                with patch("core.data.account_sync.get_runtime_setting") as mock_setting:
                    mock_setting.side_effect = lambda key, default=None: True if key == "ACCOUNT_SYNC_USE_TRADES_HISTORY" else _real_get_runtime_setting(key, default)
                    entry_meta = _compute_entry_basis(
                        client=_FailingTradesClient(),
                        symbols=["ADA/USD"],
                        asset_pairs={},
                        diagnostics=diagnostics,
                    )

        self.assertIn("ADA/USD", entry_meta)
        self.assertEqual(entry_meta["ADA/USD"]["source"], "kraken_trade_history_zip")
        self.assertAlmostEqual(float(entry_meta["ADA/USD"]["entry_price"]), 0.25, places=6)
        self.assertEqual(diagnostics.get("trades_history_source"), "zip_fallback")
        self.assertIn("trades_history_error", diagnostics)
        self.assertEqual(entry_meta["ADA/USD"]["entry_ts"], "2026-03-20T01:00:00+00:00")

    def test_normalize_entry_ts_upgrades_naive_timestamp_to_utc_iso(self) -> None:
        self.assertEqual(
            _normalize_entry_ts("2026-02-22 16:19:33.82"),
            "2026-02-22T16:19:33.820000+00:00",
        )


class _FailingTradesClient:
    def get_trades_history(self) -> dict[str, object]:
        raise RuntimeError("Kraken private API error for /0/private/TradesHistory: ['EAPI:Invalid key']")

    def get_ledgers(self, type_filter: str = "all") -> dict[str, object]:
        raise RuntimeError("Kraken private API error for /0/private/Ledgers: ['EAPI:Invalid key']")


class _FailingTradesLedgerClient:
    def get_trades_history(self) -> dict[str, object]:
        raise RuntimeError("Kraken private API error for /0/private/TradesHistory: ['EAPI:Invalid key']")

    def get_ledgers(self, type_filter: str = "all") -> dict[str, object]:
        return {
            "result": {
                "ledger": {
                    "a": {
                        "aclass": "currency",
                        "asset": "NIGHT",
                        "amount": "164.2807",
                        "balance": "164.2807",
                        "fee": "0.0000",
                        "refid": "ORDER1",
                        "time": 1774211883.472912,
                        "type": "trade",
                        "subtype": "tradespot",
                    },
                    "b": {
                        "aclass": "currency",
                        "asset": "ZUSD",
                        "amount": "-7.0000",
                        "balance": "50.9834",
                        "fee": "0.0161",
                        "refid": "ORDER1",
                        "time": 1774211883.472912,
                        "type": "trade",
                        "subtype": "tradespot",
                    },
                }
            }
        }


if __name__ == "__main__":
    unittest.main()
