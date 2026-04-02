from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.state import open_orders


class _FakeClient:
    def __init__(
        self,
        result: dict[str, object],
        *,
        open_orders: dict[str, object] | None = None,
        query_exc: Exception | None = None,
    ) -> None:
        self._result = result
        self._open_orders = open_orders or {}
        self._query_exc = query_exc
        self.query_calls: list[list[str]] = []

    def get_open_orders(self) -> dict[str, object]:
        return {"result": {"open": self._open_orders}}

    def query_orders_info(self, txids: list[str], trades: bool = False) -> dict[str, object]:
        self.query_calls.append(list(txids))
        if self._query_exc is not None:
            raise self._query_exc
        return {"result": self._result}


class OpenOrderReconcileTests(unittest.TestCase):
    def test_stale_open_order_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open_orders.json"
            with patch.object(open_orders, "OPEN_ORDERS_PATH", path):
                open_orders.save_open_orders(
                    {
                        "coid-1": {
                            "symbol": "DOGE/USD",
                            "status": "open",
                            "submitted_ts": time.time() - 1000,
                        }
                    }
                )
                data = open_orders.reconcile_open_orders(timeout_sec=60)
                self.assertEqual(data["coid-1"]["status"], "timeout")

    def test_kraken_closed_order_marks_local_filled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open_orders.json"
            with patch.object(open_orders, "OPEN_ORDERS_PATH", path):
                open_orders.save_open_orders(
                    {
                        "coid-2": {
                            "symbol": "ADA/USD",
                            "status": "open",
                            "submitted_ts": time.time(),
                            "txid": "TX123",
                        }
                    }
                )
                data = open_orders.reconcile_open_orders(
                    client=_FakeClient({"TX123": {"status": "closed"}}),
                    timeout_sec=900,
                )
                self.assertEqual(data["coid-2"]["status"], "filled")
    
    def test_open_orders_short_circuits_query_orders_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open_orders.json"
            with patch.object(open_orders, "OPEN_ORDERS_PATH", path):
                open_orders.save_open_orders(
                    {
                        "coid-3": {
                            "symbol": "LINK/USD",
                            "status": "pending",
                            "submitted_ts": time.time(),
                            "txid": "TXOPEN",
                        }
                    }
                )
                client = _FakeClient({}, open_orders={"TXOPEN": {"descr": {"pair": "LINK/USD"}}})
                data = open_orders.reconcile_open_orders(client=client, timeout_sec=900)
                self.assertEqual(data["coid-3"]["status"], "open")
                self.assertEqual(client.query_calls, [])

    def test_invalid_key_on_query_orders_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open_orders.json"
            with patch.object(open_orders, "OPEN_ORDERS_PATH", path):
                open_orders.save_open_orders(
                    {
                        "coid-4": {
                            "symbol": "BTC/USD",
                            "status": "open",
                            "submitted_ts": time.time(),
                            "txid": "TXNEEDSDETAIL",
                        }
                    }
                )
                client = _FakeClient(
                    {},
                    query_exc=RuntimeError(
                        "Kraken private API error for /0/private/QueryOrders: ['EAPI:Invalid key']"
                    ),
                )
                with self.assertLogs("core.state.open_orders", level="INFO") as captured:
                    data = open_orders.reconcile_open_orders(client=client, timeout_sec=900)
                self.assertEqual(data["coid-4"]["status"], "open")
                self.assertIn("skipping Kraken QueryOrders reconcile", "\n".join(captured.output))
                self.assertEqual(client.query_calls, [["TXNEEDSDETAIL"]])


if __name__ == "__main__":
    unittest.main()
