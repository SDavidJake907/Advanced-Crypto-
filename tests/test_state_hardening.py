import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.state import open_orders


class StateHardeningTests(unittest.TestCase):
    def test_save_open_orders_is_atomic_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "open_orders.json"
            with patch.object(open_orders, "OPEN_ORDERS_PATH", target):
                open_orders.save_open_orders({"abc": {"status": "open"}})
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"abc": {"status": "open"}})

    def test_find_stale_orders_ignores_bad_timestamp_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "open_orders.json"
            with patch.object(open_orders, "OPEN_ORDERS_PATH", target):
                target.write_text(json.dumps({"abc": {"status": "open", "submitted_ts": "not-a-time"}}), encoding="utf-8")
                self.assertEqual(open_orders.find_stale_orders(timeout_sec=1), [])


if __name__ == "__main__":
    unittest.main()
