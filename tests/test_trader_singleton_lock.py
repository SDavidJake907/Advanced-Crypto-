from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest

from core.runtime.trader_lock import TraderAlreadyRunningError, TraderSingletonLock


class TraderSingletonLockTests(unittest.TestCase):
    def test_lock_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "trader.lock"
            with TraderSingletonLock(lock_path, instance_id="primary-run") as lock:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["instance_id"], "primary-run")
                self.assertEqual(payload["pid"], lock.metadata["pid"])
                self.assertEqual(payload["lock_path"], str(lock_path))

    def test_stale_metadata_is_overwritten_on_next_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "trader.lock"
            lock_path.write_text(json.dumps({"instance_id": "stale-run", "pid": 1234}), encoding="utf-8")
            with TraderSingletonLock(lock_path, instance_id="fresh-run"):
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["instance_id"], "fresh-run")
                self.assertNotEqual(payload["pid"], 1234)

    def test_second_process_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "trader.lock"
            ready_path = Path(tmpdir) / "ready.txt"
            holder_script = "\n".join(
                [
                    "import sys, time",
                    "from pathlib import Path",
                    "from core.runtime.trader_lock import TraderSingletonLock",
                    "ready_path = Path(sys.argv[2])",
                    "lock = TraderSingletonLock(Path(sys.argv[1]), instance_id='holder-run')",
                    "lock.acquire()",
                    "ready_path.write_text('LOCKED', encoding='utf-8')",
                    "time.sleep(10)",
                    "lock.release()",
                ]
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", holder_script, str(lock_path), str(ready_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if ready_path.exists():
                        break
                    time.sleep(0.05)
                self.assertTrue(ready_path.exists())
                self.assertIsNone(proc.poll())
                with self.assertRaises(TraderAlreadyRunningError) as ctx:
                    TraderSingletonLock(lock_path, instance_id="second-run").acquire()
                self.assertEqual(ctx.exception.owner_metadata.get("instance_id"), "holder-run")
            finally:
                proc.terminate()
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
