from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from core.runtime.log_rotation import prune_directory_files


class LogRetentionTests(unittest.TestCase):
    def test_prune_directory_files_keeps_latest_and_removes_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for idx in range(4):
                path = root / f"{idx}.png"
                path.write_text("x", encoding="utf-8")
                ts = time.time() - ((idx + 1) * 86400.0)
                os_times = (ts, ts)
                path.touch()
                import os

                os.utime(path, os_times)
                files.append(path)
            prune_directory_files(root, pattern="*.png", keep_latest=1, max_age_days=2)
            remaining = sorted(path.name for path in root.glob("*.png"))
            self.assertEqual(remaining, ["0.png"])


if __name__ == "__main__":
    unittest.main()
