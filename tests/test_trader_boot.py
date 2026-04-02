from __future__ import annotations

import unittest
from unittest.mock import patch

from apps.trader import boot


class TraderBootTests(unittest.TestCase):
    def test_effective_nemotron_batch_mode_respects_enabled_flag_for_local_provider(self) -> None:
        with patch.object(boot, "NEMOTRON_BATCH_MODE", True):
            with patch("apps.trader.boot.nemotron_provider_name", return_value="local"):
                self.assertTrue(boot.effective_nemotron_batch_mode())

    def test_effective_nemotron_batch_mode_respects_disabled_flag(self) -> None:
        with patch.object(boot, "NEMOTRON_BATCH_MODE", False):
            with patch("apps.trader.boot.nemotron_provider_name", return_value="nvidia"):
                self.assertFalse(boot.effective_nemotron_batch_mode())


if __name__ == "__main__":
    unittest.main()
