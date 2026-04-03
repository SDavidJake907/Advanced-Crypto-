import unittest
from unittest.mock import patch

from core.config.runtime import get_runtime_setting, get_runtime_snapshot


class AggressionModeTests(unittest.TestCase):
    @patch("core.config.runtime.load_runtime_overrides", return_value={})
    def test_offensive_mode_lowers_entry_thresholds_and_cooldown(self, override_mock) -> None:
        with patch.dict("os.environ", {"AGGRESSION_MODE": "OFFENSIVE"}, clear=False):
            self.assertEqual(float(get_runtime_setting("ADVISORY_MIN_ENTRY_SCORE")), 42.0)
            self.assertEqual(int(get_runtime_setting("TRADER_COOLDOWN_BARS")), 2)

    @patch("core.config.runtime.load_runtime_overrides", return_value={})
    def test_defensive_mode_raises_thresholds_and_reduces_weight(self, override_mock) -> None:
        with patch.dict("os.environ", {"AGGRESSION_MODE": "DEFENSIVE"}, clear=False):
            self.assertEqual(float(get_runtime_setting("NEMOTRON_GATE_MIN_ENTRY_SCORE")), 56.0)
            self.assertAlmostEqual(float(get_runtime_setting("TRADER_PROPOSED_WEIGHT")), 0.15, places=4)

    @patch("core.config.runtime.load_runtime_overrides", return_value={})
    def test_snapshot_exposes_base_and_adjusted_values(self, override_mock) -> None:
        with patch.dict("os.environ", {"AGGRESSION_MODE": "HIGH_OFFENSIVE"}, clear=False):
            snapshot = get_runtime_snapshot()
        self.assertEqual(snapshot["aggression_mode"], "HIGH_OFFENSIVE")
        self.assertEqual(snapshot["base_values"]["ADVISORY_MIN_ENTRY_SCORE"], 45.0)
        self.assertEqual(snapshot["values"]["ADVISORY_MIN_ENTRY_SCORE"], 40.0)


if __name__ == "__main__":
    unittest.main()
