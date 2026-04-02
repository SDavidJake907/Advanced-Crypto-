import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.config import runtime
from core.llm.nemotron_client import NemotronClient
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy


class RuntimeOverrideStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.override_path = self.root / "runtime_overrides.json"
        self.proposal_path = self.root / "runtime_override_proposals.json"
        self.override_patch = patch.object(runtime, "RUNTIME_OVERRIDES_PATH", self.override_path)
        self.proposal_patch = patch.object(runtime, "RUNTIME_OVERRIDE_PROPOSALS_PATH", self.proposal_path)
        self.override_patch.start()
        self.proposal_patch.start()
        self.addCleanup(self.override_patch.stop)
        self.addCleanup(self.proposal_patch.stop)

    def test_runtime_override_proposal_requires_validation_before_apply(self) -> None:
        proposal = runtime.stage_runtime_override_proposal(
            {"PORTFOLIO_MAX_OPEN_POSITIONS": 3},
            source="unit_test",
            summary="raise slots",
        )

        with self.assertRaisesRegex(ValueError, "missing replay/shadow/human approval"):
            runtime.apply_runtime_override_proposal(proposal["id"], approved_by="tester")

        self.assertEqual(runtime.load_runtime_overrides(), {})

    def test_runtime_override_proposal_applies_after_full_validation(self) -> None:
        proposal = runtime.stage_runtime_override_proposal(
            {"PORTFOLIO_MAX_OPEN_POSITIONS": 4},
            source="unit_test",
            summary="validated change",
            validation={
                "replay_passed": True,
                "shadow_passed": True,
                "human_approved": True,
            },
        )

        applied = runtime.apply_runtime_override_proposal(
            proposal["id"],
            approved_by="tester",
            approval_note="replay and shadow reviewed",
        )

        self.assertEqual(runtime.load_runtime_overrides()["PORTFOLIO_MAX_OPEN_POSITIONS"], 4)
        self.assertEqual(applied["status"], "applied")
        self.assertEqual(applied["applied_by"], "tester")

    def test_nemotron_portfolio_adjust_stages_without_mutating_live_portfolio_config(self) -> None:
        client = NemotronClient(
            strategy=SimpleMomentumStrategy(),
            risk=BasicRiskEngine(),
            portfolio_config=PortfolioConfig(max_open_positions=2),
            execution=Mock(),
            portfolio_state=PortfolioState(),
            positions_state=PositionState(),
            symbols=["BTC/USD"],
        )

        result = client._tool_portfolio_adjust(
            key="PORTFOLIO_MAX_OPEN_POSITIONS",
            value=3,
            reason="high_score_entry_blocked",
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["staged"])
        self.assertFalse(result["effective_immediately"])
        self.assertEqual(client.portfolio_config.max_open_positions, 2)
        self.assertEqual(runtime.load_runtime_overrides(), {})
        proposals = runtime.load_runtime_override_proposals()
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["updates"]["PORTFOLIO_MAX_OPEN_POSITIONS"], 3)


if __name__ == "__main__":
    unittest.main()
