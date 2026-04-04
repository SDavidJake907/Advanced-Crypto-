from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.config.runtime import get_runtime_snapshot, load_runtime_override_proposals, stage_runtime_override_proposal
from core.llm.client import nvidia_nemotron_chat, parse_json_response
from core.llm.contracts import normalize_runtime_advice
from core.llm.prompts import NVIDIA_OPTIMIZER_SYSTEM_PROMPT
from core.policy.aggression_recommendation import recommend_aggression_mode
from core.memory.daily_review import build_daily_review_report_from_disk, write_daily_review_report
from core.memory.kelly_sizer import run_kelly_update
from core.memory.kraken_history import build_history_review_block
from core.models.xgb_entry import XGBEntryModel
from core.memory.trade_memory import TradeMemoryStore
from core.runtime.log_rotation import rotate_jsonl_if_needed
from core.state.system_record import record_optimizer_review
from core.state.store import load_universe

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
REVIEW_LOG = ROOT / "logs" / "nvidia_optimizer_reviews.jsonl"
FEATURE_IMPORTANCE_LOG = ROOT / "logs" / "feature_importance_latest.json"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    from collections import deque

    recent_lines: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                recent_lines.append(line)
    for line in recent_lines:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
        except Exception:
            continue
    return items


def _append_review(event: dict[str, Any]) -> None:
    REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(REVIEW_LOG)
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with REVIEW_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    record_optimizer_review(payload)


def _allowed_override_keys() -> set[str]:
    return set(get_runtime_snapshot()["values"].keys())


def _maybe_stage_aggression_proposal(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    runtime_values = payload.get("runtime", {}) if isinstance(payload.get("runtime", {}), dict) else {}
    universe_meta = payload.get("universe", {}) if isinstance(payload.get("universe", {}), dict) else {}
    recent_decisions = payload.get("recent_decision_debug", []) if isinstance(payload.get("recent_decision_debug", []), list) else []
    recommendation = recommend_aggression_mode(
        runtime_values=runtime_values,
        universe_meta=universe_meta,
        recent_decision_debug=recent_decisions,
    )
    current_mode = recommendation.current_mode
    if recommendation.mode == current_mode:
        return None, recommendation.to_dict()
    for proposal in reversed(load_runtime_override_proposals()):
        if not isinstance(proposal, dict):
            continue
        if proposal.get("status") != "pending":
            continue
        updates = proposal.get("updates", {}) if isinstance(proposal.get("updates", {}), dict) else {}
        if str(updates.get("AGGRESSION_MODE", "")).upper() == recommendation.mode:
            return None, recommendation.to_dict()
    staged = stage_runtime_override_proposal(
        {"AGGRESSION_MODE": recommendation.mode},
        source="deterministic_aggression",
        summary=recommendation.summary,
        context={
            "issues": recommendation.reasons,
            "confidence": recommendation.confidence,
            "current_mode": current_mode,
            "recommended_mode": recommendation.mode,
            "score": recommendation.score,
        },
    )
    return staged, recommendation.to_dict()


def _build_payload() -> dict[str, Any]:
    trade_zip = os.getenv("KRAKEN_TRADE_HISTORY_ZIP", "").strip()
    memory_block = TradeMemoryStore().build_memory_block()
    history_block = build_history_review_block(trade_zip) if trade_zip else ""
    return {
        "runtime": get_runtime_snapshot()["values"],
        "universe": load_universe().get("meta", {}),
        "recent_decision_debug": _read_jsonl_tail(ROOT / "logs" / "decision_debug.jsonl", 20),
        "recent_nemotron_debug": _read_jsonl_tail(ROOT / "logs" / "nemotron_debug.jsonl", 20),
        "trade_memory": memory_block,
        "history_review": history_block,
    }


def _run_kelly() -> None:
    trade_zip = os.getenv("KRAKEN_TRADE_HISTORY_ZIP", "").strip()
    if not trade_zip:
        return
    try:
        summary = run_kelly_update(trade_zip)
        applied = summary.get("applied", {})
        diag = summary.get("diagnostics", {})
        parts = []
        if "EXEC_RISK_PER_TRADE_PCT" in applied:
            parts.append(f"std={applied['EXEC_RISK_PER_TRADE_PCT']}% (W={diag.get('standard_win_rate', 0):.0%}, n={diag.get('standard_trips', 0)})")
        if "MEME_EXEC_RISK_PER_TRADE_PCT" in applied:
            parts.append(f"meme={applied['MEME_EXEC_RISK_PER_TRADE_PCT']}% (W={diag.get('meme_win_rate', 0):.0%}, n={diag.get('meme_trips', 0)})")
        if parts:
            print(f"[kelly_sizer] Updated position sizes: {', '.join(parts)}")
        else:
            print(f"[kelly_sizer] No update (insufficient history). diag={diag}")
    except Exception as exc:
        print(f"[kelly_sizer] Error: {exc}")


def _write_feature_importance_report() -> None:
    model = XGBEntryModel()
    model.load_or_init(str(ROOT / "models" / "xgb_entry.pkl"))
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **model.build_feature_importance_report(),
    }
    FEATURE_IMPORTANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_LOG.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _run_once(*, use_nvidia: bool) -> None:
    _run_kelly()
    write_daily_review_report(build_daily_review_report_from_disk(root=ROOT))
    _write_feature_importance_report()
    payload = _build_payload()
    staged_aggression, aggression_review = _maybe_stage_aggression_proposal(payload)
    filtered_updates: dict[str, Any] = {}
    staged_proposal: dict[str, Any] | None = None
    parsed: dict[str, Any] = {
        "summary": "",
        "issues": [],
        "recommended_overrides": {},
        "confidence": 0.0,
    }
    if use_nvidia:
        raw = nvidia_nemotron_chat(
            payload,
            system=NVIDIA_OPTIMIZER_SYSTEM_PROMPT,
            max_tokens=1200,
            model=os.getenv("NVIDIA_OPTIMIZER_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")),
        )
        allowed = _allowed_override_keys()
        parsed = normalize_runtime_advice(parse_json_response(raw), allowed_keys=allowed)
        filtered_updates = parsed.get("recommended_overrides", {})
        if filtered_updates:
            staged_proposal = stage_runtime_override_proposal(
                filtered_updates,
                source="nvidia_optimizer",
                summary=parsed.get("summary", ""),
                context={"issues": parsed.get("issues", []), "confidence": parsed.get("confidence", 0.0)},
            )
    _append_review(
        {
            "summary": parsed.get("summary", "") or aggression_review.get("summary", ""),
            "issues": parsed.get("issues", []) or aggression_review.get("reasons", []),
            "recommended_overrides": filtered_updates,
            "applied_overrides": {},
            "staged_proposal": staged_proposal or {},
            "aggression_review": aggression_review,
            "staged_aggression_proposal": staged_aggression or {},
            "confidence": parsed.get("confidence", 0.0),
        }
    )


def main() -> None:
    nvidia_enabled = _bool_env("NVIDIA_OPTIMIZER_ENABLED", True)
    nvidia_available = nvidia_enabled and bool(os.getenv("NVIDIA_API_KEY", "").strip())
    if not nvidia_enabled:
        print("NVIDIA optimizer disabled. Running deterministic aggression review only.")
    elif not nvidia_available:
        print("NVIDIA optimizer skipped: NVIDIA_API_KEY not set. Running deterministic aggression review only.")
    interval_min = max(int(os.getenv("NVIDIA_OPTIMIZER_INTERVAL_MIN", "30")), 5)
    print(f"NVIDIA optimizer scheduler running every {interval_min} minutes.")
    while True:
        try:
            _run_once(use_nvidia=nvidia_available)
        except Exception as exc:
            _append_review({"summary": "optimizer_error", "issues": [str(exc)], "recommended_overrides": {}, "applied_overrides": {}, "confidence": 0.0})
        time.sleep(interval_min * 60)


if __name__ == "__main__":
    main()
