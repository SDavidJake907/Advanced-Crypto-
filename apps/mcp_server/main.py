from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import uvicorn
from mcp.server.fastmcp import FastMCP

from core.config.runtime import (
    apply_runtime_override_proposal,
    get_runtime_snapshot,
    load_runtime_override_proposals,
    stage_runtime_override_proposal,
)
from core.execution.cpp_exec import CppExecutor
from core.llm.client import nemotron_chat, nemotron_provider_api_url, nemotron_provider_model, nemotron_provider_name
from core.llm.nemotron import NemotronStrategist
from core.llm.phi3_reflex import phi3_reflex
from core.memory.setup_reliability import build_setup_reliability_summary
from core.memory.trade_memory import TradeMemoryStore
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, Position, PositionState
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy
from core.state.store import load_universe

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8765"))

mcp = FastMCP("KrakenSK MCP", host=MCP_HOST, streamable_http_path="/")


def _mcp_llm_config() -> dict[str, Any]:
    provider = nemotron_provider_name()
    config: dict[str, Any] = {
        "provider": provider,
        "phi3_url": os.getenv("PHI3_BASE_URL", "http://127.0.0.1:8084"),
        "model": nemotron_provider_model(),
        "api_url": nemotron_provider_api_url(),
    }
    return config


def _safe_path(path: str) -> Path:
    candidate = (ROOT / path).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError("Path must stay inside project root")
    if not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


def _build_positions_state(items: list[dict[str, Any]] | None) -> PositionState:
    state = PositionState()
    for item in items or []:
        state.add_or_update(
            Position(
                symbol=str(item["symbol"]),
                side=str(item["side"]),
                weight=float(item["weight"]),
            )
        )
    return state


def _build_portfolio_state(data: dict[str, Any] | None) -> PortfolioState:
    payload = data or {}
    return PortfolioState(
        cash=float(payload.get("cash", 10_000.0)),
        positions=dict(payload.get("positions", {})),
        pnl=float(payload.get("pnl", 0.0)),
        last_fill_bar_ts=payload.get("last_fill_bar_ts"),
        last_fill_bar_idx=payload.get("last_fill_bar_idx"),
        last_fill_symbol=payload.get("last_fill_symbol"),
        last_fill_side=payload.get("last_fill_side"),
    )


def _read_jsonl_tail(path: Path, limit: int) -> list[Any]:
    if not path.exists():
        return []
    items: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            items.append({"raw": line})
    return items[-limit:]


def _optimizer_review_tail(limit: int = 20) -> list[dict[str, Any]]:
    items = _read_jsonl_tail(ROOT / "logs" / "nvidia_optimizer_reviews.jsonl", limit)
    return [item for item in items if isinstance(item, dict)]


def _lane_supervision_summary() -> dict[str, Any]:
    universe = load_universe()
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    items = meta.get("lane_supervision", [])
    if not isinstance(items, list):
        items = []
    lane_counts: dict[str, int] = {}
    conflicts = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        lane = str(item.get("lane_candidate", "unknown"))
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        if bool(item.get("lane_conflict")):
            conflicts += 1
    return {"count": len(items), "lane_counts": lane_counts, "conflicts": conflicts, "items": items}


def _outcome_review_summary(limit: int = 20) -> dict[str, Any]:
    items = _read_jsonl_tail(ROOT / "logs" / "outcome_reviews.jsonl", limit)
    reviews = [item for item in items if isinstance(item, dict)]
    outcome_counts: dict[str, int] = {}
    for item in reviews:
        review = item.get("review", {}) if isinstance(item.get("review", {}), dict) else {}
        outcome_class = str(review.get("outcome_class", "unknown"))
        outcome_counts[outcome_class] = outcome_counts.get(outcome_class, 0) + 1
    return {"count": len(reviews), "outcome_counts": outcome_counts, "reviews": reviews}


def _collector_status() -> dict[str, Any]:
    path = ROOT / "logs" / "collector_telemetry.json"
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT))}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid", "path": str(path.relative_to(ROOT)), "error": str(exc)}
    return {"status": "ok", "path": str(path.relative_to(ROOT)), "telemetry": payload}


def _visual_feed_status(limit: int = 20) -> dict[str, Any]:
    path = ROOT / "logs" / "visual_phi3_feed.jsonl"
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT))}
    items = _read_jsonl_tail(path, limit)
    return {"status": "ok", "path": str(path.relative_to(ROOT)), "items": items}


@mcp.tool()
def healthcheck() -> dict[str, Any]:
    llm = _mcp_llm_config()
    return {
        "status": "ok",
        "project_root": str(ROOT),
        "mcp_path": "/",
        "llm": llm,
        "nemotron_url": llm["api_url"],
        "phi3_url": llm["phi3_url"],
    }


@mcp.tool()
def project_overview() -> dict[str, Any]:
    top_level = sorted(
        path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    return {
        "apps": sorted(path.name for path in (ROOT / "apps").iterdir() if path.is_dir()),
        "core": sorted(path.name for path in (ROOT / "core").iterdir() if path.is_dir()),
        "top_level_dirs": top_level,
    }


@mcp.tool()
def list_project_files(pattern: str = "*.py", limit: int = 200) -> dict[str, Any]:
    files = []
    for path in ROOT.rglob(pattern):
        if path.is_file():
            files.append(str(path.relative_to(ROOT)))
        if len(files) >= limit:
            break
    return {"pattern": pattern, "count": len(files), "files": files}


@mcp.tool()
def read_project_file(path: str, start_line: int = 1, max_lines: int = 200) -> dict[str, Any]:
    target = _safe_path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    start = max(start_line - 1, 0)
    end = min(start + max_lines, len(lines))
    return {
        "path": str(target.relative_to(ROOT)),
        "start_line": start + 1,
        "end_line": end,
        "content": "\n".join(lines[start:end]),
    }


@mcp.tool()
def search_project(pattern: str, glob: str = "*.py", limit: int = 100) -> dict[str, Any]:
    regex = re.compile(pattern)
    matches: list[dict[str, Any]] = []
    for path in ROOT.rglob(glob):
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if regex.search(line):
                matches.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": lineno,
                        "text": line.strip(),
                    }
                )
                if len(matches) >= limit:
                    return {"pattern": pattern, "count": len(matches), "matches": matches}
    return {"pattern": pattern, "count": len(matches), "matches": matches}


@mcp.tool()
def phi3_reflex_tool(features: dict[str, Any]) -> dict[str, Any]:
    return phi3_reflex(features).to_dict()


@mcp.tool()
def get_runtime_config() -> dict[str, Any]:
    return get_runtime_snapshot()


@mcp.tool()
def update_runtime_config(
    updates: dict[str, Any],
    approved_by: str = "",
    validation: dict[str, Any] | None = None,
    approval_note: str = "",
) -> dict[str, Any]:
    proposal = stage_runtime_override_proposal(
        updates,
        source="mcp_manual_apply",
        summary=approval_note or "manual_runtime_update",
        validation=validation,
        context={"approval_note": approval_note},
    )
    try:
        applied = apply_runtime_override_proposal(
            proposal["id"],
            approved_by=approved_by,
            approval_note=approval_note,
        )
        status = "applied"
    except Exception as exc:
        applied = {}
        status = "blocked"
        proposal["apply_error"] = str(exc)
    snapshot = get_runtime_snapshot()
    return {
        "status": status,
        "proposal": proposal,
        "applied": applied,
        "snapshot": snapshot,
    }


@mcp.tool()
def get_runtime_override_proposals(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    proposals = load_runtime_override_proposals()
    return {
        "count": len(proposals),
        "proposals": proposals[-limit:],
    }


@mcp.tool()
def get_recent_traces(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    traces = _read_jsonl_tail(ROOT / "logs" / "decision_traces.jsonl", limit)
    return {"count": len(traces), "traces": traces}


@mcp.tool()
def get_warmup_status(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    events = _read_jsonl_tail(ROOT / "logs" / "warmup.jsonl", limit)
    latest = events[-1] if events else {}
    return {"count": len(events), "latest": latest, "events": events}


@mcp.tool()
def get_live_candles(symbol: str, timeframe: str = "1m", limit: int = 50) -> dict[str, Any]:
    token = symbol.replace("/", "").replace(":", "").upper()
    target = ROOT / "logs" / "live" / f"candles_{token}_{timeframe}.csv"
    if not target.exists():
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "path": str(target.relative_to(ROOT)),
            "rows": [],
            "count": 0,
        }
    lines = target.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {"symbol": symbol, "timeframe": timeframe, "path": str(target.relative_to(ROOT)), "rows": [], "count": 0}
    header = lines[0].split(",")
    rows = []
    for line in lines[1:][-max(1, min(limit, 500)):]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "path": str(target.relative_to(ROOT)),
        "count": len(rows),
        "rows": rows,
    }


@mcp.tool()
def get_trading_status(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    universe = load_universe()
    traces = _read_jsonl_tail(ROOT / "logs" / "decision_traces.jsonl", limit)
    recent_actions = traces[-10:]
    lane_counts: dict[str, int] = {}
    nemotron_reasons: dict[str, int] = {}
    recent_fills_exits: list[dict[str, Any]] = []
    for trace in traces:
        features = trace.get("features", {}) if isinstance(trace, dict) else {}
        lane = str(features.get("lane", "unknown"))
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        execution = trace.get("execution", {}) if isinstance(trace, dict) else {}
        nemotron = execution.get("nemotron", {}) if isinstance(execution, dict) else {}
        reason = str(nemotron.get("reason", "")).strip()
        if reason:
            nemotron_reasons[reason] = nemotron_reasons.get(reason, 0) + 1
        status = str(execution.get("status", ""))
        signal = str(trace.get("signal", ""))
        if status == "filled" or signal == "EXIT":
            recent_fills_exits.append(trace)
    memory = TradeMemoryStore().compute_track_record()
    return {
        "active_symbols": universe.get("active_pairs", []),
        "universe_reason": universe.get("reason", ""),
        "universe_meta": universe.get("meta", {}),
        "lane_supervision_summary": _lane_supervision_summary(),
        "outcome_review_summary": _outcome_review_summary(10),
        "collector_status": _collector_status(),
        "visual_feed_status": _visual_feed_status(10),
        "lane_counts": lane_counts,
        "recent_nemotron_reasons": nemotron_reasons,
        "recent_actions": recent_actions,
        "recent_fills_exits": recent_fills_exits[-10:],
        "trade_memory": [record.__dict__ for record in memory[:10]],
    }


@mcp.tool()
def get_optimizer_status(limit: int = 10) -> dict[str, Any]:
    limit = max(1, min(limit, 100))
    reviews = _optimizer_review_tail(limit)
    latest = reviews[-1] if reviews else {}
    return {
        "enabled": os.getenv("NVIDIA_OPTIMIZER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "interval_min": int(os.getenv("NVIDIA_OPTIMIZER_INTERVAL_MIN", "30")),
        "auto_apply": False,
        "review_count": len(reviews),
        "latest_review": latest,
        "recent_reviews": reviews,
    }


@mcp.tool()
def get_latest_optimizer_review() -> dict[str, Any]:
    reviews = _optimizer_review_tail(1)
    if not reviews:
        return {"status": "no_reviews", "review": {}}
    return {"status": "ok", "review": reviews[-1]}


@mcp.tool()
def get_lane_supervision_summary() -> dict[str, Any]:
    return _lane_supervision_summary()


@mcp.tool()
def get_outcome_review_summary(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    return _outcome_review_summary(limit)


@mcp.tool()
def get_setup_reliability_summary(
    lookback: int = 200,
    min_trades: int = 3,
    limit: int = 5,
) -> dict[str, Any]:
    lookback = max(1, min(lookback, 2000))
    min_trades = max(1, min(min_trades, 50))
    limit = max(1, min(limit, 20))
    return build_setup_reliability_summary(
        store=TradeMemoryStore(),
        lookback=lookback,
        min_trades=min_trades,
        limit=limit,
    )


@mcp.tool()
def get_collector_status() -> dict[str, Any]:
    return _collector_status()


@mcp.tool()
def get_visual_feed_status(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 100))
    return _visual_feed_status(limit)


@mcp.tool()
def nemotron_decide_symbol(
    symbol: str,
    features: dict[str, Any],
    portfolio_state: dict[str, Any] | None = None,
    positions_state: list[dict[str, Any]] | None = None,
    symbols: list[str] | None = None,
    proposed_weight: float = 0.1,
) -> dict[str, Any]:
    strategist = NemotronStrategist(
        strategy=SimpleMomentumStrategy(),
        risk_engine=BasicRiskEngine(),
        portfolio_config=PortfolioConfig.from_runtime(),
        executor=CppExecutor(),
    )
    portfolio = _build_portfolio_state(portfolio_state)
    positions = _build_positions_state(positions_state)
    universe = symbols or [symbol]
    decision = strategist.decide(
        symbol=symbol,
        features=features,
        portfolio_state=portfolio,
        positions_state=positions,
        symbols=universe,
        proposed_weight=float(proposed_weight),
    )
    return decision.to_dict()


@mcp.tool()
def nemotron_review_system(objective: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    system = (
        "You are Nemotron reviewing a live trading system. "
        "Return only JSON with keys summary, risks, recommendations. "
        "summary is a short string. risks and recommendations are arrays of short strings."
    )
    payload = {"objective": objective, "context": context or {}}
    raw = nemotron_chat(payload, system=system, max_tokens=500)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"summary": raw, "risks": [], "recommendations": []}


@mcp.tool()
def nemotron_adjust_runtime(objective: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    system = (
        "You are Nemotron adjusting runtime settings for a deterministic trading system. "
        "Return only JSON with keys summary, updates, cautions. "
        "updates must be an object using only approved runtime setting names. "
        "Never include prose outside JSON."
    )
    payload = {
        "objective": objective,
        "context": context or {},
        "runtime_config": get_runtime_snapshot(),
    }
    raw = nemotron_chat(payload, system=system, max_tokens=600)
    try:
        parsed = json.loads(raw)
        updates = parsed.get("updates", {})
        if not isinstance(updates, dict):
            raise ValueError("updates must be an object")
        proposal = (
            stage_runtime_override_proposal(
                updates,
                source="mcp_nemotron_runtime_advice",
                summary=parsed.get("summary", ""),
                context={"objective": objective, "cautions": parsed.get("cautions", [])},
            )
            if updates
            else {}
        )
        return {
            "summary": parsed.get("summary", ""),
            "updates": updates,
            "applied": {},
            "proposal": proposal,
            "cautions": parsed.get("cautions", []),
            "runtime_config": get_runtime_snapshot(),
        }
    except Exception:
        return {
            "summary": "nemotron_adjust_runtime_failed_to_parse",
            "updates": {},
            "applied": {},
            "proposal": {},
            "cautions": [raw],
            "runtime_config": get_runtime_snapshot(),
        }


app = mcp.streamable_http_app()


def main() -> None:
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    main()
