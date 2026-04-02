from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
UI_HOST = os.getenv("OPERATOR_UI_HOST", "127.0.0.1")
UI_PORT = int(os.getenv("OPERATOR_UI_PORT", "8780"))

app = FastAPI(title="KrakenSK Operator UI")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid", "error": str(exc)}
    if isinstance(payload, dict):
        return payload
    return {"status": "invalid", "error": "expected object"}


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items[-limit:]


def _summarize_alerts(items: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in items:
        level = str(item.get("level", "unknown") or "unknown").lower()
        source = str(item.get("source", "unknown") or "unknown").lower()
        level_counts[level] = level_counts.get(level, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "count": len(items),
        "levels": level_counts,
        "sources": source_counts,
    }


def _load_universe() -> dict[str, Any]:
    return _read_json(ROOT / "universe.json")


def _read_open_orders() -> dict[str, Any]:
    return _read_json(ROOT / "logs" / "open_orders.json")


def _summarize_positions(account_sync: dict[str, Any]) -> list[dict[str, Any]]:
    synced_usd = account_sync.get("synced_positions_usd", {})
    states = account_sync.get("positions_state", [])
    portfolio_positions = (
        account_sync.get("portfolio_state", {}).get("positions", {})
        if isinstance(account_sync.get("portfolio_state", {}), dict)
        else {}
    )
    rows: list[dict[str, Any]] = []
    for item in states:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", ""))
        rows.append(
            {
                "symbol": symbol,
                "side": str(item.get("side", "LONG")),
                "weight": float(item.get("weight", 0.0)),
                "units": float(portfolio_positions.get(symbol, 0.0)),
                "value_usd": float(synced_usd.get(symbol, 0.0)),
            }
        )
    return rows


def _summarize_order_ledger(open_orders: dict[str, Any]) -> dict[str, Any]:
    filled_buys = 0
    filled_sells = 0
    open_count = 0
    filled_notional = 0.0
    known_entries: list[dict[str, Any]] = []

    for order_id, item in open_orders.items():
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).lower()
        side = str(item.get("side", "")).upper()
        qty = float(item.get("qty", 0.0) or 0.0)
        limit_price = float(item.get("limit_price", 0.0) or 0.0)
        symbol = str(item.get("symbol", ""))
        notional = qty * limit_price

        if status == "open":
            open_count += 1
        if status == "filled":
            filled_notional += notional
            if side == "BUY":
                filled_buys += 1
                known_entries.append(
                    {
                        "symbol": symbol,
                        "qty": qty,
                        "entry_price": limit_price,
                        "notional": notional,
                        "submitted_ts": float(item.get("submitted_ts", 0.0) or 0.0),
                    }
                )
            elif side == "SELL":
                filled_sells += 1

    known_entries.sort(key=lambda item: item["submitted_ts"], reverse=True)
    return {
        "filled_buys": filled_buys,
        "filled_sells": filled_sells,
        "open_orders": open_count,
        "filled_notional_usd": filled_notional,
        "known_entries": known_entries[:8],
    }


def _build_accounting_summary(account_sync: dict[str, Any], open_orders: dict[str, Any]) -> dict[str, Any]:
    cash = float(account_sync.get("cash_usd", 0.0))
    synced_positions_usd = account_sync.get("synced_positions_usd", {})
    book_value = cash
    if isinstance(synced_positions_usd, dict):
        book_value += sum(float(value or 0.0) for value in synced_positions_usd.values())

    baseline = float(account_sync.get("initial_equity_usd", 0.0))
    open_pnl = book_value - baseline
    diagnostics = account_sync.get("diagnostics", {}) if isinstance(account_sync.get("diagnostics", {}), dict) else {}
    trades_history_error = str(diagnostics.get("trades_history_error", "")).strip()
    synced_entry_meta = account_sync.get("synced_entry_meta", {})
    has_entry_meta = isinstance(synced_entry_meta, dict) and bool(synced_entry_meta)
    order_ledger = _summarize_order_ledger(open_orders)

    confidence = "high"
    confidence_reasons: list[str] = []
    if trades_history_error:
        confidence = "low"
        confidence_reasons.append("Kraken TradesHistory unavailable")
    if not has_entry_meta:
        confidence = "low"
        confidence_reasons.append("no synced entry basis")
    if order_ledger["filled_sells"] == 0:
        confidence_reasons.append("no recorded realized exits")
    if confidence == "high" and confidence_reasons:
        confidence = "medium"

    return {
        "book_value_usd": book_value,
        "baseline_equity_usd": baseline,
        "open_pnl_usd": open_pnl,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "known_filled_buys": order_ledger["filled_buys"],
        "known_filled_sells": order_ledger["filled_sells"],
        "open_orders": order_ledger["open_orders"],
        "filled_notional_usd": order_ledger["filled_notional_usd"],
        "known_entries": order_ledger["known_entries"],
    }


def _summarize_decisions(items: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    range_blocks = 0
    fallback_holds = 0
    blocked_reasons: dict[str, int] = {}
    for item in items:
        nemotron = item.get("nemotron", {})
        if isinstance(nemotron, dict):
            action = str(nemotron.get("action", "")).strip()
            reason = str(nemotron.get("reason", "")).strip()
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if "range" in reason:
                range_blocks += 1
            if reason == "nemotron_fallback_hold":
                fallback_holds += 1
        if str(item.get("execution_status", "")).lower() == "blocked":
            raw_reason = item.get("execution_reason", "")
            if isinstance(raw_reason, list):
                for reason in raw_reason:
                    key = str(reason or "").strip()
                    if key:
                        blocked_reasons[key] = blocked_reasons.get(key, 0) + 1
            else:
                key = str(raw_reason or "").strip()
                if key:
                    blocked_reasons[key] = blocked_reasons.get(key, 0) + 1
    return {
        "actions": action_counts,
        "reasons": sorted(
            ({"reason": key, "count": value} for key, value in reason_counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        )[:8],
        "blocked_reasons": sorted(
            ({"reason": key, "count": value} for key, value in blocked_reasons.items()),
            key=lambda item: item["count"],
            reverse=True,
        )[:8],
        "range_blocks": range_blocks,
        "fallback_holds": fallback_holds,
    }


def _summarize_lane_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    per_lane: dict[str, dict[str, Any]] = {}
    for item in items:
        lane = str(item.get("lane", "unknown") or "unknown").upper()
        row = per_lane.setdefault(
            lane,
            {
                "lane": lane,
                "total": 0,
                "opens": 0,
                "blocked": 0,
                "holds": 0,
                "no_trade": 0,
                "top_reasons": {},
                "llm_approve": 0,
                "llm_reject": 0,
                "llm_defer": 0,
                "llm_normalized_fields": 0,
                "llm_defer_reasons": {},
            },
        )
        row["total"] += 1
        execution_status = str(item.get("execution_status", "") or "").lower()
        signal = str(item.get("signal", "") or "").upper()
        nemotron = item.get("nemotron", {}) if isinstance(item.get("nemotron", {}), dict) else {}
        reason = str(nemotron.get("reason", "") or item.get("execution_reason", "") or "").strip()
        if execution_status == "filled" and signal in {"LONG", "SHORT"}:
            row["opens"] += 1
        elif execution_status == "blocked":
            row["blocked"] += 1
        elif execution_status == "no_trade":
            row["no_trade"] += 1
        elif signal in {"FLAT", "HOLD", ""}:
            row["holds"] += 1
        if reason:
            top_reasons = row["top_reasons"]
            top_reasons[reason] = top_reasons.get(reason, 0) + 1
        llm_metrics = item.get("llm_metrics", {}) if isinstance(item.get("llm_metrics", {}), dict) else {}
        aggregate = llm_metrics.get("aggregate", {}) if isinstance(llm_metrics.get("aggregate", {}), dict) else {}
        row["llm_approve"] += int(aggregate.get("approve", 0) or 0)
        row["llm_reject"] += int(aggregate.get("reject", 0) or 0)
        row["llm_defer"] += int(aggregate.get("defer", 0) or 0)
        row["llm_normalized_fields"] += int(aggregate.get("normalized_fields", 0) or 0)
        roles = llm_metrics.get("roles", []) if isinstance(llm_metrics.get("roles", []), list) else []
        for role_item in roles:
            if not isinstance(role_item, dict):
                continue
            if str(role_item.get("decision", "")).lower() != "defer":
                continue
            reason_key = str(role_item.get("defer_reason_category", "") or "unknown")
            defer_reasons = row["llm_defer_reasons"]
            defer_reasons[reason_key] = defer_reasons.get(reason_key, 0) + 1
    ranked = []
    for row in per_lane.values():
        top_reasons = sorted(
            ({"reason": key, "count": value} for key, value in row["top_reasons"].items()),
            key=lambda item: item["count"],
            reverse=True,
        )[:3]
        llm_defer_reasons = sorted(
            ({"reason": key, "count": value} for key, value in row["llm_defer_reasons"].items()),
            key=lambda item: item["count"],
            reverse=True,
        )[:5]
        ranked.append({**row, "top_reasons": top_reasons, "llm_defer_reasons": llm_defer_reasons})
    ranked.sort(key=lambda item: item["lane"])
    return {"lanes": ranked}


def _summarize_universe_meta(universe: dict[str, Any]) -> dict[str, Any]:
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    phi3_scan = meta.get("phi3_scan", {}) if isinstance(meta.get("phi3_scan", {}), dict) else {}
    watchlist = phi3_scan.get("watchlist", []) if isinstance(phi3_scan.get("watchlist", []), list) else []
    top_scored = meta.get("top_scored", []) if isinstance(meta.get("top_scored", []), list) else []
    hot_candidates = meta.get("hot_candidates", []) if isinstance(meta.get("hot_candidates", []), list) else []
    avoid_candidates = meta.get("avoid_candidates", []) if isinstance(meta.get("avoid_candidates", []), list) else []
    lane_supervision = meta.get("lane_supervision", []) if isinstance(meta.get("lane_supervision", []), list) else []
    conflicts = sum(1 for item in lane_supervision if isinstance(item, dict) and bool(item.get("lane_conflict")))
    news_context = meta.get("news_context", {}) if isinstance(meta.get("news_context", {}), dict) else {}

    return {
        "watchlist": watchlist[:8],
        "top_scored": top_scored[:8],
        "hot_candidates": hot_candidates[:5],
        "avoid_candidates": avoid_candidates[:5],
        "lane_conflicts": conflicts,
        "fear_greed": {
            "value": news_context.get("fng_value"),
            "label": news_context.get("fng_label"),
            "btc_dominance": news_context.get("btc_dominance"),
            "market_cap_change_24h": news_context.get("market_cap_change_24h"),
        },
    }


def _summarize_collector_depth(collector: dict[str, Any]) -> dict[str, Any]:
    per_symbol = collector.get("per_symbol_bars", {}) if isinstance(collector.get("per_symbol_bars", {}), dict) else {}
    ranked: list[dict[str, Any]] = []
    for symbol, bars in per_symbol.items():
        if not isinstance(bars, dict):
            continue
        ranked.append(
            {
                "symbol": symbol,
                "bars_1m": int(bars.get("1m", 0) or 0),
                "bars_1h": int(bars.get("1h", 0) or 0),
                "bars_7d": int(bars.get("7d", 0) or 0),
                "bars_30d": int(bars.get("30d", 0) or 0),
            }
        )
    ranked.sort(key=lambda item: (item["bars_1m"], item["bars_1h"]), reverse=True)
    return {"top_depth": ranked[:8]}


def _summarize_collector_status(collector: dict[str, Any]) -> str:
    if not isinstance(collector, dict) or not collector:
        return "missing"
    telemetry = collector.get("trade_counts", {})
    top_depth = collector.get("top_depth", {})
    if isinstance(telemetry, dict):
        processed = int(telemetry.get("processed", 0) or 0)
        invalid_volume = int(telemetry.get("invalid_volume", 0) or 0)
        dropped = int(telemetry.get("dropped_unknown_symbol", 0) or 0)
        if processed > 0:
            if invalid_volume > 0 or dropped > 0:
                return "degraded"
            return "healthy"
    if isinstance(top_depth, dict) and top_depth:
        return "healthy"
    return "stale"


def _summarize_nemo_recommendations(items: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    for item in items:
        action = str(item.get("action", "") or "").upper()
        symbol = str(item.get("symbol", "") or "")
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1
        if symbol:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    return {
        "count": len(items),
        "actions": action_counts,
        "symbols": sorted(
            ({"symbol": key, "count": value} for key, value in symbol_counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        )[:8],
    }


def _summarize_runtime_profile() -> dict[str, Any]:
    runtime_overrides = _read_json(ROOT / "configs" / "runtime_overrides.json")
    overrides = runtime_overrides.get("overrides", {}) if isinstance(runtime_overrides.get("overrides", {}), dict) else {}
    return {
        "max_open_positions": int(overrides.get("PORTFOLIO_MAX_OPEN_POSITIONS", 0) or 0),
        "max_weight_per_symbol": float(overrides.get("PORTFOLIO_MAX_WEIGHT_PER_SYMBOL", 0.0) or 0.0),
        "max_total_gross_exposure": float(overrides.get("PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE", 0.0) or 0.0),
        "strict_entry_enabled": bool(overrides.get("STABILIZATION_STRICT_ENTRY_ENABLED", False)),
        "min_entry_score": float(overrides.get("STABILIZATION_MIN_ENTRY_SCORE", 0.0) or 0.0),
        "allowed_lanes": str(overrides.get("STABILIZATION_ALLOWED_LANES", "") or ""),
        "require_trend_confirmed": bool(overrides.get("STABILIZATION_REQUIRE_TREND_CONFIRMED", False)),
        "require_short_tf_ready_15m": bool(overrides.get("STABILIZATION_REQUIRE_SHORT_TF_READY_15M", False)),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "dashboard.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "operator_ui", "port": UI_PORT}


@app.get("/api/status")
def status() -> dict[str, Any]:
    account_sync = _read_json(ROOT / "logs" / "account_sync.json")
    collector = _read_json(ROOT / "logs" / "collector_telemetry.json")
    open_orders = _read_open_orders()
    optimizer_reviews = _read_jsonl_tail(ROOT / "logs" / "nvidia_optimizer_reviews.jsonl", 5)
    decision_debug = _read_jsonl_tail(ROOT / "logs" / "decision_debug.jsonl", 12)
    alerts = _read_jsonl_tail(ROOT / "logs" / "alerts.jsonl", 20)
    nemo_recommendations = _read_jsonl_tail(ROOT / "logs" / "nemo_recommendations.jsonl", 20)
    visual_feed = _read_jsonl_tail(ROOT / "logs" / "visual_phi3_feed.jsonl", 5)
    universe = _load_universe()
    universe_summary = _summarize_universe_meta(universe)
    collector_depth = _summarize_collector_depth(collector)
    runtime_profile = _summarize_runtime_profile()

    latest_decision = decision_debug[-1] if decision_debug else {}
    latest_visual = next(
        (
            item
            for item in reversed(visual_feed)
            if item.get("event") == "visual_review" and item.get("review_ok")
        ),
        {},
    )

    portfolio_state = (
        account_sync.get("portfolio_state", {})
        if isinstance(account_sync.get("portfolio_state", {}), dict)
        else {}
    )
    active_pairs = universe.get("active_pairs", []) if isinstance(universe.get("active_pairs", []), list) else []
    lane_supervision = (
        universe.get("meta", {}).get("lane_supervision", [])
        if isinstance(universe.get("meta", {}), dict)
        else []
    )

    return {
        "portfolio": {
            "cash_usd": float(account_sync.get("cash_usd", 0.0)),
            "initial_equity_usd": float(account_sync.get("initial_equity_usd", 0.0)),
            "positions": _summarize_positions(account_sync),
            "position_count": len(account_sync.get("positions_state", [])),
            "dust_count": len(account_sync.get("ignored_dust", {})),
        },
        "accounting": _build_accounting_summary(account_sync, open_orders),
        "runtime_profile": runtime_profile,
        "universe": {
            "active_count": len(active_pairs),
            "active_pairs": active_pairs[:20],
            "reason": universe.get("reason", ""),
            "lane_supervision_count": len(lane_supervision),
        },
        "collector": collector,
        "collector_status": _summarize_collector_status(collector),
        "collector_depth": collector_depth,
        "latest_decision": latest_decision,
        "alerts": alerts[-10:],
        "alert_summary": _summarize_alerts(alerts),
        "decision_summary": _summarize_decisions(decision_debug),
        "lane_metrics": _summarize_lane_metrics(decision_debug),
        "recent_decisions": decision_debug[-5:],
        "latest_visual_review": latest_visual,
        "optimizer_reviews": optimizer_reviews,
        "nemo_recommendations": nemo_recommendations[-10:],
        "nemo_recommendation_summary": _summarize_nemo_recommendations(nemo_recommendations),
        "universe_summary": universe_summary,
        "diagnostics": {
            "trades_history_error": (
                account_sync.get("diagnostics", {}).get("trades_history_error", "")
                if isinstance(account_sync.get("diagnostics", {}), dict)
                else ""
            ),
            "latest_market_state": latest_decision.get("market_state", {}),
            "latest_posture": latest_decision.get("posture", {}),
            "latest_timings": latest_decision.get("timings", {}),
        },
    }


def main() -> None:
    uvicorn.run(app, host=UI_HOST, port=UI_PORT)


if __name__ == "__main__":
    main()
