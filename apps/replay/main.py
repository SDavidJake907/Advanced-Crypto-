from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import uuid

import pandas as pd
from dotenv import load_dotenv

from core.config.runtime import get_proposed_weight
from core.data.loader import CandleLoader
from core.features.batch import compute_features_batch, slice_features_for_asset
from core.risk.exits import build_exit_plan, evaluate_exit, maybe_arm_break_even, maybe_update_trailing
from core.risk.portfolio import PositionState
from core.state.portfolio import PortfolioState
from core.state.system_record import record_replay_run, record_shadow_decision
from core.state.trace import DecisionTrace, append_trace
from core.validation.engine_runner import EngineComponents, build_engine_components, decide_with_engine

load_dotenv()


def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _resolve_path(symbol: str, base_path: str, template: str) -> str:
    if template:
        return template.format(symbol=_symbol_token(symbol))
    return base_path


def _load_frame(symbol: str, *, base_path: str, template: str) -> pd.DataFrame:
    return CandleLoader(_resolve_path(symbol, base_path, template)).load().reset_index(drop=True)


def _filter_to_timestamp(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    subset = frame[frame["timestamp"] <= timestamp]
    return subset.tail(200).reset_index(drop=True)


def _compute_hold_minutes(entry_bar_ts: str | None, exit_bar_ts: str | None) -> float:
    if not entry_bar_ts or not exit_bar_ts:
        return 0.0
    try:
        start = pd.Timestamp(entry_bar_ts)
        end = pd.Timestamp(exit_bar_ts)
        return max((end - start).total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0


def _compute_pnl_pct(side: str, entry_price: float | None, exit_price: float) -> float:
    if entry_price is None or entry_price <= 0.0 or exit_price <= 0.0:
        return 0.0
    if side == "LONG":
        return (exit_price / entry_price) - 1.0
    return (entry_price / exit_price) - 1.0


def _compute_equity(portfolio: PortfolioState) -> float:
    equity = float(portfolio.cash)
    for symbol, qty in (portfolio.positions or {}).items():
        mark = float((portfolio.position_marks or {}).get(symbol, 0.0) or 0.0)
        equity += float(qty) * mark
    return equity


async def replay_loop(args: argparse.Namespace) -> dict[str, object]:
    shadow_engine_name = str(args.shadow_engine or "").strip().lower()
    if shadow_engine_name and shadow_engine_name not in {"classic", "llm"}:
        raise ValueError(f"Unsupported shadow engine: {args.shadow_engine}")
    baseline = {
        "components": build_engine_components(args.engine),
        "portfolio": PortfolioState(cash=args.start_cash, initial_equity=args.start_cash),
        "positions_state": PositionState(),
        "trades": 0,
        "exits": 0,
        "hold_reasons": {},
        "outcome_rows": [],
    }
    shadow = None
    if shadow_engine_name:
        shadow = {
            "components": build_engine_components(shadow_engine_name),
            "portfolio": PortfolioState(cash=args.start_cash, initial_equity=args.start_cash),
            "positions_state": PositionState(),
            "trades": 0,
            "exits": 0,
            "hold_reasons": {},
            "outcome_rows": [],
        }

    one_m_base = os.getenv("CANDLES_PATH", "logs/candles_ETHUSD.csv")
    one_m_template = os.getenv("CANDLES_PATH_TEMPLATE", "")
    one_h_base = os.getenv("CANDLES_PATH_1H", "logs/candles_ETHUSD_1h.csv")
    one_h_template = os.getenv("CANDLES_PATH_TEMPLATE_1H", "")
    seven_d_base = os.getenv("CANDLES_PATH_7D", "logs/candles_ETHUSD_7d.csv")
    seven_d_template = os.getenv("CANDLES_PATH_TEMPLATE_7D", "")
    thirty_d_base = os.getenv("CANDLES_PATH_30D", "logs/candles_ETHUSD_30d.csv")
    thirty_d_template = os.getenv("CANDLES_PATH_TEMPLATE_30D", "")

    frames_1m = {symbol: _load_frame(symbol, base_path=one_m_base, template=one_m_template) for symbol in args.symbols}
    frames_1h = {symbol: _load_frame(symbol, base_path=one_h_base, template=one_h_template) for symbol in args.symbols}
    frames_7d = {symbol: _load_frame(symbol, base_path=seven_d_base, template=seven_d_template) for symbol in args.symbols}
    frames_30d = {symbol: _load_frame(symbol, base_path=thirty_d_base, template=thirty_d_template) for symbol in args.symbols}

    min_len = min(len(frame) for frame in frames_1m.values())
    start_idx = max(args.warmup_bars, 30)
    end_idx = min(min_len, start_idx + args.max_steps) if args.max_steps else min_len

    trace_path = Path(args.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    shadow_trace_path = Path(args.shadow_trace_path) if shadow_engine_name else None
    if shadow_trace_path is not None:
        shadow_trace_path.parent.mkdir(parents=True, exist_ok=True)
        if shadow_trace_path.exists():
            shadow_trace_path.unlink()

    shadow_comparisons: list[dict[str, object]] = []

    def _process_engine_state(
        engine_state: dict[str, object],
        *,
        symbol: str,
        features: dict[str, object],
        symbols: list[str],
        proposed_weight: float,
        trace_target: Path,
    ) -> dict[str, object]:
        positions_state = engine_state["positions_state"]
        portfolio = engine_state["portfolio"]
        assert isinstance(positions_state, PositionState)
        assert isinstance(portfolio, PortfolioState)
        existing_position = positions_state.get(symbol)
        live_qty = float(portfolio.positions.get(symbol, 0.0))
        current_price = float(features.get("price", 0.0) or 0.0)
        if current_price > 0.0:
            portfolio.mark_to_market({symbol: current_price})

        if existing_position is not None and live_qty > 0.0:
            updated_position = maybe_arm_break_even(existing_position, float(features.get("price", 0.0)))
            updated_position = maybe_update_trailing(
                updated_position,
                float(features.get("price", 0.0)),
                float(features.get("atr", 0.0)),
            )
            positions_state.add_or_update(updated_position)
            exit_reason = evaluate_exit(updated_position, float(features.get("price", 0.0)), features=features)
            if exit_reason:
                exec_result = engine_state["components"].executor.execute_exit(
                    symbol=symbol,
                    side=updated_position.side,
                    qty=abs(live_qty),
                    price=float(features.get("price", 0.0)),
                    features=features,
                    exit_reason=exit_reason,
                )
                exec_result["bar_ts"] = features.get("bar_ts")
                exec_result["bar_idx"] = features.get("bar_idx")
                state_change = portfolio.apply_execution(exec_result)
                positions_state.remove(symbol)
                pnl_pct = _compute_pnl_pct(updated_position.side, updated_position.entry_price, float(exec_result["price"]))
                hold_minutes = _compute_hold_minutes(updated_position.entry_bar_ts, features.get("bar_ts"))
                engine_state["outcome_rows"].append(
                    {
                        "symbol": symbol,
                        "lane": updated_position.lane,
                        "side": updated_position.side,
                        "pnl_pct": pnl_pct,
                        "hold_minutes": hold_minutes,
                        "exit_reason": exit_reason,
                    }
                )
                engine_state["exits"] = int(engine_state["exits"]) + 1
                append_trace(
                    str(trace_target),
                    DecisionTrace(
                        timestamp=datetime.utcnow(),
                        symbol=symbol,
                        features=features,
                        signal="EXIT",
                        risk_checks=[exit_reason],
                        execution=exec_result,
                        state_change=state_change,
                    ),
                )
                return {
                    "signal": "EXIT",
                    "risk_checks": [exit_reason],
                    "execution": exec_result,
                    "state_change": state_change,
                }

        decision = decide_with_engine(
            engine_state["components"],
            symbol=symbol,
            features=features,
            portfolio_state=portfolio,
            positions_state=positions_state,
            symbols=symbols,
            proposed_weight=proposed_weight,
        )
        exec_result = decision.execution
        state_change = {}
        if existing_position is not None and abs(live_qty) > 0.0 and exec_result.get("status") == "filled":
            exec_result = {
                "status": "no_trade",
                "reason": "already_holding_symbol",
                "bar_ts": features.get("bar_ts"),
                "bar_idx": features.get("bar_idx"),
            }
        if exec_result.get("status") == "filled":
            state_change = portfolio.apply_execution(exec_result)
            weight = proposed_weight * float(decision.portfolio_decision.get("size_factor", 1.0) or 1.0)
            entry_reason = ""
            if engine_state["components"].engine_name == "llm":
                entry_reason = str(exec_result.get("nemotron", {}).get("reason", ""))
            else:
                entry_reason = str(decision.signal).lower()
            positions_state.add_or_update(
                build_exit_plan(
                    symbol=symbol,
                    side=decision.signal,
                    weight=weight,
                    entry_price=float(exec_result["price"]),
                    atr=float(features.get("atr", 0.0)),
                    entry_bar_ts=features.get("bar_ts"),
                    entry_bar_idx=features.get("bar_idx"),
                    entry_reasons=[entry_reason, f"lane:{features.get('lane', 'L3')}"],
                    lane=features.get("lane"),
                )
            )
            engine_state["trades"] = int(engine_state["trades"]) + 1
        else:
            reason = str(exec_result.get("nemotron", {}).get("reason", exec_result.get("reason", "no_trade")))
            hold_reasons = engine_state["hold_reasons"]
            assert isinstance(hold_reasons, dict)
            hold_reasons[reason] = hold_reasons.get(reason, 0) + 1

        append_trace(
            str(trace_target),
            DecisionTrace(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                features=features,
                signal=decision.signal,
                risk_checks=decision.risk_checks,
                execution=exec_result,
                state_change=state_change,
            ),
        )
        return {
            "signal": decision.signal,
            "risk_checks": decision.risk_checks,
            "execution": exec_result,
            "state_change": state_change,
        }

    for idx in range(start_idx, end_idx):
        ohlc_by_symbol: dict[str, pd.DataFrame] = {}
        ohlc_1h_by_symbol: dict[str, pd.DataFrame] = {}
        ohlc_7d_by_symbol: dict[str, pd.DataFrame] = {}
        ohlc_30d_by_symbol: dict[str, pd.DataFrame] = {}
        current_ts: pd.Timestamp | None = None

        for symbol in args.symbols:
            frame_1m = frames_1m[symbol].iloc[: idx + 1].copy().reset_index(drop=True)
            ohlc_by_symbol[symbol] = frame_1m.tail(200)
            current_ts = pd.Timestamp(frame_1m["timestamp"].iloc[-1])
            ohlc_1h_by_symbol[symbol] = _filter_to_timestamp(frames_1h[symbol], current_ts)
            ohlc_7d_by_symbol[symbol] = _filter_to_timestamp(frames_7d[symbol], current_ts)
            ohlc_30d_by_symbol[symbol] = _filter_to_timestamp(frames_30d[symbol], current_ts)

        features_batch = compute_features_batch(
            ohlc_by_symbol,
            ohlc_1h_by_symbol=ohlc_1h_by_symbol,
            ohlc_7d_by_symbol=ohlc_7d_by_symbol,
            ohlc_30d_by_symbol=ohlc_30d_by_symbol,
        )

        for asset_idx, symbol in enumerate(features_batch["symbols"]):
            features = slice_features_for_asset(features_batch, asset_idx)
            features["proposed_weight"] = get_proposed_weight(symbol, features.get("lane"))
            proposed_weight = float(features["proposed_weight"])
            baseline_result = _process_engine_state(
                baseline,
                symbol=symbol,
                features=features,
                symbols=features_batch["symbols"],
                proposed_weight=proposed_weight,
                trace_target=trace_path,
            )
            if shadow is not None and shadow_trace_path is not None:
                shadow_result = _process_engine_state(
                    shadow,
                    symbol=symbol,
                    features=features,
                    symbols=features_batch["symbols"],
                    proposed_weight=proposed_weight,
                    trace_target=shadow_trace_path,
                )
                comparison = {
                    "ts": str(features.get("bar_ts") or current_ts or datetime.utcnow().isoformat()),
                    "symbol": symbol,
                    "baseline_engine": baseline["components"].engine_name,
                    "shadow_engine": shadow["components"].engine_name,
                    "baseline_signal": baseline_result["signal"],
                    "shadow_signal": shadow_result["signal"],
                    "baseline_status": str((baseline_result["execution"] or {}).get("status", "")),
                    "shadow_status": str((shadow_result["execution"] or {}).get("status", "")),
                    "signal_match": baseline_result["signal"] == shadow_result["signal"],
                    "status_match": str((baseline_result["execution"] or {}).get("status", "")) == str((shadow_result["execution"] or {}).get("status", "")),
                }
                shadow_comparisons.append(comparison)
                record_shadow_decision(comparison)

    final_marks = {}
    if end_idx > start_idx:
        final_idx = end_idx - 1
        for symbol in args.symbols:
            frame = frames_1m[symbol]
            if final_idx < len(frame):
                final_marks[symbol] = float(frame["close"].iloc[final_idx])

    def _liquidate_open_positions(engine_state: dict[str, object], trace_target: Path) -> None:
        portfolio = engine_state["portfolio"]
        positions_state = engine_state["positions_state"]
        assert isinstance(portfolio, PortfolioState)
        assert isinstance(positions_state, PositionState)
        portfolio.mark_to_market(final_marks)
        for position in list(positions_state.all()):
            live_qty = float(portfolio.positions.get(position.symbol, 0.0))
            mark_price = float(final_marks.get(position.symbol, 0.0) or 0.0)
            if live_qty == 0.0 or mark_price <= 0.0:
                continue
            exec_result = engine_state["components"].executor.execute_exit(
                symbol=position.symbol,
                side=position.side,
                qty=abs(live_qty),
                price=mark_price,
                features={"price": mark_price, "spread_pct": 0.0},
                exit_reason="replay_end_liquidation",
            )
            state_change = portfolio.apply_execution(exec_result)
            positions_state.remove(position.symbol)
            pnl_pct = _compute_pnl_pct(position.side, position.entry_price, float(exec_result["price"]))
            hold_minutes = _compute_hold_minutes(position.entry_bar_ts, str(frames_1m[position.symbol]["timestamp"].iloc[final_idx]))
            engine_state["outcome_rows"].append(
                {
                    "symbol": position.symbol,
                    "lane": position.lane,
                    "side": position.side,
                    "pnl_pct": pnl_pct,
                    "hold_minutes": hold_minutes,
                    "exit_reason": "replay_end_liquidation",
                }
            )
            engine_state["exits"] = int(engine_state["exits"]) + 1
            append_trace(
                str(trace_target),
                DecisionTrace(
                    timestamp=datetime.utcnow(),
                    symbol=position.symbol,
                    features={"price": mark_price},
                    signal="EXIT",
                    risk_checks=["replay_end_liquidation"],
                    execution=exec_result,
                    state_change=state_change,
                ),
            )
        portfolio.mark_to_market(final_marks)

    _liquidate_open_positions(baseline, trace_path)
    if shadow is not None and shadow_trace_path is not None:
        _liquidate_open_positions(shadow, shadow_trace_path)

    baseline_equity = _compute_equity(baseline["portfolio"])
    baseline_summary = {
        "engine": baseline["components"].engine_name,
        "final_cash": baseline["portfolio"].cash,
        "final_equity": baseline_equity,
        "final_positions": baseline["portfolio"].positions,
        "pnl": baseline["portfolio"].pnl,
        "trades": baseline["trades"],
        "exits": baseline["exits"],
        "hold_reasons_top": dict(sorted(baseline["hold_reasons"].items(), key=lambda item: item[1], reverse=True)[:10]),
        "outcomes": baseline["outcome_rows"][-20:],
        "trace_path": str(trace_path),
    }
    summary = {
        "run_id": f"replay-{uuid.uuid4().hex[:12]}",
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "symbols": args.symbols,
        "start_cash": args.start_cash,
        "bars_replayed": max(end_idx - start_idx, 0),
        "baseline_engine": baseline["components"].engine_name,
        "shadow_engine": shadow["components"].engine_name if shadow is not None else "",
        "baseline": baseline_summary,
    }
    if shadow is not None and shadow_trace_path is not None:
        divergence_count = sum(1 for item in shadow_comparisons if not item["signal_match"] or not item["status_match"])
        summary["shadow"] = {
            "engine": shadow["components"].engine_name,
            "final_cash": shadow["portfolio"].cash,
            "final_equity": _compute_equity(shadow["portfolio"]),
            "final_positions": shadow["portfolio"].positions,
            "pnl": shadow["portfolio"].pnl,
            "trades": shadow["trades"],
            "exits": shadow["exits"],
            "hold_reasons_top": dict(sorted(shadow["hold_reasons"].items(), key=lambda item: item[1], reverse=True)[:10]),
            "outcomes": shadow["outcome_rows"][-20:],
            "trace_path": str(shadow_trace_path),
        }
        summary["comparison"] = {
            "decision_count": len(shadow_comparisons),
            "divergence_count": divergence_count,
            "divergence_rate": round(divergence_count / len(shadow_comparisons), 4) if shadow_comparisons else 0.0,
            "baseline_pnl": baseline["portfolio"].pnl,
            "shadow_pnl": shadow["portfolio"].pnl,
            "pnl_delta": float(shadow["portfolio"].pnl) - float(baseline["portfolio"].pnl),
            "trade_delta": int(shadow["trades"]) - int(baseline["trades"]),
            "sample": shadow_comparisons[-20:],
        }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    record_replay_run(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay KrakenSK decisions on historical candles.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to replay, e.g. BTC/USD DOGE/USD")
    parser.add_argument("--start-cash", type=float, default=1000.0)
    parser.add_argument("--warmup-bars", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means replay all available bars")
    parser.add_argument("--engine", choices=["classic", "llm"], default="llm")
    parser.add_argument("--shadow-engine", default="", help="Optional shadow engine: classic or llm")
    parser.add_argument("--trace-path", default="logs/replay_traces.jsonl")
    parser.add_argument("--shadow-trace-path", default="logs/replay_shadow_traces.jsonl")
    parser.add_argument("--summary-path", default="logs/replay_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(replay_loop(args))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
