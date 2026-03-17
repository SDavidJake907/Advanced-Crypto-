from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core.config.runtime import get_proposed_weight
from core.data.loader import CandleLoader
from core.execution.cpp_exec import CppExecutor
from core.features.batch import compute_features_batch, slice_features_for_asset
from core.llm.nemotron import NemotronStrategist
from core.risk.basic_risk import BasicRiskEngine
from core.risk.exits import build_exit_plan, evaluate_exit, maybe_arm_break_even, maybe_update_trailing
from core.risk.portfolio import PortfolioConfig, PositionState
from core.state.portfolio import PortfolioState
from core.state.trace import DecisionTrace, append_trace
from core.strategy.simple_momo import SimpleMomentumStrategy

load_dotenv()


def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _resolve_path(symbol: str, base_path: str, template: str) -> str:
    if template:
        return template.format(symbol=_symbol_token(symbol))
    return base_path


def _load_frame(symbol: str, *, base_path: str, template: str) -> pd.DataFrame:
    return CandleLoader(_resolve_path(symbol, base_path, template)).load().tail(5000).reset_index(drop=True)


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


async def replay_loop(args: argparse.Namespace) -> dict[str, object]:
    original_execution_mode = os.environ.get("EXECUTION_MODE")
    original_kraken_env = os.environ.get("KRAKEN_ENV")
    os.environ["EXECUTION_MODE"] = "paper"
    os.environ["KRAKEN_ENV"] = "paper"
    try:
        strategy = SimpleMomentumStrategy()
        risk_engine = BasicRiskEngine()
        portfolio_config = PortfolioConfig.from_runtime()
        executor = CppExecutor()
        nemotron = NemotronStrategist(
            strategy=strategy,
            risk_engine=risk_engine,
            portfolio_config=portfolio_config,
            executor=executor,
        )
    finally:
        if original_execution_mode is None:
            os.environ.pop("EXECUTION_MODE", None)
        else:
            os.environ["EXECUTION_MODE"] = original_execution_mode
        if original_kraken_env is None:
            os.environ.pop("KRAKEN_ENV", None)
        else:
            os.environ["KRAKEN_ENV"] = original_kraken_env

    portfolio = PortfolioState(cash=args.start_cash, initial_equity=args.start_cash)
    positions_state = PositionState()

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

    trades = 0
    exits = 0
    hold_reasons: dict[str, int] = {}
    outcome_rows: list[dict[str, object]] = []

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

            existing_position = positions_state.get(symbol)
            live_qty = float(portfolio.positions.get(symbol, 0.0))

            if existing_position is not None and live_qty > 0.0:
                updated_position = maybe_arm_break_even(existing_position, float(features.get("price", 0.0)))
                updated_position = maybe_update_trailing(
                    updated_position,
                    float(features.get("price", 0.0)),
                    float(features.get("atr", 0.0)),
                )
                positions_state.add_or_update(updated_position)
                exit_reason = evaluate_exit(updated_position, float(features.get("price", 0.0)))
                if exit_reason:
                    exec_result = executor.execute_exit(
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
                    outcome_rows.append(
                        {
                            "symbol": symbol,
                            "lane": updated_position.lane,
                            "side": updated_position.side,
                            "pnl_pct": pnl_pct,
                            "hold_minutes": hold_minutes,
                            "exit_reason": exit_reason,
                        }
                    )
                    exits += 1
                    append_trace(
                        str(trace_path),
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
                    continue

            decision = nemotron.decide(
                symbol=symbol,
                features=features,
                portfolio_state=portfolio,
                positions_state=positions_state,
                symbols=features_batch["symbols"],
                proposed_weight=proposed_weight,
            )
            exec_result = decision.execution
            state_change = {}
            if exec_result.get("status") == "filled":
                state_change = portfolio.apply_execution(exec_result)
                weight = proposed_weight * decision.portfolio_decision["size_factor"]
                positions_state.add_or_update(
                    build_exit_plan(
                        symbol=symbol,
                        side=decision.signal,
                        weight=weight,
                        entry_price=float(exec_result["price"]),
                        atr=float(features.get("atr", 0.0)),
                        entry_bar_ts=features.get("bar_ts"),
                        entry_bar_idx=features.get("bar_idx"),
                        entry_reasons=[
                            str(exec_result.get("nemotron", {}).get("reason", "")),
                            f"lane:{features.get('lane', 'L3')}",
                        ],
                        lane=features.get("lane"),
                    )
                )
                trades += 1
            else:
                reason = str(exec_result.get("nemotron", {}).get("reason", exec_result.get("reason", "no_trade")))
                hold_reasons[reason] = hold_reasons.get(reason, 0) + 1

            append_trace(
                str(trace_path),
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

    summary = {
        "symbols": args.symbols,
        "start_cash": args.start_cash,
        "final_cash": portfolio.cash,
        "final_positions": portfolio.positions,
        "pnl": portfolio.pnl,
        "trades": trades,
        "exits": exits,
        "bars_replayed": max(end_idx - start_idx, 0),
        "hold_reasons_top": dict(sorted(hold_reasons.items(), key=lambda item: item[1], reverse=True)[:10]),
        "outcomes": outcome_rows[-20:],
        "trace_path": str(trace_path),
    }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay KrakenSK decisions on historical candles.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to replay, e.g. BTC/USD DOGE/USD")
    parser.add_argument("--start-cash", type=float, default=1000.0)
    parser.add_argument("--warmup-bars", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means replay all available bars")
    parser.add_argument("--trace-path", default="logs/replay_traces.jsonl")
    parser.add_argument("--summary-path", default="logs/replay_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(replay_loop(args))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
