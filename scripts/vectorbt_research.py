from __future__ import annotations

import argparse

from core.research.vectorbt_harness import (
    discover_history_symbols,
    run_batch_threshold_sweep,
    run_threshold_sweep,
    run_walk_forward_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline vectorbt threshold sweep for KrakenSK")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--history-dir", default="logs/history")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--hold-bars", type=int, default=24)
    parser.add_argument("--fee-pct", type=float, default=0.40)
    parser.add_argument("--slippage-pct", type=float, default=0.10)
    parser.add_argument("--lane", default="")
    parser.add_argument("--require-bullish-divergence", action="store_true")
    parser.add_argument("--min-net-edge-pct", type=float, default=None)
    parser.add_argument("--bullish-divergence-score-bonus", type=float, default=0.0)
    parser.add_argument("--bullish-divergence-promotion-window", type=float, default=0.0)
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--per-symbol-csv", default="")
    parser.add_argument("--aggregate-csv", default="")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-bars", type=int, default=96)
    parser.add_argument("--test-bars", type=int, default=24)
    parser.add_argument("--step-bars", type=int, default=0)
    parser.add_argument("--windows-csv", default="")
    parser.add_argument(
        "--thresholds",
        default="55,60,65,70,75",
        help="comma-separated entry_score thresholds",
    )
    args = parser.parse_args()

    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    if args.batch:
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if not symbols:
            symbols = discover_history_symbols(args.history_dir, timeframe=args.timeframe)
        result = run_batch_threshold_sweep(
            symbols,
            history_dir=args.history_dir,
            timeframe=args.timeframe,
            entry_score_thresholds=thresholds,
            lane_filter=(args.lane or None),
            require_bullish_divergence=bool(args.require_bullish_divergence),
            min_net_edge_pct=args.min_net_edge_pct,
            bullish_divergence_score_bonus=args.bullish_divergence_score_bonus,
            bullish_divergence_promotion_window=args.bullish_divergence_promotion_window,
            hold_bars=args.hold_bars,
            fee_pct=args.fee_pct,
            slippage_pct=args.slippage_pct,
            per_symbol_csv_path=(args.per_symbol_csv or None),
            aggregate_csv_path=(args.aggregate_csv or None),
        )
        print(result.aggregate_summary.to_string(index=False))
        return

    if args.walk_forward:
        result = run_walk_forward_validation(
            args.symbol,
            history_dir=args.history_dir,
            timeframe=args.timeframe,
            entry_score_thresholds=thresholds,
            lane_filter=(args.lane or None),
            require_bullish_divergence=bool(args.require_bullish_divergence),
            min_net_edge_pct=args.min_net_edge_pct,
            bullish_divergence_score_bonus=args.bullish_divergence_score_bonus,
            bullish_divergence_promotion_window=args.bullish_divergence_promotion_window,
            hold_bars=args.hold_bars,
            fee_pct=args.fee_pct,
            slippage_pct=args.slippage_pct,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=(args.step_bars or None),
            summary_csv_path=(args.summary_csv or None),
            windows_csv_path=(args.windows_csv or None),
        )
        print(result.summary.to_string(index=False))
        if not result.windows.empty:
            print()
            print(result.windows.to_string(index=False))
        return

    result = run_threshold_sweep(
        args.symbol,
        history_dir=args.history_dir,
        timeframe=args.timeframe,
        entry_score_thresholds=thresholds,
        lane_filter=(args.lane or None),
        require_bullish_divergence=bool(args.require_bullish_divergence),
        min_net_edge_pct=args.min_net_edge_pct,
        bullish_divergence_score_bonus=args.bullish_divergence_score_bonus,
        bullish_divergence_promotion_window=args.bullish_divergence_promotion_window,
        hold_bars=args.hold_bars,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        summary_csv_path=(args.summary_csv or None),
    )
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
