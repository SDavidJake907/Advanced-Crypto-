from __future__ import annotations

import argparse
import json

from core.memory.setup_reliability import build_setup_reliability_summary
from core.memory.trade_memory import TradeMemoryStore


def _print_group(name: str, payload: dict) -> None:
    print(f"[{name}] groups={payload.get('count', 0)}")
    strongest = payload.get("strongest", [])
    weakest = payload.get("weakest", [])
    if strongest:
        print("  strongest:")
        for item in strongest:
            print(
                "    "
                f"{item['key']} trades={item['trade_count']} "
                f"win={item['win_rate']:.3f} exp={item['expectancy_pct']:.4f} "
                f"verdict={item['verdict']}"
            )
    if weakest:
        print("  weakest:")
        for item in weakest:
            print(
                "    "
                f"{item['key']} trades={item['trade_count']} "
                f"win={item['win_rate']:.3f} exp={item['expectancy_pct']:.4f} "
                f"verdict={item['verdict']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report setup-level trade reliability summaries.")
    parser.add_argument("--lookback", type=int, default=200)
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    summary = build_setup_reliability_summary(
        store=TradeMemoryStore(),
        lookback=args.lookback,
        min_trades=args.min_trades,
        limit=args.limit,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    for group_name, payload in summary.items():
        _print_group(group_name, payload)


if __name__ == "__main__":
    main()
