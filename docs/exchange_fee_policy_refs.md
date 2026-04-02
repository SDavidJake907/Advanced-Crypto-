# Exchange Fee And Policy References

This file is a quick operator reference for exchange-fee assumptions and official policy links that matter to the live stack.

## Kraken Notes

Important current clarification:
- `Kraken+` does **not** apply to `Kraken Pro`, `Kraken Pro app/web`, or API-driven spot/futures trading.
- `Kraken+` fee benefits apply to Kraken web/app buy, sell, and convert flows, not to the Pro/API execution path this repository uses for live order-book trading.

That means you should **not** lower live maker/taker assumptions in this repository only because you have a `Kraken+` subscription.

## Official Kraken References

- Kraken fee schedule:
  - https://www.kraken.com/features/fee-schedule
- Kraken+ subscription overview:
  - https://support.kraken.com/articles/kraken-faq-subscription-service-overview
- Kraken+ benefits FAQ:
  - https://support.kraken.com/articles/kraken-subscription-service-benefits/

## What This Means For Runtime Config

If you are trading through Kraken Pro / order-book / API execution, fee assumptions should be based on:
- your actual Kraken Pro fee tier
- whether your traded pairs qualify for the maker rebate schedule
- whether your executions are predominantly maker or taker

Relevant runtime settings in this repo:
- [`configs/runtime_overrides.json`](../configs/runtime_overrides.json)
  - `EXEC_MAKER_FEE_PCT`
  - `EXEC_TAKER_FEE_PCT`
  - `TRADE_COST_MIN_EDGE_MULT`
  - `TRADE_COST_MIN_EXPECTED_EDGE_PCT`

## Operator Guidance

- If you only have `Kraken+`, keep Pro/API fee assumptions unchanged.
- If your Kraken Pro 30-day volume tier has changed, update fee assumptions only from the actual Pro fee table.
- If you want pair-specific fee treatment for rebate-eligible symbols, that should be implemented explicitly in code/config rather than by assuming one global zero-fee profile.
