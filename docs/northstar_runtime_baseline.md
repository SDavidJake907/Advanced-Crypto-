# North Star Runtime Baseline

Current target live shape for KrakenSK:

- deterministic scoring over the full active universe
- local advisory via `local_nemo`
- local strategist via `local`
- one small local Nemo batch on finalists only
- hard economics and risk still enforced in code

Machine-readable baseline:

- [northstar_runtime_baseline.json](/C:/Users/kitti/Desktop/KrakenSK/configs/northstar_runtime_baseline.json)

## Core Runtime Shape

- `ADVISORY_MODEL_PROVIDER=local_nemo`
- `NEMOTRON_PROVIDER=local`
- `NEMOTRON_STRATEGIST_PROVIDER=local`
- `NEMOTRON_BATCH_MODE=true`
- `NEMOTRON_BATCH_TOP_N=5`
- `NEMOTRON_TOP_CANDIDATE_COUNT=20`

Meaning:

- all 20 active symbols can still be scored and filtered in code
- only the top 5 finalists are sent to local Nemo
- this avoids both bad extremes:
  - 20 serial local model calls
  - one oversized 20-symbol local batch

## Live Gate Shape

- `STABILIZATION_STRICT_ENTRY_ENABLED=true`
- `STABILIZATION_ALLOWED_LANES=L1,L2,L3`
- `STABILIZATION_MIN_ENTRY_SCORE=64`
- `STABILIZATION_MIN_NET_EDGE_PCT=0.0`
- `STABILIZATION_REQUIRE_TP_AFTER_COST_VALID=true`
- `STABILIZATION_REQUIRE_TREND_CONFIRMED=true`
- `STABILIZATION_REQUIRE_SHORT_TF_READY_15M=true`
- `STABILIZATION_BLOCK_RANGING_MARKET=false`
- `STABILIZATION_REQUIRE_BUY_RECOMMENDATION=false`

Meaning:

- the gate stays real
- but it no longer blocks too many valid continuation or rotation setups across L1/L2/L3

## Capital Shape

Current live balance baseline: `$184.35` as of 2026-03-31.

- `PORTFOLIO_MAX_OPEN_POSITIONS=3`
- `PORTFOLIO_MAX_WEIGHT_PER_SYMBOL=0.5`
- `TRADER_PROPOSED_WEIGHT=0.5`
- `EXEC_RISK_PER_TRADE_PCT=10.0`

Meaning:

- the bot can use more of the account
- but still stays bounded by position and exposure rules
- at `$184.35`, minimum order size and dust handling still matter, but the account has more room for cleaner 3-slot deployment

## Why This Is The Right Shape

- one local model path is easier to debug than split-model live logic
- compact finalist packets are more reliable than huge prompt tables
- deterministic prefiltering is cheaper and more stable than asking the model to reject obvious bad setups
- the system keeps hard economic truth in code instead of prompt text

## Not Ready Signals

Do not treat this baseline as fully production-ready if any of these are still present:

- fresh `batch_parse_fallback_hold`
- unstable order-book telemetry
- shallow higher-timeframe history on active symbols
- repeated account-sync reconciliation issues
- churny low-edge live trades

Use [funding_readiness_checklist.md](/C:/Users/kitti/Desktop/KrakenSK/docs/funding_readiness_checklist.md) for the funding decision, not this file alone.
