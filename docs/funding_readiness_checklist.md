# Funding Readiness Checklist

Current status: not ready yet for a larger funding increase.

Current live balance baseline: `$184.35` as of 2026-03-31.

Use this checklist before increasing live capital materially.

This checklist applies to the current runtime shape:
- `ADVISORY_MODEL_PROVIDER=local_nemo`
- `NEMOTRON_PROVIDER=local`
- `NEMOTRON_STRATEGIST_PROVIDER=local`
- `NEMOTRON_BATCH_MODE=true`
- `NEMOTRON_BATCH_TOP_N=5`
- 20-symbol deterministic universe with top-5 local Nemo batch review

Do not evaluate readiness from mixed historical runs that used different model routing or broken batch behavior.

## Pass Criteria

### 1. Runtime Stability
- No fresh `batch_parse_fallback_hold` events in [logs/decision_debug.jsonl](C:/Users/kitti/Desktop/KrakenSK/logs/decision_debug.jsonl)
- No fresh `batch_parse_fallback` alerts in [logs/alerts.jsonl](C:/Users/kitti/Desktop/KrakenSK/logs/alerts.jsonl)
- `batch_nemo` continues to show `candidates: 5`, not full-universe batch load
- Cycle timing is stable enough that decisions are not materially stale

### 2. Data Quality
- [logs/collector_telemetry.json](C:/Users/kitti/Desktop/KrakenSK/logs/collector_telemetry.json) shows healthy `1h/7d/30d` depth for the active symbols
- No recurring `book_valid=false` periods
- No recurring stale or missing symbol warnings

### 3. Exchange / Sync Health
- [logs/account_sync.json](C:/Users/kitti/Desktop/KrakenSK/logs/account_sync.json) remains healthy over multiple sessions
- Order submissions, fills, and exits reconcile correctly
- `TradesHistory` permission issue is fixed, or the ledger fallback has been proven reliable over repeated cycles

### 4. Trade Quality
- No obvious same-symbol churn loop
- No repeated “open then immediate weak exit” pattern
- Entries and exits match the configured thesis and risk logic
- Cost floor is not blocking most otherwise-approved trades

### 5. Current-Config Live Sample
- At least 20-30 closed trades from the current architecture
- Sample is not polluted by old cloud/local/batch-broken periods
- Net result after fees is positive, or clearly improving with stable logic

### 6. Capital Usage
- The current `$184.35` account size is reflected in the live review baseline
- Position sizing is intentional for the larger account
- Idle cash behavior is understood
- Per-symbol exposure and total open slots are appropriate for the new funding level

## Current Recommendation

Do not add a larger amount until all of the above are true at the same time.

The system is improving, but readiness should be judged from:
- the current local-only Nemo path
- the current top-5 batch finalist flow
- the current stabilization filters

Not from older mixed runs.

With an `$184.35` balance, the immediate goal is still stable execution and clean trade quality, not aggressive scaling.

## Daily Review

Check these each day:
1. [logs/decision_debug.jsonl](C:/Users/kitti/Desktop/KrakenSK/logs/decision_debug.jsonl)
2. [logs/alerts.jsonl](C:/Users/kitti/Desktop/KrakenSK/logs/alerts.jsonl)
3. [logs/collector_telemetry.json](C:/Users/kitti/Desktop/KrakenSK/logs/collector_telemetry.json)
4. [logs/account_sync.json](C:/Users/kitti/Desktop/KrakenSK/logs/account_sync.json)

If any of these regress materially, reset the readiness clock.
