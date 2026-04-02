# KrakenSK Research Action Plan

This converts the recent research into concrete implementation priorities for KrakenSK.

## 1. Exit Architecture

Goal:
- Hard stop = thesis invalidation
- Trailing stop = later profit protection
- Structure filter = thesis still alive or not

Current status:
- Improved from before
- Still vulnerable to ATR/trailing clipping structured L2/L3 trades too early

What to change next:
- Keep `core/risk/position_monitor.py` focused on state, not micro stop placement.
- Keep `core/risk/exits.py` as the mechanical authority for:
  - hard failure exits
  - stop-loss / trail logic
  - posture tightening
- Treat structured `L2/L3` trades as:
  - no early `STALL/ROTATE/WEAKEN` while structure is intact
  - ATR/trail only after larger MFE threshold
  - structure break overrides trail patience

Concrete next changes:
- Add an explicit `mfe_pct` or `mfe_r` tracker to open positions.
- Arm trailing only after `mfe_r >= threshold` for L2/L3 structured trades.
- Separate:
  - `hard_stop_level`
  - `trail_stop`
  - `structure_break_stop`

Files:
- `core/risk/exits.py`
- `core/risk/position_monitor.py`
- `core/risk/portfolio.py`

## 2. Structure-First Exit Priority

Goal:
- Structure dominates secondary triggers.

Primary exit triggers:
- structure break
- breakout failure
- failed retest
- hard lane failure
- actual stop hit

Secondary triggers:
- stall
- spread widening
- rank decay
- replacement candidate available

Rule:
- Secondary triggers should tighten, not kill, while structure remains intact.

Concrete next changes:
- Add explicit `structure_state` to features or position context:
  - `intact`
  - `fragile`
  - `broken`
- Make `review_live_exit_state()` consume `structure_state` directly.

Files:
- `core/risk/exits.py`
- `core/features/batch.py`

## 3. Maker-First Entry and Exit

Goal:
- No casual market entries.
- Market orders only for explicit emergency exits.

Current status:
- Invalid-book entries now blocked.
- Maker preference exists.
- But maker-first is still not a full retry/reprice system.

What to add:
- Post-only entry retry loop.
- Limit reprice logic with bounded retries.
- Explicit order aging and replace policy.

Suggested policy:
- Entry:
  - try post-only limit
  - if rejected for immediate match, reprice 1-2 ticks away
  - retry bounded number of times
  - if still not fillable, skip rather than taker-chase
- Exit:
  - for non-emergency exits, rest maker limit
  - if stale, reprice on timer
  - only fallback to market for:
    - hard stop
    - hard fail
    - L4 collapse

Files:
- `core/execution/order_policy.py`
- `core/execution/kraken_live.py`
- `core/state/open_orders.py`

## 4. Order Aging / Cancel-Replace

Goal:
- Stop repeated canceled exits from degenerating into sloppy market fallback.

What to add:
- exit order age buckets
- number of reprices attempted
- last quoted spread at submit
- reason for fallback to market

Suggested rules:
- Reprice only if:
  - order age exceeds threshold
  - spread remains sane
  - structure has not broken
- Fallback to market only if:
  - emergency exit
  - stale exit exceeded max retries
  - structure has broken decisively

Files:
- `core/state/open_orders.py`
- `core/execution/kraken_live.py`

## 5. MFE / MAE Instrumentation

Goal:
- Tune exits from your own data, not instinct.

What to record per trade:
- maximum favorable excursion (MFE)
- maximum adverse excursion (MAE)
- time to MFE
- time to exit
- exit reason
- lane
- hold style

Use:
- determine whether structured L2/L3 trails arm too early
- compare realized capture vs potential capture
- choose trail arm thresholds per lane

Files:
- `apps/trader/main.py`
- `core/memory/trade_memory.py`
- `core/state/system_record.py`

## 6. Rotation Rules

Goal:
- Rotation should be cost-aware and thesis-aware.

Current principle:
- Do not rotate just because another symbol scores higher.

What to enforce:
- Only rotate when:
  - current trade is weakening or broken
  - replacement advantage exceeds a margin
  - expected edge plausibly exceeds cost

Add:
- `rotation_margin_score`
- `rotation_cost_buffer`

Files:
- `core/risk/position_monitor.py`
- `core/risk/exits.py`
- `core/risk/portfolio.py`

## 7. Small-Account Cost Rules

Goal:
- Avoid churn that can never beat fees and exchange minimums.

What to enforce:
- minimum expected move threshold by lane
- minimum expected net edge after explicit fee/slippage estimate
- skip trades whose plausible capture zone is too small

Files:
- `core/risk/fee_filter.py`
- `core/execution/clamp.py`
- `core/execution/kraken_live.py`

## 8. GPT Reviewer Workflow

Goal:
- Use GPT as auditor/reviewer, not trade decider.

Daily review tasks for GPT:
- classify exits:
  - `STRUCTURE_BREAK`
  - `VOLATILITY_STOP`
  - `TRAIL_CLIP`
  - `ROTATION_EXIT`
  - `RANK_EXIT`
  - `EMERGENCY_EXIT`
- identify contradictions:
  - maker-first intended but market used
  - structure intact but stop clipped
  - rotation fired while current trade still had healthy thesis
- summarize:
  - which lane is bleeding edge
  - which exit reasons are overrepresented
  - which fills are too fee-sensitive

Output contract:
- strict JSON only

Primary sources for GPT review:
- `logs/system_record.sqlite3`
- `logs/decision_debug.jsonl`
- `logs/account_sync.json`

## 9. Best Next Implementation Order

1. Add MFE / MAE tracking.
2. Add explicit maker-first retry / reprice logic.
3. Add exit order aging / cancel-replace rules.
4. Add structure-state field for exit logic.
5. Tighten rotation so it is cost-aware.

## 10. Success Criteria

You are improving the system if:
- market entries become rare outside emergency cases
- limit exits fill more often without panic fallback
- L2/L3 winners survive shallow pullbacks
- average MFE captured improves
- tiny green trades stop getting harvested before they cover friction
- `ROTATE` and `STALL` no longer dominate good-structure exits
