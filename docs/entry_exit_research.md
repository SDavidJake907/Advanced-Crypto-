# Entry/Exit Research

Current state: entry logic is still mostly code-shaped; exit logic is already much closer to a runtime-baseline system.

This document is meant to answer five questions:

1. Where is entry logic actually decided?
2. Where is exit logic actually decided?
3. Which parts are safe to tune with config?
4. Which parts still require code changes?
5. What should the future baseline layout look like?

## System Flow

### Entry flow

The current entry path is:

1. features are computed
2. [core/policy/entry_verifier.py](C:/Users/kitti/Desktop/KrakenSK/core/policy/entry_verifier.py) produces:
   - `entry_score`
   - `entry_recommendation`
   - `reversal_risk`
   - `promotion_tier`
   - `promotion_reason`
3. deterministic runtime gates apply
4. finalists are passed to Nemo
5. risk and execution still have final veto power

So the real ordering is:

- feature truth
- deterministic entry shaping
- deterministic gating
- model judgment
- deterministic execution legality

### Exit flow

The current exit path is:

1. [core/risk/exits.py](C:/Users/kitti/Desktop/KrakenSK/core/risk/exits.py) builds initial stop / TP / trail plan at entry
2. live position state updates MFE / MAE / ETD over time
3. live exit review classifies posture:
   - `RUN`
   - `TIGHTEN`
   - `EXIT`
   - `STALE`
4. deterministic evaluation decides whether an exit should actually fire
5. execution policy decides market-vs-limit style for the exit order

So exit is already much closer to:

- deterministic exit plan
- deterministic live review
- deterministic final firing

with the LLM only acting as an optional overlay, not the owner.

## Source Of Truth

### Entry source of truth

Primary source of truth:
- [core/policy/entry_verifier.py](C:/Users/kitti/Desktop/KrakenSK/core/policy/entry_verifier.py)

Secondary policy shaping:
- [core/policy/nemotron_gate.py](C:/Users/kitti/Desktop/KrakenSK/core/policy/nemotron_gate.py)
- [configs/runtime_overrides.json](C:/Users/kitti/Desktop/KrakenSK/configs/runtime_overrides.json)

Model role:
- final judgment only on already-shaped candidates

### Exit source of truth

Primary source of truth:
- [core/risk/exits.py](C:/Users/kitti/Desktop/KrakenSK/core/risk/exits.py)

Execution policy:
- [core/execution/order_policy.py](C:/Users/kitti/Desktop/KrakenSK/core/execution/order_policy.py)

Optional posture overlay:
- [core/llm/phi3_exit_posture.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/phi3_exit_posture.py)

Current runtime note:
- `EXIT_POSTURE_USE_PHI3=false`, so exits are effectively deterministic-first today

## Entry

### What currently owns entry decisions

Primary entry scoring and recommendation live in [core/policy/entry_verifier.py](C:/Users/kitti/Desktop/KrakenSK/core/policy/entry_verifier.py).

The file currently owns:

- score seed and score math
- lane-specific strong-buy / buy / probe thresholds
- raw feature weights
- regime penalties and bonuses
- reversal-risk classification
- late-chase penalties
- divergence bonuses/penalties
- cost-penalty adjustment before recommendation
- promotion-tier logic

### Entry mechanics that are more important than prompt changes

The current entry engine is more influenced by code and runtime than by LLM prompts.

That means when entry behavior feels wrong, the first places to inspect are:

- entry score shaping
- regime penalties
- risk/reversal thresholds
- promotion logic
- stabilization gate
- cost legality

not:

- prompt wording

This matters because the project has already reached the point where config and code boundaries are more important than adding model instructions.

### What is already runtime-configurable for entry

These live in [core/config/runtime.py](C:/Users/kitti/Desktop/KrakenSK/core/config/runtime.py) and are overridden in [configs/runtime_overrides.json](C:/Users/kitti/Desktop/KrakenSK/configs/runtime_overrides.json):

- stabilization gate:
  - `STABILIZATION_STRICT_ENTRY_ENABLED`
  - `STABILIZATION_ALLOWED_LANES`
  - `STABILIZATION_MIN_ENTRY_SCORE`
  - `STABILIZATION_MIN_NET_EDGE_PCT`
  - `STABILIZATION_REQUIRE_TP_AFTER_COST_VALID`
  - `STABILIZATION_REQUIRE_TREND_CONFIRMED`
  - `STABILIZATION_REQUIRE_SHORT_TF_READY_15M`
  - `STABILIZATION_BLOCK_RANGING_MARKET`
  - `STABILIZATION_REQUIRE_BUY_RECOMMENDATION`
- Nemo prefilter:
  - `NEMOTRON_GATE_MIN_ENTRY_SCORE`
  - `NEMOTRON_GATE_MIN_VOLUME_RATIO`
  - `NEMOTRON_TOP_CANDIDATE_COUNT`
- execution economics and sizing:
  - `EXEC_MIN_NOTIONAL_USD`
  - `EXEC_RISK_PER_TRADE_PCT`
  - `TRADER_PROPOSED_WEIGHT`
  - trade-cost floor settings
- anti-churn:
  - `REENTRY_COOLDOWN_MIN`

### Entry runtime knobs that matter most in practice

These are the highest-leverage live knobs right now:

- `STABILIZATION_MIN_ENTRY_SCORE`
- `STABILIZATION_REQUIRE_TREND_CONFIRMED`
- `STABILIZATION_REQUIRE_SHORT_TF_READY_15M`
- `STABILIZATION_BLOCK_RANGING_MARKET`
- `STABILIZATION_REQUIRE_BUY_RECOMMENDATION`
- `STABILIZATION_MIN_NET_EDGE_PCT`
- `NEMOTRON_GATE_MIN_ENTRY_SCORE`
- `EXEC_MIN_NOTIONAL_USD`
- `EXEC_RISK_PER_TRADE_PCT`
- `TRADER_PROPOSED_WEIGHT`

These are the knobs most likely to change behavior without rewriting code.

### What is still hardcoded and should become baseline-configurable next

These are the highest-value entry items to move into config:

1. Lane score thresholds
- `_lane_entry_thresholds(...)`
- `_lane_probe_threshold(...)`

2. Major score weights
- profile quality weights
- raw momentum / volume weights
- microstructure weights

3. Regime penalties
- trending/choppy/bearish/volatile adjustments

4. Reversal-risk thresholds
- z-score cutoffs
- RSI overbought cutoffs

5. Late-chase logic
- range-position and z-score thresholds

6. Cost-penalty tiers
- net-edge buckets that currently apply 0 / 5 / 10 / 15 point penalties

### Entry examples

#### Good candidate path

A good entry candidate should usually look like:

- acceptable data quality
- lane allowed by runtime
- `entry_score` above baseline
- `trend_confirmed=true`
- `short_tf_ready_15m=true`
- positive or acceptable `net_edge_pct`
- `tp_after_cost_valid=true`
- no hard reversal issue
- then Nemo chooses whether to open it

#### Bad candidate path

A bad entry candidate often fails before Nemo for reasons like:

- weak `entry_score`
- `ranging_market=true` under stricter profiles
- no 15m readiness
- bad economics
- high reversal risk
- anti-churn cooldown

That is expected and desirable.

### Entry planning conclusion

The right architecture is:

- code computes raw truth
- runtime config controls thresholds and gating
- Nemo only judges a very small finalist set

The next entry refactor should not rewrite the score engine. It should expose the lane thresholds and a small set of major penalty/bonus knobs into runtime config.

### Entry change policy

Safe to change in config:

- thresholds
- lane permissions
- stabilization rules
- risk-per-trade
- proposed weight
- cooldown

Should require code change:

- adding new raw features
- changing score architecture
- changing promotion flow semantics
- changing what fields are computed

## Exit

### What currently owns exit decisions

Primary deterministic exit logic lives in [core/risk/exits.py](C:/Users/kitti/Desktop/KrakenSK/core/risk/exits.py).

This path already looks much closer to a profile/baseline system.

It owns:

- initial stop-loss and take-profit construction
- trailing-stop arm and trail math
- live-state exit review
- fee-aware green-trade protection
- stale/no-progress handling
- final stop / TP / posture exit evaluation

The lighter posture overlay lives in [core/llm/phi3_exit_posture.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/phi3_exit_posture.py), but runtime currently has `EXIT_POSTURE_USE_PHI3=false`, so behavior is effectively heuristic/deterministic.

### Exit mechanics already feel like a baseline system

Exit is much closer to the desired architecture because:

- lane-specific stop math is already runtime-driven
- trail arm and trail distance are runtime-driven
- stale logic is runtime-driven
- fee-aware green protection is runtime-driven
- exit order TTL/chase behavior is runtime-driven

In other words, exit already behaves more like:

- mechanism in code
- behavior in config

which is exactly the direction the rest of the project should move toward.

### What is already runtime-configurable for exit

Stop / TP / trail:

- `EXIT_ATR_STOP_MULT`
- `EXIT_MIN_STOP_PCT`
- `EXIT_ATR_TAKE_PROFIT_MULT`
- `EXIT_PRIMARY_TP_ATR_MULT`
- `EXIT_TRAIL_ARM_R`
- `EXIT_TRAIL_ATR_MULT`
- lane variants for `L1`, `L2`, and meme

Profit protection / stale handling:

- `EXIT_MIN_PROFIT_AFTER_COST_PCT`
- `EXIT_STALE_MIN_HOLD_MIN`
- `EXIT_STALE_MAX_ABS_PNL_PCT`
- `EXIT_NEVER_PROFITED_MIN_HOLD_MIN`
- `EXIT_TIGHTEN_MIN_PNL_PCT`
- lane-specific tighten thresholds
- `EXIT_POSTURE_NEG_PNL_PCT`

Live-state overlay:

- `EXIT_LIVE_STATE_ENABLED`
- `EXIT_LIVE_STALL_MIN_HOLD_MIN`
- `EXIT_LIVE_STALL_MAX_PNL_PCT`
- `EXIT_LIVE_SPREAD_TIGHTEN_PCT`
- `EXIT_LIVE_SPREAD_EXIT_PCT`
- `EXIT_LIVE_RANK_DECAY_MIN_PNL_PCT`

Green-trade protection:

- `EXIT_STALE_GREEN_BLOCK`
- `EXIT_FEE_AWARE_GREEN_BLOCK`
- `EXIT_GREEN_ONLY_STOP_OR_TRAIL`
- `EXIT_GREEN_EMA_RSI_ATR_HOLD`
- `EXIT_GREEN_HOLD_MIN_RSI`
- `EXIT_GREEN_HOLD_MIN_STOP_ATR_BUFFER`

Order execution behavior:

- `ORDER_OPEN_TTL_SEC`
- `ORDER_REPRICE_TTL_SEC`
- `EXIT_ORDER_REPRICE_TTL_SEC`
- `EXIT_PROTECTIVE_ORDER_TTL_SEC`
- `EXIT_TAKE_PROFIT_ORDER_TTL_SEC`
- chase bps settings

### Exit runtime knobs that matter most in practice

These are the highest-value exit controls today:

- `EXIT_MIN_PROFIT_AFTER_COST_PCT`
- `EXIT_STALE_MIN_HOLD_MIN`
- `EXIT_NEVER_PROFITED_MIN_HOLD_MIN`
- `EXIT_LIVE_STALL_MIN_HOLD_MIN`
- `EXIT_LIVE_STALL_MAX_PNL_PCT`
- `EXIT_LIVE_RANK_DECAY_MIN_PNL_PCT`
- `EXIT_TIGHTEN_MIN_PNL_PCT`
- `EXIT_TRAIL_ARM_R`
- `EXIT_TRAIL_ATR_MULT`
- lane-specific stop and TP multipliers

These should become the backbone of any future explicit exit baseline profiles.

### What is still implicit in exit logic

These are still code-shaped and worth exposing later:

1. Structured-hold minimums
- `_structured_hold_min_minutes(...)`
- runner-specific minimum hold schedules

2. Structure-intact rules
- `_structure_intact(...)`
- these decide whether a runner deserves more room

3. Soft/hard live-state reasons
- `momentum_decay`
- `move_stall`
- `rank_decay`
- `spread_widening`
- `failed_follow_through`
- `follow_through_lost`

4. Hard-failure reason mapping
- `_is_hard_failure_reason(...)`

### Exit examples

#### Healthy runner

A healthy open position should usually:

- keep `structure_state=intact`
- avoid hard failure reasons
- avoid stale/no-progress conditions
- stay in `RUN`
- only arm a trail after real progress

#### Proper tighten

A good tighten path is:

- open profit exists
- momentum or spread deteriorates
- structure is not fully broken yet
- `review_live_exit_state(...)` returns `TIGHTEN`
- `maybe_apply_exit_posture(...)` tightens the trailing stop

#### Proper exit

A good hard exit path is:

- structure broken
- follow-through failed
- or stop/TP physically hit
- or stale loser never profited long enough

That should remain deterministic.

### Exit planning conclusion

Exit is already close to baseline-ready.

The next exit refactor should not replace the existing system. It should:

- formalize exit profiles
- move structured-hold schedules into config
- move live-state trigger thresholds into grouped profiles
- keep final exit evaluation deterministic

### Exit change policy

Safe to change in config:

- stop multipliers
- TP multipliers
- trail arm and trail distance
- stale timing
- tighten thresholds
- min-profit-after-cost
- order TTL and chase behavior

Should require code change:

- structure-state algorithm
- excursion bookkeeping
- posture state machine semantics
- final exit precedence rules

## Recommended Baseline Direction

### Entry baseline should own

- lane thresholds
- promotion thresholds
- regime penalties
- reversal-risk thresholds
- cost-penalty buckets
- stabilization gate settings
- anti-churn / cooldown

### Exit baseline should own

- stop ATR multipliers
- min stop percent
- primary TP ATR multipliers
- trail arm R
- trail ATR multipliers
- stale thresholds
- tighten thresholds
- live-state spread/rank/stall thresholds
- green-profit protection behavior

### Keep in code

- raw feature computation
- canonical candidate packet
- final deterministic exit evaluation
- execution-order construction

## Recommended File Layout

The next clean baseline structure should be:

- `configs/entry_baseline.json`
- `configs/exit_baseline.json`
- `configs/portfolio_baseline.json`

And each should answer a narrow question:

- entry baseline: when is a setup legal enough to promote?
- exit baseline: how much room does a trade get, and how does it tighten?
- portfolio baseline: how much capital can be used, and how concentrated can it get?

## Why This Matters

The project has enough code already.

The edge is more likely to come from:

- clearer operating rules
- cleaner ownership
- easier tuning
- less drift between code, prompts, and runtime behavior

than from adding more branches or more model instructions.

## Practical Read

If the project is going to stop changing code every time, the best next step is:

1. Create explicit `entry_baseline` config
2. Create explicit `exit_baseline` config
3. Move the remaining high-value hardcoded thresholds into those baselines
4. Leave raw math and final execution mechanics in code

That gives the project a cleaner control model:

- code = mechanism
- baseline config = behavior
- model = judgment inside the allowed behavior
