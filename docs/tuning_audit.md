# KrakenSK Tuning Audit

## Purpose

This document is for tuning the current engine without destabilizing the architecture.

The main idea:

- do not add more logic first
- do not move ten thresholds at once
- tune one layer at a time
- verify per-lane distributions before keeping any change

Current system status:

- architecture is materially better than the older system
- most remaining issues are calibration conflicts
- the highest-risk tuning area is the interaction between:
  - lane classification
  - lane shortlist construction
  - entry scoring
  - promotion
  - Nemotron eligibility

## Tuning Order

Tune in this order:

1. lane identity
2. scanner membership and shortlist shape
3. entry score calibration
4. promotion ladder thresholds
5. Nemotron eligibility and budget
6. execution offsets and spread constraints
7. exit behavior

Do not tune exits first. If entry composition is wrong, exit tuning will only hide the problem.

## What To Measure Every Time

For every replay or shadow comparison, capture at least:

- candidate count by lane
- shortlist count by lane
- `WATCH / BUY / STRONG_BUY` count by lane
- `skip / probe / promote` count by lane
- Nemotron reach rate by lane
- no-trade rate by lane
- fill rate by lane
- average spread at entry by lane
- missed leader rate
- after-fee expectancy by lane

If you do not have these distributions, you are tuning blind.

## Tuning Layers

### 1. Lane Identity

Primary files:

- [`core/policy/lane_classifier.py`](C:\Users\kitti\Desktop\KrakenSK\core\policy\lane_classifier.py)
- [`core/policy/lane_filters.py`](C:\Users\kitti\Desktop\KrakenSK\core\policy\lane_filters.py)
- [`tests/test_lane_identity.py`](C:\Users\kitti\Desktop\KrakenSK\tests\test_lane_identity.py)

Goal:

- `L1` = clean continuation
- `L2` = early rotation / release / takeover
- `L3` = stable balanced quality
- `L4` = fast breakout / meme / aggression

What to tune safely:

- RSI bands per lane
- low-volume warning ranges
- mild momentum thresholds for lane fit
- `L3` hotness rejection

What is risky to touch casually:

- meme symbol hard assignment to `L4`
- broad spread hard blocks
- all-lane trend assumptions

Current risk area:

- low-volume soft-fail override thresholds in lane filters are permissive enough that weak names can be treated as passing
- `L3` and `L2` are the most likely to drift into each other

Recommended workflow:

1. use lane identity tests as guardrails
2. sample 20-30 recent symbols manually by lane
3. inspect how often `L3` is receiving names that feel like early `L2` or hot `L4`
4. adjust only one lane boundary at a time

### 2. Scanner Membership And Shortlists

Primary files:

- [`apps/universe_manager/main.py`](C:\Users\kitti\Desktop\KrakenSK\apps\universe_manager\main.py)
- [`tests/test_lane_scanners.py`](C:\Users\kitti\Desktop\KrakenSK\tests\test_lane_scanners.py)

Goal:

- shortlist construction should preserve lane diversity
- `L2` should be able to catch near-L2 rotation names before they become obvious
- `L3` should not be polluted by hot fallback names

What to tune safely:

- `LANE_SHORTLIST_PCT`
- `LANE_SHORTLIST_LIMITS`
- lane rank weights for:
  - rank delta
  - momentum delta
  - volume ratio delta
  - short-timeframe acceleration

What is risky:

- removing lane diversity from merged shortlists
- making `L4` dominate merged rankings
- making `L2` require perfect pre-classification only

Current risk area:

- `scan_l2_candidates()` is currently strict enough that near-L2 names sitting in `L3` can be missed
- `L3` scoring still needs to prefer stability over heat more aggressively

Recommended workflow:

1. replay recent rotations
2. compare lane shortlist membership before and after changes
3. measure later-winner capture rate by lane
4. keep changes only if `L2` catches more movers without making `L3` noisy

### 3. Entry Score Calibration

Primary file:

- [`core/policy/entry_verifier.py`](C:\Users\kitti\Desktop\KrakenSK\core\policy\entry_verifier.py)

Goal:

- score should separate marginal `WATCH` from real `BUY`
- score should not over-promote ordinary setups into `STRONG_BUY`
- score should preserve different lane personalities

Important current built-in thresholds:

- `L1`: `72 / 56` for `STRONG_BUY / BUY`
- `L2`: `66 / 50`
- `L3`: `68 / 54`
- `L4`: from runtime:
  - `MEME_ENTRY_SCORE_STRONG_BUY_THRESHOLD`
  - `MEME_ENTRY_SCORE_BUY_THRESHOLD`

Safe to tune:

- lane thresholds
- probe thresholds
- additive score weights for:
  - volume confirmation
  - regime bonuses
  - structure bonuses
  - crowding penalties

Risky to tune:

- starting base score
- too many lane-specific bonuses at once
- removing reversal-risk penalties

Current risk area:

- positive stacking in `entry_verifier` can inflate an ordinary mover into `BUY` or `STRONG_BUY`
- that breaks the intended `WATCH -> probe` path

Recommended workflow:

1. export score distributions by lane
2. inspect 20-40 symbols around the `WATCH/BUY` boundary
3. tune one lane at a time
4. target cleaner separation, not a globally higher trade count

### 4. Promotion Ladder

Primary file:

- [`core/policy/entry_verifier.py`](C:\Users\kitti\Desktop\KrakenSK\core\policy\entry_verifier.py)

Goal:

- `skip` = not ready / not valid / not worth attention
- `probe` = promising but not yet full-strength
- `promote` = deterministic candidate ready for trade path

Safe to tune:

- `_lane_probe_threshold()`
- whether channel breakout / retest conditions promote or probe
- mover/leader criteria for `WATCH -> probe`

Risky to tune:

- bypassing hard filter severity
- letting every `WATCH` mover become `probe`
- using LLM output to restore `skip` names by default

Recommended workflow:

1. review `skip/probe/promote` distribution by lane
2. focus on cases where:
   - good `L2/L4` names die in `skip`
   - too many mediocre `L3` names become `promote`

### 5. Nemotron Eligibility

Primary file:

- [`core/policy/nemotron_gate.py`](C:\Users\kitti\Desktop\KrakenSK\core\policy\nemotron_gate.py)

Goal:

- Nemotron should review borderline but valid candidates
- Nemotron should not become a universal blocker
- Nemotron budget should concentrate on the right names

Key runtime controls:

- `NEMOTRON_TOP_CANDIDATE_COUNT`
- `NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP`
- `NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP`
- `NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT`
- `NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE`
- `NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO`
- `MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE`
- `MEME_NEMOTRON_WATCH_MIN_VOLUME_RATIO`
- `LEADER_URGENCY_OVERRIDE_THRESHOLD`

Safe to tune:

- top candidate count
- watch min entry score
- watch min volume ratio
- leader urgency override threshold

Risky to tune:

- broad “allow outside top” flags without measuring quality drift
- letting `WATCH` names through with no mover signal
- relying on candidate review to authorize ordinary deterministic flow

Current risk area:

- if these thresholds are too tight, good `L2/L4` names never reach review
- if too loose, Nemotron budget gets diluted across weak names

Recommended workflow:

1. measure Nemotron reach rate by lane
2. measure `nemo_cap_skip` frequency
3. if budget is wasted on weak candidates:
   - lower `NEMOTRON_TOP_CANDIDATE_COUNT`
4. if strong candidates are excluded:
   - lower watch thresholds slightly or raise leader override sensitivity

### 6. Execution Tuning

Primary files:

- [`core/execution/order_policy.py`](C:\Users\kitti\Desktop\KrakenSK\core\execution\order_policy.py)
- [`core/execution/kraken_live.py`](C:\Users\kitti\Desktop\KrakenSK\core\execution\kraken_live.py)
- [`core/risk/fee_filter.py`](C:\Users\kitti\Desktop\KrakenSK\core\risk\fee_filter.py)
- [`core/risk/trade_quality.py`](C:\Users\kitti\Desktop\KrakenSK\core\risk\trade_quality.py)

Key controls:

- `ORDER_PREFERENCE`
- `L1_ORDER_LIMIT_OFFSET_BPS`
- `L2_ORDER_LIMIT_OFFSET_BPS`
- `L3_ORDER_LIMIT_OFFSET_BPS`
- `MEME_ORDER_LIMIT_OFFSET_BPS`
- `EXEC_MAX_SPREAD_PCT`
- `MEME_EXEC_MAX_SPREAD_PCT`
- `EXEC_MIN_NOTIONAL_USD`
- `MEME_EXEC_MIN_NOTIONAL_USD`
- `EXEC_RISK_PER_TRADE_PCT`
- `L1_EXEC_RISK_PER_TRADE_PCT`
- `L2_EXEC_RISK_PER_TRADE_PCT`
- `MEME_EXEC_RISK_PER_TRADE_PCT`

Safe to tune:

- order limit offsets by lane
- min notional if fills are too fragmented or too blocked
- lane-specific risk-per-trade sizing

Risky to tune:

- widening max spread too aggressively
- making `L4` always market
- reducing notional floors until fills become meaningless

Recommended workflow:

1. inspect fill rate by lane
2. inspect missed fill rate by lane
3. change offsets in small steps
4. always verify after-fee capture, not just fill count

### 7. Exit Tuning

Primary files:

- [`core/risk/exits.py`](C:\Users\kitti\Desktop\KrakenSK\core\risk\exits.py)
- [`core/risk/position_monitor.py`](C:\Users\kitti\Desktop\KrakenSK\core\risk\position_monitor.py)
- [`core/llm/phi3_exit_posture.py`](C:\Users\kitti\Desktop\KrakenSK\core\llm\phi3_exit_posture.py)

Key controls:

- `EXIT_ATR_STOP_MULT`
- `EXIT_ATR_TAKE_PROFIT_MULT`
- `L1_EXIT_*`
- `L2_EXIT_*`
- `MEME_EXIT_*`
- `EXIT_BREAK_EVEN_R`
- `EXIT_TRAIL_ARM_R`
- `EXIT_TRAIL_ATR_MULT`
- `EXIT_LIVE_STALL_MIN_HOLD_MIN`
- `EXIT_LIVE_STALL_MAX_PNL_PCT`
- `EXIT_LIVE_RANK_DECAY_MIN_PNL_PCT`
- `EXIT_TIGHTEN_MIN_PNL_PCT`

Safe to tune:

- break-even and trailing thresholds
- lane-specific hold and stall windows
- tighten thresholds

Risky to tune:

- lowering stop distance before entry quality is fixed
- raising take-profit without checking capture rate

Recommended workflow:

1. measure MFE and MAE by lane
2. inspect rotate-out frequency
3. tune exits only after entry mix is stable

## Current Runtime Snapshot To Watch Closely

Current overrides in [`configs/runtime_overrides.json`](C:\Users\kitti\Desktop\KrakenSK\configs\runtime_overrides.json) make these especially important:

- `NEMOTRON_TOP_CANDIDATE_COUNT = 18`
- `NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE = 41.0`
- `NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO = 1.0`
- `EXEC_MIN_NOTIONAL_USD = 7.0`
- `MEME_EXEC_MIN_NOTIONAL_USD = 7.0`
- `ADVISORY_MIN_ENTRY_SCORE = 48.0`
- `NEMOTRON_GATE_MIN_ENTRY_SCORE = 49.0`
- `MEME_MAX_OPEN_POSITIONS = 2`

Interpretation:

- the system is already materially looser than defaults in some areas
- if trade count is still too low, the problem may be semantic conflicts more than raw threshold strictness
- if quality is unstable, these lower gates may already be allowing more than the scanner and promotion layers can properly distinguish

## Safe First Tuning Pass

If doing one careful tuning pass, this is the order I would use:

1. fix lane/shortlist semantics first
2. tighten `L3` heat rejection slightly
3. let `L2` pick up near-L2 rotation names earlier
4. lower score inflation in `entry_verifier` before lowering thresholds further
5. then adjust:
   - `NEMOTRON_TOP_CANDIDATE_COUNT`
   - `NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE`
   - `NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO`
6. only after that, tune order offsets and min notionals

## Unsafe Tuning Patterns

Avoid these:

- lowering every entry threshold at once
- increasing Nemotron candidate count to compensate for weak scanners
- widening spread allowances to compensate for missed entries
- using meme settings to fix non-meme lane problems
- tuning exits to compensate for bad entry composition
- changing aggression mode and base thresholds at the same time

## Recommended Working Method

For each tuning pass:

1. define one hypothesis
2. change one cluster only
3. replay recent data
4. compare lane distributions
5. run shadow
6. keep only changes that improve:
   - later winner capture
   - after-fee expectancy
   - lane purity
   - fill quality

## Best Next Engineering Additions

To make tuning easier, the highest-value additions would be:

- lane-by-lane score histograms
- lane-by-lane `skip/probe/promote` counts in the operator UI
- Nemotron reach-rate and budget usage metrics
- scanner hit-rate summaries by lane
- replay summary broken down by lane

Without those, the engine is tunable, but slower to tune correctly.
