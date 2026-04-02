# CoinProfile Design

## Purpose

This document defines the next signal-layer upgrade for KrakenSK:

- move from bucket-style descriptors toward continuous 0-100 dimensions
- preserve raw market measurements for hard constraints
- avoid double-counting by assigning clear ownership to each layer

This is not a rewrite of the engine into a single giant score.

It is a structured signal model:

- raw features = measurement and hard truth
- `CoinProfile` = normalized comparison layer
- deterministic policy = decision owner
- Nemo = structured reviewer, not raw-feature dump consumer

## Core Rule

Do not let the same signal dominate in three places.

Bad:

- raw momentum drives scanner rank
- normalized momentum drives entry score
- raw momentum is also emphasized in Nemo prompt

Good:

- raw momentum is measured once
- `momentum_quality` becomes the comparison-friendly representation
- deterministic policy uses either the raw hard guard or the normalized dimension, not both heavily

## CoinProfile Schema

Primary object:

```python
from dataclasses import dataclass


@dataclass(slots=True)
class CoinProfile:
    structure_quality: float
    momentum_quality: float
    volume_quality: float
    trade_quality: float
    market_support: float
    continuation_quality: float
    risk_quality: float
```

Optional future extensions:

- `rotation_quality`
- `breakout_quality`
- `mean_reversion_quality`
- `leader_quality`
- `exit_resilience`

Do not add these on day one unless they replace a real current decision need.

## Scale Definition

Every profile dimension is normalized to `0-100`.

Interpretation:

- `0-20` = very weak / strongly adverse
- `20-40` = weak
- `40-60` = neutral / mixed
- `60-80` = supportive
- `80-100` = very strong / highly favorable

These buckets are for UI and interpretation only.
Decision logic should use the continuous values.

## Normalization Rules

### Rule 1: Use Rolling Or Percentile Normalization

Do not use fixed static min/max for crypto signals.

Preferred order:

1. rolling percentile rank
2. rolling robust normalization
3. rolling min/max only where range is naturally bounded

Examples:

- RSI already bounded naturally
- volume ratio is not naturally bounded, so percentile is better
- momentum is unstable across regimes, so rolling normalization is required
- spread and ATR% often need inverted percentile-style scoring

### Rule 2: No Lookahead Leakage

Replay and backtests must use only trailing windows.

Allowed:

- last 50 bars
- last 100 bars
- last N candidate observations

Not allowed:

- using future bars to define normalization ranges

### Rule 3: Normalize By Context Where Necessary

Some signals should be lane-aware or universe-aware.

Examples:

- `trade_quality` should be lane-aware
- `market_support` should be partly global
- `continuation_quality` is more relevant to `L1` and `L4`
- `rotation-like` behavior is more relevant to `L2`

## Ownership Split

### Raw Features Own

Raw features remain authoritative for:

- data freshness
- invalid or missing book
- stale account state
- spread hard blocks
- slippage realism
- ATR%
- price
- exact RSI
- exact z-score
- exact volume ratio
- exact momentum values
- exact runtime health
- exact hard trend conflict

Use raw values for:

- hard risk gates
- execution feasibility
- fee/slippage filters
- health checks
- diagnostics
- traces

### CoinProfile Owns

`CoinProfile` should own:

- comparison between symbols
- ranking within lanes
- promotion quality context
- LLM advisory input
- UI-level “how good is this setup” visibility

Use profile values for:

- scanner ranking refinement
- `WATCH / BUY / STRONG_BUY` separation
- `skip / probe / promote` support
- Nemo structured review

### Deterministic Policy Owns

Deterministic policy remains the decision maker.

It should combine:

- raw hard guards
- selected profile dimensions
- limited raw context for exceptions

### Nemo Owns

Nemo should consume:

- lane
- setup class
- profile dimensions
- a very small raw constraints appendix

Nemo should not consume:

- the full raw feature dump plus the full normalized profile at equal importance

## Mapping Current Features

### Structure Quality

Purpose:

- quality of setup shape and structural integrity

Primary contributors:

- `ema9_above_ema20`
- `price_above_ema20`
- `ema_slope_9`
- `range_breakout_1h`
- `pivot_break`
- `pullback_hold`
- `higher_low_count`
- `range_pos_1h`
- `range_pos_4h`

Suggested construction:

```python
structure_quality =
    ema_stack_score * 25 +
    breakout_score * 20 +
    pivot_score * 15 +
    pullback_score * 20 +
    higher_low_score * 10 +
    range_position_score * 10
```

Raw-only supplements:

- exact breakout flags
- exact range position values

### Momentum Quality

Purpose:

- quality of directional thrust without collapsing to one bar

Primary contributors:

- `momentum_5`
- `momentum_14`
- `momentum_30`
- `trend_1h`
- `autocorr`

Suggested method:

- normalize each momentum series on rolling windows
- weight short and medium momentum more heavily than long momentum

Suggested shape:

```python
momentum_quality =
    mom5_norm * 40 +
    mom14_norm * 35 +
    mom30_norm * 15 +
    trend1h_norm * 10
```

Raw-only supplements:

- hard negative momentum conflict
- exact short-term crash conditions

### Volume Quality

Purpose:

- quality of participation and confirmation

Primary contributors:

- `volume_ratio`
- `volume_surge`
- `vwio`
- `sentiment_symbol_trending` only as a small add-on if kept at all

Suggested shape:

```python
volume_quality =
    volume_ratio_norm * 45 +
    volume_surge_norm * 35 +
    vwio_norm * 20
```

Raw-only supplements:

- exact low-liquidity blocks
- exact min-volume requirements

### Trade Quality

Purpose:

- can this actually be traded efficiently

Primary contributors:

- spread
- slippage expectation
- ATR%
- maybe book quality / notional achievability

Suggested shape:

```python
trade_quality =
    spread_score * 40 +
    slippage_score * 30 +
    atr_efficiency_score * 20 +
    book_quality_score * 10
```

Important:

- this is both a quality dimension and a gate family
- high `trade_quality` helps ranking
- low raw spread or invalid book can still hard-block regardless of score

### Market Support

Purpose:

- is the surrounding market helping or fighting this setup

Primary contributors:

- `regime_7d`
- `regime_state`
- `macro_30d`
- market breadth
- market cap change
- BTC dominance / market gravity if available

Suggested shape:

```python
market_support =
    breadth_score * 30 +
    regime_score * 25 +
    macro_score * 25 +
    market_cap_score * 10 +
    btc_gravity_score * 10
```

Important:

- make this partly lane-aware
- “ranging” is not equally bad for every lane

### Continuation Quality

Purpose:

- how likely is this setup to keep going instead of stalling immediately

Primary contributors:

- `higher_low_count`
- `hurst`
- `trend_confirmed`
- `ema_slope_9`
- not overextended logic from `rsi`, `price_zscore`, `range_pos`

Suggested shape:

```python
continuation_quality =
    persistence_score * 35 +
    higher_low_score * 20 +
    trend_confirmed_score * 20 +
    ema_slope_score * 10 +
    not_overextended_score * 15
```

### Risk Quality

Purpose:

- inverse of exhaustion, crowding, and adverse setup fragility

Primary contributors:

- `rsi`
- `price_zscore`
- `crowding`
- `obv_divergence`
- maybe negative `vwio`

Suggested shape:

```python
risk_quality =
    exhaustion_score * 35 +
    zscore_score * 25 +
    crowding_score * 20 +
    obv_confirmation_score * 20
```

Important:

- higher = safer / cleaner
- low risk quality should reduce promotion quality
- but raw hard limits still win when needed

## Anti-Double-Counting Rules

### Rule 1

If a decision path uses `momentum_quality`, it should not also heavily reward:

- `momentum_5`
- `momentum_14`
- `momentum_30`

in the same branch.

### Rule 2

If scanner ranking uses:

- `structure_quality`
- `momentum_quality`
- `volume_quality`

then `entry_score` should reduce repeated bonuses for:

- EMA alignment
- breakout
- pullback hold
- volume surge
- repeated momentum adds

### Rule 3

Nemo input should prefer profile dimensions over large repeated raw feature lists.

### Rule 4

Raw values stay in traces and debug output, but they should not be restated as first-class scoring inputs if the profile already represents them.

## What Current Entry Score Should Lose

When `CoinProfile` is active, these current `entry_verifier` areas should be reduced first:

- repeated momentum stacking
- repeated volume stacking
- repeated structure bonuses
- repeated regime bonuses if `market_support` already owns them
- repeated exhaustion penalties if `risk_quality` already owns them

Good transition approach:

- phase 1: add profile alongside current score
- phase 2: switch ranking and advisory to profile
- phase 3: shrink raw additive entry score
- phase 4: keep only raw hard guards plus limited lane-specific exceptions

## Recommended Deterministic Policy Split

### Hard Gates

Stay raw:

- stale data
- invalid book
- spread hard block
- min notional
- runtime health
- hard risk conflict

### Ranking Inputs

Move toward profile:

- `structure_quality`
- `momentum_quality`
- `volume_quality`
- `trade_quality`
- `market_support`
- `continuation_quality`
- `risk_quality`

### Promotion Logic

Promotion should mostly depend on:

- structure quality
- trade quality
- risk quality
- lane-aware momentum/continuation quality

With raw exceptions for:

- hard breakout
- hard invalidity
- data integrity problems

## Recommended Scanner Split

### L1

Prioritize:

- structure quality
- continuation quality
- market support
- trade quality

De-emphasize:

- raw short-term heat

### L2

Prioritize:

- momentum quality
- volume quality
- rotation-style early continuation quality
- improving structure quality

Allow near-L2 candidates that are not yet officially perfect `L2`.

### L3

Prioritize:

- trade quality
- structure quality
- market support
- risk quality

Explicitly penalize hotness and unstable short-term acceleration.

### L4

Prioritize:

- momentum quality
- volume quality
- trade quality
- continuation quality

Allow lower market support than `L1/L3`, but never ignore raw execution feasibility.

## Nemo Input Contract

Recommended compact input:

```json
{
  "symbol": "APT/USD",
  "lane": "L2",
  "setup_type": "channel_breakout",
  "promotion_tier": "probe",
  "coin_profile": {
    "structure_quality": 84,
    "momentum_quality": 77,
    "volume_quality": 63,
    "trade_quality": 88,
    "market_support": 58,
    "continuation_quality": 72,
    "risk_quality": 66
  },
  "raw_constraints": {
    "spread_pct": 0.22,
    "book_valid": true,
    "runtime_healthy": true,
    "reversal_risk": "MEDIUM",
    "price_zscore": 1.1,
    "rsi": 61.0
  }
}
```

Nemo should not need:

- every raw subfeature
- repeated regime labels
- repeated “low/medium/high” text for dimensions already represented numerically

## UI Rules

Labels may still be used in the UI:

- “strong structure”
- “mixed market”
- “weak tradeability”

But labels are presentation only.

Decision logic should run on the continuous values.

## Implementation Plan

### Phase 1

Add `CoinProfile` calculation to:

- [`core/features/batch.py`](C:\Users\kitti\Desktop\KrakenSK\core\features\batch.py)

Store:

- raw values
- normalized dimension values

No policy replacement yet.

### Phase 2

Thread `CoinProfile` into:

- scanner ranking
- decision traces
- operator UI
- Nemo payload

### Phase 3

Refactor:

- [`core/policy/entry_verifier.py`](C:\Users\kitti\Desktop\KrakenSK\core\policy\entry_verifier.py)

Reduce double-counting and move more ranking logic to profile dimensions.

### Phase 4

Retune per lane with replay and shadow.

## What Not To Do

Do not:

- normalize everything and then convert back to `low/medium/high` for policy logic
- replace hard raw gates with profile dimensions
- feed both the full raw feature dump and the full profile into Nemo at equal importance
- keep all existing score bonuses while also adding the profile
- collapse all dimensions into one super score

## First Safe Version

The safest first version is:

- add `CoinProfile`
- use it for ranking and advisory context
- keep raw hard gates intact
- keep old entry score temporarily
- then remove duplicated score terms in a second pass

That lets the system evolve without breaking the live path all at once.
