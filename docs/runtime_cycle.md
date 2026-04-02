# Runtime Cycle

Current purpose: this document describes what the live system is supposed to do on startup and during each active trading cycle.

It is a code-aligned runtime description, not a theory note.

## Startup

### Process startup

[`scripts/start_all.ps1`](C:/Users/kitti/Desktop/KrakenSK/scripts/start_all.ps1) starts the app stack:

- `universe_manager`
- `review_scheduler`
- `trader`
- optional `mcp_server`
- optional `operator_ui`
- optional visual components

Important current note:

- there is not a separate always-on standalone collector process in the normal stack
- the live collector loop is started inside the trader boot path

### Trader boot

[`apps/trader/boot.py`](C:/Users/kitti/Desktop/KrakenSK/apps/trader/boot.py) does the startup wiring:

1. loads env-driven trader constants
2. loads active trading symbols
3. initializes strategy, risk, executor, portfolio config, and model services
4. syncs account state from Kraken
5. creates `LiveMarketDataFeed`
6. seeds from:
   - `seed_feed_from_snapshots(...)`
   - `seed_feed_from_rest(...)`
7. starts `collector_loop(...)` as an async task

This is why warmup can complete quickly after restart.

### Universe

The active symbol list comes from the locked live universe state, not arbitrary ad hoc symbols.

The current runtime can still score a fairly wide active universe, but only a small finalist set should reach Nemo.

## Main Loop

### Important timing note

The trader loop sleeps for 1 second between iterations, but that does not mean the system completes one full decision cycle every second.

Real cycle time is:

- data fetch time
- feature computation time
- Nemo time
- execution / sync time

So the loop cadence is:

- 1 second sleep plus actual work time

not:

- one full completed decision cycle every second

## Per-Cycle Flow

### 1. Symbol selection

In [`apps/trader/main.py`](C:/Users/kitti/Desktop/KrakenSK/apps/trader/main.py), the trader:

- loads the current symbol universe
- keeps held positions included
- prunes stale symbol state
- selects symbols for evaluation

### 2. OHLC fetch

For live mode, the trader fetches:

- `1m`
- `5m`
- `15m`
- `1h`
- `7d`
- `30d`

This is done in parallel with `asyncio.gather(...)` in [`apps/trader/main.py`](C:/Users/kitti/Desktop/KrakenSK/apps/trader/main.py).

The fetch is:

- parallel across timeframes
- parallel across symbols within each timeframe

### 3. Feature extraction

[`core/features/batch.py`](C:/Users/kitti/Desktop/KrakenSK/core/features/batch.py) computes symbol features such as:

- RSI
- ATR
- Bollinger
- momentum
- volume ratios
- regime features
- structure features
- trade quality / continuation quality / risk quality
- entry verification inputs

The pipeline also uses CUDA-accelerated kernels for parts of the feature stack.

### 4. Deterministic entry shaping

[`core/policy/entry_verifier.py`](C:/Users/kitti/Desktop/KrakenSK/core/policy/entry_verifier.py) computes:

- `entry_score`
- `entry_recommendation`
- `reversal_risk`
- `promotion_tier`
- `promotion_reason`

This is one of the main deterministic control points in the system.

### 5. Deterministic gating

Before Nemo, runtime policy still applies:

- stabilization rules
- lane restrictions
- cost validity
- trend requirements
- short-timeframe readiness
- reentry cooldown
- other deterministic candidate filters

This means Nemo should not be deciding on the full raw universe.

### 6. Finalist selection

This is the most important current runtime shape:

- full active universe can be scored deterministically
- only the top finalists should go to local Nemo

Current intended live setting:

- `NEMOTRON_BATCH_TOP_N = 5`

So the live shape is:

- score many
- send only top 5 finalists into batch Nemo

not:

- send all 20 symbols into one big local batch

### 7. Advisory bundle

Before strategist judgment, the system builds advisory context:

- reflex / data-integrity review
- market-state review
- optional visual review state

This comes from the advisory pipeline and is passed to Nemo as supporting context.

### 8. Nemotron strategist

[`core/llm/nemotron.py`](C:/Users/kitti/Desktop/KrakenSK/core/llm/nemotron.py) runs the strategist.

Current intended local shape:

- one local batch call
- only top 5 finalists
- compact candidate payload
- strict JSON output

Current settings:

- `NEMOTRON_BATCH_MODE=true`
- `NEMOTRON_BATCH_TOP_N=5`
- `NEMOTRON_LOCAL_BATCH_SINGLE_FALLBACK=true`

If batch parsing fails:

- the system can fall back to single-symbol local decisions

### 9. Risk and portfolio gate

Even after Nemo says `OPEN`, deterministic checks still apply.

Portfolio and risk controls include:

- `PORTFOLIO_MAX_OPEN_POSITIONS`
- `PORTFOLIO_MAX_WEIGHT_PER_SYMBOL`
- `PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE`
- execution notional requirements
- account cash constraints
- correlation / replacement logic
- cooldown logic

Current live override examples in [`configs/runtime_overrides.json`](C:/Users/kitti/Desktop/KrakenSK/configs/runtime_overrides.json):

- `PORTFOLIO_MAX_OPEN_POSITIONS = 3`
- `PORTFOLIO_MAX_WEIGHT_PER_SYMBOL = 0.5`
- `PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE = 0.95`

### 10. Execution

Execution is handled through:

- [`core/execution/order_policy.py`](C:/Users/kitti/Desktop/KrakenSK/core/execution/order_policy.py)
- [`core/execution/kraken_live.py`](C:/Users/kitti/Desktop/KrakenSK/core/execution/kraken_live.py)

The executor decides:

- order type
- limit-vs-market preference
- stale TTL
- chase behavior

Important correction:

- the system does not place exchange-native stop-loss and take-profit orders immediately after every fill as a permanent bracket pair
- instead, it builds an internal exit plan and later submits exit orders when exit conditions actually fire

### 11. Exit management

For open positions, [`core/risk/exits.py`](C:/Users/kitti/Desktop/KrakenSK/core/risk/exits.py) manages:

- stop-loss behavior
- take-profit behavior
- trail arming and updates
- stale exits
- never-profited exits
- failed-follow-through exits
- live-state tighten / exit posture

Current important live setting:

- `EXIT_NEVER_PROFITED_MIN_HOLD_MIN = 180`

### 12. Order reconciliation

Open-order reconciliation is managed through:

- [`core/state/open_orders.py`](C:/Users/kitti/Desktop/KrakenSK/core/state/open_orders.py)
- calls from [`apps/trader/main.py`](C:/Users/kitti/Desktop/KrakenSK/apps/trader/main.py)

Full reconcile runs every 2 minutes.

## Learning / Feedback

### Trade memory

[`core/memory/trade_memory.py`](C:/Users/kitti/Desktop/KrakenSK/core/memory/trade_memory.py) records outcomes and builds:

- lesson blocks
- symbol lesson summaries
- behavior score blocks

### What gets sent back to Nemo

Current strategist path may receive:

- compact `lesson_summary`
- compact `behavior_score`

The local payload is supposed to stay compact.

### Optimizer

The optimizer path is advisory-only.

Current behavior:

- reviews are written to proposal/review files
- live apply is not automatic

Relevant env:

- `NVIDIA_OPTIMIZER_APPLY=false`

## Cycle Timing Reality

The live system should be thought of as:

- variable-duration cycles

not:

- constant one-second full decisions

The main timing pressure comes from:

- OHLC fetch
- feature computation
- Nemo inference

The current intended optimization is:

- small finalist batch to local Nemo

not:

- serial local full-universe decisions

and not:

- full 20-symbol local batch

## Current Correct Mental Model

The runtime is supposed to be:

1. startup seeds the live buffers fast
2. full universe is scored deterministically
3. only a small finalist set reaches local Nemo
4. Nemo judges finalists
5. code still owns legality, sizing implementation, and exits

That is the correct current runtime shape for KrakenSK.
