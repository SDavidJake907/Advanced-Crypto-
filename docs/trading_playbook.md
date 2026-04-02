# Trading Playbook

Current purpose: this is the human-readable trading behavior guide for KrakenSK.

This document is not the code itself.
It explains how the system should trade when the code, config, and LLM are all working correctly.

Use this document to answer:

- when should the bot buy?
- when should it hold through weakness?
- when should it sell?
- what kinds of coins should it focus on?
- what should be tuned in config versus changed in code?

## North Star

The system should make money by doing a small number of things well:

- buy clean structure
- avoid late chasing
- avoid weak post-cost setups
- hold valid winners longer
- cut failed ideas faster
- rotate capital into stronger opportunities only when the current position is truly weakening

This is not a “trade everything” system.

## Market Philosophy

The engine should prefer:

- structure over noise
- 15m confirmation over 1m excitement
- post-cost edge over raw signal excitement
- fewer better trades over many small churn trades

The bot should not try to predict every move.
It should try to participate in the part of the move that is:

- clean enough
- tradable enough
- liquid enough
- large enough after costs

## What To Trade

### Best symbol traits

The best live symbols are usually:

- liquid
- clean on 15m and 1h
- capable of follow-through after breakout or retest
- not excessively correlated to a bunch of held positions
- not structurally broken on higher timeframe context

### Best symbol categories

The system should prioritize:

1. clean major / high-liquidity names
2. strong rotation names with real follow-through
3. structured breakout names

The system should deprioritize:

- weak-history symbols
- thin or jumpy names
- names with poor post-cost edge
- names that repeatedly churn without trend follow-through

### What to avoid

Avoid symbols that are:

- low quality after costs
- always WATCH but rarely clean BUY
- frequently overextended before entry
- repeatedly producing tiny scalp-like churn
- missing reliable 1h / 7d / 30d context

## Entry Playbook

### A good long entry usually needs

At a high level, a clean long should usually have:

- acceptable post-cost edge
- trend or structure confirmation
- 15m readiness
- positive enough short-term momentum
- acceptable reversal risk
- clean enough location in the current range

In system terms, strong entries usually look like:

- `entry_score` at or above the active baseline
- `entry_recommendation` in `BUY` or `STRONG_BUY`
- `reversal_risk != HIGH`
- `tp_after_cost_valid = true`
- `net_edge_pct >= 0`
- `trend_confirmed = true`
- `short_tf_ready_15m = true`

### Best entry archetypes

#### 1. Breakout

Best when:

- `range_breakout_1h = true`
- price is not already too extended
- EMA alignment is supportive
- short-term momentum is positive
- volume expansion is real

Good use:

- early-to-mid breakout with room left

Bad use:

- buying after the breakout is already overextended and near exhaustion

#### 2. Pullback / retest

Best when:

- `pullback_hold = true`
- price reclaims and holds the EMA structure
- momentum is recovering after the pullback
- not a collapsing chart pretending to bounce

This is usually the cleanest entry type.

#### 3. Continuation

Best when:

- trend is already confirmed
- structure is intact
- EMA stack is aligned
- higher lows are building
- the symbol is acting like a real leader, not just a short-cover bounce

#### 4. Rotation release

Best when:

- lane is behaving like `L2`
- relative strength is improving
- volume is not dead
- short timeframe has actually turned up
- the move is early, not already exhausted

### When not to buy

Do not buy when:

- setup is only attractive before costs, not after costs
- `reversal_risk = HIGH`
- structure is falling apart
- the move is obviously late-chase
- the setup is weak on 15m and only exciting on 1m
- volume and follow-through are not there
- the symbol is only interesting because it moved, not because it structured

### When trend is going down

This is where most systems lose discipline.

When the broader local trend is still down, a long should only be considered when there is evidence of repair, not just a bounce.

Acceptable long in a downtrend:

- a clear reclaim
- then a hold
- then better short-term momentum
- then a fresh breakout or structured retest

Not acceptable:

- first green candle after heavy selling
- one sharp bounce with no structure behind it
- buying solely because RSI is low

In other words:

- downtrend bounce alone is not a buy
- downtrend repair plus structure can become a buy

### Practical buy ladder

The bot should think in three levels:

#### Level 1: Elite buy

- strong score
- strong structure
- strong economics
- low reversal risk
- clean breakout or retest

These are best candidates for automatic entry.

#### Level 2: Valid buy

- good score
- valid economics
- acceptable structure
- some uncertainty remains

These are acceptable when portfolio room exists and no better setup is present.

#### Level 3: Watch only

- improving but not proven
- or economically marginal
- or still structurally mixed

These should not become forced buys just because the market is moving.

## Hold Playbook

### When to hold

Hold when the original thesis is still alive.

That usually means:

- structure still intact
- price has not invalidated the setup
- momentum may soften, but not collapse
- the position is not obviously stale
- better replacement opportunities are not clearly taking over

### Hold through weakness only when

Hold through temporary weakness when:

- the position is still structurally healthy
- weakness looks like normal pullback, not failure
- the move has enough room and quality to justify patience
- trailing logic has not been violated

For stronger runner styles, the system should allow more room.

### Do not hold just because you want the trade to work

The system should not protect losers emotionally.

Do not keep holding when:

- structure is broken
- the move never followed through
- better candidates clearly replaced it
- the trade has been dead too long
- the thesis is no longer the same thesis

## Exit Playbook

### Sell immediately when

Hard exits should fire when:

- stop loss is hit
- trail is hit
- hard failure reason is present
- failed-follow-through logic is triggered
- stale-loser rules are triggered

These should remain deterministic.

### Tighten when

Tighten, don’t fully exit, when:

- the trade is green
- momentum is fading
- spread is worsening
- rank is decaying
- structure is getting fragile but not fully broken

This is where the system should protect profit without cutting every good winner too early.

### Let run when

Let winners run when:

- structure is still intact
- the move is healthy
- the trade is behaving like the expected hold style
- no hard exit is triggered
- green-trade protection rules say the move still deserves room

### When trend is going down after entry

If a long position was opened and trend starts turning down:

- do not auto-panic on first weakness
- first ask whether structure is still intact
- if structure is intact, keep holding or tighten
- if structure is broken, exit

The distinction is:

- soft weakness = tighten or keep room
- structural failure = sell

### Time-based exits

A position should be considered stale when:

- it has had enough time to work
- it has made little or no progress
- momentum is flat
- the capital is likely better used elsewhere

This is especially important on a small account, where idle capital hurts.

## Portfolio Playbook

### Position count

The portfolio should hold only a small number of positions at once.

That keeps:

- attention focused
- correlation lower
- capital meaningful per position
- exits easier to manage

### Use of capital

The system should use capital intentionally:

- enough size for a valid setup to matter
- not so much size that one mistake dominates the account

Exact size should stay deterministic.

### Rotation

Rotation out of a current position should happen when:

- the current position is weakening
- a replacement is clearly stronger
- the replacement advantage is persistent, not just one noisy bar

Rotation should not become random churn.

When portfolio slots are full, the system may replace the weakest held position with a materially stronger candidate, but only when:

- the held position is no longer acting like a healthy runner
- the replacement lead is clear, not marginal
- the advantage persists across the rotation checks

Healthy runners should not be sold just because a slightly higher-ranked name appears for one cycle.

## LLM Support Role

Phi-3 should help with chart interpretation only:

- verify chart structure
- verify candle confirmation in context
- verify breakout and retest quality
- warn when chart evidence is conflicting or weak

Nemo should remain the final trade judge on finalists.

Code should still own:

- entry legality
- sizing
- cost floor enforcement
- risk vetoes
- execution

That means Phi-3 helps Nemo see the chart more clearly, but it should not become the direct trade decision maker.

## Lanes

### L1

Best for:

- stronger continuation names
- major leadership
- more persistent structure

Expect:

- fewer but cleaner continuation entries
- more room on exits

### L2

Best for:

- structured rotations
- release moves
- improving relative strength

Expect:

- earlier entries than L1
- faster need for follow-through

### L3

Best for:

- balanced quality setups
- non-chaotic names
- trades that are neither pure continuation leaders nor meme accelerators

Expect:

- moderate hold style
- balanced entry requirements

### L4

Best for:

- acceleration and meme-style behavior
- very fast momentum ignition

Expect:

- faster entries
- faster exits
- stricter profit protection

## What Should Be Configurable

These should increasingly become baseline/config driven:

- entry thresholds
- promotion thresholds
- score floors
- trend / structure requirements
- cost floors
- cooldown rules
- stop / TP / trail settings
- stale thresholds
- tighten thresholds
- portfolio slot and weight limits

## What Should Stay In Code

These should remain code-owned:

- feature computation
- score calculation mechanism
- candidate packet building
- deterministic legality checks
- final execution mechanics
- final deterministic exit firing

## What Nemo Should Do

Nemo should:

- judge cleaned finalists
- choose `OPEN` or `HOLD`
- rank relative quality
- provide bounded size bias only

Nemo should not:

- own the whole system
- decide stops
- decide final exit firing
- rewrite settings by itself

## Daily Operating Questions

A healthy daily review should ask:

1. Did the system buy clean structure or weak noise?
2. Did it avoid late chasing?
3. Did it hold healthy winners long enough?
4. Did it cut failed trades fast enough?
5. Did it rotate only when replacement quality was real?
6. Were the trades large enough to matter after costs?

## Final Read

The trading edge is not likely to come from one magic indicator.

It is more likely to come from:

- disciplined entries
- post-cost filtering
- better holding behavior
- cleaner exits
- smaller amount of higher-quality action

That is the behavior this project should be moving toward.
