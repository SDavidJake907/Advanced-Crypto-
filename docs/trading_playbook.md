# Trading Playbook: The Elite Edition

This is the definitive operating guide for the KrakenSK high-performance trading engine. It defines the "Elite" behavior achieved through intensive data-driven tuning.

## The North Star
KrakenSK makes money by following **Dominance** and **Structure**. We do not chase noise; we capture confirmed shifts in market regime.
- **Follow the Lead:** BTC is the general; the alts are the fleet.
- **Buy the Breakout:** Range breakouts with volume surges are the highest priority.
- **Cut the Cancer:** If the structure breaks, we exit instantly. No exceptions.
- **Let the Winners Run:** If the structure is intact, we ignore the clock and follow the trend.

## The Core Protocols

### 1. The BTC Lead Protocol (L1 Flagship)
`BTC/USD` is the uncontested market leader.
- **Priority:** BTC bypasses all correlation and volatility downsizing.
- **Aggression:** BTC uses the most aggressive, softened filters.
- **Bonus:** High BTC Dominance provides a score boost to the entire portfolio, signaling a "Market Lead" regime.

### 2. Pure Data-Driven Exits (No Time Brakes)
Time is an arbitrary metric. We trade only on **Market Data**.
- **Instant Stops:** The `STOP_MIN_HOLD_MIN` is disabled (0.0). Stop-losses are armed and active the moment an entry is filled.
- **Structural Integrity:** We hold based on 15m EMA and support levels. If the price is negative but the structure is `intact`, we hold (The "Runner" Protocol). If the structure flips to `broken`, we exit immediately.
- **ATR-Based Goals:** We aim for realistic targets (2.5x ATR for leaders) to ensure profit covers fees and slippage.

### 3. The Elite Breakout Gate
Confirmed breakouts are the system's primary engine.
- **Dynamic Escalation:** Any coin (BTC or Alt) that breaks a 1h range with a significant volume surge is immediately escalated to L1/L2 status.
- **Context Over RSI:** We prioritize momentum and breakout structure over lagging indicators like RSI. We buy "hot" coins if the volume justifies the heat.

## Entry Playbook

### The Elite Entry Profile
A high-conviction entry must clear the **Triple Veto**:
1. **The Math (Deterministic):** High score, positive net edge, and confirmed breakout.
2. **The Vision (Phi-3):** Visual confirmation of a healthy retest or fresh breakout.
3. **The Brain (Nemotron):** Comparative ranking against the rest of the market.

### Archetypes
- **The Power Breakout:** `range_breakout_1h` + `volume_surge > 1.5x`.
- **The Structural Retest:** `pullback_hold` above the short EMA with recovering momentum.
- **The Rotation Leader:** High relative strength vs. BTC with increasing volume participation.

## Hold & Exit Playbook

### The "Runner" Mindset
We do not sell because we are "up." We sell because the move is "over."
- **RUN (Hold):** Structure is intact, price is above EMA, and macro trend is supportive.
- **TIGHTEN:** Profit target is near or momentum is starting to stall. Trailing stops move to the "noise floor."
- **FAIL (Exit):** Structural break, negative momentum crossover, or 2.2x ATR stop hit.

### Slot Velocity (Dynamic Rotation)
With a **12-slot capacity**, the bot is a constant rotation machine.
- **The Swap:** We will ruthlessly sell a "Boring" or "Stale" position to buy a 100-score "Elite" setup.
- **Follow Through:** We expect immediate momentum. If a coin sits flat while others are running, it is rotated.

## Lane Strategy (Dynamic Tiers)

- **L1 (Leaders):** Hard-coded BTC + Elite Breakouts. Highest weight, widest stops, maximum room to run.
- **L2 (Major Alts):** Confirmed trends and strong rotations. Scalable weight based on Kelly sizing.
- **L3 (Balanced):** Conservative setups. Used for "Steady" accumulation.
- **L4 (Memes/Fast-Movers):** High-heat, high-risk. Strictest filters, tightest stops, fastest exits.

## Summary Verdict
KrakenSK is now a **Structural Hunter**. We ignore 1-minute noise and 30-minute timers. We trade the **Dominance** of the leaders and the **Integrity** of the chart. 

**Follow the Data. Trust the Structure. Let it Hunt.**
