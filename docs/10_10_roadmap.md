# KrakenSK 10/10 Roadmap & Implementation Plan

This document outlines the steps necessary to elevate KrakenSK from an 8.5/10 to a 10/10 production-ready 1m scalping system based on recent architectural reviews.

## The Checklist

### Phase 1: Performance & Architecture
- [ ] Implement parallel standard OHLC fetching via `asyncio.gather`
- [ ] Implement atomic writes for `positions_state.json`
- [ ] Build decentralized RateLimiter & API Watchdog for Kraken

### Phase 2: Intelligence & LLM Tuning
- [ ] Inject `TradeMemory` "Lessons Learned" into Nemotron's strategizing context
- [ ] Simplify Nemotron 9B prompt (reduce nested logic, move bias defaults to Python)
- [ ] Adjust LLM generation settings: `top_p=1.0`, `max_tokens=800`
- [ ] Inject `hurst` and `bb_bandwidth` into `_compact_advisory_features`

### Phase 3: Strategy & Scoring Gaps
- [ ] Upgrade Universe Ranking: use 15m VWRS (Volume-Weighted Relative Strength) instead of 1h lag
- [ ] Recalibrate score weights (60% 5m momo, 20% 1m momo, 20% 1h trend)
- [ ] Integrate missing dataset: Volume Ratio log weighting

### Phase 4: Risk Management & Exits
- [ ] Replace static 25% risk with dynamic risk sizing (account size/volatilty)
- [ ] Refine Entry Gates: `NEMOTRON_GATE_MIN_ENTRY_SCORE = 52` for L1/L3, > 2.0x volume for L4
- [ ] Tighten exits: 1.8 ATR stops, 5.0 ATR targets
- [ ] Stricter Time/L4 exit heuristics (faster stale detection for Memes)

---

## Detailed Implementation Plan

### Phase 1: Performance & Robustness
#### `apps/trader/main.py`
- Convert sequential OHLC queries into parallel operations using `asyncio.gather` for symbols.
- Implement rate limiting logic and API health checking within the loop.

#### `apps/trader/portfolio.py` (or relevant state file)
- Change `json.dump` for `positions_state.json` to write to a `.tmp` file and rename it atomically to prevent corruption during race conditions.

### Phase 2: Intelligence & LLM Tuning
#### `apps/trader/nemotron.py`
- Change `max_tokens` from 2048 to 800 for faster TTFT.
- Change `top_p` from 0.9 to 1.0 to prevent probability clipping.
- Retrieve the most recent top 3 lessons from `TradeMemory` for the active pair and push them to the system prompt context as "Lessons Learned".

#### `apps/trader/prompts.py`
- Remove complex, nested heuristics from `NEMOTRON_REVIEW_CANDIDATE_SYSTEM_PROMPT`.
- Simplify structural requirements so the 9B model doesn't overthink bias extraction or hallucinate loops.

#### `apps/trader/micro_prompts.py`
- Add `hurst` and `bb_bandwidth` to `_compact_advisory_features()` so the Phi3 Reflex agent understands the volatility context.

### Phase 3: Strategy & Analytics
#### `apps/trader/features.py`
- Incorporate Volume-Weighted Relative Strength (VWRS) against BTC on a 15-minute timeframe for lead generation instead of the lagging 1h logic.
- Adjust internal score weight logic (60% 5m momentum, 20% 1m momentum, 20% 1h trend).
- Add Volume Ratio log augmentation weight.

### Phase 4: Risk & Escapes
#### `apps/trader/gates.py` 
- Adjust `NEMOTRON_GATE_MIN_ENTRY_SCORE` to 52 for tighter entry requirements.
- Enforce `volume_surge > 2.0` on L4 memes before proceeding to the NPU.

#### `apps/trader/exits.py` & `.env` configurations
- Overhaul `EXEC_RISK_PER_TRADE_PCT` for dynamic volatility scaling instead of a flat 25%.
- Shift default exit parameters: 1.8 ATR stop, 5.0 ATR target.
- Accelerate "stale" evaluation duration for L4 lanes (Meme coins shouldn't be held for 60 mins flat).
