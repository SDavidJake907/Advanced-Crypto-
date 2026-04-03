# KrakenSK System Upgrade Plan

Based on the recent comprehensive A-Z audit (Project Rating, Tuning, Architectural, and Deep Dive reviews), your system possesses a very strong and advanced local-LLM architecture, utilizing a clever dual-agent approach (Phi3 for fast reflex, Nemotron for deeper strategic evaluation). 

You are currently hovering around an 8.5/10 due to some bottlenecks directly impacting a 1-minute scalping strategy. This plan operationalizes your `10_10_roadmap.md` into concrete, actionable steps to elevate the system to production readiness.

## User Review Required

> [!WARNING]  
> **Breaking Algorithm Changes**  
> We will be shifting the 25% flat risk per trade to a dynamic risk-sizing model, and tightening your ATR exit targets. This will change the baseline performance of the bot. Are you comfortable with executing these changes immediately?

> [!IMPORTANT]  
> **Production Dependencies**  
> Phase 1 requires modifying the core async loop in `main.py`. Test integrations on paper before live trading to ensure `asyncio.gather` does not induce any unexpected rate limiting from Kraken.

## Proposed Changes

---

### Phase 1: Performance & Robustness (Highest Priority)
The 1-minute chart moves fast. If the event loop is sequentially blocking on `fetch_live_ohlc()`, signals will lag and executions points will slip.
- **`apps/trader/main.py`**
  - **[MODIFY]**: Refactor sequential OHLC queries inside the main cycle into a parallel `asyncio.gather` operation.
  - **[MODIFY]**: Inject logic for API watchdog/rate-limiting to absorb concurrent request spikes natively.
- **`apps/trader/positions.py`** (or relevant state file)
  - **[MODIFY]**: Replace standard `json.dump` with atomic writes. We will write to `.tmp` files and `os.replace` to prevent JSON corruption if the program crashes mid-write.

---

### Phase 2: Intelligence & LLM Tuning
The LLMs are over-complicating prompt payloads while missing immediate context like trade history.
- **`apps/trader/nemotron.py` & `prompts.py`**
  - **[MODIFY]**: Adjust hyperparameters (`top_p = 1.0`, `max_tokens = 800`) to increase consistency and response time.
  - **[MODIFY]**: Plumb `TradeMemory` queries to pull the top 3 "Outcome Reviews" for the specific symbol being analyzed, embedding them in the system prompt.
  - **[MODIFY]**: Strip complex "if-then" fallback heuristics from the Nemotron prompt, moving fallback routing logic strictly into Python rule gates.
- **`apps/trader/features.py` or `micro_prompts.py`**
  - **[MODIFY]**: Inject `hurst` and `bb_bandwidth` into the feature packet so the Phi3 reflex agent can assess volatility contexts properly.

---

### Phase 3: Strategy & Scoring Gaps
Current 1h lagging indicators evaluate pairs too slowly for 1m scalping.
- **`core/policy/` or `apps/trader/features.py`**
  - **[MODIFY]**: Add Volume-Weighted Relative Strength (VWRS) against BTC on a 15m window to replace 1h momentum for lead ranking.
  - **[MODIFY]**: Adjust scoring composite weights to heavily favor the 5m and 1m momentum over 30m/1h.
  - **[MODIFY]**: Incorporate a Volume Ratio `log()` modifier into the candidate composite score to emphasize volume surge validations.

---

### Phase 4: Risk Management & Exits
The current 25% flat risk model is unsuited for scalping. It must be dynamically scaled.
- **`apps/trader/gates.py`**
  - **[MODIFY]**: Increase `NEMOTRON_GATE_MIN_ENTRY_SCORE` to 52 for core pairs (L1/L3) to ensure high-quality setups.
  - **[MODIFY]**: Enforce a strict `> 2.0x` volume surge check for meme coins (L4).
- **`apps/trader/exits.py` / Environment configuration**
  - **[MODIFY]**: Overhaul risk logic to dynamically size positions based on the pair's ATR and predefined account account-risk fractions, shedding the flat 25% risk logic.
  - **[MODIFY]**: Tighten trailing/exit ATR parameters to ~1.8 ATR stop / 5.0 ATR target.
  - **[MODIFY]**: Update the stalemate detection timer for the L4 lane to force closures much sooner if price action dies.

## Open Questions

1. **Trade Memory Storage**: Is `TradeMemory` currently using a local file database like SQLite/JSON, or does it require hooking up to an external database like PostgreSQL? You mentioned Postgres in your global instructions, but it wasn't clear if `TradeMemory` is strictly using it yet.
2. **Execution Sequencing**: Should we implement Phase 1 (Parallel Data & Atomic Writes) first and verify its stability before tuning the logic (Phase 2-4)? Doing performance architecture first ensures any algorithmic changes aren't tainted by data lag.

## Verification Plan

### Automated Tests
- Run `pytest` across all core layers (`core/policy/`, `core/risk/`, `core/features/`) to ensure deterministic score weights and ATR exit parameters aren't breaking existing validations.
- Create mock test verifying `asyncio.gather` successfully requests simulated data drops concurrently without throwing errors.

### Manual Verification
- Deploy to paper trading to observe the loop cycles. Monitor the log timing timestamps for `fetch_live_ohlc` to confirm sequential delay is eradicated.
- Monitor execution sizing explicitly logging the "Dynamic Risk" dollar amount to ensure it is correctly sizing significantly lower than 25%.
