# KrakenSK Architectural Review

## 1. Gate Logic & Filtering Analysis
**Current State:** You use a three-tier gate system:
1. **Deterministic Gate:** Technical indicators pre-filter symbols (e.g., `entry_score` < 46).
2. **Advisory Gate:** Phi3 (Phi-3.5) acts as a "reflex" agent, checking data integrity and volatilty shocks.
3. **Strategic Gate:** `should_run_nemotron` decides if the full 9B Nemotron model is needed.

### Evaluation:
* **Filtering Aggressiveness:** Your thresholds are reasonable. The "Strong score override" rule in `prompts.py` is a critical safety valve that ensures high-quality technical setups aren't vetoed by a conservative LLM.
* **Lane Logic:** Excellent use of lanes (L1-L4). Specifically, the L4 (Meme) lane having lower thresholds but higher volume requirements matches market reality.
* **Refined Suggestion:** Consider moving "Ranging market" detection to the deterministic layer to save Phi3 tokens if the market is clearly sideways.

## 2. LLM Integration & Tool-Calling
**Current State:** ReAct-style loop with a maximum of 2 round trips. Uses a file-based lock (`msvcrt`) for local inference.

### Evaluation:
* **Tool-Calling Robustness:** The pattern is solid. Restricting Nemotron to "at most one tool call" is wise for a 9B model; deeper loops often hallucinate on smaller models.
* **JSON Repair:** Your `_repair_json_response` logic is a great defensive engineering piece. Small models often omit closing braces or prepend prose; this saves many failed trades.
* **Locking:** The `_LocalNemotronLock` correctly prevents the NPU and CPU from being choked by concurrent inference calls.

## 3. Risk Management & Position Management
**Current State:** ATR-based stops/takes, trailing stops, and portfolio-level caps.

### Evaluation:
* **Position Sizing:** `EXEC_RISK_PER_TRADE_PCT = 25.0` (from `.env`) is **extremely aggressive** for a production system. Even with a `PORTFOLIO_MAX_WEIGHT_PER_SYMBOL = 0.2`, a 25% risk per trade implies very tight stops or massive exposure.
* **Exit Posture:** Using Phi3 to evaluate "posture" (RUN, TIGHTEN, EXIT) for open positions is one of the strongest parts of the architecture. It allows "letting winners run" beyond static targets.

## 4. Production Readiness & Scalability
### Critical Findings:
> [!WARNING]
> **Performance Bottleneck:** In `apps/trader/main.py`, your OHLC fetching is sequential.
> ```python
> ohlc_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "1m") for symbol in eval_symbols}
> ```
> With 85 symbols, this will take several seconds per loop iteration just for data fetching, likely causing the bot to lag behind the 1m bar closing.

> [!IMPORTANT]
> **Race Conditions:** While the loop is async/single-threaded, your `save_position_state` calls write to disk frequently. In a production environment, ensure you have a backup of `positions_state.json` or use an atomic write (temp file + rename).

### Missing Production Features:
1. **Rate Limit Management:** Kraken has strict API point limits. You need a centralized `RateLimiter` to ensure your REST/WS calls don't get banned during high volatility.
2. **Health Checks:** A watchdog to ensure the Phi3 and Nemotron endpoints are actually responding (beyond just catching exceptions).
3. **Parallel Execution:** Use `asyncio.gather` for fetching data and running Phi3 advisory reviews across multiple symbols simultaneously.

## Final Verdict
The architecture is **sophisticated and well-engineered** for a local-LLM setup. The separation of concerns between "reflex" (Phi3) and "strategy" (Nemotron) is professional. Fix the sequential loop and dial back the default risk, and it's ready for serious capital.
