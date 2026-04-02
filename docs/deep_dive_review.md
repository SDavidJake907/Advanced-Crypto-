# KrakenSK Deep Dive Review

## 1. Nemotron 9B Prompt Quality
**Verdict:** High risk of "Attention Dispersion."

### Observations:
*   **Structural Complexity:** For a 9B model, the prompt is "too clever." It contains complex override rules (e.g., "defer fully to defensive advisory when entry_score < 60"). Small models often fail at nested "if-then" logic in natural language.
*   **The Hallucination Gap:** You tell the model to "Return compact JSON only" BUT also hint it should "Prefer a direct final decision." This often causes models to hallucinate the `final_decision` object without actually processing the `strategy_decision` tool output, especially when under pressure.
*   **Recommendation:** Move the "Default planner bias" out of the prompt and into your Python `strategy_decision` tool logic. Keep the LLM's goal simple: "Identify the market narrative and pick the tool/bias."

## 2. Trade Memory & Learning Loop
**Verdict:** Effective storage, under-utilized injection.

### Observations:
*   **Context Window:** Nemotron 9B (Virtuoso) typically handles 32k context. Your `trade_memory.py` is excellent at logging, but the Strategist prompt doesn't see "Historical Mistakes" yet.
*   **Efficiency:** Vector search is overkill. Your `build_memory_block` method is exactly what's needed.
*   **How to Improve:** 
    *   **Lesson Injection:** In `nemotron.py`, pull the top 3 "Outcome Reviews" for the specific symbol being traded and inject them as `Recent Lessons for <SYMBOL>`.
    *   **Momentum of Win-Rate:** If a symbol has 3 losses in a row, the LLM should see a "TEMPORARY AVOID" flag in the prompt.

## 3. Feature Engineering Gaps
**Verdict:** Good technicals, missing "Fuel" indicators.

### Missing Signals:
1.  **Funding Rates:** Crypto momentum is often driven by "short squeezes." If Funding is extremely negative while price is rising, momentum is about to explode.
2.  **Breadth Correlation:** Is the move idiosyncratic or is the whole market moving? You have `market_fingerprint`, but a simple "BTC-Beta" feature would help Nemotron ignore market-wide noise.
3.  **Volatility of Volatility (VoV):** 1m scalpers need to know if the ATR is expanding or contracting *right now*.
4.  **Order Book Velocity:** Not just depth, but how fast the bid/ask is refreshing. This is the ultimate "front-run" indicator.

## 4. Exit Logic & Posture Review
**Verdict:** Sound heuristics, but "L4 (Meme)" needs a tighter leash.

### Observations:
*   **The "Hopium" Gap:** In `_heuristic_exit_posture`, the logic `pnl_pct <= neg_pnl_exit and momentum < 0 and trend_1h <= 0` is very forgiving. If a trade is down 3% but `trend_1h` is still positive, it holds. This is dangerous in high-volatility lanes.
*   **Stale Logic:** `STALE` is detected after `EXIT_STALE_MIN_HOLD_MIN`. For Meme coins, "stale" is often 10 minutes, not 60.
*   **Recommendation:** Pass the `lane` into the heuristic. Lane L4 should have a "Hard Time Stop" and tighter stop-loss posture (aggressive TIGHTEN).

## 5. Universe Ranking & Scoring
**Verdict:** Correct intent, but "Regime Blind."

### Observations:
*   **Ranking Lag:** You use 1h candles for ranking 85 symbols. By the time a coin shows up as a "Leader" on the 1h, the 1m scalping opportunity might be 50% through its move.
*   **Improvement:** Use **"Volume-Weighted Relative Strength" (VWRS)**. Rank coins against BTC on a 15m window. If a coin is up 2% while BTC is flat, it's a leader. If it's up 2% because BTC is up 2%, it's just noise.
*   **Bucket Optimization:** Your `shortlist_candidates` logic is great. It ensures you don't just trade 20 leaders that all crash at once when BTC dips.
