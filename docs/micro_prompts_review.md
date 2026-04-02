# KrakenSK Micro-Prompts Review

## 1. Overview of Architecture
The `micro_prompts.py` file implements a sophisticated sub-agent architecture. You use **Phi3** as a "Reflex Agent" (fast, low-latency) and **Nemotron** as a "Strategic Agent" (higher reasoning). 

### Key Strengths:
*   **Heuristic Fallbacks:** Every single LLM call has a deterministic `_heuristic_...` fallback. This is **Tier-1 Engineering**. It ensures the bot never hangs or makes a random decision because an API was slow or a model hallucinated JSON.
*   **Feature Compaction:** `_compact_advisory_features` is a smart way to stay within the small 9B context window while providing the most relevant data.
*   **Visual Integration:** The hooks for `visual_context` (Phi-3.5 Vision) in market state reviews are a unique edge.

## 2. Specific Prompt Analysis

### `phi3_supervise_lane`
*   **Role:** Checks if the symbol actually belongs in its technical lane (e.g., if a Meme coin has become a "Breakout").
*   **Tuning:** The heuristic relies heavily on `momentum_5 > 0.01`. In low-volatility regimes, this might be too high, causing everything to default to the "Main" lane.
*   **Data dependency:** If `NewsSentiment` or `DexScreener` fail, the advice becomes significantly weaker.

### `nemotron_review_candidate`
*   **Role:** The "Final Gate" before the strategist loop.
*   **Weakness:** It uses `NEMOTRON_REVIEW_CANDIDATE_SYSTEM_PROMPT`. 9B models sometimes struggle with the "Bias" instruction (e.g., `action_bias: hold_preferred`). If the model is too conservative, it will demote winners before they can run.

### `nemotron_review_outcome`
*   **Role:** Post-trade analysis.
*   **Observation:** This is currently "Disconnected Intelligence." The model writes a "lesson," but that lesson isn't currently a feature or a prompt injection for the *next* trade on that symbol.
*   **Recommendation:** Capture the "lesson" string and store it in your `TradeMemory` as a "Symbol Tip."

## 3. Recommended Improvements
1.  **Divergence Check:** Ensure `_heuristic_exit_posture_review` stays in sync with `phi3_exit_posture.py`. Having two different "hardcoded" exit rules creates confusing bugs.
2.  **Add Volatility Context:** In `_compact_advisory_features`, include `hurst` and `bb_bandwidth`. These tell the model if the momentum is "real" or just noise in a tightening range.
3.  **Lane-Specific Confidence:** Adjust `lane_confidence` based on the lane. A 0.7 confidence in a Meme coin (L4) should be treated differently than 0.7 in BTC (L1).

## Final Verdict
The Micro-Prompt system is **highly robust**. It's the "secret sauce" that makes the bot feel intelligently responsive rather than just following a math formula. Focus on **closing the loop** so the "Lessons" from `review_outcome` actually feed back into `review_candidate`.
