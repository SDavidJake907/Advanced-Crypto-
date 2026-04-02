# KrakenSK Tuning Review

## 1. Gate Calibration Analysis
**Current State:** `NEMOTRON_GATE_MIN_ENTRY_SCORE = 46`, `top_n = 20`.

### Evaluation:
* **The "Top 20" Bottleneck:** You are ranking 85 symbols but only allowing the Top 20 into the LLM loop via `symbol_in_top_candidates`. If a coin explodes from rank 85 to rank 25 in one minute, the LLM will never see it until the next rebalance.
* **Score Aggressiveness:** 46 is a "Neutral-Positive" score. For 1m scalping with an account size of $61, you want to be extremely selective. 
* **Principled Thresholding:**
    * **Loose Gate (L4/Meme):** 38 is fine, but needs a "Volume Surge" requirement of > 2.0x, not just 1.8x.
    * **Tight Gate (L1/L3):** Increase `NEMOTRON_GATE_MIN_ENTRY_SCORE` to **52**. This ensures you only process setups that already have technical confirmation.

## 2. LLM Behavior (Nemotron 9B)
**Current State:** `temperature = 0.0`, `top_p = 0.9`.

### Recommendations:
* **Sampling Settings:**
    * **Temperature:** Keep `0.0`. It is the gold standard for JSON consistency.
    * **Top P:** Increase to `1.0`. Nucleus sampling can occasionally "clip" the most probable token in a 9B model if the probability mass is narrow, which is common in structured JSON responses.
    * **Presence/Frequency Penalty:** Ensure these are `0.0` to prevent the model from "getting creative" with JSON key names.
* **Output length:** 2048 is generous. Nemotron 9B is fast, but for 1m bars, you want maximum speed. Limit `max_tokens` to **800**—this is more than enough for a tool-call and a final decision, and it reduces latency.

## 3. Score Weights for 1m Scalping
**Current State:** 50% Momentum (5m), 30% Momentum (14m), 20% Momentum (30m).

### Recommendations:
* **The "Mean Reversion" Trap:** In a 1m scalping context, 30m momentum is almost "Macro." 
* **Proposed Weights:**
    * **Momentum (5m):** 60% (The primary driver)
    * **Momentum (1m):** 20% (Added new feature needed: Immediate micro-trend)
    * **Trend Confirmation (1h):** 20% (The filter: don't scalp against the 1h trend).
* **Missing Weight:** **Volume Ratio weight**. A surge in volume is often a more reliable "Confirmation" than RSI. Add `0.1 * log(volume_ratio)` to the composite score.

## 4. Exit Parameter Calibration
**Current State:** `EXIT_ATR_STOP_MULT = 2.5`, `EXIT_ATR_TAKE_PROFIT_MULT = 6.0`.

### Evaluation:
* **The $61 Account Constraint:** On a $61 account, Kraken's minimum order (e.g., 0.01 ETH) might be a significant portion of your capital. 
* **ATR Sizing:** 2.5 ATR is a standard "Swing" stop. For 1m scalping, it might be too wide, leading to "Death by a thousand cuts" if the win rate isn't > 60%.
* **Ratio:** A 1:2.4 risk-reward (2.5 stop vs 6.0 target) is healthy, but in crypto, "Liquidity Sweeps" often hit the 1.5 - 2.0 ATR range. 
* **Tuning Suggestion:** Use **Tight 1.8 ATR stops** and **Ambitious 5.0 ATR targets**. Scalping is about catching the *start* of the momentum burst; if it hasn't moved in your favor by 1.8 ATR, the "alpha" of the 1m bar is likely gone.
