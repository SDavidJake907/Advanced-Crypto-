# Round 4 Review: Exits & Nemotron Prompt

## Reply to Question #1: Exit Strategy Gaps

**What you have:**
- Standard ATR stops (1.5x) and take profit (3.0x).
- Break-even logic at 1R (`EXIT_BREAK_EVEN_R_MULT`).
- Trailing stops activating at 1.5R (`EXIT_TRAIL_ARM_R`) and trailing by 1.0 ATR.
- Phi-3 heuristics that look at time vs. progress (`STALE` after 3 hours), adverse momentum (`EXIT` at -2%), and momentum decay on winners (`TIGHTEN` at +1%).

**What's missing & What a professional system would add:**
1. **Time-Based Early Invalidation:** You have a `STALE` condition, but it waits 180 minutes. In a 1m scalping framework, if a trade hasn't moved into profit within 5-15 bars, momentum was misjudged. Professional systems will cut risk early rather than waiting hours for a scalp to work out or hit the stop.
2. **Taking Partial Profits (Scaling Out):** Your exits seem monolithic (all-in, all-out). A professional strategy scales out (e.g., closing 50% at 1R or 1.5R, and letting the runner trail) to lock in realized PnL while removing variance, which is especially important for high volatility/meme pairs.
3. **Volume Climax / Exhaustion Exits:** Right now `phi3_exit_posture.py` checks for `rsi >= 72` or negative momentum to `TIGHTEN`. It does not explicitly exit on an extreme volume climax (blow-off top). Incorporating `volume_surge_flag` + climax RSI directly into an `EXIT` posture helps capture the top before the trail gets hit.
4. **Dynamic Trailing ATRs:** Your `EXIT_TRAIL_ATR_MULT` is a static 1.0. In volatile expansion regimes, this might get chopped out prematurely by normal pullbacks. Professional systems expand the ATR multiplier dynamically as the trend strengthens and tighten it as momentum wanes (e.g., chandelier exits).

---

## Reply to Question #4: Nemotron Prompt v2

The previous logic in `NEMOTRON_STRATEGIST_SYSTEM_PROMPT` was filled with nested qualitative "prefer A even if B" logic which can confuse a 9B model. Here is the rewritten core decision-making section using a direct, prioritized rule matrix that 9B models handle much more reliably:

```text
DECISION RULES (Evaluate in order top-to-bottom):
1. REJECT: If entry_recommendation is AVOID or reversal_risk is HIGH, ALWAYS return HOLD.
2. OVERRIDE: If entry_score >= 75 AND entry_recommendation is STRONG_BUY AND reversal_risk is LOW, ALWAYS return OPEN. Ignore all advisory caution.
3. TREND ENTRY: If entry_recommendation is BUY and momentum is positive, return OPEN.
4. SELECTIVE ENTRY: If entry_recommendation is WATCH AND reversal_risk is LOW or MEDIUM AND rotation_score > 0 AND momentum_5 > 0, return OPEN (at reduced size).
5. DEFAULT: If none of the above apply, return HOLD.

ADVISORY & CONTEXT HANDLERS:
- Candidate Review Action Bias: If bias is "reduce_size", do NOT change to HOLD. Return OPEN but request a smaller size. If bias is "hold_preferred", only override if Rule #2 applies.
- Multi-Timeframe Momentum: If momentum_5m and momentum_15m are both positive, confidence is high. If momentum_5m is negative but 15m is positive, be cautious (use reduced size or HOLD).
- Model Scores: xgb_score >= 60 reinforces any OPEN signal.
- Ranging Market: This means be selective. Do not automatically HOLD if Rule #2 or Rule #3 is met.
```

This rewrite replaces overlapping suggestions with a strict 5-step waterfall and separate context modifiers.
