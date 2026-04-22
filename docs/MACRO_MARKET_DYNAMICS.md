# 🌊 Macro Market Dynamics: The Seasonal & Fractal Playbook

## Purpose
This document provides the macro-economic and seasonal context for the KrakenSK Intelligence Layer (Nemo/Phi-3). It ensures that the models do not "run blind" but instead understand the current position in the quarterly, monthly, and weekly fractal cycles.

---

## 📅 Quarterly Cycles (Q1 - Q4)

The crypto market operates on institutional quarterly rebalancing cycles. Each quarter has a distinct "personality" driven by tax cycles, bonus distributions, and institutional reporting.

### **Q1: The Ignition (Jan - Mar)**
*   **Behavior:** New capital injection. "January Effect" often leads to a strong start.
*   **Sell-off Trigger:** Mid-March tax-prep liquidity harvesting.
*   **Blow-off:** Late March institutional reporting window.
*   **Strategy:** Favor **L1 Anchors** and **L2 Rotators** as new leadership emerges.

### **Q2: The Expansion (Apr - Jun)**
*   **Behavior:** Peak trend maturity. "Sell in May and go away" often creates a mid-quarter dip followed by a recovery.
*   **Sell-off Trigger:** Mid-May "Tax Day" (US) and late June end-of-half-year rebalancing.
*   **Blow-off:** Late April "Tax Refund" momentum strikes.
*   **Strategy:** Be aggressive in **L4 Scrapers** during April; rotate to **Defense** in late May.

### **Q3: The Distribution (Jul - Sep)**
*   **Behavior:** Summer lull. Low volume, ranging markets, and "boring" price action.
*   **Sell-off Trigger:** Late August "Vacation Liquidity" exit and September "Red Month" (historically the weakest month).
*   **Blow-off:** Rare; usually localized to specific "Meme Cycles" in L4.
*   **Strategy:** Tighten **ATR Armor**. Reduce **NEMOTRON_TOP_CANDIDATE_COUNT**. Focus on **capital preservation**.

### **Q4: The Resolution (Oct - Dec)**
*   **Behavior:** The "Uptober" effect. Strong year-end performance as institutions chase benchmarks (FOMO).
*   **Sell-off Trigger:** Mid-December "Tax-Loss Harvesting."
*   **Blow-off:** "Santa Rally" (Dec 24 - Jan 2).
*   **Strategy:** Maximum **OFFENSIVE** posture. Use **TP Bypass** on L1 winners.

---

## 📉 Market Mechanics: Sell-offs vs. Blow-offs

### **1. The Sell-off (Structural Decay)**
*   **Mechanism:** Higher-lows fail. 1h EMA stack flips bearish. Volume ratio drops below 1.0.
*   **Nemo Instruction:** Identify "Lower High" signatures on the 4h scan.
*   **Action:** Immediate **L4 Liquidation**. Tighten **L1 Trailing Stops** to 1.5 ATR.

### **2. The Blow-off (Parabolic Exhaustion)**
*   **Mechanism:** Vertical price action. RSI > 85 on 1h. "Distance from EMA26" > 15%.
*   **Nemo Instruction:** Look for "Doji" or "Shooting Star" candles at the top of the range.
*   **Action:** Enable **TP Bypass** but move **Stop Loss to Break Even** immediately.
*   **Automatic Adjustment:** When BTC enters a confirmed `blowoff` regime, the system automatically triggers the following `runtime_overrides`:
    *   **Posture:** Flipped to `DEFENSIVE` immediately.
    *   **Volume Gate:** Increased to `2.5x` (only extreme institutional confirmation allowed).
    *   **Net Edge:** Floor increased to `0.5%` (must be a very high-quality setup to enter).
    *   **Stops:** `EXIT_ATR_STOP_MULT` increased to `3.0` (room for volatility) but `EXIT_TRAIL_ATR_MULT` tightened to `0.5` (lock in the parabolic move).

---

## 🔄 The Weekly Fractal (Week 1 - Week 4)

Each month is treated as a micro-representation of the 21-day hold cycle.

*   **Week 1 (Accumulation):** The first Monday/Tuesday sets the "Floor" for the month.
*   **Week 2 (Expansion):** The trend establishes itself.
*   **Week 3 (Distribution/Volatility):** The "Fractal Week 3" scan. High volatility. Expect "NYC Flushes."
*   **Week 4 (Resolution):** Monthly close positioning. Liquidate stale L2/L3 rotation names.

---

## 🏛️ Session-Aware Strategy (AKDT Time)

| Phase | Time | Posture | Task |
| :--- | :--- | :--- | :--- |
| **Asia Open** | 16:00 - 00:00 | **Defense** | Floor Accumulation (Lowest Low) |
| **London Open** | 00:00 - 04:00 | **Neutral** | Ignition / Trend Discovery |
| **Pre-NYC Pause**| 04:00 - 05:30 | **Calibration**| **NEMO TOOL CALL** / Audit |
| **NYC Flush** | 05:30 - 08:30 | **Aggressive** | Momentum Strike (Scrapers) |
| **London Close** | 08:30 | **Kill Switch** | **Liquidate L4** (Mini-Paycheck) |
| **NYC Settlement**| 11:30 - 13:00 | **Defense** | Daily Close Structural Integrity |

---

## ₿ The 25% BTC Influence Rule

In the KrakenSK framework, Bitcoin is the **Lead Engine**. Altcoin performance is weighted by BTC's current structural health.

*   **BTC Bullish Trend:** Alts receive a **+25% tailwind weighting** (approx +12 points to Final Score) if their correlation to BTC is > 0.40. This reflects institutional capital flowing from BTC into high-beta alts.
*   **BTC Bearish/Volatile:** The tailwind is removed, and **ATR Armor** is tightened by 25% across all L2-L4 lanes to protect against correlated flushes.
*   **Correlation Proxy:** The system uses real-time `btc_correlation` (from the GPU feature pipeline) to determine the exact strength of the 25% affect.

---

## 🧠 Intelligence Layer Mandate

**Nemo/Phi-3 must evaluate every candidate through these lenses:**
1.  **Quarterly Bias:** Are we in an ignition (Q1/Q4) or a distribution (Q2/Q3) phase?
2.  **Weekly Position:** Is this "Resolution Week" or "Accumulation Week"?
3.  **Session Context:** Is the current price action consistent with the session's intended logic (e.g., floor accumulation in Asia)?

*Do not trade against the macro seasonal flow unless the technical signal (Entry Score > 90) is undeniable.*
