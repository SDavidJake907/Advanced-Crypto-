# 🏛️ Advanced-Crypto: The KrakenSK Elite Framework

**Institutional-Grade Deterministic Trading & AI Strategy Orchestration**

> [!CAUTION]
> **LEGAL DISCLAIMER:** This software is for **educational and research purposes only**. I am **not a financial advisor**, and this is **not financial advice**. Algorithmic trading involves extreme risk of capital loss. The author and contributors are **not responsible** for any financial losses, technical failures, or exchange-related issues. **Use at your own risk.**

[![Production Ready](https://img.shields.io/badge/Status-Live_Trading-red.svg?style=for-the-badge)](https://github.com/SDavidJake907/Advanced-Crypto-)
[![AI Architecture](https://img.shields.io/badge/AI-Nemo_/_Phi--3-blue.svg?style=for-the-badge)](https://github.com/SDavidJake907/Advanced-Crypto-)
[![Performance](https://img.shields.io/badge/Core-GPU_Accelerated-green.svg?style=for-the-badge)](https://github.com/SDavidJake907/Advanced-Crypto-)

## 📜 Executive Overview

**Advanced-Crypto** is a high-performance, professional-grade trading architecture engineered for the **Elite 7** cryptocurrency universe. It represents a paradigm shift from traditional algorithmic "collectors" to a **Concentrated Strike Force** model. 

The framework utilizes a unique **Hybrid Intelligence** approach: a zero-latency, deterministic GPU core handles the heavy lifting of market normalization and risk gating, while an advanced AI Intelligence Layer (local Nemotron-9B and NPU-accelerated Phi-3) provides high-level strategic orchestration.

---

## 🏛️ Core Philosophy: Precision over Participation

The system is built on the **"Anchor & Scrape"** mindset. Instead of diluting capital across hundreds of low-conviction setups, KrakenSK focuses on identifying and holding 1-2 "Alpha" anchors for parabolic moves while aggressively "scrapping" the daily noise for consistent cash flow.

### **The 3-2-3 Execution Model**
The system operates an **8-Slot Execution Engine**, strictly partitioned to balance long-term capital appreciation with short-term liquidity capture:

| Lane | Purpose | Quantity | Horizon | Armor (ATR) | Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | **Alpha Anchors** | 2 Slots | 21-28 Days | **1.65x (Heavy)** | Ignore daily noise; capture the parabolic "Week 3" moves. |
| **L2/L3**| **Swing Runners** | 3 Slots | 3-7 Days | **1.50x (Mid)** | Mid-week trend rotation; capture institutional rebalancing events. |
| **L4** | **NYC Scrapers** | 3 Slots | < 8 Hours | **1.42x (Light)** | Extreme momentum strikes (NYC Flush); mandatory 08:30 AM liquidations. |

---

## ⚙️ Architectural Stack: Code + AI Synergy

### **1. The Deterministic Core (GPU-Accelerated)**
The "Engine Room" of the bot is written in optimized Python with custom **CUDA C++ kernels** for real-time feature extraction at `double` precision.
*   **Indicator Engine:** High-performance computation of Wilder's RSI, ATR, and Bollinger Bandwidth.
*   **Northstar Analysis:** Real-time calculation of **Hurst Exponents**, Shannon Entropy, and Autocorrelation to detect structural decay before price crashes.
*   **Precision Gates:** Deterministic volume gates (e.g., 1.62x threshold) and spread-aware cost filters ensure only institutional-quality liquidity is entered.

### **2. The Intelligence Layer: Multi-Tiered AI**
Strategic decisions are offloaded to a multi-tiered AI stack to prevent "indicator blindness":

#### **Local Vision (Phi-3 NPU)**
Runs on dedicated device hardware (Intel NPU) via OpenVINO to perform **Structural Verification**. It translates raw OHLC data into "Visual Patterns"—identifying the "Lowest Low" floors or "Shooting Star" exhaustion points that deterministic math might miss.

#### **Global Strategy (NVIDIA Cloud AI)**
The system integrates with **NVIDIA's high-capacity Llama-3.3-Nemotron models** to act as the **Portfolio Overwatch**. 
*   **Comparative Ranking:** Unlike the local core which sees one coin at a time, the NVIDIA layer reviews the entire "Finalist Payload" to rank assets against each other, ensuring capital is only deployed to the absolute "Alpha" of the hour.
*   **Session Calibration:** Every morning at **04:30 AM AKDT**, the system triggers an **NVIDIA Tool Call**. The AI audits the previous 24 hours of global session data and automatically adjusts `runtime_overrides` (Aggression, Volume Gates, ATR Armor) to match the day's macro narrative (e.g., Geopolitical tension vs. Corporate expansion).
*   **Narrative Processing:** It consumes macro sentiment data to determine if the market is in a "Structural Recovery" or a "Distribution Flush," adjusting the 8-slot engine's posture in real-time.

---

## ₿ Institutional Mandate: The 25% BTC Influence Rule

KrakenSK recognizes Bitcoin as the market's **Lead Engine**. Altcoin scoring is never calculated in a vacuum; it is always weighted by BTC's structural health:
*   **Bullish Tailwind:** When BTC is in a confirmed uptrend, correlated alts receive a **+25% scoring tailwind** (+12 points), reflecting institutional capital flow.
*   **Automatic "Blow-off" Protection:** The system continuously monitors BTC for parabolic exhaustion (RSI > 85, EMA26 Distance > 15%). 
*   **The Bunker Protocol:** If a blow-off is detected, the system automatically flips to **DEFENSIVE**, tightens the volume gate to **2.5x**, and slashes trailing stops to **0.5 ATR** to harvest gains before the crash.

---

## 🔄 Seasonal & Fractal Market Dynamics

The bot is programmed with an inherent understanding of the **21-Day Fractal Cycle**:
*   **Week 1 (Accumulation):** Focuses on floor identification and L1 anchor building.
*   **Week 2 (Expansion):** Increases aggression to capture trend momentum.
*   **Week 3 (Distribution):** Expects high volatility and "NYC Flushes"; tightens all exits.
*   **Week 4 (Resolution):** Mandatory liquidation of stale rotations to clean the vault for the next month.

### **Perfect Ratio Timing (HTF vs LTF)**
The system adheres to the institutional **Perfect Ratio** for timeframe analysis:
*   **L1 (Anchors):** 1-Day HTF / 1-Hour LTF.
*   **L2/L3 (Swing):** 4-Hour HTF / 15-Minute LTF.
*   **L4 (Scrapers):** 1-Hour HTF / 5-Minute LTF.

---

## 🛠️ Operational Workflow (Session Awareness)

KrakenSK operates on a **24/7 Global Session Cycle**, adjusting its posture every few hours:

1.  **🛡️ Asia Open (16:00 - 00:00 AKDT):** Defensive mode. Task: Floor accumulation.
2.  **⚖️ London Open (00:00 - 04:00 AKDT):** Neutral mode. Task: Trend discovery.
3.  **🧪 Pre-NYC Pause (04:00 - 05:30 AKDT):** **Calibration Phase.** NVIDIA AI performs the daily audit.
4.  **⚔️ NYC Flush (05:30 - 08:30 AKDT):** Aggressive mode. Task: High-beta scraper strikes.
5.  **🛑 Kill Switch (08:30 AKDT):** Mandatory L4 liquidation to lock in daily alpha.

---

## 🚀 Getting Started

### **System Requirements**
*   **OS:** Windows 11 + PowerShell.
*   **RAM:** 32GB (Min) / 64GB (Recommended).
*   **GPU:** NVIDIA RTX (16GB VRAM) for local Nemotron & CUDA indicators.
*   **NPU:** Intel Core Ultra / OpenVINO for Phi-3 advisory.

### **Quick Ignition**
1.  **Configure:** Populate `.env` with Kraken API keys.
2.  **Boot Models:** `.\scripts\start_models.ps1` (Initialize NPU and Ollama).
3.  **Launch Stack:** `.\scripts\start_all.ps1` (Ignite the Watchdog and Core Apps).

---
*Disclaimer: Advanced-Crypto is a professional-grade systematic wealth generation framework. I am not a financial advisor. Algorithmic trading involves high risk. Adherence to the Master Blueprint and technical rigor is mandatory. USE AT YOUR OWN RISK.*
