# 🏛️ Advanced-Crypto: The KrakenSK Elite Framework

**Production-Grade Deterministic Trading & AI Strategy Orchestration**

> [!IMPORTANT]
> **PRODUCTION STATUS:** This repository is currently in **LIVE TRADING** mode. All execution is deterministic and backed by institutional-grade risk parameters. Operating this system requires strict adherence to the **Master Operational CSV** and session-aware protocols.

#### **Executive Overview**

Advanced-Crypto is a high-performance trading architecture designed for the **Elite 7** cryptocurrency universe. It utilizes a **GPU-accelerated deterministic core** for execution, while offloading high-level strategic "Mindset" adjustments to a local **Nemotron (Nemo) / Phi-3 Cloud Advisory** model.

The system operates on a **24/7 Global Session Cycle**, utilizing lane-specific risk management to balance long-term capital appreciation (Anchors) with short-term liquidity capture (Scrapers).

---

#### **Core Architecture: The "Whole 9 Yards"**

*   **Deterministic Execution:** Zero-inference Python/CUDA core managing real-time volume gates and ATR-based armor.
*   **NPU-Driven Charting (Phi-3):** Local on-device intelligence identifies structural signatures (e.g., "Lowest Low" floors and "Power Close" exits).
*   **Cloud Advisory (Nemo):** Performs a **Tool Call** every morning at 04:30 AM AKDT to calculate session-specific overrides.
*   **Fractal Logic:** The system treats 8-hour sessions as micro-representations of the 21-day macro cycle, ensuring "mini-paychecks" are harvested daily.

---

#### **Dynamic Operational Modes**

1.  **🛡️ Defensive (Asian Session):** Focuses on capital preservation and floor accumulation.
2.  **⚖️ Neutral (London Session):** Focuses on trend discovery and institutional confirmation.
3.  **⚔️ Aggressive (NYC Flush):** Maximum liquidity strike using the **Confidence Multiplier** and **TP Bypass**.

---

### 🛠️ Production Configuration (CSV)

The **Master Operational CSV** (`KrakenSK_Master_Operations_V1.csv`) integrates the exact time-aware triggers and lane overrides developed for this engine. This file acts as the "Source of Truth" for the engine's auto-adjustments.

#### **Implementation Guide**

*   **The Pause-and-Adjust:** The engine recognizes the **04:30 AM AKDT** window to pause the live trader, execute the Nemo tool call, and update `runtime_overrides.json` before the NYC open.
*   **Lane Isolation:** Lane 1 (Anchors) and Lane 4 (Scrapers) utilize independent **Precision Volume Gates** as specified in the configuration.
*   **The 08:30 AM Kill Switch:** Mandatory liquidation of all L4 positions to lock in session alpha.

---

## 🏛️ Advanced Execution: The 3-2-3 Model

The system utilizes an **8-Slot Execution Engine** divided into specialized lanes to balance stability with velocity:

*   **L1 (Anchors):** 2 Slots. Focused on 21-day "Alpha" holds with Heavy Armor (1.65 ATR).
*   **L2/L3 (Swing):** 3 Slots. Mid-week trend rotators.
*   **L4 (Scrapers):** 3 Slots. NYC Flush strikes with Light Armor (1.42 ATR) and mandatory 08:30 AM Kill Switch.

## ₿ Institutional Mandate: 25% BTC Influence

Bitcoin is the **Lead Engine**. Altcoin scoring is dynamically weighted by BTC's structural health:
*   **BTC Tailwind:** Alts receive a **+25% scoring tailwind** (approx +12 points) when BTC is bullish and correlation is high (> 0.40).
*   **Automatic Protection:** The system monitors for **Blow-off Tops** (RSI > 85). If BTC goes parabolic, the engine automatically flips to **DEFENSIVE** and tightens all trailing stops to **0.5 ATR**.

---

## The Architecture: Code + AI Synergy

- **Deterministic Core:** Owns market normalization, feature extraction (GPU-accelerated), scoring, risk gating, and execution.
- **GPU Feature Pipeline:** High-performance indicator computation via custom CUDA kernels (`double` precision).
  - `cuda_rsi`: Wilder's RSI (RMA-smoothed).
  - `cuda_atr`: Average True Range lookbacks.
  - `cuda_bollinger`: Middle, Upper, Lower, and Bandwidth.
  - `cuda_correlation`: Real-time full correlation matrix across the universe.
  - `cuda_northstar`: Hurst exponent, Shannon entropy, and autocorrelation.
- **Phi-3 NPU (Advisory):** Owns visual pattern verification, structural context, and candle-evidence translation.
- **Nemotron 9B (Strategist):** Owns comparative ranking and final structured `OPEN/HOLD/FLAT` judgment on finalist candidates.
- **Data-Driven Exits:** Zero time-based locks. Exits are triggered purely by price vs. ATR and structural integrity decay.

## Advanced Structural Analysis

The system employs **Lane-Aware Multi-Timeframe Feature Sets**, ensuring each symbol is judged by its specific hold-style:

| Lane | Primary TF | Range TF | Use Case |
|------|-----------|----------|----------|
| **L4 (Meme)** | 5m | 1h | Fast momentum ignition and quick holds. |
| **L2/L3 (Swing)** | 15m | 4h | Channel continuation and structured breakouts. |
| **L1 (Blue Chip)** | 1h | 1D | Multi-day trends and persistent leadership. |

## System Flow

```text
Market Data -> GPU Feature Engine -> Dynamic Lane Classification -> Phi-3 Verification
           -> Final Candidate Payload -> Nemotron Judgment
           -> Portfolio Guard (BTC Priority) -> Execution -> Pure Data-Driven Exits
           -> Trade Memory / Replay / Shadow Review
```

## Quick Start

### 1. Clone & Configure
```powershell
git clone https://github.com/SDavidJake907/Advanced-Crypto-.git
cd Advanced-Crypto-
copy .env.example .env
# Add API credentials and adjust configs/runtime_overrides.json
```

### 2. Install Dependencies
```powershell
pip install -e .
```

### 3. Boot Local Models
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_models.ps1
```
*(Wait for "Phi-3 ready" before proceeding)*

### 4. Ignite the Stack
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_all.ps1
```

## Hardware Requirements (Recommended)

- Windows 11 + PowerShell
- Modern 8+ Core CPU (Intel Core Ultra 7 / Ryzen 7)
- 32 GB RAM
- NVIDIA RTX GPU (16GB VRAM) for local Nemotron & CUDA features.
- Intel NPU / OpenVINO setup for local Phi-3 advisory.

---
*Disclaimer: Advanced-Crypto is a professional trading system. Systematic Wealth Generation requires discipline and technical rigor.*
