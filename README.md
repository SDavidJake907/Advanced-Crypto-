# KrakenSK / Advanced-Crypto

> [!WARNING]
> **RISK DISCLAIMER:** Algorithmic trading involves significant financial risk. This repository is for educational and research purposes only. The author/operator is **not a financial advisor**. Use of this system, its code, and its strategies is at your own risk. The author is not responsible for any financial losses, technical failures, or exchange-related issues incurred through the use of this software.

KrakenSK is a deterministic-first, AI-assisted crypto trading system. It merges hard mathematical risk controls with local LLM (Nemotron/Phi-3) strategic judgment. The architecture is Kraken-first, engineered for live, autonomous scalping and swing trading.

## Third-Party Notice
This project is an orchestration layer that integrates with third-party services and models, including the Kraken API, Ollama (Nemotron), and OpenVINO (Phi-3). Users are responsible for complying with the terms of service and licenses of all third-party providers used in their local runtime.

**The Core Philosophy:** If it can be computed in code, it is computed in code. Features, scoring, portfolio limits, execution sizing, and live exits are purely deterministic. AI is reserved exclusively for chart context, structural verification, and comparative finalist ranking.

## The Architecture: Code + AI Synergy

- **Deterministic Core:** Owns market normalization, feature extraction (GPU-accelerated), scoring, risk gating, and execution.
- **Phi-3 NPU (Advisory):** Owns visual pattern verification, structural context, and candle-evidence translation.
- **Nemotron 9B (Strategist):** Owns comparative ranking and final structured `OPEN/HOLD/FLAT` judgment on finalist candidates.
- **Data-Driven Exits:** Zero time-based locks. Exits are triggered purely by price vs. ATR and structural integrity decay.

## System Flow

```text
Market Data -> GPU Feature Engine -> Dynamic Lane Classification -> Phi-3 Verification
           -> Final Candidate Payload -> Nemotron Judgment
           -> Portfolio Guard (BTC Priority) -> Execution -> Pure Data-Driven Exits
           -> Trade Memory / Replay / Shadow Review
```

## Current Decision Rules & Protocols

- **The BTC Lead Protocol:** `BTC/USD` is the uncontested flagship (L1). It is mathematically immune to correlation-based downsizing and volatility-scaling penalties. When the market moves, the system follows BTC with full weight.
- **Pure Data-Driven Exits:** The system does not hold trades based on a clock. Stop-losses are armed immediately upon entry and triggered by structural breaks or ATR-based invalidation.
- **Dynamic Breakout Priority:** Range breakouts supported by volume surges bypass standard trend/RSI filters, immediately escalating candidates to L1/L2 leader status.
- **Kelly Criterion Sizing:** Position weights are dynamically calculated using half-Kelly fractions derived from rolling historical trade performance.
- **Asymmetric Risk Management:** Strict filters on low-cap "meme" coins (L4) combined with loosened, aggressive filters for market leaders (L1/L2) ensure capital is deployed where momentum is highest.

## Repository Layout

- `apps/` - Live trader loop, universe manager, review scheduler, and operator UI.
- `configs/` - Hot-reloadable runtime configuration and exchange metadata.
- `core/` - The brain: policy gates, risk engines, LLM adapters, feature extraction, and trade memory.
- `cpp/` - Custom CUDA kernels for blazing-fast technical indicator computation.
- `docs/` - Deep-dive architecture notes, operating models, and playbooks.
- `scripts/` - PowerShell utilities for booting models and the trading stack.
- `tests/` - Comprehensive regression suite and replay harness.
- `logs/` - SQLite system record, decision traces, and synced state.

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

## Operational Safety & Shadow Mode

KrakenSK is built for production, but never blindly trust live capital to new logic.
- **Shadow Validation:** Run `SHADOW_VALIDATION_ENABLED=true` to compare baseline vs. LLM engine decisions live without placing real orders.
- **Replay Mode:** Use `apps/replay/main.py` to backtest against historical ticks.
- **Runtime Overrides:** The `review_scheduler` stages strategy proposals. They must pass shadow validation before being applied to the live trading engine.

## Hardware Requirements (Recommended)

- Windows 11 + PowerShell
- Modern 8+ Core CPU (Intel Core Ultra 7 / Ryzen 7)
- 32 GB RAM
- NVIDIA RTX GPU (16GB VRAM) for local Nemotron & CUDA features.
- Intel NPU / OpenVINO setup for local Phi-3 advisory.

---
*Disclaimer: KrakenSK is not financial advice. Use this system at your own risk. Live trading incurs financial risk, and algorithmic execution can result in capital loss.*
