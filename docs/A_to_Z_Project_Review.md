# KrakenSK A-to-Z Project Review

## 1. System Status & Runtime Overview
- **Active Processes**: The system is currently running (`node` and `python3.12` instances active). The stack includes `universe_manager`, `review_scheduler`, and the core `trader`.
- **Environment Context**: Operating in `live` mode with Kraken (`KRAKEN_ENV=live`, `EXECUTION_MODE=live`).
- **Data & Positions**: The `open_orders.json` shows recent extensive activity across coins like BTC, AAVE, ZRO, FET, ZEC, XRP, ADA, DOGE. Exits are regularly hitting `stop_loss`, `take_profit`, and `exit_posture:time_stop_no_progress`. The `position_state.json` is currently flat (no open positions at the moment of scan), meaning risk is completely reset. 
- **AI Backend**: Utilizing `phi3` on NPU for fast reflex and `NVIDIA-Nemotron-Nano-9B-v2` locally for strategic candidate advisory. 

## 2. Codebase & Architectural Analysis

### A. Universe & Lane Classification (`core/policy/lane_classifier.py`)
- **Current State**: Pairs are hard-classified into L1 (Breakout), L2 (Reversion), L3 (Main), and L4 (Meme).
- **Observation**: The conditions for L1 and L2 rely heavily on hardcoded thresholds (e.g., `volume_ratio >= 1.6 and momentum_5 > 0.006` for L1). 
- **Required Adjustment**: To fix the issues where L2 and L3 assignments feel inconsistent or lists appear empty, these thresholds need to be smoothed out or dynamically calibrated based on actual market regime (`regime_7d`), rather than relying on static scalar boundaries that pairs frequently float in and out of.

### B. Nemotron Gating & Logic (`core/policy/nemotron_gate.py` & `nemotron.py`)
- **Current State**: `passes_deterministic_candidate_gate()` and `should_run_nemotron()` filter heavily. Symbols need a minimum score (`NEMOTRON_GATE_MIN_ENTRY_SCORE`) or need to be shortlisted.
- **Observation**: There are overrides like `NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP` which allow edge cases to bypass the strict scoring if they have a low reversal risk. 
- **Required Adjustment**: The deep nesting of boolean conditions (e.g. `lane == "L1" and trend_confirmed and momentum_5 > 0.0 or lane == "L2" ...`) makes tuning difficult. Prompt V2 and the corresponding gate logic need refactoring to reduce this "spaghetti" boolean logic into a generalized score-weight system to align with the goal of JSON-only strategy decisions.

### C. Execution & Position Sizing (`core/execution/clamp.py` & `portfolio.py`)
- **Current State**: Uses a flat `EXEC_RISK_PER_TRADE_PCT` (e.g., 25% for main, 20% for memes) applied directly to equity. 
- **Observation**: While a `kelly_fraction` is accepted in the signature, it mainly relies on hard caps. A flat 25% risk per trade is extremely aggressive and does not account for continuous account growth or dynamic win-rate tracking.
- **Required Adjustment**: Move from aggressive fixed risk to dynamic sizing. Position sizing needs a volatility-adjusted model where `EXEC_RISK_PER_TRADE_PCT` shrinks as exposure logic tightens or if win-rate changes logic (closed-loop learning).

### D. Exit Strategy (`core/risk/exits.py`)
- **Current State**: Exit state manages Breakeven, Trailing Stops, and Phi3 visual/heuristic exit postures (`RUN`, `TIGHTEN`, `EXIT`, `STALE`).
- **Observation**: The live exit posture employs `rank_decay`, `move_stall`, and `spread_widening` as soft reasons to tighten limits or trigger an exit. 
- **Required Adjustment**: The time-stop logic (`min_hold` parameter before a stop is armed) and the ATR trailing logic need fine-tuning for 1m scalping context to prevent profitable trades from retracing into stop losses. 

## 3. Actionable Adjustment Roadmap

To achieve the 10/10 rating and hit the profitability goals ($1k milestone), the following adjustments should be prioritized:

1. **Refine Exit Strategy (Highest Priority)**
   - Transition to dynamic position sizing, moving away from the flat 25% risk limit.
   - Tighten ATR trailing stop logic for the 1-minute scalping timeframe to lock in profits quicker once a move stalls (`review_live_exit_state`).
2. **Nemotron Prompt Redesign**
   - Refactor `micro_prompts.py` to simplify nested if/then logic.
   - Guarantee valid JSON outputs by restricting the strategist prompt's permissible response keys and recalibrating temperature/top_p parameters.
3. **Lane Classification Stability**
   - Address the L2/L3 flip-flopping by widening the hysteresis in `lane_classifier.py` and verifying that `universe_manager` successfully populates shortlists even when volume ratios dip slightly below the hard 1.05/1.6 barriers.
4. **Closed-Loop Learning**
   - Currently, outcomes are reviewed (`nemotron_review_outcome`), but the feedback isn't strictly altering the global model parameters in real-time. Integrate the lesson strings back into the `BasicRiskEngine` or the pre-gate criteria dynamically.
