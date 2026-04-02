# KrakenSK - Comprehensive Code Audit Findings

## 1. Bugs Identified

### `core/policy/entry_verifier.py` (Critical Crash Bug)
- **NameError (Runtime Crash):** On line 262, the variable `trend_confirmed` is referenced but never defined:
  `mover_signal = rotation_score > 0.0 or momentum_5 > 0.0 or volume_surge >= 0.2 or trend_confirmed`
  This will crash the trading loop whenever execution reaches this code path and `rotation_score`, `momentum_5`, and `volume_surge` are all below their thresholds, forcing Python to evaluate `trend_confirmed`.

## 2. Memory Leaks Identified

### `apps/trader/main.py` (Unbounded Memory Growth)
- **Zombie State in Tracking Dictionaries:** The `last_prices`, `last_bar_keys`, and `last_exit_ts` dictionaries accumulate keys for every traded symbol. When `current_symbols` updates (e.g. universe rotation drops a symbol), old symbols are never removed from these dictionaries. Over a long runtime with many unique symbols rotating in and out, this acts as a slow memory leak.

## 3. Trading & Lane Issues

### `apps/trader/main.py` (Lane Override Inconsistency)
- **Dynamically Calculated Lanes are Ignored:** The live features pipeline computes the current lane (L1, L2, L3) via `lane_classifier.py`. However, in `main.py` lines 828-829, `_apply_lane_supervision` completely overwrites `features["lane"]` with the static `universe_lane`. This means real-time structural shifts (e.g., a coin dynamically heating up to L1 intrabar) are ignored because the trading phase stubbornly reverts to whatever the universe manager assigned it at rest.

### `core/execution/cpp_exec.py` (Silent Fallback)
- **Fallback Obscurity:** If `krakencpp` fails to load, `self.engine` is initialized as `None`. The execute function silently delegates to `MockExecutor` (`self._fallback.execute`). If this happens in production, the system will silently paper trade instead of executing live trades, with no clear runtime error emitted.

## 4. Synchronization (Async) Issues

### `apps/trader/main.py` & `core/llm/nemotron.py` (Event Loop Starvation)
- **Blocking the Async Loop with Synchronous IO:** Inside the `trader_loop()` (an `async` function), the code iterates through `active_eval_symbols` one by one. For each symbol, if `DECISION_ENGINE == "llm"`, it calls `nemotron.decide(...)`.
- `nemotron.decide(...)` eventually calls `run_nemotron_tool_loop`, which is purely synchronous and makes blocking HTTP requests to Ollama/OpenAI. 
- *Impact:* While the LLM inference is running (which could take hundreds of milliseconds to seconds per symbol), the entire Python `asyncio` event loop is frozen. This means Websockets (`LiveMarketDataFeed`), collectors, watchdog timers, and background tasks are blocked. The trader will miss market updates and latency will spike dramatically.
