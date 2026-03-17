# KrakenSK

KrakenSK is a live Kraken spot trading stack with:
- `Phi-3` on NPU for fast reflex and market-state review
- `Nemotron` for candidate advisory and final strategist decisions
- deterministic policy, portfolio, and execution guards
- a universe manager, review scheduler, trader, and optional operator UI / visual services

This repo is no longer just a scaffold. The current runtime is driven by `scripts/start_all.ps1`.

## Current Runtime

The normal stack started by [`scripts/start_all.ps1`](C:\Users\kitti\Desktop\KrakenSK\scripts\start_all.ps1) is:
- `phi3_npu`
- `ollama` when `NEMOTRON_PROVIDER=local`
- `universe_manager`
- `review_scheduler`
- `trader`
- optional `visual_phil`
- optional `visual_feed`
- optional `operator_ui`

Startup now prints the active:
- decision engine
- policy profile
- Nemotron provider / model / backend mode
- key policy thresholds

## Main Components

- [`apps/trader`](C:\Users\kitti\Desktop\KrakenSK\apps\trader): live trading loop, account sync, exit management, execution
- [`apps/universe_manager`](C:\Users\kitti\Desktop\KrakenSK\apps\universe_manager): active pair selection and lane supervision metadata
- [`apps/review_scheduler`](C:\Users\kitti\Desktop\KrakenSK\apps\review_scheduler): review / outcome scheduling
- [`core/llm`](C:\Users\kitti\Desktop\KrakenSK\core\llm): Phi-3 + Nemotron orchestration
- [`core/policy`](C:\Users\kitti\Desktop\KrakenSK\core\policy): canonical policy pipeline, verdicts, Nemotron gating, trade-plan metadata
- [`core/risk`](C:\Users\kitti\Desktop\KrakenSK\core\risk): exit plans, portfolio gating, fee filter
- [`core/execution`](C:\Users\kitti\Desktop\KrakenSK\core\execution): Kraken order sizing, order policy, live executor
- [`core/state`](C:\Users\kitti\Desktop\KrakenSK\core\state): portfolio state, open orders, persisted position state

## Decision Flow

Hot path for a tradable symbol:
1. feature batch compute
2. policy pipeline
3. `Phi-3` reflex / market-state review
4. optional Nemotron advisory
5. Nemotron strategist
6. deterministic risk / portfolio checks
7. execution

Important current behavior:
- local Nemotron supports completion-fallback mode
- advisory payloads are compacted
- Nemotron tool use is reduced to the main strategy tool path
- unchanged non-executed Nemotron decisions are short-term cached
- position exit plans persist across restart

## Local Start / Stop

From the repo root:

```powershell
.\scripts\stop_all.ps1
.\scripts\start_all.ps1
```

If PowerShell blocks scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_all.ps1
```

## Key Environment

Common runtime controls in `.env`:
- `TRADER_DECISION_ENGINE=llm`
- `POLICY_PROFILE=custom`
- `NEMOTRON_PROVIDER=local` or `nvidia`
- `NEMOTRON_BASE_URL=http://127.0.0.1:11434`
- `NEMOTRON_MODEL=mirage335/NVIDIA-Nemotron-Nano-9B-v2-virtuoso:latest`
- `VISUAL_PHI3_DEVICE=NPU`
- `START_VISUAL_PHIL_ON_START=true`
- `START_VISUAL_PHIL_FEED_ON_START=true`
- `START_OPERATOR_UI_ON_START=true|false`
- `ACTIVE_MAX=85`
- `ROTATION_SHORTLIST_SIZE=85`
- `MEME_SYMBOLS=DOGE/USD,SHIB/USD,PEPE/USD,WIF/USD,BONK/USD,FARTCOIN/USD,TRUMP/USD,PENGU/USD,PUMP/USD,USELESS/USD,SPX/USD`

Selected runtime settings come from [`core/config/runtime.py`](C:\Users\kitti\Desktop\KrakenSK\core\config\runtime.py) and can be overridden via env or `configs/runtime_overrides.json`.

Notable current controls:
- `EXEC_RISK_PER_TRADE_PCT`
- `MEME_EXEC_RISK_PER_TRADE_PCT`
- `PORTFOLIO_MAX_OPEN_POSITIONS`
- `PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE`
- `PORTFOLIO_AVG_CORR_SCALE_THRESHOLD`
- `PORTFOLIO_AVG_CORR_SCALE_DOWN`
- `PORTFOLIO_MAX_POSITIONS_PER_SECTOR`
- `NEMOTRON_VERDICT_CACHE_TTL_SEC`
- `ORDER_PREFERENCE`
- `ORDER_POST_ONLY`

## Account Sync Notes

Live account bootstrap is handled in [`core/data/account_sync.py`](C:\Users\kitti\Desktop\KrakenSK\core\data\account_sync.py).

Current behavior:
- spot equity is resolved from cash plus synced holdings
- synced holdings seed live position marks
- persisted position state preserves exit-plan context across restart

Kraken API keys should have:
- `Query Funds`
- `Query closed orders & trades`
- trading permission if live execution is enabled

If `TradesHistory` fails, synced holdings may still be missing true entry prices, but persisted position state now prevents restart from wiping stop / TP / trailing context.

## Execution Notes

- entry sizing is clamped through [`core/execution/clamp.py`](C:\Users\kitti\Desktop\KrakenSK\core\execution\clamp.py)
- per-trade risk guard is active
- zero-quantity rounded orders are rejected
- entry order policy now biases toward limit orders when a valid quote exists
- live Kraken entries can send post-only maker flags

## Logs

Common logs:
- `logs/decision_traces.jsonl`
- `logs/decision_debug.jsonl`
- `logs/nemotron_debug.jsonl`
- `logs/account_sync.json`
- `logs/open_orders.json`
- `logs/position_state.json`

## Universe

The active trading universe is 85 Kraken USD spot pairs, all verified with >$300K 24h volume.

Key sectors:
- **AI/DePIN**: TAO, FET, RENDER, VIRTUAL, ONDO
- **Meme**: DOGE, SHIB, PEPE, WIF, BONK, FARTCOIN, TRUMP, PENGU, PUMP, USELESS, SPX
- **Solana ecosystem**: SOL, JUP, BONK, WIF
- **L1/L2**: HYPE, SUI, APT, SEI, TON, MNT, POL, ARB
- **DeFi blue chips**: AAVE, UNI, CRV, PENDLE, MORPHO, ENA, AERO
- **Core majors**: BTC, ETH, XRP, SOL, DOGE, ADA, AVAX, LINK, DOT, LTC

Universe is managed by `apps/universe_manager` and written to `universe.json`.

## Status

The repo currently reflects:
- Phase 1 backend stability work
- Phase 2 policy de-bloat and reduced over-safe gating
- Phase 3 structural cleanup with canonical policy verdicts and better runtime clarity
- Phase 4 universe expansion to 85 verified Kraken pairs with sector-aware scan prompts

`Atlas` may still be unavailable depending on the local environment. That is separate from the core trader stack.
