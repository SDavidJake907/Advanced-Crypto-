# Open Position Flow

This doc maps the live open-position lifecycle in the order it actually runs.

Use this when you want to understand:
- where an open position is handled each loop
- where Phi is used
- where the real exit authority lives
- where state is persisted
- where portfolio replacement and exposure logic fit

## Overview

For an already-open trade, the runtime path is:

1. [`apps/trader/positions.py`](../apps/trader/positions.py)
2. [`core/risk/position_monitor.py`](../core/risk/position_monitor.py)
3. [`core/risk/exits.py`](../core/risk/exits.py)
4. [`core/state/position_state_store.py`](../core/state/position_state_store.py)
5. [`core/risk/portfolio.py`](../core/risk/portfolio.py)

That is the clean mental model.

Important role split:
- `positions.py` is the loop-facing coordinator for an existing position
- `position_monitor.py` updates live position state
- `exits.py` is the hard exit authority
- `position_state_store.py` persists local state
- `portfolio.py` is mainly entry/replacement/exposure logic, not the core hold-manager loop

## 1. `positions.py`

Primary file:
- [`apps/trader/positions.py`](../apps/trader/positions.py)

This is the open-position entry point used by the trader loop.

It does these things:
- checks whether the symbol already has an open tracked position
- computes live context such as:
  - current price
  - ATR
  - hold time
  - PnL
- builds the exit-posture payload for Phi
- calls `phi3_review_exit_posture(...)`
- calls `monitor_open_position(...)`
- calls `evaluate_exit(...)`
- records exit outcomes and lessons if the position is closed

Key point:
- this file coordinates the open-position lifecycle
- it does not own the final exit trigger by itself

## 2. `position_monitor.py`

Primary file:
- [`core/risk/position_monitor.py`](../core/risk/position_monitor.py)

This module updates the live position state machine.

It is responsible for:
- updating excursions:
  - MFE
  - MAE
  - ETD
- tracking structure/health state
- combining:
  - lane monitor state
  - Phi posture
  - deterministic live exit posture
- returning the updated position plus the current posture/state decision

Think of it as:
- the live state updater
- not the final execution authority

## 3. `exits.py`

Primary file:
- [`core/risk/exits.py`](../core/risk/exits.py)

This is the mechanical exit authority.

It owns:
- stop-loss logic
- trailing-stop logic
- breakeven arming
- stale/no-progress logic
- hard failure logic
- deterministic tighten/exit conditions

This is the module that answers:
- “is there a real exit reason now?”

Important rule:
- Phi can advise posture
- `exits.py` decides the actual exit trigger

## 4. `position_state_store.py`

Primary file:
- [`core/state/position_state_store.py`](../core/state/position_state_store.py)

This module persists local tracked position state.

It is responsible for:
- loading persisted tracked positions at startup
- saving updated state after monitor/entry/exit changes
- keeping the local runtime state durable across restarts

This is persistence, not strategy.

Important detail:
- this store reflects the bot’s tracked internal position model
- account sync still comes separately from exchange/ledger state

## 5. `portfolio.py`

Primary file:
- [`core/risk/portfolio.py`](../core/risk/portfolio.py)

This module is important, but it sits beside the open-position hold loop more than inside it.

It owns:
- max open positions
- per-symbol weight caps
- total gross exposure
- scale-down vs allow vs block
- replacement selection

This matters most when:
- a new candidate wants a slot
- a trade would exceed exposure caps
- the system decides whether to replace a weaker held position

So for open positions:
- `portfolio.py` is not the routine “watch this trade every bar” file
- it is the exposure and replacement authority around the book

## Where Phi Fits

Phi is used in the open-position path through:
- [`core/llm/micro_prompts.py`](../core/llm/micro_prompts.py)
- specifically `phi3_review_exit_posture(...)`

Phi receives a position payload built in [`apps/trader/positions.py`](../apps/trader/positions.py) using:
- current feature values
- hold time
- PnL
- entry thesis / invalidate-on fields

Phi does:
- `RUN`
- `TIGHTEN`
- `EXIT`
- `STALE`

Phi does not own:
- actual forced exit execution
- stop placement
- portfolio replacement

## Where Nemo Fits

Nemo is not part of the live exit path anymore.

Nemo remains for:
- entry/finalist judgment

Nemo does not own:
- live exit posture
- live exit execution

## Logs And State Files

Main places to inspect open positions:

- local tracked state:
  - [`logs/position_state.json`](../logs/position_state.json)
- synced account/book state:
  - [`logs/account_sync.json`](../logs/account_sync.json)
- decision trail:
  - [`logs/decision_debug.jsonl`](../logs/decision_debug.jsonl)

## Practical Summary

If you want the shortest version:

1. `positions.py` manages the open trade in the loop
2. `position_monitor.py` updates state and posture
3. `exits.py` decides whether to actually exit
4. `position_state_store.py` persists the result
5. `portfolio.py` governs replacement/exposure around the book

That is the intended live ownership model.
