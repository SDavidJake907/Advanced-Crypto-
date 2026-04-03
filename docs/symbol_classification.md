# Symbol Classification

This document explains how the current tracked universe is intended to be understood at a human/operator level.

It is not the only source of truth for live behavior. Runtime classification still comes from code and runtime settings. This file exists so operators can quickly see what each tracked symbol is supposed to be.

Current live note:
- meme trading is disabled in runtime overrides
- non-meme names remain live candidates
- meme names may still appear in research lists, but they should not be active live entries until explicitly re-enabled

## Majors

These are core market anchors and generally the first names to compare against broader market structure:

| Symbol | Segment | Typical role | Expected default lane |
|---|---|---|---|
| `BTC/USD` | major | market anchor / primary risk barometer | `L1` or `L3` depending on setup |
| `ETH/USD` | major | core large-cap trend / structure name | `L1` or `L3` depending on setup |
| `SOL/USD` | major | high-beta major | `L2` or `L3` |
| `XRP/USD` | major | conditional/cleanliness-gated major | `L2` or `L3` |
| `ADA/USD` | major | liquid major alt | `L2` or `L3` |
| `LINK/USD` | major | liquid major alt / infrastructure leader | `L2` or `L3` |
| `AVAX/USD` | major | liquid major alt / beta continuation name | `L2` or `L3` |
| `LTC/USD` | major | older liquid major alt | `L3` |
| `DOT/USD` | major | liquid large-cap alt | `L2` or `L3` |
| `TRX/USD` | major | liquid large-cap alt | `L2` or `L3` |
| `ALGO/USD` | major | liquid legacy infrastructure major alt | `L2` or `L3` |

## Core Alts

These are non-meme tradable alts that can still be strong leaders, but are not treated as the core major sleeve:

| Symbol | Segment | Typical role | Expected default lane |
|---|---|---|---|
| `SUI/USD` | core alt | fast-moving momentum alt | `L2` or `L3` |
| `AAVE/USD` | core alt | DeFi leader / thinner than majors | `L3` |
| `ATOM/USD` | core alt | infrastructure / rotation alt | `L2` or `L3` |
| `INJ/USD` | core alt | higher-beta infrastructure alt | `L2` or `L3` |
| `NEAR/USD` | core alt | platform / AI-adjacent rotation alt | `L2` or `L3` |
| `ARB/USD` | core alt | L2 ecosystem alt | `L2` or `L3` |
| `SEI/USD` | core alt | higher-beta momentum alt | `L2` or `L3` |
| `ONDO/USD` | core alt | narrative / RWAs rotation alt | `L2` or `L3` |
| `RENDER/USD` | core alt | AI / compute narrative alt | `L3` in current live logs |
| `PENDLE/USD` | core alt | yield / DeFi rotation alt | `L2` or `L3` |
| `UNI/USD` | core alt | large-cap DeFi exchange token | `L3` |
| `FET/USD` | core alt | AI narrative alt | `L2` or `L3` |
| `TIA/USD` | core alt | infrastructure / data availability alt | `L3` |

## Meme / Research Disabled

These names are tracked in research and may still exist in `MEME_SYMBOLS`, but they should not be active live-trading candidates while meme trading is disabled.

| Symbol | Segment | Live status | Expected default lane if re-enabled |
|---|---|---|---|
| `DOGE/USD` | meme | disabled for live meme trading | `L4` |

## How To Read This

- `major` means liquid core market names that matter most for broader regime context
- `core alt` means non-meme tradable alts that can still be strong live candidates
- `meme` means high-beta narrative/speculation names that should use their own stricter research and automation policy

## Source Notes

This document reflects the current tracked universe from runtime config and recent live behavior:
- [`configs/runtime_overrides.json`](../configs/runtime_overrides.json)
- [`core/config/runtime.py`](../core/config/runtime.py)
- [`apps/universe_manager/main.py`](../apps/universe_manager/main.py)
