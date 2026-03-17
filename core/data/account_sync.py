from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any

from core.data.kraken_rest import KrakenRestClient
from core.risk.portfolio import Position, PositionState
from core.state.portfolio import PortfolioState
from core.symbols import normalize_symbol, to_kraken_symbol


def _normalize_pair(pair: str) -> str:
    return normalize_symbol(pair)


def _extract_numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, dict):
        for key in ("balance", "total", "value"):
            if key in value:
                return _extract_numeric(value[key], default=default)
        return default
    try:
        return float(value)
    except Exception:
        return default


def _first_numeric(mapping: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        value = _extract_numeric(mapping.get(key), default=0.0)
        if value > 0.0:
            return value
    return 0.0


@dataclass
class AccountBootstrap:
    portfolio_state: PortfolioState
    positions_state: PositionState
    cash_usd: float
    initial_equity_usd: float
    ignored_dust: dict[str, float]
    synced_positions_usd: dict[str, float]
    synced_entry_meta: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_state": self.portfolio_state.to_dict(),
            "positions_state": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "weight": pos.weight,
                }
                for pos in self.positions_state.all()
            ],
            "cash_usd": self.cash_usd,
            "initial_equity_usd": self.initial_equity_usd,
            "ignored_dust": self.ignored_dust,
            "synced_positions_usd": self.synced_positions_usd,
            "synced_entry_meta": self.synced_entry_meta,
            "diagnostics": self.diagnostics,
        }


def _build_trade_pair_map(asset_pairs: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key, value in asset_pairs.items():
        wsname = str(value.get("wsname") or "").upper()
        altname = str(value.get("altname") or "").upper()
        base = str(value.get("base") or "").upper()
        quote = str(value.get("quote") or "").upper()
        normalized = ""
        if wsname:
            normalized = _normalize_pair(wsname)
            mapping[wsname] = normalized
        if altname and normalized:
            mapping[altname] = normalized
        if key and normalized:
            mapping[str(key).upper()] = normalized
        if base and quote and normalized:
            mapping[f"{base}{quote}"] = normalized
    return mapping


def _compute_entry_basis(
    *,
    client: KrakenRestClient,
    symbols: list[str],
    asset_pairs: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        history = client.get_trades_history().get("result", {})
    except Exception as exc:
        diagnostics["trades_history_error"] = str(exc)
        return {}

    trades = history.get("trades", {})
    if not isinstance(trades, dict):
        return {}

    pair_map = _build_trade_pair_map(asset_pairs)
    target_symbols = {str(symbol).strip().upper() for symbol in symbols}
    inventories: dict[str, dict[str, Any]] = {
        symbol: {"qty": 0.0, "cost": 0.0, "last_buy_ts": None}
        for symbol in target_symbols
    }

    ordered = sorted(
        (payload for payload in trades.values() if isinstance(payload, dict)),
        key=lambda item: float(item.get("time", 0.0) or 0.0),
    )

    for trade in ordered:
        raw_pair = str(trade.get("pair") or "").upper()
        symbol = pair_map.get(raw_pair, _normalize_pair(raw_pair))
        if symbol not in inventories:
            continue
        side = str(trade.get("type") or "").lower()
        qty = _extract_numeric(trade.get("vol"), default=0.0)
        price = _extract_numeric(trade.get("price"), default=0.0)
        ts_value = trade.get("time")
        if qty <= 0.0 or price <= 0.0 or side not in {"buy", "sell"}:
            continue

        inventory = inventories[symbol]
        if side == "buy":
            inventory["qty"] += qty
            inventory["cost"] += qty * price
            inventory["last_buy_ts"] = ts_value
            continue

        current_qty = float(inventory["qty"])
        if current_qty <= 0.0:
            continue
        avg_cost = float(inventory["cost"]) / current_qty if current_qty > 0.0 else 0.0
        reduction = min(qty, current_qty)
        remaining_qty = max(current_qty - reduction, 0.0)
        if remaining_qty <= 1e-12 or reduction >= current_qty:
            inventory["qty"] = 0.0
            inventory["cost"] = 0.0
        else:
            inventory["qty"] = remaining_qty
            inventory["cost"] = max(float(inventory["cost"]) - (avg_cost * reduction), 0.0)

    entry_meta: dict[str, dict[str, Any]] = {}
    for symbol, inventory in inventories.items():
        qty = float(inventory["qty"])
        cost = float(inventory["cost"])
        if qty <= 0.0 or cost <= 0.0:
            continue
        entry_meta[symbol] = {
            "entry_price": cost / qty,
            "entry_ts": datetime.fromtimestamp(float(inventory["last_buy_ts"]), tz=timezone.utc).isoformat()
            if inventory.get("last_buy_ts") is not None
            else None,
            "source": "kraken_trade_history",
        }
    diagnostics["trades_history_count"] = len(trades)
    return entry_meta


def bootstrap_account_state(
    *,
    client: KrakenRestClient,
    symbols: list[str],
    dust_usd_threshold: float | None = None,
) -> AccountBootstrap:
    dust_threshold = (
        float(dust_usd_threshold)
        if dust_usd_threshold is not None
        else float(os.getenv("ACCOUNT_DUST_USD_THRESHOLD", "2.0"))
    )
    diagnostics: dict[str, Any] = {"balance_source": None, "cash_keys_checked": ["ZUSD", "USD", "FUSD"]}
    asset_pairs = client.get_asset_pairs().get("result", {})
    meta_map: dict[str, dict[str, Any]] = {}
    for key, value in asset_pairs.items():
        wsname = str(value.get("wsname") or "").upper()
        altname = str(value.get("altname") or "").upper()
        if wsname:
            meta_map[wsname] = {"key": key, **value}
        if altname:
            meta_map[altname] = {"key": key, **value}

    normalized_pairs: list[tuple[str, str, dict[str, Any]]] = []
    unsupported_symbols: list[str] = []
    for original_symbol in symbols:
        normalized_symbol = _normalize_pair(original_symbol)
        meta = meta_map.get(to_kraken_symbol(normalized_symbol))
        if not meta:
            unsupported_symbols.append(original_symbol)
            continue
        normalized_pairs.append((original_symbol, normalized_symbol, meta))

    diagnostics["unsupported_symbols"] = unsupported_symbols
    ticker_pairs = list(dict.fromkeys(str(meta.get("key") or normalized_symbol) for _, normalized_symbol, meta in normalized_pairs))
    diagnostics["ticker_pairs"] = ticker_pairs
    ticker = client.get_ticker(ticker_pairs).get("result", {}) if ticker_pairs else {}
    try:
        balances = client.get_balance_ex().get("result", {})
        diagnostics["balance_source"] = "BalanceEx"
    except Exception as exc:
        diagnostics["balance_ex_error"] = str(exc)
        balances = client.get_balance().get("result", {})
        diagnostics["balance_source"] = "Balance"

    trade_balance: dict[str, Any] = {}
    trade_balance_errors: dict[str, str] = {}
    for asset in ("ZUSD", "USD"):
        try:
            trade_balance = client.get_trade_balance(asset).get("result", {})
            diagnostics["trade_balance_asset"] = asset
            break
        except Exception as exc:
            trade_balance_errors[asset] = str(exc)
    if trade_balance_errors:
        diagnostics["trade_balance_errors"] = trade_balance_errors

    cash_usd = _first_numeric(balances, ["ZUSD", "USD", "FUSD"])
    if cash_usd <= 0.0:
        cash_usd = _first_numeric(trade_balance, ["tb", "eb", "mf"])
    diagnostics["cash_usd_resolved"] = cash_usd

    positions: dict[str, float] = {}
    positions_state = PositionState()
    synced_positions_usd: dict[str, float] = {}
    ignored_dust: dict[str, float] = {}
    synced_entry_meta = _compute_entry_basis(
        client=client,
        symbols=[normalized_symbol for _, normalized_symbol, _ in normalized_pairs],
        asset_pairs=asset_pairs,
        diagnostics=diagnostics,
    )

    for original_symbol, normalized_symbol, meta in normalized_pairs:
        base_asset = str(meta.get("base") or "")
        qty = _extract_numeric(balances.get(base_asset), default=0.0)
        if qty <= 0.0:
            continue
        ticker_key = str(meta.get("altname") or meta.get("wsname") or meta.get("key") or normalized_symbol).upper()
        tick = ticker.get(ticker_key) or ticker.get(normalized_symbol) or ticker.get(meta.get("key"))
        if not isinstance(tick, dict):
            continue
        price = _extract_numeric((tick.get("c") or [0.0])[0], default=0.0)
        if price <= 0.0:
            continue
        notional_usd = qty * price
        if notional_usd < dust_threshold:
            ignored_dust[original_symbol] = notional_usd
            continue
        positions[original_symbol] = qty
        synced_positions_usd[original_symbol] = notional_usd

    position_marks: dict[str, float] = {}
    for symbol in positions:
        normalized_symbol = _normalize_pair(symbol)
        for _, candidate_symbol, meta in normalized_pairs:
            if candidate_symbol != normalized_symbol:
                continue
            ticker_key = str(meta.get("altname") or meta.get("wsname") or meta.get("key") or normalized_symbol).upper()
            tick = ticker.get(ticker_key) or ticker.get(normalized_symbol) or ticker.get(meta.get("key"))
            if not isinstance(tick, dict):
                break
            mark_price = _extract_numeric((tick.get("c") or [0.0])[0], default=0.0)
            if mark_price > 0.0:
                position_marks[symbol] = mark_price
            break

    spot_equity_usd = cash_usd + sum(synced_positions_usd.values()) + sum(ignored_dust.values())
    trade_balance_equity_usd = _extract_numeric(trade_balance.get("e"), default=0.0)
    diagnostics["trade_balance_equity_usd"] = trade_balance_equity_usd
    diagnostics["spot_equity_usd"] = spot_equity_usd

    initial_equity_usd = spot_equity_usd if spot_equity_usd > 0.0 else trade_balance_equity_usd
    if initial_equity_usd <= 0.0:
        initial_equity_usd = cash_usd
    diagnostics["initial_equity_usd_resolved"] = initial_equity_usd
    diagnostics["balances_keys_sample"] = sorted(str(key) for key in list(balances.keys())[:20])

    for symbol, notional_usd in synced_positions_usd.items():
        weight = notional_usd / initial_equity_usd if initial_equity_usd > 0.0 else 0.0
        entry_meta = synced_entry_meta.get(symbol, {})
        positions_state.add_or_update(
            Position(
                symbol=symbol,
                side="LONG",
                weight=weight,
                entry_price=_extract_numeric(entry_meta.get("entry_price"), default=0.0) or None,
                entry_bar_ts=str(entry_meta.get("entry_ts")) if entry_meta.get("entry_ts") is not None else None,
                entry_reasons=[f"source:{entry_meta.get('source', 'account_sync')}"] if entry_meta else ["source:account_sync"],
            )
        )

    portfolio_state = PortfolioState(
        cash=cash_usd,
        positions=positions,
        position_marks=position_marks,
        pnl=0.0,
        initial_equity=initial_equity_usd if initial_equity_usd > 0.0 else cash_usd,
    )
    return AccountBootstrap(
        portfolio_state=portfolio_state,
        positions_state=positions_state,
        cash_usd=cash_usd,
        initial_equity_usd=portfolio_state.initial_equity,
        ignored_dust=ignored_dust,
        synced_positions_usd=synced_positions_usd,
        synced_entry_meta=synced_entry_meta,
        diagnostics=diagnostics,
    )
