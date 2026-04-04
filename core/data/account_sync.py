from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any

from core.config.runtime import get_runtime_setting, get_symbol_lane
from core.data.kraken_rest import KrakenRestClient
from core.memory.kraken_history import load_trades_from_zip
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


def _normalize_entry_ts(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _bootstrap_exit_profile(
    *,
    symbol: str,
    side: str,
    entry_price: float | None,
    lane: str | None = None,
) -> dict[str, float | None]:
    resolved_lane = str(lane or get_symbol_lane(symbol) or "L3").upper()
    price = float(entry_price or 0.0)
    if price <= 0.0:
        return {"lane": resolved_lane, "risk_r": None, "stop_loss": None, "take_profit": None}

    if resolved_lane == "L1":
        min_stop_pct = float(get_runtime_setting("L1_EXIT_MIN_STOP_PCT")) / 100.0
    elif resolved_lane == "L2":
        min_stop_pct = float(get_runtime_setting("L2_EXIT_MIN_STOP_PCT")) / 100.0
    elif resolved_lane == "L4":
        min_stop_pct = float(get_runtime_setting("MEME_EXIT_MIN_STOP_PCT")) / 100.0
    else:
        min_stop_pct = float(get_runtime_setting("EXIT_MIN_STOP_PCT")) / 100.0

    risk_r = max(price * min_stop_pct, 0.0)
    if risk_r <= 0.0:
        return {"lane": resolved_lane, "risk_r": None, "stop_loss": None, "take_profit": None}

    if str(side).upper() == "LONG":
        stop_loss = price - risk_r
    else:
        stop_loss = price + risk_r
    return {
        "lane": resolved_lane,
        "risk_r": risk_r,
        "stop_loss": stop_loss,
        "take_profit": None,
    }


@dataclass
class AccountBootstrap:
    portfolio_state: PortfolioState
    positions_state: PositionState
    cash_usd: float
    initial_equity_usd: float
    ignored_dust: dict[str, float]
    ignored_dust_qty: dict[str, float]
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
            "ignored_dust_qty": self.ignored_dust_qty,
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


def _build_base_asset_symbol_map(
    asset_pairs: dict[str, Any],
    target_symbols: set[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in asset_pairs.values():
        wsname = str(value.get("wsname") or "").upper()
        if not wsname:
            continue
        normalized = _normalize_pair(wsname)
        if normalized not in target_symbols:
            continue
        base = str(value.get("base") or "").upper()
        if not base:
            continue
        mapping[base] = normalized
        stripped = base.lstrip("XZ")
        if stripped:
            mapping[stripped] = normalized
    return mapping


def _apply_inventory_update(
    inventories: dict[str, dict[str, Any]],
    *,
    symbol: str,
    side: str,
    qty: float,
    effective_cost: float,
    ts_value: Any,
) -> None:
    if symbol not in inventories or qty <= 0.0 or effective_cost <= 0.0:
        return
    inventory = inventories[symbol]
    if side == "buy":
        inventory["qty"] += qty
        inventory["cost"] += effective_cost
        inventory["last_buy_ts"] = ts_value
        return
    current_qty = float(inventory["qty"])
    if current_qty <= 0.0:
        return
    avg_cost = float(inventory["cost"]) / current_qty if current_qty > 0.0 else 0.0
    reduction = min(qty, current_qty)
    remaining_qty = max(current_qty - reduction, 0.0)
    if remaining_qty <= 1e-12 or reduction >= current_qty:
        inventory["qty"] = 0.0
        inventory["cost"] = 0.0
    else:
        inventory["qty"] = remaining_qty
        inventory["cost"] = max(float(inventory["cost"]) - (avg_cost * reduction), 0.0)


def _entry_meta_from_inventories(
    inventories: dict[str, dict[str, Any]],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    entry_meta: dict[str, dict[str, Any]] = {}
    for symbol, inventory in inventories.items():
        qty = float(inventory["qty"])
        cost = float(inventory["cost"])
        if qty <= 0.0 or cost <= 0.0:
            continue
        last_buy_ts = inventory.get("last_buy_ts")
        if isinstance(last_buy_ts, (int, float)):
            entry_ts = _normalize_entry_ts(datetime.fromtimestamp(float(last_buy_ts), tz=timezone.utc))
        else:
            entry_ts = _normalize_entry_ts(last_buy_ts)
        entry_meta[symbol] = {
            "entry_price": cost / qty,
            "entry_ts": entry_ts,
            "source": source,
        }
    return entry_meta


def _compute_entry_basis_from_ledgers(
    *,
    client: KrakenRestClient,
    symbols: set[str],
    asset_pairs: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        history = client.get_ledgers(type_filter="trade").get("result", {})
    except Exception as exc:
        diagnostics["ledgers_error"] = str(exc)
        return {}

    ledger = history.get("ledger", {})
    if not isinstance(ledger, dict):
        diagnostics["ledgers_error"] = "invalid ledger payload"
        return {}

    base_asset_map = _build_base_asset_symbol_map(asset_pairs, symbols)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in ledger.values():
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").lower() != "trade":
            continue
        refid = str(item.get("refid") or "").strip()
        if not refid:
            continue
        grouped.setdefault(refid, []).append(item)

    inventories: dict[str, dict[str, Any]] = {
        symbol: {"qty": 0.0, "cost": 0.0, "last_buy_ts": None}
        for symbol in symbols
    }
    applied_groups = 0
    for rows in sorted(
        grouped.values(),
        key=lambda items: min(float(item.get("time", 0.0) or 0.0) for item in items),
    ):
        symbol = ""
        qty = 0.0
        quote_amount = 0.0
        quote_fee = 0.0
        ts_value = None
        for item in rows:
            asset = str(item.get("asset") or "").upper()
            amount = _extract_numeric(item.get("amount"), default=0.0)
            fee = _extract_numeric(item.get("fee"), default=0.0)
            ts_value = item.get("time", ts_value)
            mapped_symbol = base_asset_map.get(asset)
            if mapped_symbol and mapped_symbol in inventories:
                symbol = mapped_symbol
                qty = abs(amount)
                continue
            if asset in {"ZUSD", "USD"}:
                quote_amount += amount
                quote_fee += fee
        if not symbol or qty <= 0.0 or quote_amount == 0.0:
            continue
        side = "buy" if quote_amount < 0.0 else "sell"
        effective_cost = abs(quote_amount) + max(quote_fee, 0.0)
        _apply_inventory_update(
            inventories,
            symbol=symbol,
            side=side,
            qty=qty,
            effective_cost=effective_cost,
            ts_value=ts_value,
        )
        applied_groups += 1

    diagnostics["ledgers_count"] = len(ledger)
    diagnostics["ledgers_trade_groups"] = len(grouped)
    diagnostics["ledgers_applied_groups"] = applied_groups
    if applied_groups <= 0:
        return {}
    diagnostics["trades_history_source"] = "ledgers_fallback"
    return _entry_meta_from_inventories(inventories, source="kraken_ledgers")


def _compute_entry_basis(
    *,
    client: KrakenRestClient,
    symbols: list[str],
    asset_pairs: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    target_symbols = {str(symbol).strip().upper() for symbol in symbols}
    if not target_symbols:
        diagnostics["trades_history_source"] = "not_needed"
        diagnostics["trades_history_count"] = 0
        return {}
    if not bool(get_runtime_setting("ACCOUNT_SYNC_USE_TRADES_HISTORY")):
        diagnostics["trades_history_source"] = "disabled_ledger_only"
        diagnostics["trades_history_count"] = 0
        ledgers_fallback = _compute_entry_basis_from_ledgers(
            client=client,
            symbols=target_symbols,
            asset_pairs=asset_pairs,
            diagnostics=diagnostics,
        )
        if ledgers_fallback:
            return ledgers_fallback
        fallback = _compute_entry_basis_from_zip(
            symbols=target_symbols,
            diagnostics=diagnostics,
        )
        if fallback:
            diagnostics["trades_history_source"] = "zip_fallback"
        return fallback
    try:
        history = client.get_trades_history().get("result", {})
    except Exception as exc:
        diagnostics["trades_history_error"] = str(exc)
        ledgers_fallback = _compute_entry_basis_from_ledgers(
            client=client,
            symbols=target_symbols,
            asset_pairs=asset_pairs,
            diagnostics=diagnostics,
        )
        if ledgers_fallback:
            return ledgers_fallback
        fallback = _compute_entry_basis_from_zip(
            symbols=target_symbols,
            diagnostics=diagnostics,
        )
        if fallback:
            diagnostics["trades_history_source"] = "zip_fallback"
        return fallback

    trades = history.get("trades", {})
    if not isinstance(trades, dict):
        return _compute_entry_basis_from_zip(symbols=target_symbols, diagnostics=diagnostics)

    pair_map = _build_trade_pair_map(asset_pairs)
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

        _apply_inventory_update(
            inventories,
            symbol=symbol,
            side=side,
            qty=qty,
            effective_cost=qty * price,
            ts_value=ts_value,
        )

    entry_meta = _entry_meta_from_inventories(inventories, source="kraken_trade_history")
    diagnostics["trades_history_count"] = len(trades)
    diagnostics["trades_history_source"] = "api"
    return entry_meta


def _compute_entry_basis_from_zip(
    *,
    symbols: set[str],
    diagnostics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    zip_path = str(os.getenv("KRAKEN_TRADE_HISTORY_ZIP", "") or "").strip()
    if not zip_path:
        diagnostics["trades_history_zip_error"] = "KRAKEN_TRADE_HISTORY_ZIP not set"
        return {}

    try:
        trades = load_trades_from_zip(zip_path)
    except Exception as exc:
        diagnostics["trades_history_zip_error"] = str(exc)
        return {}

    diagnostics["trades_history_zip_path"] = zip_path
    diagnostics["trades_history_zip_count"] = len(trades)
    inventories: dict[str, dict[str, Any]] = {
        symbol: {"qty": 0.0, "cost": 0.0, "last_buy_ts": None}
        for symbol in symbols
    }

    for trade in trades:
        symbol = str(trade.symbol or "").strip().upper()
        if symbol not in inventories:
            continue
        side = str(trade.side or "").lower()
        qty = float(trade.volume or 0.0)
        price = float(trade.price or 0.0)
        if qty <= 0.0 or price <= 0.0 or side not in {"buy", "sell"}:
            continue
        _apply_inventory_update(
            inventories,
            symbol=symbol,
            side=side,
            qty=qty,
            effective_cost=qty * price,
            ts_value=trade.ts or inventories[symbol].get("last_buy_ts"),
        )

    return _entry_meta_from_inventories(inventories, source="kraken_trade_history_zip")


def bootstrap_account_state(
    *,
    client: KrakenRestClient,
    symbols: list[str],
    dust_usd_threshold: float | None = None,
) -> AccountBootstrap:
    dust_threshold = (
        float(dust_usd_threshold)
        if dust_usd_threshold is not None
        else float(os.getenv("ACCOUNT_DUST_USD_THRESHOLD", "5.0"))
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
    ignored_dust_qty: dict[str, float] = {}

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
            ignored_dust_qty[original_symbol] = qty
            continue
        positions[original_symbol] = qty
        synced_positions_usd[original_symbol] = notional_usd

    synced_entry_meta = _compute_entry_basis(
        client=client,
        symbols=list(synced_positions_usd.keys()),
        asset_pairs=asset_pairs,
        diagnostics=diagnostics,
    )

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

    # Exclude ignored_dust from equity — dust is untradeable and inflates position sizing if included
    spot_equity_usd = cash_usd + sum(synced_positions_usd.values())
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
        entry_price = _extract_numeric(entry_meta.get("entry_price"), default=0.0) or None
        exit_profile = _bootstrap_exit_profile(
            symbol=symbol,
            side="LONG",
            entry_price=entry_price,
        )
        positions_state.add_or_update(
            Position(
                symbol=symbol,
                side="LONG",
                weight=weight,
                lane=str(exit_profile.get("lane") or get_symbol_lane(symbol) or "L3"),
                entry_price=entry_price,
                entry_bar_ts=str(entry_meta.get("entry_ts")) if entry_meta.get("entry_ts") is not None else None,
                stop_loss=exit_profile.get("stop_loss"),
                take_profit=exit_profile.get("take_profit"),
                risk_r=exit_profile.get("risk_r"),
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
        ignored_dust_qty=ignored_dust_qty,
        synced_positions_usd=synced_positions_usd,
        synced_entry_meta=synced_entry_meta,
        diagnostics=diagnostics,
    )
