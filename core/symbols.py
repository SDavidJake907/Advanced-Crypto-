from __future__ import annotations

KRAKEN_TO_CANONICAL = {
    "XBT/USD": "BTC/USD",
    "XDG/USD": "DOGE/USD",
}

CANONICAL_TO_KRAKEN = {value: key for key, value in KRAKEN_TO_CANONICAL.items()}


def normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    return KRAKEN_TO_CANONICAL.get(value, value)


def to_kraken_symbol(symbol: str) -> str:
    value = normalize_symbol(symbol)
    return CANONICAL_TO_KRAKEN.get(value, value)
