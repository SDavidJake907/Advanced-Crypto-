"""
Download historical OHLCV data from Binance public S3.
No API key required. Saves CSV files compatible with KrakenSK replay.

Usage:
    python scripts/download_history.py
    python scripts/download_history.py --symbols BTC/USD ETH/USD SOL/USD
    python scripts/download_history.py --no-1m
    python scripts/download_history.py --months 6

Output directory default: logs/history/
File naming: candles_{SYMBOL}_{TF}.csv (for example candles_BTCUSD_1h.csv)
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd


KRAKEN_TO_BINANCE: dict[str, str] = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "XRP/USD": "XRPUSDT",
    "ADA/USD": "ADAUSDT",
    "AVAX/USD": "AVAXUSDT",
    "DOGE/USD": "DOGEUSDT",
    "LINK/USD": "LINKUSDT",
    "DOT/USD": "DOTUSDT",
    "LTC/USD": "LTCUSDT",
    "BCH/USD": "BCHUSDT",
    "ATOM/USD": "ATOMUSDT",
    "UNI/USD": "UNIUSDT",
    "ETC/USD": "ETCUSDT",
    "FIL/USD": "FILUSDT",
    "ALGO/USD": "ALGOUSDT",
    "APT/USD": "APTUSDT",
    "ARB/USD": "ARBUSDT",
    "CELO/USD": "CELOUSDT",
    "CRV/USD": "CRVUSDT",
    "EGLD/USD": "EGLDUSDT",
    "GMT/USD": "GMTUSDT",
    "LDO/USD": "LDOUSDT",
    "LSK/USD": "LSKUSDT",
    "OP/USD": "OPUSDT",
    "PENDLE/USD": "PENDLEUSDT",
    "INJ/USD": "INJUSDT",
    "SAND/USD": "SANDUSDT",
    "SUI/USD": "SUIUSDT",
    "STORJ/USD": "STORJUSDT",
    "TIA/USD": "TIAUSDT",
    "SEI/USD": "SEIUSDT",
    "TRX/USD": "TRXUSDT",
    "XLM/USD": "XLMUSDT",
    "ZEC/USD": "ZECUSDT",
    "HBAR/USD": "HBARUSDT",
    "ICP/USD": "ICPUSDT",
    "ONDO/USD": "ONDOUSDT",
    "WIF/USD": "WIFUSDT",
    "FET/USD": "FETUSDT",
    "RENDER/USD": "RENDERUSDT",
    "ENA/USD": "ENAUSDT",
    "JUP/USD": "JUPUSDT",
    "TON/USD": "TONUSDT",
    "AAVE/USD": "AAVEUSDT",
    "WLD/USD": "WLDUSDT",
    "QNT/USD": "QNTUSDT",
    "MNT/USD": "MNTUSDT",
    "POL/USD": "POLUSDT",
    "KAS/USD": "KASUSDT",
    "TRUMP/USD": "TRUMPUSDT",
    "PENGU/USD": "PENGUUSDT",
    "PEPE/USD": "PEPEUSDT",
    "BONK/USD": "BONKUSDT",
    "SHIB/USD": "SHIBUSDT",
    "FARTCOIN/USD": "FARTCOINUSDT",
    "SPX/USD": "SPX6900USDT",
    "HYPE/USD": "HYPEUSDT",
    "ZK/USD": "ZKUSDT",
    "NEAR/USD": "NEARUSDT",
    "PYTH/USD": "PYTHUSDT",
    "HNT/USD": "HNTUSDT",
    "GALA/USD": "GALAUSDT",
    "DEGEN/USD": "DEGENUSDT",
    "VIRTUAL/USD": "VIRTUALUSDT",
    "MORPHO/USD": "MORPHOUSDT",
}

TF_TO_BINANCE: dict[str, str] = {
    "1m": "1m",
    "1h": "1h",
    "1d": "1d",
}

BINANCE_BASE = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_DIR = Path("logs/history")


def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _months_range(n_months: int) -> list[tuple[int, int]]:
    now = datetime.now(timezone.utc)
    months: list[tuple[int, int]] = []
    year, month = now.year, now.month - 1
    if month == 0:
        month = 12
        year -= 1
    for _ in range(n_months):
        months.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def _download_monthly(binance_sym: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
    filename = f"{binance_sym}-{interval}-{year:04d}-{month:02d}"
    url = f"{BINANCE_BASE}/{binance_sym}/{interval}/{filename}.zip"
    try:
        with urlopen(url, timeout=30) as resp:
            content = resp.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = f"{filename}.csv"
            if csv_name not in zf.namelist():
                csv_name = zf.namelist()[0]
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh, header=None)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        print(f"    ERROR fetching {url}: HTTP {exc.code}")
        return None
    except URLError as exc:
        print(f"    ERROR fetching {url}: {exc}")
        return None
    except Exception as exc:
        print(f"    ERROR fetching {url}: {exc}")
        return None

    df = df.iloc[:, :6].copy()
    df.columns = ["ts_raw", "open", "high", "low", "close", "volume"]

    ts_numeric = pd.to_numeric(df["ts_raw"], errors="coerce")
    if ts_numeric.isna().all():
        return None

    max_ts = float(ts_numeric.dropna().max())
    if max_ts >= 1e15:
        ts_unit = "us"
    elif max_ts >= 1e12:
        ts_unit = "ms"
    else:
        ts_unit = "s"

    df["timestamp"] = pd.to_datetime(ts_numeric, unit=ts_unit, utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df


def download_symbol(
    kraken_sym: str,
    *,
    timeframes: list[str],
    months: list[tuple[int, int]],
    output_dir: Path,
) -> None:
    binance_sym = KRAKEN_TO_BINANCE.get(kraken_sym)
    if not binance_sym:
        print(f"  [{kraken_sym}] No Binance mapping - skipping")
        return

    token = _symbol_token(kraken_sym)
    for tf in timeframes:
        binance_interval = TF_TO_BINANCE.get(tf, tf)
        frames: list[pd.DataFrame] = []
        missing = 0

        print(f"  [{kraken_sym}] {tf} ({binance_interval}) - {len(months)} months", end="", flush=True)
        for year, month in months:
            df = _download_monthly(binance_sym, binance_interval, year, month)
            if df is not None and not df.empty:
                frames.append(df)
                print(".", end="", flush=True)
            else:
                missing += 1
                print("x", end="", flush=True)
            time.sleep(0.05)
        print()

        if not frames:
            print(f"    No data found for {binance_sym} {binance_interval} - skipping")
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        out_path = output_dir / f"candles_{token}_{tf}.csv"
        combined.to_csv(out_path, index=False)
        print(f"    Saved {len(combined):,} bars -> {out_path} ({missing} months missing)")

        if tf == "1d":
            for ctx in ("7d", "30d"):
                ctx_path = output_dir / f"candles_{token}_{ctx}.csv"
                combined.to_csv(ctx_path, index=False)
            print("    Also wrote 7d + 30d context files from daily bars")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data for KrakenSK replay")
    parser.add_argument("--symbols", nargs="+", help="Kraken symbols (for example BTC/USD ETH/USD).")
    parser.add_argument("--months", type=int, default=12, help="Number of months to download.")
    parser.add_argument("--no-1m", action="store_true", help="Skip 1m data.")
    parser.add_argument("--only-1d", action="store_true", help="Download daily bars only.")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = args.symbols
    else:
        env_syms = os.getenv("TRADER_SYMBOLS", "")
        if env_syms:
            symbols = [s.strip() for s in env_syms.split(",") if s.strip()]
        else:
            symbols = list(KRAKEN_TO_BINANCE.keys())

    if args.only_1d:
        timeframes = ["1d"]
    elif args.no_1m:
        timeframes = ["1h", "1d"]
    else:
        timeframes = ["1m", "1h", "1d"]

    months = _months_range(args.months)
    print(f"Downloading {len(symbols)} symbols x {timeframes} x {len(months)} months")
    print(f"Date range: {months[0][0]}-{months[0][1]:02d} -> {months[-1][0]}-{months[-1][1]:02d}")
    print(f"Output: {output_dir.resolve()}")
    print()

    for idx, sym in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {sym}")
        download_symbol(sym, timeframes=timeframes, months=months, output_dir=output_dir)

    print()
    print("Done. To use in replay, set these env vars:")
    print(f"  CANDLES_PATH_TEMPLATE={output_dir}/candles_{{symbol}}_1m.csv")
    print(f"  CANDLES_PATH_TEMPLATE_1H={output_dir}/candles_{{symbol}}_1h.csv")
    print(f"  CANDLES_PATH_TEMPLATE_7D={output_dir}/candles_{{symbol}}_7d.csv")
    print(f"  CANDLES_PATH_TEMPLATE_30D={output_dir}/candles_{{symbol}}_30d.csv")


if __name__ == "__main__":
    main()
