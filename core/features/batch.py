from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

import cuda_features
from core.features.cuda_atr import compute_atr_gpu
from core.features.cuda_bollinger import compute_bollinger_gpu
from core.features.cuda_correlation import compute_correlation_gpu
from core.features.cuda_rsi import compute_rsi_gpu
from core.features.multi_timeframe.trend_1h import compute_trend_1h_batch
from core.features.multi_timeframe.trend_7d import compute_regime_7d_batch
from core.features.multi_timeframe.trend_30d import compute_macro_30d_batch
from core.features.trend_state import compute_trend_state
from core.policy.pipeline import apply_policy_pipeline

try:
    from core.features.cuda_northstar import (
        compute_northstar_batch_features_gpu,
        compute_northstar_fingerprint_gpu,
    )
except Exception:  # pragma: no cover - build/runtime fallback
    def compute_northstar_batch_features_gpu(prices: np.ndarray) -> dict[str, np.ndarray]:
        n_assets = int(np.asarray(prices).shape[0]) if np.asarray(prices).ndim == 2 else 0
        return {
            "hurst": np.full(n_assets, 0.5, dtype=np.float64),
            "entropy": np.full(n_assets, 0.5, dtype=np.float64),
            "autocorr": np.zeros(n_assets, dtype=np.float64),
        }

    def compute_northstar_fingerprint_gpu(
        prices: np.ndarray,
        *,
        btc_idx: int,
        eth_idx: int,
    ) -> dict[str, Any]:
        return {
            "metrics": np.zeros(8, dtype=np.float64),
            "r_mkt": 0.0,
            "r_btc": 0.0,
            "r_eth": 0.0,
            "breadth": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "rv_mkt": 0.0,
            "corr_avg": 0.0,
        }


def _align_frames(ohlc_by_symbol: dict[str, pd.DataFrame]) -> tuple[list[str], list[pd.DataFrame]]:
    symbols = list(ohlc_by_symbol.keys())
    if not symbols:
        return [], []

    min_len = min(len(frame) for frame in ohlc_by_symbol.values())
    if min_len == 0:
        return symbols, [frame.iloc[0:0].copy() for frame in ohlc_by_symbol.values()]

    frames = [ohlc_by_symbol[symbol].tail(min_len).reset_index(drop=True) for symbol in symbols]
    return symbols, frames


def _compute_momentum(prices: np.ndarray, lookback: int) -> np.ndarray:
    n_assets, n_points = prices.shape
    if n_points < lookback + 1:
        return np.zeros(n_assets, dtype=np.float64)
    latest = prices[:, -1]
    previous = prices[:, -(lookback + 1)]
    valid = previous > 0.0
    momentum = np.zeros(n_assets, dtype=np.float64)
    momentum[valid] = (latest[valid] / previous[valid]) - 1.0
    return momentum


def _compute_rotation_score(
    *,
    momentum_5: np.ndarray,
    momentum_14: np.ndarray,
    momentum_30: np.ndarray,
    rsi: np.ndarray,
    trend_1h: np.ndarray,
    regime_7d: list[str],
    correlation: np.ndarray,
) -> np.ndarray:
    n_assets = momentum_5.shape[0]
    if n_assets == 0:
        return np.array([], dtype=np.float64)

    trend_bonus = np.where(trend_1h > 0, 0.15, np.where(trend_1h < 0, -0.15, 0.0))
    regime_bonus = np.asarray([0.1 if regime == "trending" else 0.0 for regime in regime_7d], dtype=np.float64)
    overextension_penalty = np.clip((rsi - 70.0) / 30.0, 0.0, 1.0) * 0.2

    if correlation.size:
        crowding_penalty = np.clip((np.mean(np.abs(correlation), axis=1) - 0.5) / 0.5, 0.0, 1.0) * 0.15
    else:
        crowding_penalty = np.zeros(n_assets, dtype=np.float64)

    return (
        (0.5 * momentum_5)
        + (0.3 * momentum_14)
        + (0.2 * momentum_30)
        + trend_bonus
        + regime_bonus
        - overextension_penalty
        - crowding_penalty
    )


def _compute_volume_ratio(frames: list[pd.DataFrame], lookback: int = 20) -> np.ndarray:
    ratios: list[float] = []
    for frame in frames:
        if "volume" not in frame.columns or frame.empty:
            ratios.append(1.0)
            continue
        series = frame["volume"].astype(float)
        window = series.tail(lookback + 1)
        baseline = float(window.iloc[:-1].mean()) if len(window) > 1 else float(window.mean())
        current = float(series.iloc[-1])
        if baseline <= 0.0:
            ratios.append(1.0)
        else:
            ratios.append(current / baseline)
    return np.asarray(ratios, dtype=np.float64)


def _compute_volume_surge(frames: list[pd.DataFrame], lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    surge_scores: list[float] = []
    surge_flags: list[bool] = []
    for frame in frames:
        if "volume" not in frame.columns or frame.empty:
            surge_scores.append(0.0)
            surge_flags.append(False)
            continue
        series = frame["volume"].astype(float)
        window = series.tail(lookback + 1)
        baseline = float(window.iloc[:-1].mean()) if len(window) > 1 else float(window.mean())
        current = float(series.iloc[-1])
        if baseline <= 0.0:
            surge_scores.append(0.0)
            surge_flags.append(False)
            continue
        surge = max((current / baseline) - 1.0, 0.0)
        surge_scores.append(surge)
        surge_flags.append(surge >= 0.5)
    return np.asarray(surge_scores, dtype=np.float64), np.asarray(surge_flags, dtype=bool)


def _compute_price_zscore(
    prices: np.ndarray,
    *,
    middle: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
) -> np.ndarray:
    denom = (upper - lower) / 4.0
    zscore = np.zeros(prices.shape[0], dtype=np.float64)
    valid = denom > 1e-12
    zscore[valid] = (prices[:, -1][valid] - middle[valid]) / denom[valid]
    return zscore


def _compute_short_tf_features(
    ohlc_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    lookback: int = 5,
    min_bars: int = 6,
) -> tuple[list[float], list[int], list[bool]]:
    """Compute momentum and trend for a short timeframe per-symbol. Returns (momentum, trend, ready) lists."""
    momentums: list[float] = []
    trends: list[int] = []
    readiness: list[bool] = []
    for symbol in symbols:
        frame = ohlc_by_symbol.get(symbol)
        required = max(lookback + 1, min_bars)
        if frame is None or frame.empty or len(frame) < required:
            momentums.append(0.0)
            trends.append(0)
            readiness.append(False)
            continue
        closes = frame["close"].to_numpy(dtype=np.float64)
        prev = closes[-(lookback + 1)]
        latest = closes[-1]
        mom = (latest / prev - 1.0) if prev > 0.0 else 0.0
        trend = 1 if latest > closes[-lookback] else (-1 if latest < closes[-lookback] else 0)
        momentums.append(float(mom))
        trends.append(int(trend))
        readiness.append(True)
    return momentums, trends, readiness


def compute_features_batch(
    ohlc_by_symbol: dict[str, pd.DataFrame],
    *,
    ohlc_5m_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_15m_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_1h_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_7d_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_30d_by_symbol: dict[str, pd.DataFrame] | None = None,
    lookback_mom: int = 14,
    lookback_vol: int = 20,
    lookback_rsi: int = 14,
    lookback_atr: int = 14,
    lookback_bollinger: int = 20,
    bollinger_num_std: float = 2.0,
    finbert_scores: dict[str, float] | None = None,
    xgb_model: Any = None,
) -> dict[str, Any]:
    symbols, frames = _align_frames(ohlc_by_symbol)
    n_assets = len(symbols)
    if n_assets == 0:
        return {
            "symbols": [],
            "momentum": np.array([], dtype=np.float64),
            "momentum_5": np.array([], dtype=np.float64),
            "momentum_14": np.array([], dtype=np.float64),
            "momentum_30": np.array([], dtype=np.float64),
            "rotation_score": np.array([], dtype=np.float64),
            "volatility": np.array([], dtype=np.float64),
            "volume": np.array([], dtype=np.float64),
            "volume_ratio": np.array([], dtype=np.float64),
            "volume_surge": np.array([], dtype=np.float64),
            "volume_surge_flag": np.array([], dtype=bool),
            "price_zscore": np.array([], dtype=np.float64),
            "history_points": np.array([], dtype=np.int64),
            "indicators_ready": np.array([], dtype=bool),
            "rsi": np.array([], dtype=np.float64),
            "atr": np.array([], dtype=np.float64),
            "hurst": np.array([], dtype=np.float64),
            "entropy": np.array([], dtype=np.float64),
            "autocorr": np.array([], dtype=np.float64),
            "bb_middle": np.array([], dtype=np.float64),
            "bb_upper": np.array([], dtype=np.float64),
            "bb_lower": np.array([], dtype=np.float64),
            "bb_bandwidth": np.array([], dtype=np.float64),
            "price": np.array([], dtype=np.float64),
            "bar_ts": [],
            "bar_idx": [],
            "bar_interval_seconds": [],
            "correlation": np.zeros((0, 0), dtype=np.float64),
            "market_fingerprint": np.zeros(8, dtype=np.float64),
            "market_regime": {
                "r_mkt": 0.0,
                "r_btc": 0.0,
                "r_eth": 0.0,
                "breadth": 0.0,
                "median": 0.0,
                "iqr": 0.0,
                "rv_mkt": 0.0,
                "corr_avg": 0.0,
            },
            "trend_1h": np.array([], dtype=np.int64),
            "regime_7d": [],
            "macro_30d": [],
            "ma_7": np.array([], dtype=np.float64),
            "ma_26": np.array([], dtype=np.float64),
            "macd": np.array([], dtype=np.float64),
            "macd_signal": np.array([], dtype=np.float64),
            "macd_hist": np.array([], dtype=np.float64),
            "trend_confirmed": np.array([], dtype=bool),
            "ranging_market": np.array([], dtype=bool),
            "momentum_5m": np.array([], dtype=np.float64),
            "trend_5m": np.array([], dtype=np.int64),
            "short_tf_ready_5m": np.array([], dtype=bool),
            "momentum_15m": np.array([], dtype=np.float64),
            "trend_15m": np.array([], dtype=np.int64),
            "short_tf_ready_15m": np.array([], dtype=bool),
            "finbert_score": np.array([], dtype=np.float64),
            "xgb_score": np.array([], dtype=np.float64),
        }

    n_points = len(frames[0])
    prices = np.vstack([frame["close"].to_numpy(dtype=np.float64) for frame in frames])
    highs = np.vstack([frame["high"].to_numpy(dtype=np.float64) for frame in frames])
    lows = np.vstack([frame["low"].to_numpy(dtype=np.float64) for frame in frames])
    volumes = np.asarray(
        [float(frame["volume"].iloc[-1]) if len(frame) and "volume" in frame.columns else 0.0 for frame in frames],
        dtype=np.float64,
    )

    bar_ts: list[str | None] = []
    bar_idx: list[int | None] = []
    bar_interval_seconds: list[int | None] = []
    for frame in frames:
        if n_points == 0 or "timestamp" not in frame.columns:
            bar_ts.append(None)
            bar_idx.append(None)
            bar_interval_seconds.append(None)
            continue
        last_ts = pd.Timestamp(frame["timestamp"].iloc[-1])
        bar_ts.append(last_ts.isoformat())
        if n_points > 1:
            prev_ts = pd.Timestamp(frame["timestamp"].iloc[-2])
            interval = int((last_ts - prev_ts).total_seconds())
            bar_interval_seconds.append(interval)
            bar_idx.append(int(last_ts.timestamp() // interval) if interval > 0 else None)
        else:
            bar_interval_seconds.append(None)
            bar_idx.append(None)

    required_points = max(
        lookback_mom + 1,
        lookback_vol + 1,
        lookback_rsi + 1,
        lookback_atr + 1,
        lookback_bollinger,
    )
    if n_points < required_points:
        zeros = np.zeros(n_assets, dtype=np.float64)
        return {
            "symbols": symbols,
            "momentum": zeros.copy(),
            "momentum_5": zeros.copy(),
            "momentum_14": zeros.copy(),
            "momentum_30": zeros.copy(),
            "rotation_score": zeros.copy(),
            "volatility": zeros.copy(),
            "volume": volumes.copy() if n_assets else zeros.copy(),
            "volume_ratio": np.ones(n_assets, dtype=np.float64),
            "volume_surge": zeros.copy(),
            "volume_surge_flag": np.zeros(n_assets, dtype=bool),
            "price_zscore": zeros.copy(),
            "history_points": np.full(n_assets, n_points, dtype=np.int64),
            "indicators_ready": np.zeros(n_assets, dtype=bool),
            "rsi": zeros.copy(),
            "atr": zeros.copy(),
            "hurst": np.full(n_assets, 0.5, dtype=np.float64),
            "entropy": np.full(n_assets, 0.5, dtype=np.float64),
            "autocorr": zeros.copy(),
            "bb_middle": zeros.copy(),
            "bb_upper": zeros.copy(),
            "bb_lower": zeros.copy(),
            "bb_bandwidth": zeros.copy(),
            "price": prices[:, -1] if n_points else zeros.copy(),
            "bar_ts": bar_ts,
            "bar_idx": bar_idx,
            "bar_interval_seconds": bar_interval_seconds,
            "correlation": np.eye(n_assets, dtype=np.float64),
            "market_fingerprint": np.zeros(8, dtype=np.float64),
            "market_regime": {
                "r_mkt": 0.0,
                "r_btc": 0.0,
                "r_eth": 0.0,
                "breadth": 0.0,
                "median": 0.0,
                "iqr": 0.0,
                "rv_mkt": 0.0,
                "corr_avg": 0.0,
            },
            "trend_1h": np.zeros(n_assets, dtype=np.int64),
            "regime_7d": ["unknown"] * n_assets,
            "macro_30d": ["sideways"] * n_assets,
            "ma_7": prices[:, -1].copy() if n_points else zeros.copy(),
            "ma_26": prices[:, -1].copy() if n_points else zeros.copy(),
            "macd": zeros.copy(),
            "macd_signal": zeros.copy(),
            "macd_hist": zeros.copy(),
            "trend_confirmed": np.zeros(n_assets, dtype=bool),
            "ranging_market": np.zeros(n_assets, dtype=bool),
            "momentum_5m": zeros.copy(),
            "trend_5m": np.zeros(n_assets, dtype=np.int64),
            "short_tf_ready_5m": np.zeros(n_assets, dtype=bool),
            "momentum_15m": zeros.copy(),
            "trend_15m": np.zeros(n_assets, dtype=np.int64),
            "short_tf_ready_15m": np.zeros(n_assets, dtype=bool),
            "finbert_score": zeros.copy(),
            "xgb_score": np.full(n_assets, 50.0, dtype=np.float64),
        }

    cfg = cuda_features.FeatureConfig()
    cfg.lookback_mom = lookback_mom
    cfg.lookback_vol = lookback_vol
    out = cuda_features.compute_features_gpu(prices.flatten().tolist(), n_assets, n_points, cfg)
    rsi = compute_rsi_gpu(prices, lookback=lookback_rsi)
    atr = compute_atr_gpu(highs, lows, prices, lookback=lookback_atr)
    northstar = compute_northstar_batch_features_gpu(prices)
    bollinger = compute_bollinger_gpu(prices, lookback=lookback_bollinger, num_std=bollinger_num_std)
    correlation = compute_correlation_gpu(prices)
    volume_ratio = _compute_volume_ratio(frames, lookback=lookback_vol)
    volume_surge, volume_surge_flag = _compute_volume_surge(frames, lookback=lookback_vol)
    price_zscore = _compute_price_zscore(
        prices,
        middle=np.asarray(bollinger["middle"], dtype=np.float64),
        upper=np.asarray(bollinger["upper"], dtype=np.float64),
        lower=np.asarray(bollinger["lower"], dtype=np.float64),
    )
    btc_idx = symbols.index("BTC/USD") if "BTC/USD" in symbols else 0
    eth_idx = symbols.index("ETH/USD") if "ETH/USD" in symbols else (1 if n_assets > 1 else 0)
    market_fingerprint = compute_northstar_fingerprint_gpu(prices, btc_idx=btc_idx, eth_idx=eth_idx)
    trend_1h = compute_trend_1h_batch(ohlc_1h_by_symbol or ohlc_by_symbol)["trend_1h"]
    regime_7d = compute_regime_7d_batch(ohlc_7d_by_symbol or ohlc_by_symbol)["regime_7d"]
    macro_30d = compute_macro_30d_batch(ohlc_30d_by_symbol or ohlc_by_symbol)["macro_30d"]
    trend_state = compute_trend_state(prices, np.asarray(bollinger["bandwidth"], dtype=np.float64))
    momentum_5 = _compute_momentum(prices, 5)
    momentum_14 = _compute_momentum(prices, 14)
    momentum_30 = _compute_momentum(prices, 30)
    rotation_score = _compute_rotation_score(
        momentum_5=momentum_5,
        momentum_14=momentum_14,
        momentum_30=momentum_30,
        rsi=np.asarray(rsi, dtype=np.float64),
        trend_1h=np.asarray(trend_1h, dtype=np.int64),
        regime_7d=regime_7d,
        correlation=np.asarray(correlation, dtype=np.float64),
    )

    short_tf_min_bars_5m = int(os.getenv("WARMUP_MIN_BARS_5M", "6"))
    short_tf_min_bars_15m = int(os.getenv("WARMUP_MIN_BARS_15M", "4"))
    # 5m/15m short timeframe features
    if ohlc_5m_by_symbol:
        momentum_5m_list, trend_5m_list, short_tf_ready_5m = _compute_short_tf_features(
            ohlc_5m_by_symbol,
            symbols,
            lookback=5,
            min_bars=short_tf_min_bars_5m,
        )
    else:
        momentum_5m_list = [0.0] * n_assets
        trend_5m_list = [0] * n_assets
        short_tf_ready_5m = [False] * n_assets

    if ohlc_15m_by_symbol:
        momentum_15m_list, trend_15m_list, short_tf_ready_15m = _compute_short_tf_features(
            ohlc_15m_by_symbol,
            symbols,
            lookback=5,
            min_bars=short_tf_min_bars_15m,
        )
    else:
        momentum_15m_list = [0.0] * n_assets
        trend_15m_list = [0] * n_assets
        short_tf_ready_15m = [False] * n_assets

    # Per-symbol finbert scores
    finbert_per_symbol = [
        float(finbert_scores.get(sym, 0.0)) if finbert_scores else 0.0
        for sym in symbols
    ]

    # Per-symbol XGBoost scores
    xgb_per_symbol: list[float] = []
    for i, sym in enumerate(symbols):
        if xgb_model is not None:
            try:
                # Build a minimal feature dict for xgb prediction
                xgb_feat: dict[str, Any] = {
                    "momentum_5": float(momentum_5[i]),
                    "momentum_14": float(momentum_14[i]),
                    "momentum_30": float(momentum_30[i]),
                    "rsi": float(np.asarray(rsi, dtype=np.float64)[i]),
                    "atr": float(np.asarray(atr, dtype=np.float64)[i]),
                    "volume_surge": float(volume_surge[i]),
                    "book_imbalance": 0.0,
                    "rotation_score": float(rotation_score[i]),
                    "hurst": float(northstar["hurst"][i]),
                    "entropy": float(northstar["entropy"][i]),
                    "trend_1h": int(np.asarray(trend_1h, dtype=np.int64)[i]),
                    "finbert_score": finbert_per_symbol[i],
                    "autocorr": float(northstar["autocorr"][i]),
                    "price_zscore": float(price_zscore[i]),
                    "volume_ratio": float(volume_ratio[i]),
                }
                xgb_per_symbol.append(float(xgb_model.predict(xgb_feat)))
            except Exception:
                xgb_per_symbol.append(50.0)
        else:
            xgb_per_symbol.append(50.0)

    return {
        "symbols": symbols,
        "momentum": np.asarray(out.momentum, dtype=np.float64),
        "momentum_5": momentum_5,
        "momentum_14": momentum_14,
        "momentum_30": momentum_30,
        "rotation_score": rotation_score,
        "volatility": np.asarray(out.volatility, dtype=np.float64),
        "volume": volumes,
        "volume_ratio": volume_ratio,
        "volume_surge": volume_surge,
        "volume_surge_flag": volume_surge_flag,
        "price_zscore": price_zscore,
        "history_points": np.full(n_assets, n_points, dtype=np.int64),
        "indicators_ready": np.ones(n_assets, dtype=bool),
        "rsi": np.asarray(rsi, dtype=np.float64),
        "atr": np.asarray(atr, dtype=np.float64),
        "hurst": np.asarray(northstar["hurst"], dtype=np.float64),
        "entropy": np.asarray(northstar["entropy"], dtype=np.float64),
        "autocorr": np.asarray(northstar["autocorr"], dtype=np.float64),
        "bb_middle": np.asarray(bollinger["middle"], dtype=np.float64),
        "bb_upper": np.asarray(bollinger["upper"], dtype=np.float64),
        "bb_lower": np.asarray(bollinger["lower"], dtype=np.float64),
        "bb_bandwidth": np.asarray(bollinger["bandwidth"], dtype=np.float64),
        "price": prices[:, -1],
        "bar_ts": bar_ts,
        "bar_idx": bar_idx,
        "bar_interval_seconds": bar_interval_seconds,
        "correlation": np.asarray(correlation, dtype=np.float64),
        "market_fingerprint": np.asarray(market_fingerprint["metrics"], dtype=np.float64),
        "market_regime": market_fingerprint,
        "trend_1h": np.asarray(trend_1h, dtype=np.int64),
        "regime_7d": regime_7d,
        "macro_30d": macro_30d,
        "ma_7": np.asarray(trend_state["ma_7"], dtype=np.float64),
        "ma_26": np.asarray(trend_state["ma_26"], dtype=np.float64),
        "macd": np.asarray(trend_state["macd"], dtype=np.float64),
        "macd_signal": np.asarray(trend_state["macd_signal"], dtype=np.float64),
        "macd_hist": np.asarray(trend_state["macd_hist"], dtype=np.float64),
        "trend_confirmed": np.asarray(trend_state["trend_confirmed"], dtype=bool),
        "ranging_market": np.asarray(trend_state["ranging_market"], dtype=bool),
        "momentum_5m": np.asarray(momentum_5m_list, dtype=np.float64),
        "trend_5m": np.asarray(trend_5m_list, dtype=np.int64),
        "short_tf_ready_5m": np.asarray(short_tf_ready_5m, dtype=bool),
        "momentum_15m": np.asarray(momentum_15m_list, dtype=np.float64),
        "trend_15m": np.asarray(trend_15m_list, dtype=np.int64),
        "short_tf_ready_15m": np.asarray(short_tf_ready_15m, dtype=bool),
        "finbert_score": np.asarray(finbert_per_symbol, dtype=np.float64),
        "xgb_score": np.asarray(xgb_per_symbol, dtype=np.float64),
    }


def slice_features_for_asset(features_batch: dict[str, Any], asset_idx: int) -> dict[str, Any]:
    symbol = features_batch["symbols"][asset_idx]
    features = {
        "symbol": symbol,
        "momentum": float(features_batch["momentum"][asset_idx]),
        "momentum_5": float(features_batch["momentum_5"][asset_idx]),
        "momentum_14": float(features_batch["momentum_14"][asset_idx]),
        "momentum_30": float(features_batch["momentum_30"][asset_idx]),
        "rotation_score": float(features_batch["rotation_score"][asset_idx]),
        "volatility": float(features_batch["volatility"][asset_idx]),
        "volume": float(features_batch["volume"][asset_idx]),
        "volume_ratio": float(features_batch["volume_ratio"][asset_idx]),
        "volume_surge": float(features_batch["volume_surge"][asset_idx]),
        "volume_surge_flag": bool(features_batch["volume_surge_flag"][asset_idx]),
        "price_zscore": float(features_batch["price_zscore"][asset_idx]),
        "history_points": int(features_batch["history_points"][asset_idx]),
        "indicators_ready": bool(features_batch["indicators_ready"][asset_idx]),
        "rsi": float(features_batch["rsi"][asset_idx]),
        "atr": float(features_batch["atr"][asset_idx]),
        "hurst": float(features_batch["hurst"][asset_idx]),
        "entropy": float(features_batch["entropy"][asset_idx]),
        "autocorr": float(features_batch["autocorr"][asset_idx]),
        "bb_middle": float(features_batch["bb_middle"][asset_idx]),
        "bb_upper": float(features_batch["bb_upper"][asset_idx]),
        "bb_lower": float(features_batch["bb_lower"][asset_idx]),
        "bb_bandwidth": float(features_batch["bb_bandwidth"][asset_idx]),
        "price": float(features_batch["price"][asset_idx]),
        "bar_ts": features_batch["bar_ts"][asset_idx],
        "bar_idx": features_batch["bar_idx"][asset_idx],
        "bar_interval_seconds": features_batch["bar_interval_seconds"][asset_idx],
        "correlation_row": features_batch["correlation"][asset_idx],
        "correlation_symbols": features_batch["symbols"],
        "market_fingerprint": features_batch["market_fingerprint"],
        "market_regime": features_batch["market_regime"],
        "trend_1h": int(features_batch["trend_1h"][asset_idx]),
        "regime_7d": features_batch["regime_7d"][asset_idx],
        "macro_30d": features_batch["macro_30d"][asset_idx],
        "ma_7": float(features_batch["ma_7"][asset_idx]),
        "ma_26": float(features_batch["ma_26"][asset_idx]),
        "macd": float(features_batch["macd"][asset_idx]),
        "macd_signal": float(features_batch["macd_signal"][asset_idx]),
        "macd_hist": float(features_batch["macd_hist"][asset_idx]),
        "trend_confirmed": bool(features_batch["trend_confirmed"][asset_idx]),
        "ranging_market": bool(features_batch["ranging_market"][asset_idx]),
        "momentum_5m": float(features_batch["momentum_5m"][asset_idx]),
        "trend_5m": int(features_batch["trend_5m"][asset_idx]),
        "short_tf_ready_5m": bool(features_batch["short_tf_ready_5m"][asset_idx]),
        "momentum_15m": float(features_batch["momentum_15m"][asset_idx]),
        "trend_15m": int(features_batch["trend_15m"][asset_idx]),
        "short_tf_ready_15m": bool(features_batch["short_tf_ready_15m"][asset_idx]),
        "finbert_score": float(features_batch["finbert_score"][asset_idx]),
        "xgb_score": float(features_batch["xgb_score"][asset_idx]),
    }
    return apply_policy_pipeline(symbol, features)
