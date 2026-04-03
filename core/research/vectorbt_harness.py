from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from core.data.loader import CandleLoader
from core.features.divergence import detect_rsi_divergence
from core.llm.micro_prompts import deterministic_market_state_review
from core.policy.pipeline import apply_policy_pipeline
from core.risk.fee_filter import evaluate_trade_cost


def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _resolve_history_path(symbol: str, timeframe: str, history_dir: str | Path = "logs/history") -> Path:
    return Path(history_dir) / f"candles_{_symbol_token(symbol)}_{timeframe}.csv"


def load_history_frame(symbol: str, timeframe: str = "1h", history_dir: str | Path = "logs/history") -> pd.DataFrame:
    path = _resolve_history_path(symbol, timeframe, history_dir)
    return CandleLoader(str(path)).load().reset_index(drop=True)


def _simple_momentum(closes: pd.Series, lookback: int) -> pd.Series:
    return closes / closes.shift(lookback) - 1.0


def _simple_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _price_zscore(closes: pd.Series, lookback: int = 20) -> pd.Series:
    mean = closes.rolling(lookback, min_periods=lookback).mean()
    std = closes.rolling(lookback, min_periods=lookback).std(ddof=0).replace(0.0, np.nan)
    return ((closes - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _volume_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    baseline = volume.rolling(lookback, min_periods=lookback).mean().replace(0.0, np.nan)
    return (volume / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _atr_pct(frame: pd.DataFrame, lookback: int = 14) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean()
    return (atr / close.replace(0.0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_policy_feature_frame(symbol: str, frame_1h: pd.DataFrame) -> pd.DataFrame:
    frame = frame_1h.copy()
    frame["momentum_5"] = _simple_momentum(frame["close"], 5)
    frame["momentum_14"] = _simple_momentum(frame["close"], 14)
    frame["momentum_30"] = _simple_momentum(frame["close"], 30)
    frame["trend_1h"] = np.sign(frame["momentum_14"].fillna(0.0)).astype(int)
    frame["volume_ratio"] = _volume_ratio(frame["volume"])
    frame["price_zscore"] = _price_zscore(frame["close"])
    frame["rsi"] = _simple_rsi(frame["close"])
    frame["atr_pct"] = _atr_pct(frame)
    frame["rotation_score"] = frame["momentum_14"].fillna(0.0)
    bullish: list[bool] = []
    bearish: list[bool] = []
    strengths: list[float] = []
    ages: list[int] = []
    closes = frame["close"].astype(float).to_numpy()
    rsi_values = frame["rsi"].astype(float).to_numpy()
    for idx in range(len(frame)):
        signal = detect_rsi_divergence(closes[: idx + 1], rsi_values=rsi_values[: idx + 1])
        bullish.append(bool(signal.bullish_divergence))
        bearish.append(bool(signal.bearish_divergence))
        strengths.append(float(signal.divergence_strength))
        ages.append(int(signal.divergence_age_bars))
    frame["bullish_divergence"] = bullish
    frame["bearish_divergence"] = bearish
    frame["divergence_strength"] = strengths
    frame["divergence_age_bars"] = ages
    frame["ranging_market"] = (
        frame["momentum_5"].abs().fillna(0.0).le(0.0025)
        & frame["momentum_14"].abs().fillna(0.0).le(0.006)
        & frame["atr_pct"].fillna(0.0).le(2.5)
    )
    frame["symbol_group"] = _classify_symbol_group(symbol)
    frame["regime_bucket"] = frame.apply(_classify_regime_bucket, axis=1)
    return frame


def _policy_row(symbol: str, row: pd.Series) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "momentum": float(row["momentum_14"]),
        "momentum_5": float(row["momentum_5"]),
        "momentum_14": float(row["momentum_14"]),
        "momentum_30": float(row["momentum_30"]),
        "trend_1h": int(row["trend_1h"]),
        "volume_ratio": float(row["volume_ratio"]),
        "price_zscore": float(row["price_zscore"]),
        "rsi": float(row["rsi"]),
        "regime_7d": "unknown",
        "macro_30d": "sideways",
        "price": float(row["close"]),
        "atr": float(row["close"]) * (float(row["atr_pct"]) / 100.0),
        "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
        "bb_bandwidth": 0.02,
        "spread_pct": 0.10,
        "rotation_score": float(row["rotation_score"]),
        "correlation_row": [],
        "hurst": 0.5,
        "autocorr": 0.0,
        "entropy": 0.5,
    }


def add_policy_outputs(symbol: str, feature_frame: pd.DataFrame) -> pd.DataFrame:
    output = feature_frame.copy()
    entry_scores: list[float] = []
    recommendations: list[str] = []
    reversal_risks: list[str] = []
    lanes: list[str] = []
    net_edges: list[float] = []
    tp_after_cost_valid_values: list[bool] = []
    expected_move_pcts: list[float] = []
    required_edge_pcts: list[float] = []
    expected_edge_pcts: list[float] = []
    phi3_market_states: list[str] = []
    phi3_lane_biases: list[str] = []
    phi3_market_confidences: list[float] = []
    phi3_alignment_flags: list[bool] = []
    phi3_reasons: list[str] = []

    for _, row in output.iterrows():
        feature_row = _policy_row(symbol, row)
        policy = apply_policy_pipeline(symbol, feature_row)
        entry_scores.append(float(policy.get("entry_score", 0.0) or 0.0))
        recommendations.append(str(policy.get("entry_recommendation", "WATCH")))
        reversal_risks.append(str(policy.get("reversal_risk", "MEDIUM")))
        lanes.append(str(policy.get("lane", "L3")))
        cost_features = {
            **feature_row,
            "lane": str(policy.get("lane", "L3")),
            "entry_recommendation": str(policy.get("entry_recommendation", "WATCH")),
            "promotion_tier": str(policy.get("promotion_tier", "skip")),
            "structure_quality": float(policy.get("structure_quality", 50.0) or 50.0),
            "continuation_quality": float(policy.get("continuation_quality", 50.0) or 50.0),
            "momentum_quality": float(policy.get("momentum_quality", 50.0) or 50.0),
            "trade_quality": float(policy.get("trade_quality", 50.0) or 50.0),
        }
        cost = evaluate_trade_cost(cost_features, "LONG")
        net_edges.append(float(cost.net_edge_pct))
        tp_after_cost_valid_values.append(bool(cost.actionable))
        expected_move_pcts.append(float(cost.expected_move_pct))
        required_edge_pcts.append(float(cost.required_edge_pct))
        expected_edge_pcts.append(float(cost.expected_edge_pct))
        phi3_features = {
            **cost_features,
            "entry_score": float(policy.get("entry_score", 0.0) or 0.0),
            "reversal_risk": str(policy.get("reversal_risk", "MEDIUM")),
            "risk_quality": float(policy.get("risk_quality", 50.0) or 50.0),
            "ranging_market": bool(row.get("ranging_market", False)),
            "trend_confirmed": bool(int(row.get("trend_1h", 0) or 0) > 0 and float(row.get("momentum_14", 0.0) or 0.0) > 0.0),
            "volume_surge": max(float(row.get("volume_ratio", 1.0) or 1.0) - 1.0, 0.0),
            "sentiment_symbol_trending": False,
        }
        phi3_review = deterministic_market_state_review(phi3_features).to_dict()
        phi3_market_state = str(phi3_review.get("market_state", "transition") or "transition")
        phi3_lane_bias = str(phi3_review.get("lane_bias", "favor_selective") or "favor_selective")
        regime_bucket = str(row.get("regime_bucket", "unknown") or "unknown")
        phi3_market_states.append(phi3_market_state)
        phi3_lane_biases.append(phi3_lane_bias)
        phi3_market_confidences.append(float(phi3_review.get("confidence", 0.0) or 0.0))
        phi3_alignment_flags.append(_phi3_regime_alignment(regime_bucket, phi3_market_state, phi3_lane_bias))
        phi3_reasons.append(str(phi3_review.get("reason", "")))

    output["entry_score"] = entry_scores
    output["entry_recommendation"] = recommendations
    output["reversal_risk"] = reversal_risks
    output["lane"] = lanes
    output["net_edge_pct"] = net_edges
    output["tp_after_cost_valid"] = tp_after_cost_valid_values
    output["expected_move_pct"] = expected_move_pcts
    output["required_edge_pct"] = required_edge_pcts
    output["expected_edge_pct"] = expected_edge_pcts
    output["phi3_market_state"] = phi3_market_states
    output["phi3_lane_bias"] = phi3_lane_biases
    output["phi3_market_confidence"] = phi3_market_confidences
    output["phi3_regime_alignment"] = phi3_alignment_flags
    output["phi3_reason"] = phi3_reasons
    return output


@dataclass
class SweepResult:
    summary: pd.DataFrame
    portfolio: vbt.Portfolio
    feature_frame: pd.DataFrame


@dataclass
class BatchSweepResult:
    per_symbol_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    feature_rows: dict[str, pd.DataFrame]


@dataclass
class WalkForwardResult:
    summary: pd.DataFrame
    windows: pd.DataFrame
    regime_summary: pd.DataFrame
    phi3_summary: pd.DataFrame
    feature_frame: pd.DataFrame


@dataclass
class BatchWalkForwardResult:
    per_symbol_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    regime_summary: pd.DataFrame
    phi3_summary: pd.DataFrame
    windows: pd.DataFrame


def _format_variant_component(prefix: str, value: float) -> str:
    token = str(float(value)).replace(".", "p")
    return f"{prefix}_{token}"


_MAJOR_SYMBOLS = {
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "ADA/USD",
    "LINK/USD",
    "AVAX/USD",
    "LTC/USD",
    "DOT/USD",
    "TRX/USD",
    "ALGO/USD",
    "XLM/USD",
    "HBAR/USD",
    "TAO/USD",
}
_MEME_SYMBOLS = {
    "DOGE/USD",
    "SHIB/USD",
    "PEPE/USD",
    "WIF/USD",
    "BONK/USD",
    "FARTCOIN/USD",
    "TRUMP/USD",
    "SPX/USD",
    "PENGU/USD",
}


def _classify_symbol_group(symbol: str) -> str:
    normalized = str(symbol or "").upper()
    if normalized in _MEME_SYMBOLS:
        return "meme"
    if normalized in _MAJOR_SYMBOLS:
        return "major"
    return "core_alt"


def _classify_regime_bucket(row: pd.Series) -> str:
    if bool(row.get("ranging_market", False)):
        return "ranging"
    trend_1h = int(row.get("trend_1h", 0) or 0)
    momentum_14 = float(row.get("momentum_14", 0.0) or 0.0)
    momentum_5 = float(row.get("momentum_5", 0.0) or 0.0)
    if trend_1h > 0 and momentum_14 > 0.0:
        return "trend"
    if trend_1h < 0 or (momentum_14 < 0.0 and momentum_5 <= 0.0):
        return "red"
    return "transition"


def _phi3_regime_alignment(regime_bucket: str, market_state: str, lane_bias: str) -> bool:
    regime = str(regime_bucket or "unknown")
    state = str(market_state or "transition")
    bias = str(lane_bias or "favor_selective")
    if regime == "trend":
        return state == "trending"
    if regime == "ranging":
        return state == "ranging"
    if regime == "transition":
        return state == "transition"
    if regime == "red":
        return state != "trending" and bias != "favor_trend"
    return False


def _dominant_label(series: pd.Series, default: str = "unknown") -> str:
    if series.empty:
        return default
    counts = series.astype(str).value_counts()
    if counts.empty:
        return default
    return str(counts.index[0])


def _build_entry_mask(
    features: pd.DataFrame,
    *,
    threshold: float,
    lane_filter: str | None = None,
    require_bullish_divergence: bool = False,
    min_net_edge_pct: float | None = None,
    bullish_divergence_score_bonus: float = 0.0,
    bullish_divergence_promotion_window: float = 0.0,
) -> pd.Series:
    bullish_divergence = features["bullish_divergence"].astype(bool)
    effective_entry_score = features["entry_score"].astype(float) + (
        bullish_divergence.astype(float) * float(bullish_divergence_score_bonus)
    )
    buy_recommendation = features["entry_recommendation"].isin(["BUY", "STRONG_BUY"])
    promoted_watch = (
        bullish_divergence
        & features["entry_recommendation"].astype(str).eq("WATCH")
        & features["entry_score"].astype(float).ge(float(threshold) - float(bullish_divergence_promotion_window))
    )
    entry_mask = (
        (effective_entry_score.ge(float(threshold)) | promoted_watch)
        & (buy_recommendation | promoted_watch)
        & features["reversal_risk"].astype(str).ne("HIGH")
    )
    if lane_filter:
        entry_mask &= features["lane"].astype(str).str.upper().eq(lane_filter.upper())
    if require_bullish_divergence:
        entry_mask &= bullish_divergence
    if min_net_edge_pct is not None:
        entry_mask &= features["net_edge_pct"].astype(float).ge(float(min_net_edge_pct))
    entry_mask &= features["tp_after_cost_valid"].astype(bool)
    return entry_mask


def _portfolio_from_entry_mask(
    features: pd.DataFrame,
    *,
    entry_mask: pd.Series,
    hold_bars: int,
    fee_pct: float,
    slippage_pct: float,
) -> vbt.Portfolio:
    features = features.reset_index(drop=True)
    close = features["close"].astype(float)
    index = pd.DatetimeIndex(features["timestamp"])
    entries = pd.Series(entry_mask.to_numpy(dtype=bool), index=index)
    exits = pd.Series(entry_mask.shift(hold_bars, fill_value=False).to_numpy(dtype=bool), index=index)
    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=fee_pct / 100.0,
        slippage=slippage_pct / 100.0,
        init_cash=1_000.0,
        direction="longonly",
    )


def _portfolio_metrics(portfolio: vbt.Portfolio) -> dict[str, float]:
    return {
        "total_return_pct": float(portfolio.total_return() * 100.0),
        "win_rate_pct": float(portfolio.trades.win_rate() * 100.0),
        "total_trades": float(portfolio.trades.count()),
        "max_drawdown_pct": float(portfolio.max_drawdown() * 100.0),
        "sharpe_ratio": float(portfolio.sharpe_ratio()),
    }


def discover_history_symbols(
    history_dir: str | Path = "logs/history",
    timeframe: str = "1h",
) -> list[str]:
    root = Path(history_dir)
    suffix = f"_{timeframe}.csv"
    symbols: list[str] = []
    for path in sorted(root.glob(f"candles_*{suffix}")):
        token = path.name[len("candles_") : -len(suffix)]
        if not token:
            continue
        symbols.append(f"{token[:-3]}/{token[-3:]}" if token.endswith("USD") and len(token) > 3 else token)
    return symbols


def run_threshold_sweep(
    symbol: str,
    *,
    history_dir: str | Path = "logs/history",
    timeframe: str = "1h",
    entry_score_thresholds: list[float] | None = None,
    lane_filter: str | None = None,
    require_bullish_divergence: bool = False,
    min_net_edge_pct: float | None = None,
    bullish_divergence_score_bonus: float = 0.0,
    bullish_divergence_promotion_window: float = 0.0,
    hold_bars: int = 24,
    fee_pct: float = 0.40,
    slippage_pct: float = 0.10,
    summary_csv_path: str | Path | None = None,
) -> SweepResult:
    thresholds = entry_score_thresholds or [55.0, 60.0, 65.0, 70.0, 75.0]
    frame = load_history_frame(symbol, timeframe=timeframe, history_dir=history_dir)
    features = add_policy_outputs(symbol, build_policy_feature_frame(symbol, frame))
    features = features.dropna(subset=["close"]).reset_index(drop=True)

    close = features["close"].astype(float)
    index = pd.DatetimeIndex(features["timestamp"])

    entries: dict[str, pd.Series] = {}
    exits: dict[str, pd.Series] = {}
    lane_label = (lane_filter or "ALL").upper()
    divergence_label = "bulldiv" if require_bullish_divergence else "anydiv"
    edge_label = "edge_any" if min_net_edge_pct is None else f"edge_{str(min_net_edge_pct).replace('.', 'p')}"
    bonus_label = _format_variant_component("divbonus", bullish_divergence_score_bonus)
    promotion_label = _format_variant_component("divpromo", bullish_divergence_promotion_window)
    for threshold in thresholds:
        key = (
            f"entry_score_{int(threshold)}__lane_{lane_label}__{divergence_label}"
            f"__{edge_label}__{bonus_label}__{promotion_label}"
        )
        entry_mask = _build_entry_mask(
            features,
            threshold=float(threshold),
            lane_filter=lane_filter,
            require_bullish_divergence=require_bullish_divergence,
            min_net_edge_pct=min_net_edge_pct,
            bullish_divergence_score_bonus=bullish_divergence_score_bonus,
            bullish_divergence_promotion_window=bullish_divergence_promotion_window,
        )
        entries[key] = pd.Series(entry_mask.to_numpy(dtype=bool), index=index)
        exits[key] = pd.Series(entry_mask.shift(hold_bars, fill_value=False).to_numpy(dtype=bool), index=index)

    entries_df = pd.DataFrame(entries)
    exits_df = pd.DataFrame(exits)

    portfolio = vbt.Portfolio.from_signals(
        close=pd.DataFrame({column: close.to_numpy() for column in entries_df.columns}, index=index),
        entries=entries_df,
        exits=exits_df,
        fees=fee_pct / 100.0,
        slippage=slippage_pct / 100.0,
        init_cash=1_000.0,
        direction="longonly",
    )

    summary = pd.DataFrame(
        {
            "total_return_pct": portfolio.total_return() * 100.0,
            "win_rate_pct": portfolio.trades.win_rate() * 100.0,
            "total_trades": portfolio.trades.count(),
            "max_drawdown_pct": portfolio.max_drawdown() * 100.0,
            "sharpe_ratio": portfolio.sharpe_ratio(),
        }
    ).reset_index(names="variant")
    summary = summary.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    if summary_csv_path is not None:
        target = Path(summary_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target, index=False)
    return SweepResult(summary=summary, portfolio=portfolio, feature_frame=features)


def run_batch_threshold_sweep(
    symbols: list[str],
    *,
    history_dir: str | Path = "logs/history",
    timeframe: str = "1h",
    entry_score_thresholds: list[float] | None = None,
    lane_filter: str | None = None,
    require_bullish_divergence: bool = False,
    min_net_edge_pct: float | None = None,
    bullish_divergence_score_bonus: float = 0.0,
    bullish_divergence_promotion_window: float = 0.0,
    hold_bars: int = 24,
    fee_pct: float = 0.40,
    slippage_pct: float = 0.10,
    per_symbol_csv_path: str | Path | None = None,
    aggregate_csv_path: str | Path | None = None,
) -> BatchSweepResult:
    per_symbol_frames: list[pd.DataFrame] = []
    feature_rows: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        result = run_threshold_sweep(
            symbol,
            history_dir=history_dir,
            timeframe=timeframe,
            entry_score_thresholds=entry_score_thresholds,
            lane_filter=lane_filter,
            require_bullish_divergence=require_bullish_divergence,
            min_net_edge_pct=min_net_edge_pct,
            bullish_divergence_score_bonus=bullish_divergence_score_bonus,
            bullish_divergence_promotion_window=bullish_divergence_promotion_window,
            hold_bars=hold_bars,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
        )
        summary = result.summary.copy()
        summary.insert(0, "symbol", symbol)
        per_symbol_frames.append(summary)
        feature_rows[symbol] = result.feature_frame

    per_symbol_summary = pd.concat(per_symbol_frames, ignore_index=True) if per_symbol_frames else pd.DataFrame()
    aggregate_summary = pd.DataFrame()
    if not per_symbol_summary.empty:
        aggregate_summary = (
            per_symbol_summary.groupby("variant", as_index=False)
            .agg(
                symbols=("symbol", "count"),
                avg_total_return_pct=("total_return_pct", "mean"),
                median_total_return_pct=("total_return_pct", "median"),
                avg_win_rate_pct=("win_rate_pct", "mean"),
                avg_total_trades=("total_trades", "mean"),
                avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
                avg_sharpe_ratio=("sharpe_ratio", "mean"),
            )
            .sort_values("avg_total_return_pct", ascending=False)
            .reset_index(drop=True)
        )

    if per_symbol_csv_path is not None:
        target = Path(per_symbol_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        per_symbol_summary.to_csv(target, index=False)
    if aggregate_csv_path is not None:
        target = Path(aggregate_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        aggregate_summary.to_csv(target, index=False)

    return BatchSweepResult(
        per_symbol_summary=per_symbol_summary,
        aggregate_summary=aggregate_summary,
        feature_rows=feature_rows,
    )


def run_walk_forward_validation(
    symbol: str,
    *,
    history_dir: str | Path = "logs/history",
    timeframe: str = "1h",
    entry_score_thresholds: list[float] | None = None,
    lane_filter: str | None = None,
    require_bullish_divergence: bool = False,
    min_net_edge_pct: float | None = None,
    bullish_divergence_score_bonus: float = 0.0,
    bullish_divergence_promotion_window: float = 0.0,
    hold_bars: int = 24,
    fee_pct: float = 0.40,
    slippage_pct: float = 0.10,
    train_bars: int = 96,
    test_bars: int = 24,
    step_bars: int | None = None,
    summary_csv_path: str | Path | None = None,
    windows_csv_path: str | Path | None = None,
) -> WalkForwardResult:
    thresholds = entry_score_thresholds or [55.0, 60.0, 65.0, 70.0, 75.0]
    step = step_bars or test_bars
    frame = load_history_frame(symbol, timeframe=timeframe, history_dir=history_dir)
    features = add_policy_outputs(symbol, build_policy_feature_frame(symbol, frame))
    features = features.dropna(subset=["close"]).reset_index(drop=True)

    total_bars = len(features)
    rows: list[dict[str, Any]] = []
    start = int(train_bars)
    while start + int(test_bars) <= total_bars:
        train_slice = features.iloc[start - int(train_bars) : start].reset_index(drop=True)
        test_slice = features.iloc[start : start + int(test_bars)].reset_index(drop=True)
        best_threshold = float(thresholds[0])
        best_train_return = float("-inf")
        best_train_metrics: dict[str, float] = {}

        for threshold in thresholds:
            train_mask = _build_entry_mask(
                train_slice,
                threshold=float(threshold),
                lane_filter=lane_filter,
                require_bullish_divergence=require_bullish_divergence,
                min_net_edge_pct=min_net_edge_pct,
                bullish_divergence_score_bonus=bullish_divergence_score_bonus,
                bullish_divergence_promotion_window=bullish_divergence_promotion_window,
            )
            train_portfolio = _portfolio_from_entry_mask(
                train_slice,
                entry_mask=train_mask,
                hold_bars=hold_bars,
                fee_pct=fee_pct,
                slippage_pct=slippage_pct,
            )
            train_metrics = _portfolio_metrics(train_portfolio)
            train_return = float(train_metrics["total_return_pct"])
            if train_return > best_train_return:
                best_train_return = train_return
                best_threshold = float(threshold)
                best_train_metrics = train_metrics

        test_mask = _build_entry_mask(
            test_slice,
            threshold=best_threshold,
            lane_filter=lane_filter,
            require_bullish_divergence=require_bullish_divergence,
            min_net_edge_pct=min_net_edge_pct,
            bullish_divergence_score_bonus=bullish_divergence_score_bonus,
            bullish_divergence_promotion_window=bullish_divergence_promotion_window,
        )
        test_portfolio = _portfolio_from_entry_mask(
            test_slice,
            entry_mask=test_mask,
            hold_bars=hold_bars,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
        )
        test_metrics = _portfolio_metrics(test_portfolio)
        rows.append(
            {
                "symbol": symbol,
                "symbol_group": _classify_symbol_group(symbol),
                "lane_filter": str((lane_filter or "ALL")).upper(),
                "train_start": str(train_slice["timestamp"].iloc[0]),
                "train_end": str(train_slice["timestamp"].iloc[-1]),
                "test_start": str(test_slice["timestamp"].iloc[0]),
                "test_end": str(test_slice["timestamp"].iloc[-1]),
                "test_regime_bucket": _dominant_label(test_slice["regime_bucket"], default="unknown"),
                "test_lane_mode": _dominant_label(test_slice["lane"], default="unknown"),
                "test_phi3_market_state": _dominant_label(test_slice["phi3_market_state"], default="unknown"),
                "test_phi3_lane_bias": _dominant_label(test_slice["phi3_lane_bias"], default="unknown"),
                "test_phi3_alignment_rate": round(float(test_slice["phi3_regime_alignment"].astype(float).mean()), 4),
                "test_phi3_market_confidence": round(float(test_slice["phi3_market_confidence"].astype(float).mean()), 4),
                "selected_threshold": best_threshold,
                "train_total_return_pct": round(best_train_metrics.get("total_return_pct", 0.0), 4),
                "train_win_rate_pct": round(best_train_metrics.get("win_rate_pct", 0.0), 4),
                "train_total_trades": round(best_train_metrics.get("total_trades", 0.0), 4),
                "test_total_return_pct": round(test_metrics.get("total_return_pct", 0.0), 4),
                "test_win_rate_pct": round(test_metrics.get("win_rate_pct", 0.0), 4),
                "test_total_trades": round(test_metrics.get("total_trades", 0.0), 4),
                "test_max_drawdown_pct": round(test_metrics.get("max_drawdown_pct", 0.0), 4),
                "test_sharpe_ratio": round(test_metrics.get("sharpe_ratio", 0.0), 4),
            }
        )
        start += int(step)

    windows = pd.DataFrame(rows)
    regime_summary = pd.DataFrame()
    phi3_summary = pd.DataFrame()
    if not windows.empty:
        regime_summary = (
            windows.groupby("test_regime_bucket", as_index=False)
            .agg(
                windows=("symbol", "count"),
                avg_test_total_return_pct=("test_total_return_pct", "mean"),
                median_test_total_return_pct=("test_total_return_pct", "median"),
                avg_test_win_rate_pct=("test_win_rate_pct", "mean"),
                avg_test_total_trades=("test_total_trades", "mean"),
                avg_test_max_drawdown_pct=("test_max_drawdown_pct", "mean"),
                avg_test_sharpe_ratio=("test_sharpe_ratio", "mean"),
            )
            .sort_values("avg_test_total_return_pct", ascending=False)
            .reset_index(drop=True)
        )
        phi3_summary = (
            windows.groupby(["test_phi3_market_state", "test_phi3_lane_bias"], as_index=False)
            .agg(
                windows=("symbol", "count"),
                avg_test_total_return_pct=("test_total_return_pct", "mean"),
                median_test_total_return_pct=("test_total_return_pct", "median"),
                avg_test_win_rate_pct=("test_win_rate_pct", "mean"),
                avg_test_total_trades=("test_total_trades", "mean"),
                avg_test_max_drawdown_pct=("test_max_drawdown_pct", "mean"),
                avg_test_sharpe_ratio=("test_sharpe_ratio", "mean"),
                avg_phi3_alignment_rate=("test_phi3_alignment_rate", "mean"),
                avg_phi3_market_confidence=("test_phi3_market_confidence", "mean"),
            )
            .sort_values("avg_test_total_return_pct", ascending=False)
            .reset_index(drop=True)
        )
    summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "symbol_group": _classify_symbol_group(symbol),
                "lane_filter": str((lane_filter or "ALL")).upper(),
                "windows": int(len(windows)),
                "avg_test_total_return_pct": float(windows["test_total_return_pct"].mean()) if not windows.empty else 0.0,
                "median_test_total_return_pct": float(windows["test_total_return_pct"].median()) if not windows.empty else 0.0,
                "avg_test_win_rate_pct": float(windows["test_win_rate_pct"].mean()) if not windows.empty else 0.0,
                "avg_test_total_trades": float(windows["test_total_trades"].mean()) if not windows.empty else 0.0,
                "avg_test_max_drawdown_pct": float(windows["test_max_drawdown_pct"].mean()) if not windows.empty else 0.0,
                "avg_test_sharpe_ratio": float(windows["test_sharpe_ratio"].mean()) if not windows.empty else 0.0,
                "avg_phi3_alignment_rate": float(windows["test_phi3_alignment_rate"].mean()) if not windows.empty else 0.0,
                "avg_phi3_market_confidence": float(windows["test_phi3_market_confidence"].mean()) if not windows.empty else 0.0,
            }
        ]
    )
    if summary_csv_path is not None:
        target = Path(summary_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target, index=False)
    if windows_csv_path is not None:
        target = Path(windows_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        windows.to_csv(target, index=False)
    return WalkForwardResult(
        summary=summary,
        windows=windows,
        regime_summary=regime_summary,
        phi3_summary=phi3_summary,
        feature_frame=features,
    )


def run_batch_walk_forward_validation(
    symbols: list[str],
    *,
    history_dir: str | Path = "logs/history",
    timeframe: str = "1h",
    entry_score_thresholds: list[float] | None = None,
    lane_filter: str | None = None,
    require_bullish_divergence: bool = False,
    min_net_edge_pct: float | None = None,
    bullish_divergence_score_bonus: float = 0.0,
    bullish_divergence_promotion_window: float = 0.0,
    hold_bars: int = 24,
    fee_pct: float = 0.40,
    slippage_pct: float = 0.10,
    train_bars: int = 96,
    test_bars: int = 24,
    step_bars: int | None = None,
    per_symbol_csv_path: str | Path | None = None,
    aggregate_csv_path: str | Path | None = None,
    regime_csv_path: str | Path | None = None,
    windows_csv_path: str | Path | None = None,
) -> BatchWalkForwardResult:
    per_symbol_frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []
    phi3_frames: list[pd.DataFrame] = []
    windows_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        result = run_walk_forward_validation(
            symbol,
            history_dir=history_dir,
            timeframe=timeframe,
            entry_score_thresholds=entry_score_thresholds,
            lane_filter=lane_filter,
            require_bullish_divergence=require_bullish_divergence,
            min_net_edge_pct=min_net_edge_pct,
            bullish_divergence_score_bonus=bullish_divergence_score_bonus,
            bullish_divergence_promotion_window=bullish_divergence_promotion_window,
            hold_bars=hold_bars,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
        )
        per_symbol_frames.append(result.summary.copy())
        if not result.regime_summary.empty:
            regime = result.regime_summary.copy()
            regime.insert(0, "symbol", symbol)
            regime.insert(1, "symbol_group", _classify_symbol_group(symbol))
            regime_frames.append(regime)
        if not result.phi3_summary.empty:
            phi3 = result.phi3_summary.copy()
            phi3.insert(0, "symbol", symbol)
            phi3.insert(1, "symbol_group", _classify_symbol_group(symbol))
            phi3.insert(2, "lane_filter", str((lane_filter or "ALL")).upper())
            phi3_frames.append(phi3)
        if not result.windows.empty:
            windows_frames.append(result.windows.copy())

    per_symbol_summary = pd.concat(per_symbol_frames, ignore_index=True) if per_symbol_frames else pd.DataFrame()
    regime_summary = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    phi3_summary = pd.concat(phi3_frames, ignore_index=True) if phi3_frames else pd.DataFrame()
    windows = pd.concat(windows_frames, ignore_index=True) if windows_frames else pd.DataFrame()

    aggregate_summary = pd.DataFrame()
    if not per_symbol_summary.empty:
        aggregate_summary = (
            per_symbol_summary.groupby(["symbol_group", "lane_filter"], as_index=False)
            .agg(
                symbols=("symbol", "count"),
                avg_test_total_return_pct=("avg_test_total_return_pct", "mean"),
                median_test_total_return_pct=("median_test_total_return_pct", "median"),
                avg_test_win_rate_pct=("avg_test_win_rate_pct", "mean"),
                avg_test_total_trades=("avg_test_total_trades", "mean"),
                avg_test_max_drawdown_pct=("avg_test_max_drawdown_pct", "mean"),
                avg_test_sharpe_ratio=("avg_test_sharpe_ratio", "mean"),
            )
            .sort_values("avg_test_total_return_pct", ascending=False)
            .reset_index(drop=True)
        )

    if per_symbol_csv_path is not None:
        target = Path(per_symbol_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        per_symbol_summary.to_csv(target, index=False)
    if aggregate_csv_path is not None:
        target = Path(aggregate_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        aggregate_summary.to_csv(target, index=False)
    if regime_csv_path is not None:
        target = Path(regime_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        regime_summary.to_csv(target, index=False)
    if windows_csv_path is not None:
        target = Path(windows_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        windows.to_csv(target, index=False)

    return BatchWalkForwardResult(
        per_symbol_summary=per_symbol_summary,
        aggregate_summary=aggregate_summary,
        regime_summary=regime_summary,
        phi3_summary=phi3_summary,
        windows=windows,
    )
