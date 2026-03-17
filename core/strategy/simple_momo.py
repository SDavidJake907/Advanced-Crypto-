from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from core.config.runtime import get_runtime_setting, get_symbol_lane
from core.strategy.base import Strategy
from core.strategy.smoothing import SignalSmoother


@dataclass
class SimpleMomoConfig:
    base_momo_threshold: float = 0.0
    atr_volatility_scale: float = 0.5
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    meme_base_momo_threshold: float = -0.002
    meme_atr_volatility_scale: float = 0.02
    meme_rsi_overbought: float = 85.0
    meme_rsi_oversold: float = 20.0
    bb_breakout_zscore: float = 1.0
    smoothing_bars: int = 2

    @classmethod
    def from_env(cls) -> "SimpleMomoConfig":
        return cls(
            base_momo_threshold=float(get_runtime_setting("TRADER_BASE_MOMO_THRESHOLD")),
            atr_volatility_scale=float(get_runtime_setting("TRADER_ATR_VOLATILITY_SCALE")),
            rsi_overbought=float(get_runtime_setting("TRADER_RSI_OVERBOUGHT")),
            rsi_oversold=float(get_runtime_setting("TRADER_RSI_OVERSOLD")),
            meme_base_momo_threshold=float(get_runtime_setting("MEME_BASE_MOMO_THRESHOLD")),
            meme_atr_volatility_scale=float(get_runtime_setting("MEME_ATR_VOLATILITY_SCALE")),
            meme_rsi_overbought=float(get_runtime_setting("MEME_RSI_OVERBOUGHT")),
            meme_rsi_oversold=float(get_runtime_setting("MEME_RSI_OVERSOLD")),
            bb_breakout_zscore=float(get_runtime_setting("TRADER_BB_BREAKOUT_ZSCORE")),
            smoothing_bars=int(get_runtime_setting("TRADER_SMOOTHING_BARS")),
        )


class SimpleMomentumStrategy(Strategy):
    """
    Multi-indicator momentum strategy using:
      - momentum
      - RSI
      - ATR
      - Bollinger bands
      - smoothing
    """

    def __init__(self, config: SimpleMomoConfig | None = None):
        self.config = config or SimpleMomoConfig.from_env()
        self.smoother = SignalSmoother(window_bars=self.config.smoothing_bars)

    def _refresh_runtime_config(self) -> None:
        updated = SimpleMomoConfig.from_env()
        if updated.smoothing_bars != self.config.smoothing_bars:
            self.smoother = SignalSmoother(window_bars=updated.smoothing_bars)
        self.config = updated

    def _compute_momo_threshold(self, features: Dict[str, Any], lane: str) -> float:
        if lane == "L4":
            base = self.config.meme_base_momo_threshold
            scale = self.config.meme_atr_volatility_scale
        else:
            base = self.config.base_momo_threshold
            scale = self.config.atr_volatility_scale
        atr = float(features.get("atr", 0.0))
        vol = float(features.get("volatility", 0.0))

        if atr <= 0.0 or vol <= 0.0:
            return base

        adj = scale * (atr / vol)
        return base + adj

    def _raw_signal(self, features: Dict[str, Any]) -> str:
        if not bool(features.get("indicators_ready", False)):
            return "FLAT"
        lane = str(features.get("lane") or get_symbol_lane(str(features.get("symbol", ""))))
        trend_1h = int(features.get("trend_1h", 0))
        regime_7d = str(features.get("regime_7d", "unknown"))
        macro_30d = str(features.get("macro_30d", "sideways"))
        trend_confirmed = bool(features.get("trend_confirmed", False))
        ranging_market = bool(features.get("ranging_market", False))
        mom = float(features.get("momentum", 0.0))
        rsi = float(features.get("rsi", 50.0))
        price = float(features.get("price", 0.0))

        bb_mid = float(features.get("bb_middle", 0.0))
        bb_up = float(features.get("bb_upper", 0.0))
        bb_low = float(features.get("bb_lower", 0.0))

        if price <= 0.0 or bb_mid <= 0.0 or bb_up <= 0.0 or bb_low <= 0.0:
            return "FLAT"

        momo_thresh = self._compute_momo_threshold(features, lane)
        if macro_30d == "bull":
            momo_thresh *= 0.85
        elif macro_30d == "bear":
            momo_thresh *= 1.15
        if trend_confirmed:
            momo_thresh *= 0.9
        if ranging_market:
            momo_thresh *= 1.1

        if lane == "L4":
            overbought = rsi >= self.config.meme_rsi_overbought
            oversold = rsi <= self.config.meme_rsi_oversold
        else:
            overbought = rsi >= self.config.rsi_overbought
            oversold = rsi <= self.config.rsi_oversold

        above_upper = price >= bb_up
        below_lower = price <= bb_low
        near_mid = bb_low < price < bb_up
        band_half_width = max(bb_up - bb_mid, bb_mid - bb_low, 0.0)
        breakout_z = max(float(self.config.bb_breakout_zscore), 0.0)
        upper_breakout_level = bb_mid + (band_half_width * breakout_z)
        lower_breakout_level = bb_mid - (band_half_width * breakout_z)
        breakout_long = price >= upper_breakout_level
        breakout_short = price <= lower_breakout_level

        raw_signal = "FLAT"
        if regime_7d == "choppy":
            if price <= bb_low and not oversold:
                raw_signal = "LONG"
            elif price >= bb_up and not overbought:
                raw_signal = "SHORT"
        elif mom > momo_thresh:
            if overbought:
                raw_signal = "FLAT"
            elif breakout_long or (near_mid and not overbought):
                raw_signal = "LONG"
        elif mom < -momo_thresh:
            if oversold:
                raw_signal = "FLAT"
            elif breakout_short or (near_mid and not oversold):
                raw_signal = "SHORT"

        if ranging_market and lane in {"L1", "L3"} and raw_signal in {"LONG", "SHORT"}:
            return "FLAT"

        if lane != "L4" and trend_1h == -1 and raw_signal == "LONG":
            return "FLAT"
        if lane != "L4" and trend_1h == 1 and raw_signal == "SHORT":
            return "FLAT"
        return raw_signal

    def generate_signal(self, features: Dict[str, Any]) -> str:
        self._refresh_runtime_config()
        raw = self._raw_signal(features)
        smoothed = self.smoother.update(raw)
        return smoothed
