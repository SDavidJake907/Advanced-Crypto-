"""AI Behavior Score — tracks Nemo's decision quality from trade outcomes.

Computes a 0-100 score across five dimensions:
  precision        — win rate quality (entering good setups)
  capture_quality  — how much of the available move was captured
  cost_awareness   — penalty for short-hold losing trades (cost-dominated losses)
  tail_control     — penalty for large drawdown losses
  regime_quality   — win rate in trending vs choppy regimes

The overall score drives a threshold_adjustment that Nemo receives in its
batch prompt so it can self-calibrate entry strictness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.trade_memory import OutcomeRecord


@dataclass
class BehaviorScore:
    # Individual dimension scores (0-100 each)
    precision: float           # win rate quality
    capture_quality: float     # exit timing (captured % of MFE)
    cost_awareness: float      # penalty for quick losers (cost-dominated exits)
    tail_control: float        # penalty for big losses
    regime_quality: float      # trending vs choppy performance delta

    overall: float             # weighted composite 0-100
    trade_count: int           # how many outcomes were analysed
    threshold_adjustment: float  # suggested delta for NEMOTRON_GATE_MIN_ENTRY_SCORE


_SHORT_HOLD_MIN = 20.0   # trades held < this are "quick" exits
_QUICK_LOSS_PCT = -0.5   # quick loss floor (cost-dominated if worse)
_BIG_LOSS_PCT   = -2.5   # tail loss floor


def compute_behavior_score(
    outcomes: list["OutcomeRecord"],
    lookback: int = 50,
) -> BehaviorScore | None:
    """Return a BehaviorScore from the most recent *lookback* outcomes.

    Returns None when there are fewer than 5 outcomes (not enough data).
    """
    recent = outcomes[-lookback:] if len(outcomes) > lookback else list(outcomes)
    n = len(recent)
    if n < 5:
        return None

    wins   = [o for o in recent if o.pnl_pct > 0.001]
    losses = [o for o in recent if o.pnl_pct < -0.001]
    win_rate = len(wins) / n

    # --- precision (0-100) -----------------------------------------------
    # Linear: 0% WR → 0 pts, 50% WR → 50 pts, 80% WR → 100 pts
    precision = min(win_rate * 125.0, 100.0)

    # --- capture_quality (0-100) ------------------------------------------
    # capture_vs_mfe_pct: 100 means we exited exactly at MFE (perfect);
    # 0 means we gave back everything.  Average across winners only.
    captures = [o.capture_vs_mfe_pct for o in wins if o.capture_vs_mfe_pct > 0.0]
    if captures:
        avg_capture = sum(captures) / len(captures)
        capture_quality = min(avg_capture, 100.0)
    else:
        capture_quality = 50.0  # neutral when unknown

    # --- cost_awareness (0-100) -------------------------------------------
    # Penalise quick-hold losing trades — those are cost-dominated exits where
    # Nemo entered at the wrong moment.
    quick_losers = [
        o for o in losses
        if o.hold_minutes < _SHORT_HOLD_MIN and o.pnl_pct < _QUICK_LOSS_PCT
    ]
    quick_loss_rate = len(quick_losers) / n
    # 0% quick losses → 100 pts; 25%+ quick losses → 0 pts
    cost_awareness = max(0.0, 100.0 - quick_loss_rate * 400.0)

    # --- tail_control (0-100) ---------------------------------------------
    # Big losses hurt risk-adjusted returns disproportionately.
    tail_losers = [o for o in losses if o.pnl_pct < _BIG_LOSS_PCT]
    tail_rate = len(tail_losers) / n
    # 0% big losses → 100 pts; 20%+ big losses → 0 pts
    tail_control = max(0.0, 100.0 - tail_rate * 500.0)

    # --- regime_quality (0-100) -------------------------------------------
    # Trending regimes should have higher win rates than choppy ones.
    trending = [o for o in recent if o.regime_label in ("trending", "strong_trend", "breakout")]
    choppy   = [o for o in recent if o.regime_label in ("ranging", "choppy", "consolidation")]
    if len(trending) >= 3 and len(choppy) >= 3:
        trend_wr = sum(1 for o in trending if o.pnl_pct > 0.001) / len(trending)
        choppy_wr = sum(1 for o in choppy if o.pnl_pct > 0.001) / len(choppy)
        delta = trend_wr - choppy_wr   # should be positive (better in trends)
        # +0.3 delta → 100 pts; -0.3 delta → 0 pts
        regime_quality = min(max((delta + 0.3) / 0.6 * 100.0, 0.0), 100.0)
    else:
        regime_quality = 50.0  # neutral when insufficient regime data

    # --- overall (weighted average) --------------------------------------
    overall = (
        precision       * 0.25
        + capture_quality * 0.20
        + cost_awareness  * 0.20
        + tail_control    * 0.20
        + regime_quality  * 0.15
    )

    # --- threshold_adjustment --------------------------------------------
    # Suggest adding to NEMOTRON_GATE_MIN_ENTRY_SCORE when score is low
    # (Nemo is entering bad trades) or relaxing it when score is high.
    if overall < 35:
        threshold_adjustment = +8.0   # Nemo is taking poor trades — raise the bar
    elif overall < 50:
        threshold_adjustment = +5.0
    elif overall < 60:
        threshold_adjustment = +2.0
    elif overall > 75:
        threshold_adjustment = -3.0   # Nemo is performing well — can be slightly more aggressive
    else:
        threshold_adjustment = 0.0

    return BehaviorScore(
        precision=round(precision, 1),
        capture_quality=round(capture_quality, 1),
        cost_awareness=round(cost_awareness, 1),
        tail_control=round(tail_control, 1),
        regime_quality=round(regime_quality, 1),
        overall=round(overall, 1),
        trade_count=n,
        threshold_adjustment=threshold_adjustment,
    )


def format_behavior_score_block(bs: BehaviorScore) -> str:
    """Return a compact text block suitable for injection into an LLM prompt."""
    grade = (
        "EXCELLENT" if bs.overall >= 75
        else "GOOD" if bs.overall >= 60
        else "FAIR" if bs.overall >= 45
        else "POOR"
    )
    adj_str = (
        f"+{bs.threshold_adjustment:.0f} (be MORE selective)"
        if bs.threshold_adjustment > 0
        else f"{bs.threshold_adjustment:.0f} (maintain or relax slightly)"
        if bs.threshold_adjustment < 0
        else "no change needed"
    )
    lines = [
        f"=== NEMO BEHAVIOR SCORE ({bs.trade_count} trades) ===",
        f"Overall: {bs.overall:.0f}/100 [{grade}]",
        f"  precision={bs.precision:.0f}  capture={bs.capture_quality:.0f}"
        f"  cost_aware={bs.cost_awareness:.0f}  tail={bs.tail_control:.0f}"
        f"  regime={bs.regime_quality:.0f}",
        f"Threshold advice: {adj_str}",
    ]
    if bs.cost_awareness < 50:
        lines.append("⚠ HIGH quick-loss rate — many entries are cost-dominated, prefer HOLD on marginal setups")
    if bs.tail_control < 50:
        lines.append("⚠ TAIL RISK — large losses occurring too frequently, tighten stop discipline")
    if bs.precision < 40:
        lines.append("⚠ LOW win rate — raise entry bar, only take high-conviction setups (score ≥ 70+)")
    return "\n".join(lines)
