from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.data.kraken_rest import KrakenRestClient
from core.data.dexscreener import DexScreenerFeed
from core.data.news_sentiment import NewsSentimentFeed
from core.features.momentum import risk_adjusted_momentum
from core.llm.phi3_scan import ScanCandidate, phi3_scan_market, phi3_supervise_lanes
from core.policy.pipeline import apply_policy_pipeline
from core.policy.universe_policy import UniversePolicy, apply_churn_threshold, clamp_active_size
from core.symbols import normalize_symbol, to_kraken_symbol
from core.state.age_cache import load_age_cache, save_age_cache
from core.state.store import load_universe, save_universe


@dataclass
class Candidate:
    pair: str
    score: float
    volume_usd: float
    last: float
    momentum_5: float = 0.0
    momentum_14: float = 0.0
    momentum_30: float = 0.0
    trend_1h: int = 0
    volume_ratio: float = 1.0
    price_zscore: float = 0.0
    rsi: float = 50.0
    spread_bps: float = 0.0
    lane: str = "L3"
    candidate_score: float = 0.0
    candidate_recommendation: str = "WATCH"
    candidate_risk: str = "MEDIUM"
    candidate_reasons: list[str] | None = None


def bucket_candidates(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    buckets: dict[str, list[Candidate]] = {
        "leaders": [],
        "emerging": [],
        "meme_heat": [],
        "losers": [],
        "reversion_candidates": [],
    }
    for candidate in candidates:
        lane = candidate.lane
        if lane == "L4" and (candidate.momentum_5 > 0.01 or candidate.score > 0.02):
            buckets["meme_heat"].append(candidate)
        if lane == "L1" or (candidate.momentum_5 > 0.015 and candidate.momentum_14 > 0 and candidate.trend_1h >= 0):
            buckets["leaders"].append(candidate)
            continue
        if lane == "L3" and candidate.momentum_5 > 0.005 and candidate.momentum_14 > -0.005:
            buckets["emerging"].append(candidate)
            continue
        if candidate.momentum_5 < -0.01 and candidate.momentum_14 < 0:
            buckets["losers"].append(candidate)
            if candidate.momentum_30 < 0 and candidate.score > -0.05:
                buckets["reversion_candidates"].append(candidate)
            continue
        if lane == "L2" or (candidate.momentum_5 < 0 and candidate.momentum_14 > 0):
            buckets["reversion_candidates"].append(candidate)

    for values in buckets.values():
        values.sort(key=lambda item: item.candidate_score, reverse=True)
    return buckets


def shortlist_candidates(
    buckets: dict[str, list[Candidate]],
    *,
    shortlist_size: int,
) -> tuple[list[Candidate], dict[str, list[str]]]:
    ordered_bucket_names = ["leaders", "emerging", "meme_heat", "reversion_candidates", "losers"]
    bucket_limits = {
        "leaders": max(3, shortlist_size // 3),
        "emerging": max(2, shortlist_size // 4),
        "meme_heat": max(2, shortlist_size // 5),
        "reversion_candidates": max(1, shortlist_size // 6),
        "losers": max(1, shortlist_size // 8),
    }
    shortlisted: list[Candidate] = []
    seen: set[str] = set()

    for bucket_name in ordered_bucket_names:
        for candidate in buckets[bucket_name][: bucket_limits[bucket_name]]:
            if candidate.pair in seen:
                continue
            shortlisted.append(candidate)
            seen.add(candidate.pair)
            if len(shortlisted) >= shortlist_size:
                break
        if len(shortlisted) >= shortlist_size:
            break

    if len(shortlisted) < shortlist_size:
        remaining = []
        for bucket_name in ordered_bucket_names:
            remaining.extend(buckets[bucket_name])
        remaining.sort(key=lambda item: item.candidate_score, reverse=True)
        for candidate in remaining:
            if candidate.pair in seen:
                continue
            shortlisted.append(candidate)
            seen.add(candidate.pair)
            if len(shortlisted) >= shortlist_size:
                break

    bucket_meta = {
        name: list(
            dict.fromkeys(
                candidate.pair for candidate in buckets[name][: bucket_limits.get(name, shortlist_size)]
            )
        )
        for name in ordered_bucket_names
    }
    return shortlisted, bucket_meta


def _simple_momentum(closes: list[float], lookback: int) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    prev = float(closes[-(lookback + 1)])
    last = float(closes[-1])
    if prev <= 0.0:
        return 0.0
    return (last / prev) - 1.0


def _simple_rsi(closes: list[float], lookback: int = 14) -> float:
    if len(closes) < lookback + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    window = closes[-(lookback + 1) :]
    for idx in range(1, len(window)):
        delta = float(window[idx]) - float(window[idx - 1])
        if delta > 0:
            gains += delta
        elif delta < 0:
            losses += abs(delta)
    if losses == 0.0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _price_zscore(closes: list[float], lookback: int = 20) -> float:
    if len(closes) < lookback:
        return 0.0
    window = closes[-lookback:]
    mean = sum(window) / len(window)
    variance = sum((value - mean) ** 2 for value in window) / len(window)
    if variance <= 0.0:
        return 0.0
    std = variance ** 0.5
    if std <= 0.0:
        return 0.0
    return (closes[-1] - mean) / std


def _volume_ratio(rows: list[list[str | float]], lookback: int = 20) -> float:
    if len(rows) < lookback + 1:
        return 1.0
    volumes = [float(r[6]) for r in rows[-(lookback + 1) :]]
    current = volumes[-1]
    baseline = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
    if baseline <= 0.0:
        return 1.0
    return current / baseline


def load_pair_pool() -> list[str]:
    pool_file = os.getenv("PAIR_POOL_FILE", "pair_pool_usd.txt")
    path = Path(pool_file)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def get_policy() -> UniversePolicy:
    return UniversePolicy(
        active_min=int(os.getenv("ACTIVE_MIN", "10")),
        active_max=int(os.getenv("ACTIVE_MAX", "15")),
        max_adds=int(os.getenv("MAX_ADDS_PER_REBALANCE", "5")),
        max_removes=int(os.getenv("MAX_REMOVES_PER_REBALANCE", "5")),
        cooldown_minutes=int(os.getenv("PAIR_COOLDOWN_MINUTES", "180")),
        min_volume_usd=float(os.getenv("MIN_QUOTE_VOLUME_USD", "500000")),
        max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", "12")),
        min_price=float(os.getenv("MIN_PRICE", "0.05")),
        churn_threshold=float(os.getenv("CHURN_THRESHOLD", "1.20")),
    )


def normalize_pair(pair: str) -> str:
    return normalize_symbol(pair)


def load_cached_asset_pairs(cache_path: Path, ttl_min: int) -> dict | None:
    if not cache_path.exists():
        return None
    age_s = time.time() - cache_path.stat().st_mtime
    if age_s > ttl_min * 60:
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def save_cached_asset_pairs(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))


def filter_pairs_with_kraken(client: KrakenRestClient, pairs: list[str]) -> tuple[list[str], list[str], dict]:
    # Use Kraken AssetPairs to validate available USD pairs
    cache_file = Path(os.getenv("ASSET_PAIRS_CACHE_FILE", "configs/kraken_asset_pairs.cache.json"))
    ttl_min = int(os.getenv("ASSET_PAIRS_CACHE_TTL_MIN", "30"))
    resp = load_cached_asset_pairs(cache_file, ttl_min)
    if resp is None:
        try:
            resp = client.get_asset_pairs()
            save_cached_asset_pairs(cache_file, resp)
        except Exception:
            # If REST fails and cache exists, try using stale cache
            if cache_file.exists():
                try:
                    resp = json.loads(cache_file.read_text())
                except Exception:
                    raise
            else:
                raise

    result = resp.get("result", {})
    meta_map: dict[str, dict] = {}
    for k, v in result.items():
        wsname = v.get("wsname")
        altname = v.get("altname")
        status = (v.get("status") or "online").lower()
        if status != "online":
            continue
        if wsname:
            meta_map[wsname.upper()] = {"key": k, **v}
        if altname:
            meta_map[altname.upper()] = {"key": k, **v}

    normalized = [normalize_pair(p) for p in pairs]
    valid = [p for p in normalized if to_kraken_symbol(p) in meta_map]
    invalid = [p for p in normalized if to_kraken_symbol(p) not in meta_map]
    return valid, invalid, meta_map


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def calc_spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 1e9
    return ((ask - bid) / mid) * 10_000.0


def first_ohlc_timestamp(client: KrakenRestClient, pair: str, interval_min: int) -> int | None:
    try:
        res = client.get_ohlc(pair, interval_min=interval_min, since=0)
        result = res.get("result", {})
        for k, v in result.items():
            if k == "last":
                continue
            if isinstance(v, list) and v:
                return int(v[0][0])
    except Exception:
        return None
    return None


def validate_and_filter_bench(
    client: KrakenRestClient, bench_pairs: list[str]
) -> tuple[list[str], list[str], dict]:
    require_marginable = os.getenv("REQUIRE_MARGINABLE", "false").lower() == "true"
    min_notional = float(os.getenv("MIN_NOTIONAL_USD", "25"))
    max_ordermin_notional = float(os.getenv("MAX_ORDERMIN_NOTIONAL_USD", "25"))
    min_vol_usd = float(os.getenv("MIN_QUOTE_VOLUME_USD", "500000"))
    max_spread_bps = float(os.getenv("MAX_SPREAD_BPS", "12"))
    min_price = float(os.getenv("MIN_PRICE", "0.05"))

    min_age_days = int(os.getenv("MIN_AGE_DAYS", "30"))
    age_interval = int(os.getenv("AGE_CHECK_INTERVAL_MINUTES", "60"))
    age_cache_file = os.getenv("AGE_CACHE_FILE", "configs/pair_age_cache.json")
    age_cache_ttl = int(os.getenv("AGE_CACHE_TTL_HOURS", "24"))

    valid_pool, invalid_pool, meta_map = filter_pairs_with_kraken(client, bench_pairs)

    # Marginable filter (optional)
    if require_marginable:
        valid_pool = [
            p for p in valid_pool if bool(meta_map[p].get("marginable") or meta_map[p].get("margin"))
        ]

    # Batch ticker lookup
    eligible: list[str] = []
    for batch in chunked(valid_pool, 20):
        ticker_pairs = [
            str(meta_map[to_kraken_symbol(p)].get("key") or to_kraken_symbol(p))
            for p in batch
            if to_kraken_symbol(p) in meta_map
        ]
        tick = client.get_ticker(ticker_pairs).get("result", {}) if ticker_pairs else {}
        for p in batch:
            meta = meta_map.get(to_kraken_symbol(p), {})
            ticker_key = (meta.get("altname") or meta.get("wsname") or meta.get("key") or p).upper()
            t = tick.get(ticker_key) or tick.get(p) or tick.get(meta.get("key"))
            if not t:
                continue

            try:
                ask = float(t["a"][0])
                bid = float(t["b"][0])
                last = float(t["c"][0])
                vol_base = float(t["v"][1])
            except Exception:
                continue

            if last < min_price:
                continue
            quote_vol = vol_base * last
            if quote_vol < min_vol_usd:
                continue
            if calc_spread_bps(bid, ask) > max_spread_bps:
                continue

            ordermin = float(meta.get("ordermin") or 0.0)
            ordermin_notional = ordermin * last if ordermin > 0 else 0.0
            if ordermin_notional > max_ordermin_notional:
                continue
            if min_notional < ordermin_notional:
                # This pair is still ok; execution should clamp upward.
                pass

            eligible.append(p)

    # Age filter with caching
    age_cache = load_age_cache(age_cache_file, ttl_hours=age_cache_ttl)
    aged: list[str] = []
    now = int(time.time())
    for p in eligible:
        rec = age_cache.get(p)
        first_ts = int(rec["first_ts"]) if rec and "first_ts" in rec else None
        if not first_ts:
            first_ts = first_ohlc_timestamp(client, p, interval_min=age_interval)
            if first_ts:
                age_cache[p] = {"first_ts": first_ts, "checked_at": now}
        if not first_ts:
            aged.append(p)
            continue
        age_days = (now - first_ts) / 86400.0
        if age_days < min_age_days:
            continue
        aged.append(p)

    save_age_cache(age_cache_file, age_cache)
    return aged, invalid_pool, meta_map


def rank_pairs(client: KrakenRestClient, pairs: list[str], meta_map: dict[str, dict]) -> list[Candidate]:
    # Basic ranking using OHLC 1h candles over 24h
    candidates: list[Candidate] = []
    ticker_pairs = [
        str(meta_map[to_kraken_symbol(pair)].get("key") or to_kraken_symbol(pair))
        for pair in pairs
        if to_kraken_symbol(pair) in meta_map
    ]
    ticker_result = client.get_ticker(ticker_pairs).get("result", {}) if ticker_pairs else {}
    for pair in pairs:
        try:
            meta = meta_map.get(to_kraken_symbol(pair), {})
            pair_key = str(meta.get("key") or pair)
            ohlc = client.get_ohlc(pair_key, interval_min=60)
            result = ohlc.get("result", {})
            rows = None
            for key, value in result.items():
                if key == "last":
                    continue
                if key in {pair, pair_key, meta.get("key"), meta.get("altname"), meta.get("wsname"), to_kraken_symbol(pair)}:
                    rows = value
                    break
            if rows is None:
                continue
            closes = [float(r[4]) for r in rows][-25:]  # ~24h + 1
            score = risk_adjusted_momentum(closes)
            momentum_5 = _simple_momentum(closes, 5)
            momentum_14 = _simple_momentum(closes, 14)
            momentum_30 = _simple_momentum(closes, min(30, len(closes) - 1))
            last = closes[-1]
            volume_ratio = _volume_ratio(rows)
            price_zscore = _price_zscore(closes)
            rsi = _simple_rsi(closes)
            # Volume is base volume; convert to quote using last
            vol_base = sum(float(r[6]) for r in rows[-24:])
            vol_quote = vol_base * last
            trend_1h = 1 if momentum_14 > 0 else (-1 if momentum_14 < 0 else 0)
            ticker_key = (meta.get("altname") or meta.get("wsname") or meta.get("key") or pair).upper()
            ticker = (
                ticker_result.get(ticker_key)
                or ticker_result.get(pair_key)
                or ticker_result.get(meta.get("key"))
                or {}
            )
            try:
                ask = float(ticker["a"][0])
                bid = float(ticker["b"][0])
                spread_bps = calc_spread_bps(bid, ask)
            except Exception:
                spread_bps = 0.0
            policy = apply_policy_pipeline(
                pair,
                {
                    "symbol": pair,
                    "momentum": momentum_14,
                    "momentum_5": momentum_5,
                    "momentum_14": momentum_14,
                    "momentum_30": momentum_30,
                    "trend_1h": trend_1h,
                    "volume_ratio": volume_ratio,
                    "price_zscore": price_zscore,
                    "rsi": rsi,
                    "regime_7d": "unknown",
                    "macro_30d": "sideways",
                    "price": last,
                    "atr": 0.0,
                    "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                    "bb_bandwidth": 0.02,
                    "spread_pct": spread_bps / 100.0,
                    "rotation_score": score,
                    "correlation_row": [],
                    "hurst": 0.5,
                    "autocorr": 0.0,
                    "entropy": 0.5,
                },
            )
            lane = str(policy.get("lane", "L3"))
            candidate_reasons = list(policy.get("entry_reasons", []))
            lane_filter_reason = str(policy.get("lane_filter_reason", "") or "")
            if lane_filter_reason and lane_filter_reason not in candidate_reasons:
                candidate_reasons.append(lane_filter_reason)
            regime_state_value = str(policy.get("regime_state", "") or "")
            if regime_state_value:
                candidate_reasons.append(regime_state_value)
            candidates.append(
                Candidate(
                    pair=pair,
                    score=score,
                    volume_usd=vol_quote,
                    last=last,
                    momentum_5=momentum_5,
                    momentum_14=momentum_14,
                    momentum_30=momentum_30,
                    trend_1h=trend_1h,
                    volume_ratio=volume_ratio,
                    price_zscore=price_zscore,
                    rsi=rsi,
                    spread_bps=spread_bps,
                    lane=lane,
                    candidate_score=float(policy.get("entry_score", 0.0) or 0.0),
                    candidate_recommendation=str(policy.get("entry_recommendation", "WATCH")),
                    candidate_risk=str(policy.get("reversal_risk", "MEDIUM")),
                    candidate_reasons=candidate_reasons,
                )
            )
        except Exception:
            continue
    candidates.sort(key=lambda c: c.candidate_score, reverse=True)
    return candidates


async def build_scan_meta(candidates: list[Candidate], bucket_meta: dict[str, list[str]]) -> dict:
    news = NewsSentimentFeed(fetch_interval_sec=0.0)
    dex = DexScreenerFeed()
    try:
        await news.maybe_update()
        news_context = news.snapshot.to_dict()
    except Exception:
        news_context = {}
    try:
        dex_context = await dex.fetch_market_summary([candidate.pair for candidate in candidates[:15]])
    except Exception:
        dex_context = {}
    scan_candidates = [
        ScanCandidate(
            symbol=item.pair,
            lane=item.lane,
            momentum_5=float(item.momentum_5),
            momentum_14=float(item.momentum_14),
            momentum_30=float(item.momentum_30),
            rotation_score=float(item.candidate_score / 100.0),
            rsi=float(item.rsi),
            volume=float(item.volume_usd),
            trend_1h=int(item.trend_1h),
            regime_7d="unknown",
            macro_30d="sideways",
        )
        for item in candidates[:25]
    ]
    scan = phi3_scan_market(
        scan_candidates,
        news_context=news_context,
        market_context={"candidate_count": len(candidates), "bucket_meta": bucket_meta},
        dex_context=dex_context,
    )
    lane_supervision = [
        item.to_dict()
        for item in phi3_supervise_lanes(
            scan_candidates,
            news_context=news_context,
            dex_context=dex_context,
        )
    ]
    hot_candidates = [
        {
            "symbol": candidate.pair,
            "lane": candidate.lane,
            "candidate_score": candidate.candidate_score,
            "recommendation": candidate.candidate_recommendation,
            "risk": candidate.candidate_risk,
            "reasons": candidate.candidate_reasons or [],
        }
        for candidate in candidates
        if candidate.candidate_recommendation in {"BUY", "STRONG_BUY"}
    ][:10]
    avoid_candidates = [
        {
            "symbol": candidate.pair,
            "lane": candidate.lane,
            "candidate_score": candidate.candidate_score,
            "recommendation": candidate.candidate_recommendation,
            "risk": candidate.candidate_risk,
            "reasons": candidate.candidate_reasons or [],
        }
        for candidate in sorted(candidates, key=lambda item: item.candidate_score)
        if candidate.candidate_recommendation == "AVOID"
    ][:10]
    return {
        "phi3_scan": scan.to_dict(),
        "lane_supervision": lane_supervision,
        "news_context": news_context,
        "dex_context": dex_context,
        "buckets": bucket_meta,
        "top_ranked": [candidate.pair for candidate in candidates[:10]],
        "top_scored": [
            {
                "symbol": candidate.pair,
                "lane": candidate.lane,
                "candidate_score": candidate.candidate_score,
                "recommendation": candidate.candidate_recommendation,
                "risk": candidate.candidate_risk,
                "reasons": candidate.candidate_reasons or [],
            }
            for candidate in candidates[:10]
        ],
        "hot_candidates": hot_candidates,
        "avoid_candidates": avoid_candidates,
    }


def rebalance_universe(candidates: list[Candidate], policy: UniversePolicy) -> dict:
    current = load_universe()
    active = [normalize_pair(pair) for pair in current.get("active_pairs", [])]

    # Filter by liquidity and price
    filtered = [
        c for c in candidates if c.volume_usd >= policy.min_volume_usd and c.last >= policy.min_price
    ]

    valid_filtered_pairs = {candidate.pair for candidate in filtered}
    active = [pair for pair in active if pair in valid_filtered_pairs]

    if not filtered:
        return save_universe(active, "no candidates passed filters", meta={"top_ranked": []})

    shortlist_size = max(policy.active_min, min(policy.active_max, int(os.getenv("ROTATION_SHORTLIST_SIZE", "20"))))
    buckets = bucket_candidates(filtered)
    shortlisted, bucket_meta = shortlist_candidates(buckets, shortlist_size=shortlist_size)
    candidate_pool = shortlisted or filtered

    # Determine worst active score among active pairs
    score_map = {c.pair: c.candidate_score for c in candidate_pool}
    active_scores = [score_map.get(p, float("-inf")) for p in active]
    worst_active = min(active_scores) if active_scores else float("-inf")

    adds: list[str] = []
    removes: list[str] = []

    for c in candidate_pool:
        if len(adds) >= policy.max_adds:
            break
        if c.pair in active:
            continue
        if worst_active != float("-inf") and not apply_churn_threshold(
            c.candidate_score, worst_active, policy.churn_threshold
        ):
            continue
        adds.append(c.pair)

    # Remove the weakest active if we add new ones
    if adds:
        active_scored = sorted(
            [(p, score_map.get(p, float("-inf"))) for p in active],
            key=lambda x: x[1],
        )
        for p, _ in active_scored:
            if len(removes) >= policy.max_removes or len(active) - len(removes) <= policy.active_min:
                break
            removes.append(p)

    next_active = [p for p in active if p not in removes] + adds
    next_active = clamp_active_size(next_active, policy.active_max)

    reason = f"adds={adds} removes={removes}"
    meta = asyncio.run(build_scan_meta(candidate_pool, bucket_meta))
    return save_universe(next_active, reason, meta=meta)


def main() -> None:
    load_dotenv()
    client = KrakenRestClient()
    policy = get_policy()
    pool = load_pair_pool()
    if not pool:
        print("pair pool is empty")
        return
    valid_pool, invalid_pool, _meta = validate_and_filter_bench(client, pool)
    if invalid_pool:
        print(
            f"invalid pairs in pool (not on Kraken USD): {len(invalid_pool)} "
            f"{invalid_pool}"
        )
    if not valid_pool:
        print("no valid pairs after Kraken validation")
        return
    if os.getenv("WRITE_VALIDATED_BENCH", "false").lower() == "true":
        out_file = Path(os.getenv("VALIDATED_BENCH_FILE", "pair_pool_usd.valid.txt"))
        out_file.write_text("\n".join(valid_pool) + "\n")
        print(f"wrote validated bench: {out_file} ({len(valid_pool)} pairs)")
    candidates = rank_pairs(client, valid_pool, _meta)
    result = rebalance_universe(candidates, policy)
    print(result)


if __name__ == "__main__":
    main()
