# Research Baseline

Current offline research baseline as of 2026-03-27:

- Baseline variant: `entry_score >= 70`, `lane = L3`, `hold_bars = 24`
- High-confidence variant: `entry_score >= 70`, `lane = L3`, `bullish_divergence = required`, `hold_bars = 24`
- Research runner: [scripts/vectorbt_research.py](/C:/Users/kitti/Desktop/KrakenSK/scripts/vectorbt_research.py)
- Machine-readable settings: [configs/research_baseline.json](/C:/Users/kitti/Desktop/KrakenSK/configs/research_baseline.json)

Summary:

- The broader `L3 @ 70` baseline trades more, but performed materially worse than the stricter divergence-filtered version in the current simplified harness.
- The stricter `L3 + bullish divergence + 70` version had much lower drawdown and much better aggregate return, but trade count became extremely sparse.
- Treat bullish divergence as a high-confidence filter, not automatically the only allowed live path.

Key artifacts:

- [research_l3_70_h24_nodiv_aggregate.csv](/C:/Users/kitti/Desktop/KrakenSK/logs/research_l3_70_h24_nodiv_aggregate.csv)
- [research_l3_70_h24_nodiv_per_symbol.csv](/C:/Users/kitti/Desktop/KrakenSK/logs/research_l3_70_h24_nodiv_per_symbol.csv)
- [research_l3_div70_h24_aggregate.csv](/C:/Users/kitti/Desktop/KrakenSK/logs/research_l3_div70_h24_aggregate.csv)
- [research_l3_div70_h24_per_symbol.csv](/C:/Users/kitti/Desktop/KrakenSK/logs/research_l3_div70_h24_per_symbol.csv)

Notable symbol behavior from the strict divergence-filtered run:

- Strong: `BCH/USD`
- Weak: `INJ/USD`, `ETH/USD`
- Many symbols produced zero trades under the strict filter, confirming that it is highly selective.

Recommended next comparisons:

1. Compare `L3 @ 70` with divergence as a bonus versus a hard gate.
2. Test whether `INJ/USD` and `ETH/USD` should be excluded from the strict divergence variant.
3. Add a multi-tier research view:
   - Tier 1: `L3 + bullish divergence + 70`
   - Tier 2: `L3 + 70` without divergence gate
