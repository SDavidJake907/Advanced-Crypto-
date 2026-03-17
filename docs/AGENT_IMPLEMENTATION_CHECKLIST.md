## KrakenSK Manual Implementation Checklist

Source of truth: [AGENT_TRADING_MANUAL.md](C:/Users/kitti/Desktop/KrakenSK/docs/AGENT_TRADING_MANUAL.md)

### Agent Split
- Phi-3 advisory reflex/scan/lane support
  - Enforced by: `core/llm/phi3_reflex.py`, `core/llm/phi3_scan.py`, `core/llm/micro_prompts.py`
  - Status: enforced
- Nemotron final planner
  - Enforced by: `core/llm/nemotron.py`, `core/llm/nemotron_client.py`, `core/runtime/tools.py`
  - Status: enforced
- Hard deterministic risk
  - Enforced by: `core/risk/basic_risk.py`, `core/risk/portfolio.py`, `core/risk/exits.py`
  - Status: enforced

### Data Integrity
- Hard block on bad price / bad volume / malformed state
  - Enforced by: `core/llm/phi3_reflex.py`, `apps/collector/main.py`, `core/data/live_buffer.py`
  - Status: partial
  - Next patch: add explicit per-symbol invalid-volume counters and collector diagnostics

### Lane Logic
- Lane classifier and lane-aware behavior
  - Enforced by: `core/policy/lane_classifier.py`, `core/policy/lane_filters.py`, `core/strategy/simple_momo.py`, `core/policy/entry_verifier.py`
  - Status: enforced
- Phi-3 lane supervision
  - Enforced by: `core/llm/micro_prompts.py`, `core/llm/phi3_scan.py`, `apps/universe_manager/main.py`
  - Status: enforced

### Entry Verification
- Entry score / recommendation / reversal risk
  - Enforced by: `core/policy/entry_verifier.py`, `core/policy/candidate_score.py`
  - Status: enforced

### Planner Logic
- Final decision validation and coercion to HOLD on invalid contracts
  - Enforced by: `core/llm/nemotron.py`
  - Status: enforced
- Candidate review micro-prompt for top-ranked symbols
  - Enforced by: `core/llm/micro_prompts.py`, `core/llm/nemotron.py`
  - Status: enforced

### Universe / Rotation
- Broad pool -> filtered active universe -> ranked candidates
  - Enforced by: `apps/universe_manager/main.py`, `core/policy/universe_policy.py`, `core/state/store.py`
  - Status: enforced

### Execution
- Live/paper mode separation
  - Enforced by: `core/execution/cpp_exec.py`, `core/execution/kraken_live.py`, `core/data/kraken_rest.py`
  - Status: enforced
- Cost/spread/size validation
  - Enforced by: `core/execution/order_policy.py`, `core/risk/fee_filter.py`, `core/execution/kraken_rules.py`, `core/execution/clamp.py`
  - Status: enforced

### Exits
- ATR stop / TP / break-even / trailing
  - Enforced by: `core/risk/exits.py`, `apps/trader/main.py`
  - Status: enforced
- True manual-position fill-basis sync
  - Enforced by: partial in `core/data/account_sync.py`, `apps/trader/main.py`
  - Status: partial
  - Next patch: import Kraken fill history into synced position seed

### Outcome Learning
- Trade memory
  - Enforced by: `core/memory/trade_memory.py`
  - Status: enforced
- Outcome review micro-prompt
  - Enforced by: `core/llm/micro_prompts.py`, `apps/trader/main.py`
  - Status: enforced
- Scheduled review loop
  - Enforced by: `apps/review_scheduler/main.py`, `core/memory/kraken_history.py`
  - Status: partial
  - Next patch: feed optimizer outcomes back into compact operator status

### Observability
- Decision traces/debug
  - Enforced by: `apps/trader/main.py`, `core/runtime/log_rotation.py`
  - Status: enforced
- Operator/MCP inspection
  - Enforced by: `apps/mcp_server/main.py`
  - Status: partial
  - Next patch: expose lane-supervision and outcome-review summaries directly
