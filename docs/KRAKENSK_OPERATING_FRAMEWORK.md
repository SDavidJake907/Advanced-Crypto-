# KrakenSK Operating Framework

## Purpose

This document defines the operating framework for KrakenSK as a live crypto trading engine.

It sits above the trading manual.

Use this document to keep the system coherent across:
- live trading
- AI reasoning
- policy enforcement
- review/adaptation
- operator visibility
- incident handling

The goal is not abstract orchestration.
The goal is a practical, stable framework for running KrakenSK safely and improving it over time.

---

## 1. Framework Goals

KrakenSK should operate as:
- a live lane-based trading engine
- a deterministic risk-controlled execution system
- a structured AI-assisted decision system
- a review-driven adaptive system
- a crypto-native always-on system

The framework should provide:
- clear agent roles
- clear runtime paths
- clear review paths
- clear schedules
- clear output contracts
- clear operator expectations

---

## 2. Core Layers

### 2.1 Live Layer

This is the fast path.

Responsibilities:
- collect market data
- normalize symbols
- build candles/features
- run local advisory models
- run final planner
- enforce deterministic risk
- place or reject orders
- monitor exits

Requirements:
- fast
- narrow
- explicit
- safe
- deterministic where possible

Modules:
- `apps/collector`
- `apps/trader`
- `core/features`
- `core/policy`
- `core/risk`
- `core/execution`

### 2.2 Intelligence Layer

This is the model reasoning layer.

Responsibilities:
- Phi-3 reflex
- Phi-3 lane supervision
- Phi-3 market-state review
- Nemotron candidate review
- Nemotron posture review
- Nemotron final planning
- cloud optimizer review

Requirements:
- advisory roles must remain advisory
- hard-risk authority must stay outside model outputs
- outputs must stay structured and auditable

Modules:
- `core/llm`
- `apps/review_scheduler`

### 2.3 Policy Layer

This is the deterministic interpretation layer.

Responsibilities:
- lane classification
- lane-specific filtering
- regime state
- verifier scoring
- promotion logic
- sizing logic
- exit logic

Requirements:
- policy must be explicit
- lane behavior must be consistent
- market-state logic must be reusable
- thresholds must be observable and reviewable

Modules:
- `core/policy`
- `core/strategy`
- `core/risk`

### 2.4 Review Layer

This is the slow adaptation layer.

Responsibilities:
- replay/backtesting
- trade memory
- outcome review
- optimizer review
- posture review
- missed-trade review

Requirements:
- advisory first
- structured outputs
- repeated evidence before changes
- no uncontrolled self-modifying behavior

Modules:
- `apps/replay`
- `core/memory`
- `apps/review_scheduler`

### 2.5 Operator Layer

This is the system visibility layer.

Responsibilities:
- status visibility
- lane summaries
- posture summaries
- collector telemetry
- account sync state
- open positions
- recent reviews

Requirements:
- easy to inspect
- low-friction
- no need to dig through raw logs for normal operation

Modules:
- `apps/mcp_server`
- `logs/*`

---

## 3. Agent Roles

### 3.1 Phi-3

Phi-3 is:
- reflex advisor
- lane helper
- market-state helper
- anomaly detector

Phi-3 is not:
- final trade authority
- hard-risk authority
- execution authority

Main duties:
- `phi3_reflex`
- `phi3_supervise_lane`
- `phi3_review_market_state`
- scan/watchlist guidance

### 3.2 Nemotron

Nemotron is:
- candidate reviewer
- posture reviewer
- final local planner

Nemotron is not:
- a hard-risk override
- an execution bypass

Main duties:
- `nemotron_review_candidate`
- `nemotron_set_posture`
- final `OPEN/HOLD/CLOSE` contract generation
- structured trade reasoning

### 3.3 Cloud NVIDIA Review

Cloud review is:
- optimizer coach
- lesson extractor
- review assistant

Cloud review is not:
- live execution authority
- direct runtime controller by default

Main duties:
- periodic review
- runtime adjustment suggestions
- lane/performance summaries
- threshold advice

### 3.4 Risk Layer

Risk is:
- deterministic
- absolute

Main duties:
- exposure caps
- correlation caps
- max open positions
- execution contract validity
- position sizing caps
- execution blocking

---

## 4. Runtime Paths

### 4.1 Live Decision Path

The live decision path is:

`collector -> features -> policy -> Phi-3 -> Nemotron -> risk -> execution -> sync -> exits`

Intent:
- small surface area
- fast response
- stable behavior

Design rules:
- no review-loop logic in the hot path
- no cloud dependency in the hot path
- no prompt drift in the hot path

### 4.2 Review Path

The review path is:

`trades + skips + memory + replay + cloud review -> structured suggestions`

Intent:
- learn slowly
- detect drift
- tune carefully

Design rules:
- advisory by default
- repeated evidence over single events
- explicit suggested changes only

---

## 5. Crypto-Native Schedule

KrakenSK is not a stock-market engine.
It should run on crypto-native timing.

### 5.1 Event-Driven / Continuous

Always-on:
- live data collection
- live feature building
- Phi-3 reflex
- Nemotron final planning
- account sync
- exit monitoring

### 5.2 Every 5 Minutes

Tasks:
- lane strength review
- market-state refresh
- candidate ranking refresh
- active universe quality check

### 5.3 Every 15-30 Minutes

Tasks:
- cloud optimizer review
- missed-trade review
- funnel/promotion review

### 5.4 On Every Exit

Tasks:
- outcome classification
- lesson capture
- trade memory update

### 5.5 Every 4-6 Hours

Tasks:
- posture review
- lane performance summary
- universe quality review

### 5.6 Daily

Tasks:
- summary of winners and losers
- strongest lane
- weakest lane
- one suggested threshold change
- one suggested policy note

---

## 6. Playbooks

### 6.1 Trading Manual

Source of truth for:
- trade rules
- risk doctrine
- lane doctrine
- model duties

Document:
- `docs/AGENT_TRADING_MANUAL.md`

### 6.2 Lane Playbook

Should define:
- what each lane is for
- when a lane is valid
- how aggressive it should be
- how sizing differs
- how exits differ

Current source:
- trading manual + policy code

### 6.3 Market-State Playbook

Should define what to do in:
- trending
- ranging
- transition
- bullish
- bearish
- volatile

Current source:
- trading manual + `trend_state` + regime logic

### 6.4 Incident Playbook

Should define what to do when:
- data integrity fails
- model endpoint fails
- exchange rejects orders
- account sync mismatches positions
- cloud review errors

Current source:
- partly implicit in code and logs

---

## 7. Operator Responsibilities

Operators should be able to answer:
- what is the active universe
- what are the top candidates
- what lane is each symbol in
- why was a symbol skipped
- what is current posture
- what positions are live
- what was the latest optimizer guidance
- what happened on the latest exit

Operator tools:
- MCP status tools
- JSONL traces
- account sync
- collector telemetry
- optimizer reviews

Operator doctrine:
- do not force changes from one trade
- prefer evidence from logs and reviews
- restart only when code changes require it
- use reload for state refresh only

---

## 8. Framework Boundaries

The framework should avoid:
- giant generic agent systems
- vague role overlap
- hidden hard rules inside prompts
- cloud dependency for live safety
- too many simultaneous agents

The framework should prefer:
- a few clear agents
- strict JSON contracts
- deterministic enforcement
- explicit logs
- replay and review over improvisation

---

## 9. Current Framework Shape

KrakenSK currently has the right core shape:
- live engine
- lane system
- local AI roles
- cloud review
- memory direction
- operator visibility
- replay harness
- trading manual

That means the framework is now about:
- clarity
- maintainability
- observability
- controlled adaptation

Not about inventing a new architecture again.

---

## 10. Next Framework Steps

Best next improvements under this framework:

1. central trace writer
- one place for decision/review/replay logging

2. operator dashboard
- one compact runtime status surface

3. replay summary surfacing in MCP
- easier comparison of changes without raw file inspection

4. lane playbook split-out
- separate lane handbook from the main manual

5. incident playbook
- explicit recovery steps for live operational issues

---

## 11. Framework Principle

KrakenSK should evolve by:
- tightening
- observing
- replaying
- reviewing
- making small deliberate changes

Not by:
- large rewrites
- uncontrolled prompt drift
- adding agents faster than structure

That is the operating framework.
