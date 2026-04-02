# LLM Operating Model

Current state: the project should be treated as a deterministic trading system with one active local LLM path layered on top, not as an LLM-first bot.

This document is meant to answer:

1. What the LLM is allowed to do
2. What the LLM is not allowed to do
3. Which model role owns which part of the pipeline
4. How much context the model should receive
5. How failures must degrade

## North Star

The correct control model is:

- code computes truth
- runtime config defines behavior
- LLM judges only the finalists
- execution and risk remain deterministic

That means the LLM is not allowed to:

- invent indicators
- recalculate score math
- override hard risk rules
- rescue candidates already rejected by deterministic policy

## End-To-End Decision Flow

The current intended live flow is:

1. market data and features are computed deterministically
2. deterministic policy shapes and filters candidates
3. advisory provides light context and data-integrity screening
4. deterministic prefilter cuts the universe down to finalists
5. strategist judges only those finalists
6. deterministic risk/execution still decides whether an order is legal

So the LLM is not the top-level owner.

It sits in the middle of the pipeline, not at the beginning and not at the end.

## Source Of Truth

### Code truth

Code is the source of truth for:

- features
- scores
- cost math
- promotion legality
- sizing
- stops / exits
- execution legality

### LLM truth

The LLM is only a judgment layer over already-computed truth.

It should interpret:

- candidate quality
- relative attractiveness
- whether a finalist deserves `OPEN` or `HOLD`

It should never fabricate system state.

## Current Active Roles

### Advisory

Advisory is currently the lightweight local model path.

Relevant code:
- [core/llm/prompts.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/prompts.py)
- [core/llm/orchestrator.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/orchestrator.py)

Current responsibilities:

- reflex / data-integrity screening
- scout/watchlist triage
- lane supervision
- market-state review
- chart-pattern verification and candle-context validation as a secondary structure layer

Advisory is not the final trade decision maker.

Its job is to:

- reject broken data
- provide light context
- avoid making the strategist do unnecessary screening work
- verify structured chart evidence produced by code, not rediscover trade setups from scratch

### What advisory must not do

Advisory must not:

- decide final entry
- decide final exit legality
- invent risk sizing
- override stabilization gates
- block on normal market weakness unless the task explicitly requires it

### Strategist

Strategist is the main trade-decision path.

Relevant code:
- [core/llm/nemotron.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/nemotron.py)
- [apps/trader/boot.py](C:/Users/kitti/Desktop/KrakenSK/apps/trader/boot.py)

Current responsibilities:

- decide OPEN vs HOLD on finalists
- compare a small finalist set in batch mode
- return strict JSON only

Current runtime shape:

- `advisory_provider = local_nemo`
- `strategist_provider = local`
- `NEMOTRON_BATCH_MODE = true`
- `NEMOTRON_BATCH_TOP_N = 5`

This means:

- full universe is scored deterministically
- only top 5 finalists go to local Nemo batch

### What strategist must not do

Strategist must not:

- rescore candidates from scratch
- widen the universe
- bypass deterministic hold blocks
- invent a side or size outside the contract
- treat prompt examples as live symbols

The contract in [core/llm/nemotron.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/nemotron.py) already validates this, and that contract should stay strict.

## Finalist Policy

The strategist should not see the full universe.

Current intended pipeline:

1. deterministic scoring over the active universe
2. deterministic gating and stabilization checks
3. top-N finalist selection
4. one local strategist batch over those finalists

This is the correct batch shape for current hardware.

Bad shape:

- send all active symbols to local batch
- or run 20 serial local strategist calls every cycle

Good shape:

- score many
- judge few

### Why top 5 matters

Top 5 is not arbitrary.

It protects:

- prompt size
- local latency
- JSON reliability
- freshness of the cycle

Top 5 is a throughput-control rule, not just a ranking preference.

## Packet Policy

### Canonical packet

The full canonical packet remains the system truth.

Relevant code:
- [core/policy/candidate_packet.py](C:/Users/kitti/Desktop/KrakenSK/core/policy/candidate_packet.py)

It exists for:

- deterministic state
- logging
- audits
- future adapters

### Local strategist packet

The local strategist does not receive the full packet.

Relevant code:
- [core/policy/candidate_packet.py](C:/Users/kitti/Desktop/KrakenSK/core/policy/candidate_packet.py)
- [core/llm/nemotron.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/nemotron.py)

Current rule:

- local Nemo gets a compact packet only

This packet is meant to keep:

- symbol identity
- lane
- entry score / recommendation
- reversal risk
- net edge and cost facts
- structure/trend confirmation
- compact lesson summary
- compact behavior score

It is meant to drop:

- low-value score duplication
- long lesson blocks
- oversized context
- bloat that does not improve a 9B local decision

### Packet design rule

The packet should answer one question:

- what does the strategist need to decide this finalist now?

If a field does not materially improve that decision, it should not be in the local packet.

## Prompt Policy

### Local strategist prompts

Local strategist prompts must stay:

- short
- strict
- JSON-only
- one decision contract

Relevant code:
- [core/llm/prompts.py](C:/Users/kitti/Desktop/KrakenSK/core/llm/prompts.py)

The local strategist should be told:

- deterministic policy outranks it
- use provided fields as truth
- do not recompute
- do not invent
- keep reason short
- return one exact JSON object

### Prompt design rule

Prompts should be written as operating contracts, not essays.

Good prompt behavior:

- short
- literal
- contract-first
- JSON-only
- clear hard-hold rules

Bad prompt behavior:

- long market philosophy
- duplicated scoring explanation
- too many exceptions
- narrative bloat

### Cloud prompts

Cloud prompt variants exist, but the current operating model is local/local.

Cloud is not the active strategist path right now.

## Batch Policy

Batch is acceptable only when:

- finalist count is small
- prompt is compact
- output contract is strict

Current target:

- `NEMOTRON_BATCH_TOP_N = 5`

This is the working compromise between:

- serial local calls being too slow
- oversized local batch becoming unstable

### Batch design rule

Batch is acceptable only when it improves cycle freshness.

If batch becomes:

- too large
- too slow
- too unstable
- too hard to parse

then it is working against the trading system, not for it.

## Code vs Model Boundaries

### Code owns

- feature computation
- score computation
- cost math
- gating
- stabilization rules
- position sizing rules
- risk vetoes
- exit mechanics

### Runtime/config owns

Runtime/config should own:

- finalist count
- batch mode on/off
- timeouts
- stabilization gates
- candidate limits
- portfolio limits
- reentry cooldown

That is what keeps this system tunable without code churn.

### LLM owns

- final judgment among already-clean candidates
- light contextual explanation
- posture bias where explicitly requested

## Control Split

The intended operating split is:

- semi-automatic entries
- automatic sizing implementation
- automatic exits

### What Nemo should control

Nemo should control:

- whether a finalist deserves `OPEN` or `HOLD`
- relative preference between finalists
- concise reasoning for why a setup deserves capital now
- a bounded aggressiveness hint only

Nemo should have real control over:

- final judgment on cleaned finalists
- whether Phi-3 chart verification meaningfully supports or weakens a finalist setup

Nemo should not have direct control over:

- raw legality
- raw portfolio exposure
- exact stop placement
- exact trail placement
- exact exit firing

In the current charting split:

- code detects structure and candle facts deterministically
- Phi-3 verifies whether those facts support the claimed setup
- Nemo decides whether that verified setup deserves `OPEN` or `HOLD`
- code still decides whether the trade is legal to place

### What code should control

Code should control:

- hard entry legality
- hard exit legality
- exact position sizing formula
- exact notional and risk-budget enforcement
- exact stop-loss placement
- exact take-profit placement
- exact trailing logic
- stale / failed-follow-through exits
- execution order behavior

This also includes rotation enforcement when slots are full: code may decide to replace a weakening hold with a materially stronger candidate, while preserving healthy runners and applying the configured persistence thresholds.

### Why this split is safer

This split prevents two bad failure modes:

1. Nemo becomes powerless
- the model sees candidates but cannot materially influence anything

2. Nemo becomes dangerous
- the model controls too much and can drift, stall, or overtrade

The correct middle ground is:

- Nemo decides whether a cleaned setup deserves entry
- code decides how that decision is implemented safely

## Semi-Automatic Entry Model

The best operating model for this project is:

1. code scores the full active universe
2. code filters out illegal or weak setups
3. Nemo evaluates only the finalists
4. Nemo returns `OPEN` or `HOLD`
5. code still enforces:
   - cash
   - slot limits
   - notional
   - cost legality
   - portfolio concentration

This keeps Nemo important without making it the owner of the whole trading loop.

## Automatic Sizing Model

Sizing should be implemented automatically by code.

Nemo may supply only a bounded bias:

- `full`
- `reduced`
- `minimal`

or the equivalent via a limited size hint.

But code should still translate that into actual size using:

- `EXEC_RISK_PER_TRADE_PCT`
- `TRADER_PROPOSED_WEIGHT`
- max weight limits
- open slot constraints
- minimum notional rules

That prevents:

- random oversized entries
- under-allocating strong setups
- inconsistent behavior across sessions

## Automatic Exit Model

Exits should remain automatic and deterministic.

Nemo should not be able to delay a hard exit once:

- stop loss is hit
- trail is hit
- stale-loser rules are hit
- failed-follow-through rules are hit

The model may still influence:

- posture bias
- whether a healthy winner should keep more room

But the actual fire/no-fire exit decision should remain in code.

### LLM must not own

- hard entry legality
- hard exit legality
- arithmetic
- account economics
- hidden policy changes

## Failure And Fallback Rules

The LLM path must fail safely.

That means:

- malformed strategist output should degrade to `HOLD`
- parse failures must be visible in logs
- deterministic legality checks must still fire
- no failure mode should accidentally increase aggressiveness

The preferred failure direction is:

- less action

not:

- guessed action

### Required logging

The system should make these failure states obvious:

- malformed JSON
- empty response
- symbol mismatch
- invalid action contract
- timeout
- stale batch behavior

## Failure Policy

When LLM output fails, the system should degrade safely.

Required behavior:

- malformed batch output must not create forced opens
- hard risk checks still veto
- deterministic path remains source of truth
- logs must make failure visible

The system should prefer:

- HOLD on ambiguous model output

over:

- accidental OPEN on bad parsing

## What To Keep Small

To improve local Nemo quality, keep these small:

- finalist count
- prompt length
- memory/lesson payload
- duplicated score language
- descriptive prose

The model should not carry scanner, ranker, reviewer, and strategist duties all at once.

## Examples

### Good use of the LLM

Good:

- 20 symbols scored in code
- top 5 finalists survive deterministic filters
- local Nemo compares those 5
- returns 1-2 valid `OPEN` ideas or all `HOLD`
- execution still vetoes if cash/risk/cost fail

### Bad use of the LLM

Bad:

- send the whole live universe
- include long memory/history blocks
- ask the model to infer economics already computed in code
- let malformed output trigger fallback opens

### Good architecture change

Good change:

- reduce packet size
- reduce finalist count
- move thresholds into config
- tighten contract validation

### Bad architecture change

Bad change:

- add more prose to prompts to fix deterministic problems
- ask the model to do screening that code can already do
- add a second model path before the first is boring and reliable

## Recommended Operating Discipline

For the current project state:

1. deterministic universe scoring first
2. deterministic prefilter second
3. top-5 finalist batch to local Nemo
4. deterministic risk/execution veto last

This is the operating model that should be preserved in docs, prompts, and config unless there is a deliberate architecture change.

## Change Policy

Safe to change without architecture drift:

- finalist count
- prompt wording within the same contract
- compact packet fields
- timeout values
- batch mode settings
- bounded size-bias wording

Should be treated as architecture changes:

- giving advisory final trade authority
- moving hard risk rules into the model
- letting the model own scoring math
- widening local strategist context back toward full-universe judgment

## Practical Conclusion

The project should treat the LLM as:

- a narrow, high-leverage judgment layer

not:

- the brain of the whole bot

That distinction is the main thing that keeps local Nemo useful instead of overloaded.
