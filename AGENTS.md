# AGENTS.md

## Project purpose
This repo is a modular crypto trading system with deterministic scoring, AI-assisted screening/ranking, risk vetoes, execution, and feedback loops.

## Architecture rules

### Layer ownership
- `core/features/`
  - technical indicators
  - feature extraction
  - deterministic market-state calculations
  - no execution logic
  - no trade placement decisions

- `core/policy/`
  - deterministic scoring
  - final score calculation
  - entry/scan policy
  - non-LLM decision rules
  - payload shaping for downstream consumers

- `core/risk/`
  - cost floors
  - spread/slippage rejection
  - sizing and exposure rules
  - exits and protection logic
  - veto power over all model outputs

- `core/execution/`
  - exchange adapters
  - order placement
  - fill tracking
  - no strategy logic unless explicitly requested

- `core/memory/`
  - trade outcomes
  - symbol reliability
  - realized vs expected edge tracking
  - behavior scoring
  - historical reports

- `core/llm/`
  - model adapters
  - prompt construction
  - response parsing
  - no hidden business logic that should live in scoring/policy

## Core design principles
1. Deterministic logic first. If a value can be computed in code, compute it before sending to any model.
2. Advisory and Nemo must share the same economic truth where relevant:
   - `final_score`
   - `net_edge_pct`
   - `reliability_bonus`
   - compact packet fields, not duplicated score math
3. Keep advisory lightweight. It is a fast integrity/context layer, not the main portfolio decision engine.
4. Nemo should judge/rank finalists, not recompute arithmetic already available in code.
5. Local Nemo should see compact finalist payloads, not full wide candidate tables, unless a task explicitly changes that.
6. Risk/execution layers can veto model outputs.

## Scoring rules
- Prefer precomputed scores over prompt-side weighting.
- Keep score components explicit and inspectable.
- Cap new score modifiers unless the task explicitly changes score ranges.
- Avoid duplicated score math across multiple modules.

## Prompting rules
- Do not move deterministic scoring into prompt text.
- Keep prompt payloads compact.
- Use short lesson summaries instead of long text dumps when possible.
- Prefer `lesson_summary` over long `learned_lessons` blocks.
- Preserve shared field names across advisory and Nemo payloads unless explicitly refactoring all consumers.
- Phi should translate chart data into structured evidence for Nemo:
  - pattern
  - quality
  - context
  - warnings
  - compact recommended interpretation
- Do not let Phi respond with vague chart prose like "looks bullish" when a structured evidence field can be emitted instead.

## Implementation rules
- Make the smallest safe change set.
- Inspect relevant files first.
- Summarize current data flow before risky edits.
- Preserve backward compatibility where practical.
- Add or update tests for scoring, persistence, exit logic, and payload schema changes.

## Change discipline
- Prefer one variable change at a time for live-trading behavior.
- Do not loosen multiple independent gates at once unless the task explicitly requires it.
- Separate diagnosis changes from behavior changes where practical.
- When a model path is involved, prefer improving observability before changing core trade behavior.
- Keep runtime tweaks reversible and easy to explain from logs.

## Evidence rules
- Use recent live logs before changing gates, fees, sizing, or exits.
- Distinguish clearly between:
  - candidate discovery failure
  - model judgment failure
  - execution/cost rejection
  - portfolio/correlation rejection
- Do not call something a strategist failure if the execution or portfolio layer is the real blocker.
- Prefer concrete blocker labels over vague summaries.

## Prompt hygiene
- Prompts should request specific reasons, not vague catch-all labels.
- If a HOLD reason can be made more specific, prefer specificity over generic wording.
- Do not let prompt examples normalize lazy outputs that weaken diagnosis.
- Keep reason taxonomies short, stable, and log-friendly.

## Runtime safety
- Treat live fee assumptions, spread assumptions, and cost floors as production risk settings.
- When changing live cost assumptions, prefer conservative buffers over literal best-case values.
- Do not widen position count, sizing, and entry aggression in the same change unless explicitly required.
- Respect the distinction between local runtime overrides and committed repo defaults.

## Validation expectations
For any non-trivial task:
1. identify relevant files
2. explain the planned change
3. implement
4. run or update tests
5. summarize what changed and any risks remaining

## Task templates

### 1. Safe feature add
Task:
Add a new deterministic feature to the scanner.

Context:
This is a modular trading system. Feature extraction belongs in `core/features/`. Scoring belongs in `core/policy/`. Do not place strategy logic in LLM prompts.

Requirements:
- Add the feature in the correct module
- Wire it into the candidate feature packet
- Add a bounded score impact in final scoring
- Update tests

Definition of done:
- feature exists
- scanner sees it
- scoring uses it
- tests pass

### 2. Plan-first change
Task:
Plan first, then implement.

Before editing:
1. inspect relevant files
2. summarize current data flow
3. list exact files to change
4. identify risk points
5. then implement the smallest safe change set

Goal:
[replace with current task]

### 3. Scoring change
Task:
Add a new score modifier without changing unrelated behavior.

Rules:
- scoring logic belongs in `core/policy/`
- keep modifier bounded and explicit
- return `score_breakdown` updates
- do not duplicate logic in LLM prompt construction
- update tests

### 4. Payload alignment
Task:
Align Phi-3 and Nemo payloads to the same core economics.

Requirements:
- both should consume the same truth fields where relevant
- keep Phi-3 compact
- do not overload Phi-3 with deep basket reasoning
- preserve existing payload compatibility where possible
- summarize any field additions

### 5. Bug trace
Task:
Trace and fix a data propagation bug.

Observed issue:
[describe issue]

Investigate:
- where field is created
- where it is stored
- where it is transformed
- where it is lost

Fix:
- preserve the field end-to-end
- add a regression test

### 6. Refactor without behavior change
Task:
Refactor for clarity without changing behavior.

Rules:
- preserve output fields
- preserve score values
- split large functions only if it improves readability
- add tests confirming equivalence

### 7. Add persistence field
Task:
Add a new field to persisted trade/outcome records.

Requirements:
- add field to dataclass/model
- wire through creation, serialization, deserialization, and reporting
- keep backward compatibility where practical
- update tests

### 8. Exit logic change
Task:
Modify exit behavior carefully.

Requirements:
- explain current exit flow first
- make the smallest safe logic change
- preserve unrelated exit paths
- add tests for edge cases
- summarize expected live behavior differences

## Best way to use this now
Since the system is mostly complete, use Codex for:
- tightening modules
- enforcing architecture
- adding bounded features
- removing duplication
- regression-proofing with tests

Not for:
- giant redesigns
- vague "make it better" changes
- moving business logic into prompts

## Recommended workflow
Do these next:
- Keep a small text file of favorite Codex prompts
- For every change, force:
  - plan
  - files
  - constraints
  - done criteria
- Let the bot run and only tune based on live data

That will push the repo from advanced working system to maintainable professional system.
