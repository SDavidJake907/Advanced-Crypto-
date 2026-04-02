# KrakenSK — Claude Instructions

## Project architecture
- Feature extraction belongs in `core/features/`
- Deterministic scoring and trade policy belong in `core/policy/`
- Risk and veto logic belong in `core/risk/`
- Execution logic belongs in `core/execution/`
- Trade memory and historical analytics belong in `core/memory/`
- LLM integrations belong in `core/llm/`

## Rules
- Do not move trading logic into LLM prompt text if it can be computed deterministically.
- Prefer precomputed scores over model-side arithmetic.
- Keep advisory lightweight and aligned with the same economic truth as Nemo.
- Prefer compact finalist packets for local Nemo instead of wide candidate payloads.
- Prefer `lesson_summary` over long lesson dumps in model-facing payloads.
- Do not change execution behavior unless the task explicitly requests it.
- Add tests for any scoring, persistence, or exit-logic change.
- Preserve existing payload field names unless explicitly refactoring consumers too.

## Prompting expectations
- Inspect relevant files first.
- Summarize the current data flow before major edits.
- Make the smallest safe change set.
- Prefer explicit score fields and compact payloads.
