# Docs Index

This file is the complete documentation front door for the repository.

## Start Here

- [README.md](../README.md): project overview, runtime shape, screenshots, system requirements, and startup guidance
- [AGENT_IMPLEMENTATION_CHECKLIST.md](AGENT_IMPLEMENTATION_CHECKLIST.md): implementation status of the main manual/checklist items
- [AGENT_TRADING_MANUAL.md](AGENT_TRADING_MANUAL.md): operating rules for the lane-based agent stack
- [funding_readiness_checklist.md](funding_readiness_checklist.md): whether the current stack is ready for more live capital

## Architecture And Runtime

- [runtime_cycle.md](runtime_cycle.md): startup flow and live decision-cycle behavior
- [llm_operating_model.md](llm_operating_model.md): what Phi-3, Nemo, and deterministic code are each allowed to do
- [architectural_review.md](architectural_review.md): architecture-level review notes
- [KRAKENSK_OPERATING_FRAMEWORK.md](KRAKENSK_OPERATING_FRAMEWORK.md): structured operating framework for the repo
- [northstar_runtime_baseline.md](northstar_runtime_baseline.md): intended target runtime profile

## Trading Behavior

- [trading_playbook.md](trading_playbook.md): practical trading behavior and expectations
- [entry_exit_research.md](entry_exit_research.md): where entry and exit behavior live in code and config
- [exchange_fee_policy_refs.md](exchange_fee_policy_refs.md): official exchange fee and subscription-policy references for live cost assumptions
- [symbol_classification.md](symbol_classification.md): current tracked-symbol reference, including majors, core alts, and meme/research-disabled names
- [coin_profile_design.md](coin_profile_design.md): coin profile and feature design notes
- [micro_prompts_review.md](micro_prompts_review.md): prompt-shape review and compact advisory guidance

## Tuning And Improvement

- [tuning_review.md](tuning_review.md): high-level tuning observations
- [tuning_audit.md](tuning_audit.md): lane-by-lane tuning audit and next adjustments
- [10_10_roadmap.md](10_10_roadmap.md): larger implementation roadmap toward production hardening
- [research_action_plan.md](research_action_plan.md): research and follow-up action items
- [research_baseline.md](research_baseline.md): research baseline assumptions and setup
- [round_4_review.md](round_4_review.md): later review round notes

## Reviews And Project Evaluation

- [A_to_Z_Project_Review.md](A_to_Z_Project_Review.md): broader project review
- [deep_dive_review.md](deep_dive_review.md): detailed review notes
- [project_rating.md](project_rating.md): overall project quality and risk assessment
- [audit_findings/01_code_audit.md](audit_findings/01_code_audit.md): focused code audit findings

## How To Use The Docs

Suggested reading order for a new operator:
1. [README.md](../README.md)
2. [runtime_cycle.md](runtime_cycle.md)
3. [llm_operating_model.md](llm_operating_model.md)
4. [trading_playbook.md](trading_playbook.md)
5. [entry_exit_research.md](entry_exit_research.md)
6. [funding_readiness_checklist.md](funding_readiness_checklist.md)

Suggested reading order for tuning live behavior:
1. [tuning_audit.md](tuning_audit.md)
2. [tuning_review.md](tuning_review.md)
3. [10_10_roadmap.md](10_10_roadmap.md)
4. [research_action_plan.md](research_action_plan.md)
