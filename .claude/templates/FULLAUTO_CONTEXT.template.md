# Full Auto Context

> This file is used by `/fullauto` mode to maintain persistent memory across conversation compactions.
> Claude updates this file as work progresses. Do not delete while a fullauto session is active.

## Task

[Description of the main task being worked on]

## Status

| Field | Value |
|-------|-------|
| Started | [YYYY-MM-DD HH:MM] |
| Last Updated | [YYYY-MM-DD HH:MM] |
| Current Phase | [Planning / Implementation / Testing / Review] |
| Progress | [X/Y steps complete] |
| Status | [Active / Paused / Completed / Blocked] |

## High-Level Plan

*Approved by Gemini Architect*

1. [ ] Step 1
2. [ ] Step 2
3. [ ] Step 3

## Architectural Decisions

| # | Decision | Options Considered | Rationale | Source |
|---|----------|-------------------|-----------|--------|
| 1 | [Decision made] | [A, B, C] | [Why this choice] | Gemini |

## Completed Work

### [Phase/Step Name]
- **What**: [Description]
- **Files Changed**: [list of files]
- **Notes**: [Any important observations]

## Current Focus

**Working On**: [Current task]

**Context**:
- [Relevant context point 1]
- [Relevant context point 2]

**Next Action**: [Immediate next step]

## Blockers & Resolutions

### [Blocker Title] - [RESOLVED/ACTIVE]
- **Issue**: [Description of the blocker]
- **Attempted**: [What was tried]
- **Resolution**: [How it was resolved, or current status]

## Gemini Consultations Log

### Consultation #1 - [Topic]
**Question**: [What was asked]
**Gemini's Response**: [Summary of response]
**Action Taken**: [What Claude did based on this]

## Divergences from Architect Guidance

*Document any cases where Claude's implementation differed from Gemini's recommendation*

| Recommendation | Actual Implementation | Reason for Divergence |
|---------------|----------------------|----------------------|
| [What Gemini suggested] | [What was actually done] | [Why] |

## Next Steps

1. [ ] [Next step 1]
2. [ ] [Next step 2]
3. [ ] [Next step 3]

## Session Notes

*Freeform notes that might be useful for context recovery*

---

## Quick Reference

**Project**: CT Scan MLOps - Lung tumor classification (4 classes)
**Stack**: Python 3.12, PyTorch Lightning, Hydra, W&B, DVC
**Key Paths**:
- Source: `src/ct_scan_mlops/`
- Configs: `configs/`
- Tests: `tests/`

**Common Commands**:
```bash
invoke quality.ruff   # Lint + format
invoke quality.test   # Run tests
invoke train.train    # Train model
```
