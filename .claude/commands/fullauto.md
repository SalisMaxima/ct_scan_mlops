---
description: Autonomous mode with Gemini Pro as lead architect for strategic oversight
argument-hint: <task description or path to plan.md>
allowed-tools: Bash(gemini*), Read, Write, Edit, Glob, Grep, Task
---

# Full Auto Mode: Claude + Gemini Architect

You are entering **Full Auto Mode** - an autonomous development workflow where you (Claude) handle implementation while consulting Gemini Pro as your Lead Architect at key decision points.

## The Task

**$ARGUMENTS**

## How Full Auto Works

### Role Separation
- **You (Claude)**: Implementation, coding, testing, debugging
- **Gemini (Architect)**: Strategic decisions, architecture review, course correction

### Persistent Memory
You MUST maintain `FULLAUTO_CONTEXT.md` in the project root. This file survives conversation compactions and keeps track of:
- Current progress and completed steps
- Architectural decisions made
- Blockers and how they were resolved
- Next steps

### When to Consult the Architect

Consult Gemini at these decision points:

1. **Before starting** - Get high-level approach validation
2. **At architectural crossroads** - When multiple valid approaches exist
3. **When stuck** - If you hit a blocker for more than 2 attempts
4. **Before major refactors** - Get approval for significant changes
5. **After completing major milestones** - Validate direction

### Consultation Format

```bash
gemini -p "You are the Lead Architect for a CT scan ML classification project.

CONTEXT FROM FULLAUTO_CONTEXT.md:
[Include relevant context]

CURRENT SITUATION:
[What you're working on and the decision point]

OPTIONS I SEE:
1. [Option A]
2. [Option B]

QUESTION: Which approach should I take and why?

Be direct and decisive. Give me a clear recommendation with brief rationale."
```

## Full Auto Protocol

### Step 1: Initialize Context
First, read or create `FULLAUTO_CONTEXT.md`:

```markdown
# Full Auto Context

## Task
[The main task being worked on]

## Status
- Started: [timestamp]
- Current Phase: [phase]
- Progress: [X/Y steps complete]

## Architectural Decisions
| Decision | Rationale | Approved By |
|----------|-----------|-------------|
| ... | ... | Gemini |

## Completed Steps
- [ ] Step 1...

## Current Focus
[What you're actively working on]

## Blockers & Resolutions
[Any issues encountered and how they were solved]

## Next Steps
[Upcoming work]
```

### Step 2: Get Initial Architect Approval
Before coding, consult Gemini on the overall approach:

```bash
gemini -p "Review this task and provide a high-level implementation plan:

TASK: $ARGUMENTS

PROJECT CONTEXT: CT scan ML classification (PyTorch Lightning, 4-class tumor detection)

Provide:
1. Recommended approach (3-5 steps)
2. Key risks to watch for
3. Success criteria

Be concise and actionable."
```

### Step 3: Execute with Checkpoints
- Work through the plan step by step
- Update `FULLAUTO_CONTEXT.md` after each major step
- Consult Gemini when hitting decision points

### Step 4: Course Correction
If Gemini's guidance seems misaligned:
1. Document the misalignment in FULLAUTO_CONTEXT.md
2. Explain your reasoning
3. Ask for clarification with more specific context
4. If still misaligned, proceed with your judgment and note it

## Auto-Adjustment Rules

Sometimes the architect's guidance may not fit the specific situation. When this happens:

1. **Trust but verify**: Implement the suggestion, but validate it works
2. **Document divergence**: If you must deviate, explain why in FULLAUTO_CONTEXT.md
3. **Re-consult with context**: Provide more specific context if guidance was off
4. **Escalate to user**: For major disagreements, ask the user to decide

## Example Session Flow

```
1. /fullauto Implement feature extraction with SHAP interpretability

2. [Claude reads/creates FULLAUTO_CONTEXT.md]

3. [Claude consults Gemini for initial plan]
   Gemini: "1. Add SHAP dependency, 2. Create explainer wrapper, 3. Integrate with training loop, 4. Add visualization"

4. [Claude implements step 1, updates context]

5. [Claude hits decision point: TreeExplainer vs DeepExplainer]
   [Claude consults Gemini]
   Gemini: "Use DeepExplainer for CNN models - TreeExplainer is for tree-based models"

6. [Claude continues, updates context after each step]

7. [Claude completes, final context update with summary]
```

## Start Now

Begin by:
1. Reading or creating `FULLAUTO_CONTEXT.md`
2. Consulting Gemini for the initial approach
3. Executing the plan with architectural oversight

Remember: You handle the implementation. Gemini handles the strategy. Together you're unstoppable.
