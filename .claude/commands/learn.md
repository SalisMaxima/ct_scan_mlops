---
description: Extract learnings from this conversation and save to project docs
---

# Extract Learnings

Analyze this conversation and extract reusable insights that would help future sessions. Focus on:

1. **Gotchas discovered** - Bugs, edge cases, or unexpected behaviors found
2. **Architecture decisions** - Why something was built a certain way
3. **Patterns used** - Reusable patterns specific to this codebase
4. **Commands/workflows** - Non-obvious commands or multi-step processes

## Output Format

For each learning, write a concise entry with:
- **What**: One-line summary
- **Why**: Brief context (1-2 sentences max)
- **Example**: Code snippet or command if applicable

## Where to Save

Append learnings to `.claude/docs/learnings.md`. Create the file if it doesn't exist.

Use this format:
```markdown
## [Category] Title
What was learned and why it matters.
```

Categories: `Gotcha`, `Pattern`, `Architecture`, `Workflow`

Only save genuinely useful insights - skip trivial or one-off fixes.
