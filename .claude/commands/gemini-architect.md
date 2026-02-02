---
description: Consult Gemini Pro as a lead architect for strategic decisions
argument-hint: <question about architecture, design, or strategy>
allowed-tools: Bash(gemini*)
---

You are consulting Gemini Pro as your **Lead Software Architect**. Gemini excels at high-level strategic thinking, architecture decisions, and providing guidance that keeps implementations on track.

## Your Question for the Architect

**$ARGUMENTS**

## Instructions

1. **Format the question** for maximum architectural insight:
   - Include relevant context about the current state
   - Mention constraints (performance, maintainability, ML-specific concerns)
   - Ask for trade-offs and recommendations

2. **Execute the consultation** using the Gemini CLI:

```bash
gemini -p "You are a Lead Software Architect reviewing a CT scan ML classification project (PyTorch Lightning, Hydra configs, W&B tracking).

Question: $ARGUMENTS

Provide:
1. Your recommended approach with rationale
2. Key trade-offs to consider
3. Potential pitfalls to avoid
4. Implementation priorities (what to do first)

Be concise but thorough. Focus on architectural decisions, not implementation details."
```

3. **Interpret the response** and summarize:
   - Extract the key recommendation
   - Note any warnings or concerns
   - Identify actionable next steps

## Example Usages

- `/gemini-architect Should we use a dual-pathway CNN or single backbone for multi-class CT classification?`
- `/gemini-architect How should we structure the feature extraction pipeline for interpretability?`
- `/gemini-architect What's the best way to handle class imbalance in our 4-class tumor classification?`

## If Gemini CLI is Not Available

If the `gemini` command fails, inform the user they need to install Gemini CLI:
```bash
npm install -g @google/generative-ai-cli
# or
pip install google-generativeai
```
And configure their API key with `gemini auth login` or by setting `GEMINI_API_KEY`.
