---
name: repo-consolidation
description: "Analyze repository directory structure and suggest consolidation opportunities to keep the codebase simple and manageable. Use when the repository has grown organically and needs structural cleanup."
trigger: "/consolidate"
---

# Repository Structure Consolidation Skill

## Purpose

Analyze the current repository directory structure and provide actionable recommendations to consolidate directories, reduce clutter, and maintain a clean, manageable codebase organization.

## When to Use This Skill

- After significant project growth or multiple feature additions
- When onboarding reveals confusing directory organization
- Before major refactoring efforts
- When preparing for project handoff or documentation
- Periodic maintenance (quarterly/semi-annual structure reviews)

## Instructions for the Agent

When this skill is invoked, launch the **repo-structure-enforcer** agent with the following comprehensive task:

### Phase 1: Discovery and Analysis

1. **Map Current Structure**
   - Use `tree -L 3 -d` or equivalent to get a comprehensive directory overview
   - Identify all top-level directories and their purposes
   - Document the current organizational philosophy (if discernible)

2. **Identify Redundancy and Clutter**
   - Find directories with similar/overlapping purposes
   - Locate nearly empty directories (< 3 files)
   - Identify single-purpose directories that could be consolidated
   - Flag directories with unclear naming or purpose

3. **Analyze Functional Separation**
   - Determine which directories have clear, distinct functional boundaries
   - Identify cases where separation is arbitrary vs. necessary
   - Review module coupling between directories

4. **Review Dependencies**
   - Use `grep -r "from .* import\|import .*" --include="*.py"` to map import relationships
   - Identify which directories are tightly coupled
   - Find circular dependencies or unexpected cross-directory imports
   - Document which consolidations would break imports

### Phase 2: Generate Consolidation Plan

Create a detailed consolidation proposal including:

1. **Directories to Merge**
   - Source directory → Target directory mappings
   - Justification for each merge (reduce clutter, similar purpose, etc.)
   - Impact assessment (number of files affected, import changes needed)

2. **Directories to Keep Separated**
   - Explicit justification for why certain directories should remain separate
   - Functional boundaries that should be preserved
   - Architectural reasons for separation

3. **New Directory Structure** (if needed)
   - Proposed simplified hierarchy
   - Before/after comparison
   - Directory purpose documentation

4. **Migration Actions Required**
   - Specific `git mv` commands for each file/directory move
   - Import statement updates needed (with file locations)
   - Configuration file updates (paths in configs, tasks.py, etc.)
   - Documentation updates (README, Structure.md, etc.)

### Phase 3: Risk Assessment

For EACH proposed change, meticulously document:

1. **Import Breakage Analysis**
   - List all Python files that import from affected directories
   - Show exact import statements that will need updating
   - Identify any circular dependency risks

2. **Configuration Impact**
   - Check `tasks.py` for hardcoded paths
   - Review `pyproject.toml`, `setup.py` for package structure references
   - Examine config files (YAML, JSON, TOML) for path dependencies
   - Check `.gitignore`, `.dockerignore` for path patterns

3. **CI/CD Impact**
   - Review GitHub Actions workflows for path-specific jobs
   - Check Docker build contexts and COPY commands
   - Examine test discovery patterns

4. **External Dependencies**
   - DVC file paths and data versioning
   - W&B artifact paths and experiment tracking
   - Pre-commit hook configurations

### Phase 4: User Confirmation

Before making ANY changes:

1. **Present the Plan**
   ```
   ## Repository Consolidation Plan

   ### Summary
   - X directories will be merged
   - Y import statements require updating
   - Z configuration files need changes

   ### Proposed Changes
   [Detailed list with justifications]

   ### Risk Assessment
   [Comprehensive dependency analysis]

   ### Estimated Effort
   - File moves: N files
   - Import updates: N files
   - Config updates: N files
   - Documentation updates: N files
   ```

2. **Ask for Explicit Approval**
   Use the AskUserQuestion tool with options:
   - "Approve entire plan - proceed with all changes"
   - "Approve with modifications - I'll specify which parts to do"
   - "Review individual changes - ask me before each directory merge"
   - "Reject - keep current structure"

3. **Wait for Response**
   - DO NOT make any file moves without explicit approval
   - If modifications requested, revise plan and re-confirm
   - If individual review requested, confirm each change separately

### Phase 5: Implementation (Only After Approval)

1. **Pre-Implementation Checklist**
   - Ensure working directory is clean (`git status`)
   - Create a new branch for consolidation work
   - Run tests to establish baseline (all passing)

2. **Execute Changes in Order**

   **Step 1: Move Files**
   - Use `git mv` to preserve history
   - Move entire directories when possible
   - For partial moves, move files individually

   **Step 2: Update Imports**
   - Update all `import` and `from ... import` statements
   - Use automated find/replace where safe
   - Manually verify complex import patterns

   **Step 3: Update Configurations**
   - Update `tasks.py` path references
   - Update config files (YAML, JSON, TOML)
   - Update `.gitignore` patterns if needed
   - Update `pyproject.toml` package structure

   **Step 4: Update Documentation**
   - Update `docs/Structure.md` with new organization
   - Update README.md if it references directory structure
   - Update any inline code comments with old paths

3. **Verification After Each Major Change**
   - Run `invoke ruff` to catch any linting errors
   - Run full test suite: `uv run pytest`
   - Check for import errors: `python -c "import <module>"`
   - Manually test key functionality

4. **Final Validation**
   - Full test suite passes
   - All imports resolve correctly
   - No broken path references in configs
   - Documentation updated
   - `git status` shows only intended changes

### Phase 6: Documentation and Handoff

Create a summary report:

```markdown
# Repository Consolidation Summary

## Changes Made
- [List of directory merges]
- [Files moved: before → after paths]
- [Import updates: N files modified]
- [Config updates: files and changes]

## New Structure Benefits
- [Reduced directory count: X → Y]
- [Clearer functional separation]
- [Improved navigability]

## Migration Notes
- [Breaking changes for contributors]
- [New import patterns]
- [Updated documentation locations]

## Verification Checklist
- [x] All tests passing
- [x] Imports resolve
- [x] Configs updated
- [x] Documentation current
```

## Example Consolidation Scenarios

### Scenario 1: Multiple Script Directories

**Before:**
```
scripts/
utils/
tools/
helpers/
```

**Analysis:** All contain utility scripts with no clear functional difference.

**Recommendation:** Consolidate into `scripts/` with optional subdirectories if needed:
```
scripts/
  data/         # Data processing utilities
  analysis/     # Analysis utilities
  deployment/   # Deployment helpers
```

### Scenario 2: Overlapping Test Directories

**Before:**
```
tests/
test/
unit_tests/
integration_tests/
```

**Analysis:** Multiple test directories create confusion about test organization.

**Recommendation:** Consolidate into single `tests/` with subdirectories:
```
tests/
  unit/
  integration/
  fixtures/
```

### Scenario 3: Documentation Sprawl

**Before:**
```
docs/
documentation/
guides/
README.md
CONTRIBUTING.md
notes/
planning/
```

**Analysis:** Documentation scattered across multiple directories.

**Recommendation:** Consolidate into `docs/` with clear structure:
```
docs/
  guides/
  api/
  planning/
  CONTRIBUTING.md (symlink to root if needed)
```

## Safety Guardrails

### Never Consolidate Without Justification

These directories should typically remain separate (require strong justification to merge):

- `src/` and `tests/` - Core separation between production and test code
- `data/` and `outputs/` - Input vs. generated artifacts
- `configs/` and `src/` - Configuration vs. code
- `.github/` and application code - CI/CD isolation
- `scripts/` and `src/` - Utilities vs. core application (usually)

### Always Preserve Git History

- Use `git mv` instead of `mv` + `git add`
- Commit directory moves separately from content changes
- Use descriptive commit messages explaining the consolidation

### Test Rigorously

After consolidation:
- Full test suite must pass
- Import statements must resolve
- CLI commands must work
- Docker builds must succeed
- CI/CD pipeline must pass

## Output Format

The agent should provide a structured report:

1. **Executive Summary** (1-2 paragraphs)
2. **Current Structure Analysis** (directory tree + issues found)
3. **Consolidation Recommendations** (specific, actionable)
4. **Dependency Impact Analysis** (comprehensive risk assessment)
5. **Implementation Plan** (step-by-step with commands)
6. **User Approval Request** (clear options)

Only proceed with implementation after explicit user approval.
