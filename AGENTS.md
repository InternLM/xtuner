# AGENTS.md - Project Guidelines for AI Assistants

This file contains project-specific guidelines and conventions for AI assistants working on this codebase.

## Git Commit Message Guidelines

### Format

```
[<Type>] <Short summary>

<Long description explaining what and why (not how)>

- Bullet point for specific changes
- Another bullet point
```

### Types

- `[Feature]` - New feature
- `[Fix]` - Bug fix
- `[Refactor]` - Code refactoring
- `[Docs]` - Documentation changes
- `[Test]` - Test changes
- `[Chore]` - Build/tooling changes

### Guidelines

1. **Short summary**: Concise description of the change (50 chars or less)
2. **Long description**: Explain **what** changed and **why**, not **how**
3. **No bullet points**: Do not list specific changes in commit message
4. **No file lists**: Do not include file names or "Files modified:" section
5. **Keep it brief**: Only high-level functional description, details go to PR description

### Example

```
[Fix] Muon optimizer per-expert orthogonalization for MoE models

Fix Muon optimizer to apply orthogonalization per expert matrix instead of
to the concatenated large matrix for MoE models.
```

## PR Description Guidelines

The PR description should contain:

1. **Summary**: Brief overview of the changes
2. **Motivation**: Why this change is needed
3. **Changes**: Detailed list of what changed
4. **Files modified**: List of files changed
5. **Testing**: How the changes were tested

## Code Style

- Follow existing code style in the project
- Add type hints for new functions
- Add docstrings for public functions and classes
- Keep functions focused and small
