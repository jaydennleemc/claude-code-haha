---
name: commit-message
description: Use when writing or improving a git commit message.
---

## Commit Message Format

```
<type>(<scope>): <subject>     ← 50 chars, imperative, no period

<body — wrap at 72 chars; explain what & why, not how>     ← blank line above

<footer — references, breaking changes, co-authors>
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`

**Subject rules**:
- Imperative mood: "add" not "added"
- No trailing period
- Lowercase after the type
- The subject completes the sentence: *"This commit will ___."*

**Body rules**:
- Explain motivation — what problem, why this approach
- Reference issues/PRs: `Closes #123`, `Refs #456`
- For breaking changes, start the body with `BREAKING CHANGE:`
