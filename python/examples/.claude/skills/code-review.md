---
name: code-review
description: Use when reviewing code for quality, security, or performance issues.
when_to_use: User asks for a code review, audit, or critique.
tags: [review, quality]
---

## Code Review Checklist

When reviewing code, systematically check:

1. **Correctness**
   - Edge cases (empty input, max size, unicode, etc.)
   - Error paths — every `raise` has a corresponding `except`
   - Off-by-one errors in loops/ranges
   - Mutable default arguments (Python classic foot-gun)

2. **Resource safety**
   - File handles, sockets, DB connections — all closed on error paths
   - Context managers (`with`) for everything that supports them
   - No leaked subprocesses / async tasks

3. **Security**
   - User input never concatenated into shell commands or SQL
   - Secrets not logged or committed
   - Path traversal: any path from user input must be normalized + checked

4. **Performance**
   - O(n²) loops over large collections — flag and suggest
   - Unnecessary I/O inside hot paths
   - String concatenation in loops (use `"".join`)

5. **Style**
   - Naming is self-documenting
   - Functions do one thing; if "and" appears in the name, split it
   - Comments explain *why*, not *what*
