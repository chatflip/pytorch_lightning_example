---
description: "Run markdownlint to validate formatting when editing Markdown files"
globs:
  - "**/*.md"
alwaysApply: false
---

# Markdown Formatting with markdownlint

After editing any Markdown file, you MUST validate formatting with markdownlint.

## Instructions

1. Run the lint script:

   @scripts/lint.sh

2. If markdownlint modifies files or reports errors:
   - Review the changes or error messages
   - Fix any issues in the Markdown files
   - Re-run the script

3. Repeat until the command exits with code 0 (success)

## Important

- NEVER consider the task complete until markdownlint passes
- markdownlint with `--fix` may auto-format files; always verify the final state
