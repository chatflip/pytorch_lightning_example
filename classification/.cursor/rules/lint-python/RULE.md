---
description: "Run formatter and linter when editing Python files"
globs:
  - "**/*.py"
alwaysApply: false
---

# Python Formatting and Linting

After editing any Python file, you MUST run formatter and linter.

## Instructions

1. Run the format and lint script:

   @scripts/lint.sh

2. If formatter modifies files or linter reports errors:
   - Review the changes or error messages
   - Fix any issues in the Python files
   - Re-run the script

3. Repeat until both commands exit with code 0 (success)

## Important

- NEVER consider the task complete until both formatter and linter pass
- Formatter may auto-format files; always verify the final state
- Do not return to user until all checks pass
