---
description: "Run Prettier to validate formatting when editing YAML files"
globs:
  - "**/*.yaml"
  - "**/*.yml"
alwaysApply: false
---

# YAML Formatting with Prettier

After editing any YAML file, you MUST validate formatting with Prettier.

## Instructions

1. Run the lint script:

   @scripts/lint.sh

2. If Prettier modifies files or reports errors:
   - Review the changes or error messages
   - Fix any issues in the YAML files
   - Re-run the script

3. Repeat until the command exits with code 0 (success)

## Important

- NEVER consider the task complete until Prettier passes
- Prettier may auto-format files; always verify the final state
