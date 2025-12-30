---
description: "Run tests after modifying test files or source code"
globs:
  - "tests/**/*.py"
  - "src/**/*.py"
alwaysApply: false
---

# Testing with pytest

After editing test files or source code, you MUST run tests to verify changes.

## Instructions

1. Run the test script:

   @scripts/test.sh

2. If tests fail:
   - Review the error messages
   - Fix any issues in the code
   - Re-run the script

3. Repeat until all tests pass (exit code 0)

## Important

- NEVER consider the task complete until all tests pass
- Check test coverage if applicable
