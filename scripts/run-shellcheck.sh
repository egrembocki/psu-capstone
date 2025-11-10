#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if ! command -v shellcheck >/dev/null 2>&1; then
  echo "shellcheck not installed. Install it (brew install shellcheck / apt install shellcheck)."
  exit 1
fi

# If filenames were provided by pre-commit, use them
if [ "$#" -gt 0 ]; then
  shellcheck -x -S error "$@"
  exit $?
fi

# No filenames passed: find tracked shell scripts (allow common extensions)
files=$(git ls-files -- '*.sh' '*.bash' '*.zsh' || true)

if [ -z "$files" ]; then
  echo "No shell files to check."
  exit 0
fi

# Run shellcheck on the list (note: filenames with newlines/spaces are uncommon for scripts)
echo "$files" | xargs shellcheck -x -S error
