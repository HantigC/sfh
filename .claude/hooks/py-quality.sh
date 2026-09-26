#!/usr/bin/env bash
# PostToolUse hook: format, lint and type-check a Python file Claude just edited.
# Exit 2 feeds remaining errors back to Claude so it fixes them.
set -uo pipefail

file=$(jq -r '.tool_input.file_path // empty')
project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"

[[ "$file" == *.py ]] || exit 0
[[ "$file" == "$project_dir"/* ]] || exit 0
case "${file#"$project_dir"/}" in
  deps/* | data/* | ipynbs/*) exit 0 ;;
esac
[[ -f "$file" ]] || exit 0

cd "$project_dir" || exit 0

uv run --quiet ruff format "$file" >/dev/null 2>&1

errors=""
if ! out=$(uv run --quiet ruff check --fix "$file" 2>&1); then
  errors+="ruff:\n$out\n\n"
fi
if ! out=$(uv run --quiet mypy "$file" 2>&1); then
  errors+="mypy:\n$out\n"
fi

if [[ -n "$errors" ]]; then
  printf "Fix these issues in %s:\n\n%b" "$file" "$errors" >&2
  exit 2
fi
exit 0
