#!/usr/bin/env bash
# Snapshot the LIVE Claude Code project memory into the repo so it travels with a clone.
# Run this before switching machines / committing, then `git add docs/claude-memory`.
#
# The live memory lives at ~/.claude/projects/<abs-repo-path-with-/-as->/memory/ and is
# NOT tracked by git on its own. Restore it on a new machine with restore-claude-memory.sh.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
key="$(printf '%s' "$repo" | sed 's#/#-#g')"
src="$HOME/.claude/projects/$key/memory"
dest="$repo/docs/claude-memory"

if [ ! -d "$src" ]; then
  echo "No live memory found at: $src" >&2
  echo "(Nothing to snapshot — has this project accumulated memory yet?)" >&2
  exit 1
fi

mkdir -p "$dest"
rm -f "$dest"/*.md
cp -v "$src"/*.md "$dest"/
echo
echo "Snapshotted memory: $src -> $dest"
echo "Now: git add docs/claude-memory && git commit"
