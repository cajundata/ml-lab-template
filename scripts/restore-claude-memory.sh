#!/usr/bin/env bash
# Re-seed the LIVE Claude Code project memory from the committed snapshot.
# Run this once on a NEW machine after `git clone`, so Claude Code picks up this
# project's memory (progress, preferences, GPU account-reality) from the start.
#
# The destination path is derived from THIS clone's absolute location, so it works
# even if the new machine clones to a different path than the original.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
key="$(printf '%s' "$repo" | sed 's#/#-#g')"
src="$repo/docs/claude-memory"
dest="$HOME/.claude/projects/$key/memory"

if [ ! -d "$src" ] || ! ls "$src"/*.md >/dev/null 2>&1; then
  echo "No committed snapshot at: $src" >&2
  echo "(Run scripts/sync-claude-memory.sh on the source machine first.)" >&2
  exit 1
fi

mkdir -p "$dest"
cp -v "$src"/*.md "$dest"/
echo
echo "Restored memory: $src -> $dest"
echo "Claude Code will load it for this project from the next session."
