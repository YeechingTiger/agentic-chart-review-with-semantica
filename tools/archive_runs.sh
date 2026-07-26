#!/usr/bin/env bash
# Move run directories into runs/_archive/<timestamp>/ instead of deleting them.
#
# Never `rm -rf runs/...`. Run output is an experimental record: a trace that documented a
# bug in code that has since been fixed can never be regenerated, because the code path that
# produced it no longer exists.
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="runs/_archive/$STAMP"
mkdir -p "$DEST"
moved=0
for d in "$@"; do
  [ -d "$d" ] || { echo "skip (not a directory): $d"; continue; }
  mv "$d" "$DEST/"; moved=$((moved+1)); echo "archived: $d -> $DEST/"
done
echo "$moved directory(ies) archived. Nothing was deleted."
