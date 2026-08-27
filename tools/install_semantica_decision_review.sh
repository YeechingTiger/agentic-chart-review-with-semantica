#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
semantica_repo="https://github.com/semantica-agi/semantica.git"
semantica_ref="f187d4b5da5027618f8af5689006a8105efb5ec9"
patch_path="$repo_root/patches/semantica-decision-review-f187d4b.patch"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/acr-semantica.XXXXXX")"

cleanup() {
  rm -rf "$temporary_root"
}
trap cleanup EXIT

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v npm >/dev/null || { echo "npm is required to build Semantica Explorer" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
test -f "$patch_path" || { echo "missing patch: $patch_path" >&2; exit 1; }

source_dir="$temporary_root/semantica"
git clone --quiet --filter=blob:none --no-checkout "$semantica_repo" "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$semantica_ref"
git -C "$source_dir" apply --check "$patch_path"
git -C "$source_dir" apply "$patch_path"

(
  cd "$source_dir/explorer"
  npm ci
  npm run build
)

cd "$repo_root"
uv pip install --no-deps --force-reinstall "$source_dir"
uv run --no-sync python -c \
  "from semantica.explorer.decision_review import NARRATIVE_SCHEMA; print(NARRATIVE_SCHEMA)"

echo "Installed Semantica $semantica_ref with native decision narrative review."
