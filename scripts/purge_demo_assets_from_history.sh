#!/usr/bin/env bash
# Strip the demo's heavy GENERATED assets out of every commit, shrinking the repo.
# The site stays in source control — only the ~123 MB that scripts/build_demo.py can
# regenerate is removed:
#
#     demo/data/rollouts/**      demo/data/carry_*      demo/data/subleq_fib.json
#
# The small hand-exported weight JSONs (adder8.json, gates, manifest.json, ...) and
# all HTML/CSS/JS/assets are KEPT.
#
# THIS REWRITES HISTORY. Every commit SHA after the first touched commit changes, so
# the remote needs a force-push and existing clones/forks must re-clone. Read the
# whole script before running it.
#
#     bash scripts/purge_demo_assets_from_history.sh --yes
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [[ "${1:-}" != "--yes" ]]; then
  echo "This rewrites every commit and requires a force-push. Re-run with --yes to proceed."
  exit 1
fi
command -v git-filter-repo >/dev/null || { echo "git-filter-repo not on PATH"; exit 1; }

# 1. filter-repo refuses to touch a dirty tree, and it ends in a hard reset — commit first.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Working tree has uncommitted changes. Commit or stash them first:"
  git status --short --untracked-files=no
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="../ncpu-purge-backup-${STAMP}"
mkdir -p "${BACKUP_DIR}"

# 2. Full recoverable backup of history, plus the generated assets themselves: the
#    hard reset at the end of filter-repo deletes files that are no longer in HEAD.
echo "==> backing up to ${BACKUP_DIR}"
git bundle create "${BACKUP_DIR}/history.bundle" --all
tar czf "${BACKUP_DIR}/demo-data.tar.gz" demo/data
git remote -v >"${BACKUP_DIR}/remotes.txt"
du -sh "${BACKUP_DIR}"/*

BEFORE=$(du -sm .git | cut -f1)

# 3. The rewrite. --invert-paths turns the list into "remove these".
echo "==> rewriting history"
git filter-repo --force --invert-paths \
  --path demo/data/rollouts \
  --path demo/data/subleq_fib.json \
  --path-glob 'demo/data/carry_*'

# 4. filter-repo drops the remote so you cannot push by reflex. Put it back.
if ! git remote | grep -qx origin; then
  git remote add origin "$(awk '/^origin.*\(fetch\)/{print $2; exit}' "${BACKUP_DIR}/remotes.txt")"
fi

# 5. Restore the generated assets (now gitignored) so the local demo still runs.
echo "==> restoring demo/data working files"
tar xzf "${BACKUP_DIR}/demo-data.tar.gz"

git reflog expire --expire=now --all
git gc --prune=now --aggressive
AFTER=$(du -sm .git | cut -f1)

echo
echo "==> .git: ${BEFORE} MB -> ${AFTER} MB"
echo "Verify the demo still builds and the site still loads, then publish with:"
echo
echo "    git push --force-with-lease origin $(git rev-parse --abbrev-ref HEAD)"
echo
echo "Backup kept at ${BACKUP_DIR} — delete it once you are happy."
echo "To undo before pushing: rm -rf .git && git clone ${BACKUP_DIR}/history.bundle recovered"
