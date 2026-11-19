#!/usr/bin/env bash
# Creates the public GitHub repository only after the authors confirm data rights.
set -euo pipefail

REPOSITORY="${1:-tsad-calibration-study}"
TAG="v0.1.0-paa-candidate"

if [[ "${CONFIRM_DERIVED_DATA_RIGHTS:-}" != "YES" ]]; then
  cat >&2 <<'EOF'
Refusing to create a public repository. Confirm the right to redistribute every
score/label derivative by setting CONFIRM_DERIVED_DATA_RIGHTS=YES in your own
terminal after reviewing DATA_RIGHTS_REVIEW.csv and THIRD_PARTY_NOTICES.md.
EOF
  exit 2
fi

command -v gh >/dev/null || {
  echo "GitHub CLI is required. Install it, then run: gh auth login" >&2
  exit 3
}
gh auth status

if [[ -n "$(git remote)" ]]; then
  echo "A git remote already exists. Review it before pushing." >&2
  git remote -v >&2
  exit 4
fi

gh repo create "$REPOSITORY" --public --source=. --remote=origin --push
git tag -a "$TAG" -m "PAA public score-level release candidate"
git push origin "$TAG"
echo "Remote repository and tag created. Create a Zenodo new-version draft next; do not cite a DOI until the record is published and files are verified."
