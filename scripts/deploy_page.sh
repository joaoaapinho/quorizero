#!/usr/bin/env bash
# Deploy the page from this machine with the live model, not whatever frontend/model
# holds: after a retrain the bucket has a newer one. Use this, never the bare
# `wrangler pages deploy`, once the retrain job runs.
#
#   QZ_BUCKET=gs://<bucket>/runs bash scripts/deploy_page.sh
#   bash scripts/deploy_page.sh        # before the cloud job exists: frontend/ as it is
#
# Needs the cloud extra (pip install -e ".[cloud]") and gcloud auth application-default login.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${QZ_RUN:-pilot-final}

if [ -n "${QZ_BUCKET:-}" ]; then
  run=$(mktemp -d)
  python scripts/sync_run.py down "$QZ_BUCKET/$RUN" "$run"
  if [ -f "$run/best.txt" ]; then  # a promoted retrain: that is the live model
    best=$(cat "$run/best.txt")
    python -m model.export "$run/$best" frontend/model --version "$RUN/$best"
  else
    echo "no retrain promoted yet, keeping frontend/model"
  fi
  rm -rf "$run"
fi

# the retrain job deploys the page code on GitHub main, so local-only changes get undone
git fetch -q origin main || true
if ! git diff --quiet origin/main -- frontend; then
  echo "warning: frontend/ differs from GitHub main, push it or the next retrain undoes it"
fi

echo "deploying model $(cat frontend/model/version.json)"
npx wrangler pages deploy frontend --project-name "${QZ_PAGES_PROJECT:-quorizero}" --branch main
